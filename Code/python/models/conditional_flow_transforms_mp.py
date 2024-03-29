### Conditional flow transforms based on AC flow:
# AC Flow: https://arxiv.org/abs/1909.06319
# Removing unnecessary optimization/scaffolding

import copy
import numpy as np
import scipy
import torch
from torch import nn, optim
import time

from models import torch_models, flow_likelihood, flow_transforms
from models.model_base import ModelException, BaseConfig
from utils import math_utils, torch_utils


import IPython


# Options for designing conditional transforms:
# 1. Meta parameterization:
#    F_o(x_o) gives parameters \theta of function F_u(x_u):
#    -    z | x_o = F_u(x_u; F_o(x_o))
# 2. Joint parameterization:
#    F(x_u, x_o) takes both parameters:
#    -    z | x_o = F(x_u, x_o)

# Here, we do (1) -- meta parameterization.


_BASE_DISTS = flow_transforms._BASE_DISTS
# Utils:


################################################################################
class CTfmMPConfig(BaseConfig):
  def __init__(
      self, tfm_type="scale_shift_coupling", neg_slope=0.01, is_sigmoid=False,
      func_nn_config=None, meta_parameters=False, has_bias=True,
      base_dist="gaussian", reg_coeff=.1, lr=1e-3, batch_size=50,
      max_iters=1000, stopping_eps=1e-5, num_stopping_iter=10, grad_clip=5.,
      verbose=True, *args, **kwargs):

    super(CTfmMPConfig, self).__init__(*args, **kwargs)

    self.tfm_type = tfm_type.lower()
    # LeakyReLU params:
    self.neg_slope = neg_slope
    # Sigmoid/logit params:
    self.is_sigmoid = is_sigmoid

    # Shift Scale params:
    self.func_nn_config = func_nn_config
    # self.shift_config = shift_config
    # self.shared_wts = shared_wts

    # Flag for whether to have two-step parameter generation for conditioning
    # or not. True means two step.
    self.meta_parameters = meta_parameters


    # Fixed Linear Transform params:
    self.has_bias = has_bias

    # Some misc. parameters if training transforms only
    self.base_dist = "gaussian"
    self.reg_coeff = reg_coeff
    self.lr = lr
    self.batch_size = batch_size
    self.max_iters = max_iters
    self.stopping_eps = stopping_eps
    self.num_stopping_iter = num_stopping_iter
    self.grad_clip = grad_clip

    self.verbose = verbose


# Functions of the form q(x_u | x_o)
class ConditionalInvertibleTransformMP(flow_transforms.InvertibleTransform):
  def __init__(self, config):
    super(flow_transforms.InvertibleTransform, self).__init__()
    self.config = config
    self._dim = None

  def initialize(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def _get_params(self, x_o, rtn_torch=True):
    raise NotImplementedError("Abstract class method")

  def forward(self, x, x_o, rtn_torch=True, rtn_logdet=False):
    raise NotImplementedError("Abstract class method")

  def inverse(self, z, x_o):
    raise NotImplementedError("Abstract class method")

  def log_prob(self, x, x_o):
    z, log_det = self(x, x_o, rtn_torch=True, rtn_logdet=True)
    return self.base_log_prob(z) + log_det

  def loss_nll(self, z, log_jac_det):
    z_ll = self.base_log_prob(z)
    nll_orig = -torch.mean(log_jac_det + z_ll)
    return nll_orig

  def loss_err(self, x_tfm, y, log_jac_det):
    recon_error = self.recon_criterion(x_tfm, y)
    logdet_reg = -torch.mean(log_jac_det)
    loss_val = recon_error + self.config.reg_coeff * torch.abs(logdet_reg)
    return loss_val

  def _train_loop(self):
    shuffle_inds = np.random.permutation(self._npts)
    x = self._x[shuffle_inds]
    x_o = self._x_o[shuffle_inds]
    y = None if self._y is None else self._y[shuffle_inds]

    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      x_batch = x[b_start:b_end]
      x_o_batch = x_o[b_start:b_end]
      x_tfm_batch, jac_logdet = self.forward(
          x_batch, x_o_batch, rtn_torch=True, rtn_logdet=True)
      if y is not None:
        y_batch = y[b_start:b_end]

      self.opt.zero_grad()
      loss_val = (
          # Reconstruction loss
          self.loss_err(x_tfm_batch, y_batch, jac_logdet)
          if y is not None else
          # Generative model -- log likelihood loss
          self.loss_nll(x_tfm_batch, jac_logdet)
      )

      loss_val.backward()
      self._avg_grad = self._get_avg_grad_val()
      self._max_grad = self._get_max_grad_val()
      nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
      self.opt.step()

      self.itr_loss += loss_val * x_batch.shape[0]
    self.itr_loss /= self._npts

    curr_loss = float(self.itr_loss.detach())
    if np.abs(self._prev_loss - curr_loss) < self.config.stopping_eps:
      self._stop_iters += 1
      if self._stop_iters >= self.config.num_stopping_iter:
        self._finished_training = True
    else:
      self._stop_iters = 0
    self._prev_loss = curr_loss

  def _get_avg_grad_val(self):
    # Just for debugging
    p_grads = [p.grad for p in self.parameters()]
    num_params = sum([pg.numel() for pg in p_grads])
    abs_sum_grad = sum([pg.abs().sum() for pg in p_grads])
    return abs_sum_grad / num_params

  def _get_max_grad_val(self):
    # Just for debugging
    p_grads = [p.grad for p in self.parameters()]
    return max([pg.abs().max() for pg in p_grads])

  def fit(self, x, x_o, y=None, mask_func=None, lhood_model=None):
    # Simple fitting procedure for transforming x to y
    if self.config.verbose:
      all_start_time = time.time()
    self._x = torch_utils.numpy_to_torch(x)
    self._x_o = torch_utils.numpy_to_torch(x_o)
    self._y = None if y is None else torch_utils.numpy_to_torch(y)
    self._npts, self._dim = self._x.shape

    self._mask_func = mask_func

    if y is None:
      if lhood_model is None:
        if self.config.base_dist not in _BASE_DISTS:
          raise NotImplementedError(
              "Base dist. type %s not implemented and likelihood model"
              " not provided." % self.config.base_dist)
        else:
          loc, scale = torch.zeros(self._dim), torch.eye(self._dim)
          self.base_dist = torch.distributions.MultivariateNormal(loc, scale)
      else:
        self.base_dist = lhood_model

      self.recon_criterion = None
      self.base_log_prob = (
          lambda Z: self.base_dist.log_prob(torch_utils.numpy_to_torch(Z)))
    else:
      self.recon_criterion = nn.MSELoss(reduction="mean")
      self.base_log_prob = None

    self.opt = optim.Adam(self.parameters(), self.config.lr)
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))

    self._finished_training = False
    self._loss_history = []
    self._avg_grad_history = []
    self._prev_loss = np.inf
    self._stop_iters = 0
    try:
      itr = -1
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()

        self._train_loop()
        self._loss_history.append(float(self.itr_loss.detach()))
        self._avg_grad_history.append(float(self._get_avg_grad_val().detach()))

        if self._finished_training:
          if self.config.verbose:
            print("\nLoss change < stopping eps for multiple iters."
                  "Finished training.")
          break
        if self.config.verbose:
          itr_diff_time = time.time() - itr_start_time
          loss_val = float(self.itr_loss.detach())
          print("Iteration %i out of %i (in %.2fs). Loss: %.5f. "
                "Avg/Max grad: %.5f / %.5f. Stop iter: %i" %
                (itr + 1, self.config.max_iters, itr_diff_time, loss_val,
                  self._avg_grad, self._max_grad, self._stop_iters),
                end='\r')
      if self.config.verbose and itr >= 0:
        print("Iteration %i out of %i (in %.2fs). Loss: %.5f. Avg grad: %.5f."
              "Stop iter: %i" % (itr + 1, self.config.max_iters,
                  itr_diff_time, loss_val, self._avg_grad, self._stop_iters))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
    return self

  def sample(self, x_o, n_samples=1, inverted=True, rtn_torch=False):
    if isinstance(n_samples, tuple):
      z_samples = self.base_dist.sample(*n_samples)
    else:
      z_samples = self.base_dist.sample(n_samples)
    if inverted:
      return self.inverse(z_samples, x_o, rtn_torch)
    return z_samples if rtn_torch else torch_utils.torch_to_numpy(z_samples)
  # def log_prob(self, x_u, x_o):
  #   z, log_det = self(x, rtn_torch=True, rtn_logdet=True)
  #   return self.base_log_prob(z) + log_det
 
  # def loss_nll(self, z, log_jac_det):
  #   z_ll = self.base_log_prob(z)
  #   nll_orig = -torch.mean(log_jac_det + z_ll)
  #   return nll_orig

  # def loss_err(self, x_tfm, x_u, log_jac_det):
  #   recon_error = self.recon_criterion(x_tfm, y)
  #   logdet_reg = -torch.mean(log_jac_det)
  #   loss_val = recon_error + self.config.reg_coeff * torch.abs(logdet_reg)
  #   return loss_val
  # def sample_z(self, x_o, n_samples=1, rtn_torch=False):
  #   pass

  # def sample_xu(self, x_o, n_samples=1, rtn_torch=False):
  #   pass
    # if isinstance(n_samples, tuple):
    #   z_samples = self.base_dist.sample(*n_samples)
    # else:
    #   z_samples = self.base_dist.sample(n_samples)
    # if inverted:
    #   return self.inverse(z_samples, rtn_torch)
    # return z_samples if rtn_torch else torch_utils.torch_to_numpy(z_samples)


# Simple transforms with no parameters:
class ReverseTransform(ConditionalInvertibleTransformMP):
  def __init__(self, config):
    super(ReverseTransform, self).__init__(config)

  def initialize(self, *args, **kwargs):
    pass

  def forward(self, x, x_o, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    reverse_idx = torch.arange(x.size(-1) -1, -1, -1).long()
    z = x.index_select(-1, reverse_idx)

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)

    # Determinant of Jacobian reverse transform is 1.
    return (z, 0.) if rtn_logdet else z

  def inverse(self, z, rtn_torch=True):
    return self(z, rtn_torch=rtn_torch, rtn_logdet=False)


class LeakyReLUTransform(ConditionalInvertibleTransformMP):
  def __init__(self, config):
    super(LeakyReLUTransform, self).__init__(config)

  def initialize(self, *args, **kwargs):
    neg_slope = self.config.neg_slope
    self._relu_func = torch.nn.LeakyReLU(negative_slope=neg_slope)
    self._inv_func = torch.nn.LeakyReLU(negative_slope=(1. / neg_slope))
    self._log_slope = np.log(np.abs(neg_slope))

  def forward(self, x, x_o, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    z = self._relu_func(x)

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)
    if rtn_logdet:
      neg_elements = torch.as_tensor((x < 0), dtype=torch_utils._DTYPE)
      jac_logdet = neg_elements.sum(1) * self._log_slope
      return z, jac_logdet
    return z

  def inverse(self, z, x_o, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x = self._inv_func(z)

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


class SigmoidLogitTransform(ConditionalInvertibleTransformMP):
  def __init__(self, config):
    super(SigmoidLogitTransform, self).__init__(config)

  def initialize(self, *args, **kwargs):
    pass

  def _sigmoid(self, x, rtn_logdet=False):
    y = torch.sigmoid(x)
    if rtn_logdet:
      jac = torch.abs(y * (1 - y))
      log_det = torch.log(jac)
      return y, log_det

    return y

  def _logit(self, x, rtn_logdet=False):
    x_odds = x / (1 - x)
    y = torch.log(x_odds)
    if rtn_logdet:
      jac = torch.abs(1 / ((x - 1) * (x)))
      log_det = torch.log(jac)
      return y, log_det

    return y

  def forward(self, x, x_o, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    
    y = (
        self._sigmoid(x, rtn_logdet=rtn_logdet)
        if self.config.is_sigmoid else
        self._logit(x, rtn_logdet=rtn_logdet))

    if rtn_logdet:
      y, jac_logdet = y
      y = y if rtn_torch else torch_utils.torch_to_numpy(y)
      return y, jac_logdet

    y = y if rtn_torch else torch_utils.torch_to_numpy(y)
    return y

  def inverse(self, y, x_o, rtn_torch=True):
    x = (
        self._logit(y, rtn_logdet=False)
        if self.config.is_sigmoid else
        self._sigmoid(y, rtn_logdet=False))
    x = x if rtn_torch else torch_utils.torch_to_numpy(x)
    return x


# Class for function parameters for unobserved covariates given observed
# covariates
_FUNC_TYPES = ["linear", "scale_shift"]
class FunctionParamNet(nn.Module):
  def __init__(self, nn_config, *args, **kwargs):
    self.config = nn_config
    super(FunctionParamNet, self).__init__(*args, **kwargs)

  def initialize(self, func_type, input_dims, output_dims, **kwargs):
    if func_type not in _FUNC_TYPES:
      raise ModelException("Unknown function type %s." % func_type)

    self.func_type = func_type
    self.input_dims = input_dims
    self.output_dims = output_dims

    if self.config.use_vae:
      print("Warning: Disabling VAE for param network.")
    self.config.use_vae = False

    if func_type == "linear":
      mat_size = output_dims ** 2
      self._has_bias = kwargs.get("bias", False)
      if self._has_bias:
        mat_size += output_dims
      self.config.set_sizes(input_size=input_dims, output_size=mat_size)
      self._param_net = torch_models.MultiLayerNN(self.config)
    elif func_type == "scale_shift":
      hidden_sizes = kwargs.get("hidden_sizes", None)
      fixed_dims = kwargs.get("fixed_dims", None)
      if hidden_sizes is None:
        raise ModelException("Need hidden sizes for scale-shift function.")
      if fixed_dims is None:
        raise ModelException("Need fixed dim size for scale-shift function.")
      self.hidden_sizes = hidden_sizes
      self.fixed_dims = fixed_dims
      # activations = kwargs.get("activations", torch_models._IDENTITY)
      activations = kwargs.get("activations", nn.Tanh())
      # Either single activation function or a list of activations of the same
      # size as len(hidden_sizes) + 1
      self.activations = activations

      self.config.set_sizes(input_size=input_dims)
      # The output dim is doubled, one set for (scale) and one set for (shift)
      all_sizes = [fixed_dims] + hidden_sizes + [output_dims * 2]
      param_sizes = [
          all_sizes[i] * all_sizes[i+1] for i in range(len(all_sizes) - 1)]
      self._param_net = torch_models.MultiOutputMLNN(self.config, param_sizes)

  def _get_lin_params(self, x):
    # output: n_pts x mat_size x mat_size
    lin_params = self._param_net(x)
    bias_dim = 1 if self._has_bias else 0
    lin_params = torch.reshape(
        lin_params, (-1, self.output_dims, self.output_dims + bias_dim))
    return lin_params

  def _get_ss_params(self, x):
    # try:
    ss_params = self._param_net(x)
    # except Exception as e:
    #   IPython.embed()
    #   raise(e)
    all_sizes = (
        [self.fixed_dims] + self.hidden_sizes + [self.output_dims * 2])
    ss_params = [
        torch.reshape(param, (-1, all_sizes[i + 1], all_sizes[i]))
        for i, param in enumerate(ss_params)
    ]
    return ss_params, self.activations

  def get_params(self, x):
    if self.func_type == "linear":
      return self._get_lin_params(x)
    elif self.func_type == "scale_shift":
      return self._get_ss_params(x)

  def forward(self, x):
    return self.get_params(x)


class ConditionalLinearTransformMP(ConditionalInvertibleTransformMP):
  # Model linear transform as L U matrix decomposition where L is lower
  # triangular with unit diagonal and U is upper triangular with arbitrary
  # non-zero diagonal elements.
  def __init__(self, config):
    super(ConditionalLinearTransformMP, self).__init__(config)

  def initialize(
      self, obs_dim, unobs_dim, nn_config, init_lin_param=None,
      init_bias_param=None, *args, **kwargs):

    self._param_net = FunctionParamNet(nn_config)
    self._obs_dim = obs_dim
    self._dim = unobs_dim

    self._param_net.initialize(
        func_type="linear", input_dims=obs_dim, output_dims=unobs_dim,
        bias=self.config.has_bias)
    # init_lin_param = (
    #     torch.eye(dim) if init_lin_param is None else
    #     torch_utils.numpy_to_torch(init_lin_param))
    # self._lin_param = torch.nn.Parameter(init_lin_param)

    # if self.config.has_bias:
    #   init_bias_param = (
    #       torch.zeros(dim) if init_bias_param is None else
    #       torch_utils.numpy_to_torch(init_bias_param))
    #   self._b = torch.nn.Parameter(init_bias_param)
    # else:
    #   self._b = torch.zeros(dim)

  def _get_params(self, x_o):
    lin_params = self._param_net.get_params(x_o)
    if self.config.has_bias:
      b = lin_params[:, :, -1]
      lin_params = lin_params[:, :, :-1]
    else:
      b = 0

    L, U = torch_utils.LU_split(lin_params)
    return L, U, b

  def forward(self, x, x_o, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    x_o = torch_utils.numpy_to_torch(x_o)
    L, U, b = self._get_params(x_o)

    try:
      # x_ = x.transpose(0, 1) if len(x.shape) > 1 else x.view(-1, 1)
      x_ = x.view(x.shape[0], -1, 1)
      z = L.matmul(U.matmul(x_)) + b.view(b.shape[0], -1, 1)
      # Undoing dimension changes to be consistent with input
      z = z.squeeze()
      # z = z.transpose(0, 1) if len(x.shape) > 1 else z.squeeze()
      z = z if rtn_torch else torch_utils.torch_to_numpy(z)
    except Exception as e:
      IPython.embed()
      raise(e)

    if rtn_logdet:
      # Determinant of triangular jacobian is product of the diagonals.
      # Since both L and U are triangular and L has unit diagonal,
      # this is just the product of diagonal elements of U.
      U_diag = torch.diagonal(U, offset=0, dim1=1, dim2=2)
      # IPython.embed()
      jac_logdet = torch.sum(torch.log(torch.abs(U_diag)), axis=1)
      return z, jac_logdet
    return z

  def inverse(self, z, x_o, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x_o = torch_utils.numpy_to_torch(x_o)
    L, U, b = self._get_params(x_o)

    try:
      z_ = z.view(z.shape[0], -1, 1)
      z_b = z_ - b.view(b.shape[0], -1, 1)
      # z_b_t = z_b.transpose(0, 1) if len(z_b.shape) > 1 else z_b
      # Torch solver for triangular system of equations
      # IPython.embed()
      sol_L = torch.triangular_solve(z_b, L, upper=False, unitriangular=True)[0]
      x = torch.triangular_solve(sol_L, U, upper=True)[0]
      # trtrs always returns 2-D output, even if input is 1-D. So we do this:
      # x = x_t.transpose(0, 1) if len(z.shape) > 1 else x_t.squeeze()
      x = x.squeeze()
      x = x if rtn_torch else torch_utils.torch_to_numpy(x)
    except Exception as e:
      IPython.embed()
      raise(e)

    return x


class ConditionalSSCTransformMP(ConditionalInvertibleTransformMP):
  def __init__(self, config, *args, **kwargs):
    super(ConditionalSSCTransformMP, self).__init__(config)
    # self.config = config

  def set_fixed_inds(self, index_mask):
    # Converting mask to boolean:
    index_mask = np.where(index_mask, 1, 0).astype(int)
    self._fixed_inds = np.nonzero(index_mask)[0]
    self._tfm_inds = np.nonzero(index_mask == 0)[0]

    self._fixed_dim = self._fixed_inds.shape[0]
    self._output_dim = self._tfm_inds.shape[0]

  def initialize(
      self, obs_dim, index_mask, hidden_sizes, activation, nn_config,
      *args, **kwargs):

    self._param_net = FunctionParamNet(nn_config)
    self._obs_dim = obs_dim
    self._dim = index_mask.shape[0]

    self.set_fixed_inds(index_mask)
    # The input size is the size of the observed covariates
    # The output size is the size of the partition of unobserved covariates
    # from the coupling that are fed into the SS network.
    self._param_net.initialize(
        func_type="scale_shift", input_dims=obs_dim, fixed_dims=self._fixed_dim,
        output_dims=self._output_dim, hidden_sizes=hidden_sizes,
        activation=activation)

  def _get_params(self, x_o):
    return self._param_net.get_params(x_o)

  def _transform(self, x_fixed, ss_params, activations):
    # This function manually computes the foward pass of the network produced by
    # parameters from the param_net.
    x_ss = x_fixed.view(x_fixed.shape[0], -1, 1)
    if not isinstance(activations, list):
      activations = [activations] * len(ss_params)

    try:
      for layer_params, activation in zip(ss_params, activations):
        x_ss = activation(layer_params.matmul(x_ss))
    except Exception as e:
      IPython.embed()
      raise(e)

    x_ss = torch.squeeze(x_ss)
    log_scale = x_ss[:, :self._output_dim]
    shift = x_ss[:, self._output_dim:]
    return log_scale, shift

  def forward(self, x, x_o, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    x_o = torch_utils.numpy_to_torch(x_o)

    z = torch.zeros_like(x)
    x_fixed = x[:, self._fixed_inds]

    ss_params, activations = self._get_params(x_o)
    log_scale, shift = self._transform(x_fixed, ss_params, activations)
    scale = torch.exp(log_scale)

    z[:, self._fixed_inds] = x_fixed
    z[:, self._tfm_inds] = scale * x[:, self._tfm_inds] + shift

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)

    if rtn_logdet:
      jac_logdet = log_scale.sum(1)
      return z, jac_logdet
    return z

  def inverse(self, z, x_o, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x_o = torch_utils.numpy_to_torch(x_o)

    x = torch.zeros_like(z)
    z_fixed = z[:, self._fixed_inds]

    ss_params, activations = self._get_params(x_o)
    log_scale, shift = self._transform(z_fixed, ss_params, activations)
    scale = torch.exp(-log_scale)

    x[:, self._fixed_inds] = z_fixed
    x[:, self._tfm_inds] = scale * (z[:, self._tfm_inds] - shift)

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


class CompositionConditionalTransformMP(ConditionalInvertibleTransformMP):
  def __init__(self, config, tfm_list=[], init_args=None):
    super(CompositionConditionalTransformMP, self).__init__(config)
    if tfm_list or init_args:
      self.initialize(tfm_list, init_args)

  def _set_transform_ordered_list(self, tfm_list):
    self._tfm_list = nn.ModuleList(tfm_list)

  def _set_dim(self):
    # Check dims:
    dims = [tfm._dim for tfm in self._tfm_list if tfm._dim is not None]
    if len(dims) > 0:
      print(dims)
      if any([dim != dims[0] for dim in dims]):
        raise ModelException("Not all transforms have the same dimension.")
      self._dim = dims[0]

  def initialize(self, tfm_list, init_args=None, *args, **kwargs):
    if tfm_list:
      self._set_transform_ordered_list(tfm_list)
    if init_args:
      for tfm, arg in zip(self._tfm_list, init_args):
        if isinstance(arg, tuple):
          tfm.initialize(*arg)
        else:
          tfm.initialize(arg)
    self._set_dim()

  def forward(self, x, x_o, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    x_o = torch_utils.numpy_to_torch(x_o)
    z = x
    if rtn_logdet:
      jac_logdet = 0
    for tfm in self._tfm_list:
      z = tfm(z, x_o, rtn_torch=True, rtn_logdet=rtn_logdet)
      try:
        if rtn_logdet:
          z, tfm_jlogdet = z
          jac_logdet += tfm_jlogdet
      except Exception as e:
        IPython.embed()
        raise(e)

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)
    return (z, jac_logdet) if rtn_logdet else z

  def inverse(self, z, x_o, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x_o = torch_utils.numpy_to_torch(x_o)
    x = z
    for tfm in self._tfm_list[::-1]:
      try:
        x = tfm.inverse(x, x_o)
      except Exception as e:
        IPython.embed()
        raise(e)

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


### Transformation construction utilities:
_TFM_TYPES = {
    "reverse": ReverseTransform,
    "leaky_relu": LeakyReLUTransform,
    "sigmoid": SigmoidLogitTransform,
    "logit": SigmoidLogitTransform,
    "scale_shift_coupling": ConditionalSSCTransformMP,
    "scale_shift": ConditionalSSCTransformMP,
    "fixed_linear": ConditionalLinearTransformMP,
    "linear": ConditionalLinearTransformMP,
}
def make_transform(config, init_args=None, comp_config=None):
  if isinstance(config, list):
    tfm_list = [make_transform(cfg) for cfg in config]
    comp_config = (
        CTfmMPConfig("composition") if comp_config is None else comp_config)
    return CompositionConditionalTransformMP(comp_config, tfm_list, init_args)

  if config.tfm_type not in _TFM_TYPES:
    raise TypeError(
        "%s not a valid transform. Available transforms: %s" %
        (config.tfm_type, list(_TFM_TYPES.keys())))

  tfm = _TFM_TYPES[config.tfm_type](config)
  if init_args is not None:
    print(config.tfm_type, init_args)
    tfm.initialize(init_args)
  return tfm

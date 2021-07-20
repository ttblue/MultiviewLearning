### Conditional flow transforms as in AC Flow.
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
from utils import math_utils, torch_utils, utils


import IPython


# Options for designing conditional transforms:
# 1. Meta parameterization:
#    F_o(x_o) gives parameters \theta of function F_u(x_u):
#    -    z | x_o = F_u(x_u; F_o(x_o))
# 2. Joint parameterization:
#    F(x_u, x_o) takes both parameters:
#    -    z | x_o = F(x_u, x_o)

# Here, we do (2) -- joint parameterization.


_BASE_DISTS = flow_transforms._BASE_DISTS
# Utils:

# # Some utilities:
# def MVZeroImpute(xvs, v_dims, ignored_view=None, expand_b=True):
#   # @expand_b: Blow up binary flag to size of views if True, keep single bit
#   #     for each view otherwise.
#   # @ignored_view: If not None, view to ignore while zero-imputing and
#   #     concatenating.
#   tot_dim = sum([dim for vi, dim in v_dims.items() if vi != ignored_view])
#   npts = xvs[utils.get_any_key(xvs)].shape[0]

#   is_torch = not isinstance(xvs[utils.get_any_key(xvs)], np.ndarray)
#   xvs = torch_utils.dict_torch_to_numpy(xvs)

#   b_vals = np.empty((npts, 0))
#   imputed_vals = np.empty((npts, 0))
#   vi_idx = 0
#   for vi, dim in v_dims.items():
#     if vi == ignored_view:
#       continue
#     vi_vals = xvs[vi] if vi in xvs else np.zeros((npts, dim))
#     imputed_vals = np.concatenate([imputed_vals, vi_vals], axis=1)

#     vi_b = np.ones((npts, 1)) * (vi in xvs)
#     if expand_b:
#       vi_b = np.tile(vi_b, (1, dim))
#     b_vals = np.concatenate([b_vals, vi_b], axis=1)

#   output = np.c_[imputed_vals, b_vals].astype(_NP_DTYPE)
#   # IPython.embed()
#   if is_torch:
#     output = torch.from_numpy(output)

#   return output

# def MVZeroImpute(x_vs, b_available, v_dims, ignored_view=None, expand_b=True):
#   # @expand_b: Blow up binary flag to size of views if True, keep single bit
#   #     for each view otherwise.
#   # @ignored_view: If not None, view to ignore while zero-imputing and
#   #     concatenating.
#   tot_dim = sum([dim for vi, dim in v_dims.items() if vi != ignored_view])
#   npts = x_vs[utils.get_any_key(x_vs)].shape[0]

#   is_torch = not isinstance(x_vs[utils.get_any_key(x_vs)], np.ndarray)
#   x_vs = torch_utils.dict_torch_to_numpy(x_vs)

#   b_vals = np.empty((npts, 0))
#   imputed_vals = np.empty((npts, 0))
#   vi_idx = 0
#   for vi, dim in v_dims.items():
#     if vi == ignored_view:
#       continue
#     vi_b = b_available[vi]
#     vi_vals = x_vs[vi] * vi_available
#     imputed_vals = np.concatenate([imputed_vals, vi_vals], axis=1)
#     if expand_b:
#       vi_b = np.tile(vi_b, (1, dim))
#     b_vals = np.concatenate([b_vals, vi_b], axis=1)

#   output = np.c_[imputed_vals, b_vals].astype(_NP_DTYPE)
#   # IPython.embed()
#   if is_torch:
#     output = torch.from_numpy(output)

#   return output

# Some utilities:
def MVZeroImpute(
    x_vs, b_available, v_dims, ignored_bitflags=[], expand_b=True):
  # @expand_b: Blow up binary flag to size of views if True, keep single bit
  #     for each view otherwise.
  # @ignored_view: If not None, view to ignore while zero-imputing and
  #     concatenating.
  # tot_dim = sum([dim for vi, dim in v_dims.items() if vi != ignored_view])
  npts = x_vs[utils.get_any_key(x_vs)].shape[0]

  is_torch = not isinstance(x_vs[utils.get_any_key(x_vs)], np.ndarray)

  #   pass
  # else:
  b_vals = [] #math_module.empty((npts, 0))
  imputed_vals = [] #math_module.empty((npts, 0))
  vi_idx = 0
  sorted_keys = sorted(v_dims.keys())
  for vi in sorted_keys:
    dim = v_dims[vi]
    vi_b = b_available[vi]
    vi_vals = x_vs[vi] * vi_b.view(-1, 1)
    imputed_vals.append(vi_vals)
    # imputed_vals = np.concatenate([imputed_vals, vi_vals], axis=1)

    if vi in ignored_bitflags:
      continue
    vi_b = vi_b.view(-1, 1) if is_torch else vi_b.reshape(-1, 1)
    if expand_b:
      vi_b = torch.tile(vi_b, (1, dim)) if is_torch else np.tile(vi_b, (1, dim))
    b_vals.append(vi_b)
      # b_vals = np.concatenate([b_vals, vi_b], axis=1)

  all_vals = imputed_vals + b_vals
  output = (
      torch.cat(all_vals, dim=1) if is_torch else
      np.concatenate(all_vals, axis=1))

    # output = np.c_[imputed_vals, b_vals].astype(_NP_DTYPE)

  # IPython.embed()
  # if is_torch:
  #   output = torch.from_numpy(output)

  return output


################################################################################
class CTfmConfig(BaseConfig):
  def __init__(
      self, tfm_type="scale_shift_coupling", neg_slope=0.01, is_sigmoid=False,
      func_nn_config=None, has_bias=True, base_dist="gaussian", reg_coeff=.1,
      lr=1e-3, batch_size=50, max_iters=1000, stopping_eps=1e-5,
      num_stopping_iter=10, grad_clip=5., verbose=True, *args, **kwargs):

    super(CTfmConfig, self).__init__(*args, **kwargs)

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
    # self.meta_parameters = meta_parameters

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
class ConditionalInvertibleTransform(flow_transforms.InvertibleTransform):
  def __init__(self, config):
    super(ConditionalInvertibleTransform, self).__init__(config)
    # self.config = config
    # self._dim = None

  def initialize(self, view_id, view_sizes, dev, *args, **kwargs):
    # raise NotImplementedError("Abstract class method")
    self.view_id = view_id
    self.view_sizes = view_sizes
    self._dim = view_sizes[view_id]
    self._dev = dev

  # def _get_params(self, x_o, rtn_torch=True):
  #   raise NotImplementedError("Abstract class method")

  def forward(self, x, x_o, b_o=None, rtn_torch=True, rtn_logdet=False):
    raise NotImplementedError("Abstract class method")

  def inverse(self, z, x_o):
    raise NotImplementedError("Abstract class method")

  def _base_log_prob(self, Z):
    return self.base_dist.log_prob(torch_utils.numpy_to_torch(Z))

  def log_prob(self, x, x_o):
    z, log_det = self(x, x_o, rtn_torch=True, rtn_logdet=True)
    return self._base_log_prob(z) + log_det

  def loss_nll(self, z, log_jac_det):
    z_ll = self._base_log_prob(z)
    nll_orig = -torch.mean(log_jac_det + z_ll)
    return nll_orig

  def loss_err(self, x_tfm, y, log_jac_det):
    recon_error = self.recon_criterion(x_tfm, y)
    logdet_reg = -torch.mean(log_jac_det)
    loss_val = recon_error + self.config.reg_coeff * torch.abs(logdet_reg)
    return loss_val

  def _train_loop(self):
    try:  # For debugging
      self._shuffle_inds = np.random.permutation(self._npts)
      x = self._x_vs[self.view_id][self._shuffle_inds]
      x_o = {
          vi:x_vi[self._shuffle_inds]
          for vi, x_vi in self._x_vs.items() if vi != self.view_id
      }
      b_o = {
          vi:b_o_vi[self._shuffle_inds]
          for vi, b_o_vi in self._b_o.items()
          if vi != self.view_id
      }
      # y = None if self._y is None else self._y[self._shuffle_inds]

      self.itr_loss = 0.
      for self._bidx in range(self._n_batches):
        b_start = self._bidx * self.config.batch_size
        b_end = b_start + self.config.batch_size
        x_batch = x[b_start:b_end]
        x_o_batch = {
            vi:x_o_vi[b_start:b_end] for vi, x_o_vi in x_o.items()}
        b_o_batch = {
            vi:b_o_vi[b_start:b_end] for vi, b_o_vi in b_o.items()}
        x_tfm_batch, jac_logdet = self.forward(
            x_batch, x_o_batch, b_o_batch, rtn_torch=True, rtn_logdet=True)
        # if y is not None:
        #   y_batch = y[b_start:b_end]

        self.opt.zero_grad()
        loss_val = self.loss_nll(x_tfm_batch, jac_logdet)  # (
            # Reconstruction loss
            # self.loss_err(x_tfm_batch, y_batch, jac_logdet)
            # if y is not None else
            # Generative model -- log likelihood loss
            # self.loss_nll(x_tfm_batch, jac_logdet)
        # )

        loss_val.backward()
        self._avg_grad = self._get_avg_grad_val()
        self._max_grad = self._get_max_grad_val()

        if torch.isnan(self._avg_grad) or torch.isinf(self._avg_grad):
          IPython.embed()
          raise ModelException("nan/inf gradient detected.")

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
    except Exception as e:
      print(e)
      IPython.embed()
      raise(e)

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

  def load_state_dict(self, state_dict, dev=None):
    super(ConditionalInvertibleTransform, self).load_state_dict(state_dict)
    self._dev = dev
    if not hasattr(self, "base_dist"):
      loc, scale = torch.zeros(self._dim, device=dev), torch.eye(self._dim, device=dev)
      self.base_dist = torch.distributions.MultivariateNormal(loc, scale)
    if dev:
      self.cuda(dev)
    self.eval()


  def fit(self, x_vs, b_o, lhood_model=None, dev=None):
    # @b_o: dictionary of bit flags of length n_pts, denoting availablity of 
    #     view for each data-point
    # Simple fitting procedure for transforming x to y
    torch.autograd.set_detect_anomaly(True)
    if self.config.verbose:
      all_start_time = time.time()

    self._x_vs = torch_utils.dict_numpy_to_torch(x_vs, dev=dev)
    self._b_o = torch_utils.dict_numpy_to_torch(b_o, dev=dev)
    # if dev is not None:
    #   for vi in self._x_vs:
    #     self._x_vs[vi] = self._x_vs[vi].cuda(dev)
    #     self._b_o[vi] = self._b_o[vi].cuda(dev)

    # self._y = None if y is None else torch_utils.numpy_to_torch(y)
    self._npts, self._dim = self._x_vs[self.view_id].shape

    # if y is None:
    if lhood_model is None:
      if self.config.base_dist not in _BASE_DISTS:
        raise NotImplementedError(
            "Base dist. type %s not implemented and likelihood model"
            " not provided." % self.config.base_dist)
      else:
        loc = torch.zeros(self._dim, device=dev)
        scale = torch.eye(self._dim, device=dev)
        self.base_dist = torch.distributions.MultivariateNormal(loc, scale)
    else:
      self.base_dist = lhood_model

    self.recon_criterion = None
    # self._base_log_prob = (
    #     lambda Z: self.base_dist.log_prob(torch_utils.numpy_to_torch(Z)))
    # else:
    #   self.recon_criterion = nn.MSELoss(reduction="mean")
    #   self.base_log_prob = None

    self.opt = optim.Adam(self.parameters(), self.config.lr)
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))

    self._finished_training = False
    self._loss_history = []
    self._avg_grad_history = []
    self._prev_loss = np.inf
    self._stop_iters = 0

    if dev is not None:
      self.cuda(dev)

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

  def sample(
      self, x_o, b_o=None, use_mean=True, rtn_torch=False):
    # if not isinstance(n_samples, int) or n_samples <= 0:
    #   raise ValueError("n_samples must be a positive integer.")
    n_samples = x_o[utils.get_any_key(x_o)].shape[0]
    if use_mean:
      z_samples = flow_likelihood.get_mean(self.base_dist, n_samples, x_o)
    else:
      z_samples = self.base_dist.sample((n_samples,))

    # if inverted:
    return self.inverse(z_samples, x_o, b_o, rtn_torch)
    # return z_samples if rtn_torch else torch_utils.torch_to_numpy(z_samples)
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
class ReverseTransform(ConditionalInvertibleTransform):
  def __init__(self, config):
    super(ReverseTransform, self).__init__(config)

  def initialize(self, view_id, view_sizes, dev, *args, **kwargs):
    super(ReverseTransform, self).initialize(view_id, view_sizes, dev)

  def forward(self, x, x_o, b_o=None, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    reverse_idx = torch.arange(x.size(-1) -1, -1, -1, device=self._dev).long()
    z = x.index_select(-1, reverse_idx)

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)

    # Determinant of Jacobian reverse transform is 1.
    return (z, 0.) if rtn_logdet else z

  def inverse(self, z, x_o, b_o=None, rtn_torch=True):
    return self(z, x_o, b_o, rtn_torch=rtn_torch, rtn_logdet=False)


class LeakyReLUTransform(ConditionalInvertibleTransform):
  def __init__(self, config):
    super(LeakyReLUTransform, self).__init__(config)

  def initialize(self, view_id, view_sizes, dev, *args, **kwargs):
    super(LeakyReLUTransform, self).initialize(view_id, view_sizes, dev)
    neg_slope = self.config.neg_slope
    self._relu_func = torch.nn.LeakyReLU(negative_slope=neg_slope)
    self._inv_func = torch.nn.LeakyReLU(negative_slope=(1. / neg_slope))
    self._log_slope = np.log(np.abs(neg_slope))

  def forward(self, x, x_o, b_o=None, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    z = self._relu_func(x)

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)
    if rtn_logdet:
      neg_elements = torch.as_tensor(
          (x < 0), dtype=torch_utils._DTYPE, device=self._dev)
      jac_logdet = neg_elements.sum(1) * self._log_slope
      return z, jac_logdet
    return z

  def inverse(self, z, x_o, b_o=None, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x = self._inv_func(z)

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


class SigmoidLogitTransform(ConditionalInvertibleTransform):
  def __init__(self, config):
    raise NotImplementedError("Not yet implemented.")
    super(SigmoidLogitTransform, self).__init__(config)

  def initialize(self, view_id, view_sizes, dev, *args, **kwargs):
    super(SigmoidLogitTransform, self).initialize(view_id, view_sizes, dev)

  def _sigmoid(self, x, rtn_logdet=False):
    z = torch.sigmoid(x)
    if rtn_logdet:
      jac = torch.abs(z * (1 - z))
      log_det = torch.log(jac)
      return z, log_det

    return z

  def _logit(self, x, rtn_logdet=False):
    x_odds = x / (1 - x)
    z = torch.log(x_odds)
    if rtn_logdet:
      jac = torch.abs(1 / ((x - 1) * (x)))
      log_det = torch.log(jac)
      return z, log_det

    return z

  def forward(self, x, x_o, b_o=None, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    
    z = (
        self._sigmoid(x, rtn_logdet=rtn_logdet)
        if self.config.is_sigmoid else
        self._logit(x, rtn_logdet=rtn_logdet))

    if rtn_logdet:
      z, jac_logdet = z
      z = z if rtn_torch else torch_utils.torch_to_numpy(z)
      return z, jac_logdet

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)
    return z

  def inverse(self, z, x_o, b_o=None, rtn_torch=True):
    x = (
        self._logit(z, rtn_logdet=False)
        if self.config.is_sigmoid else
        self._sigmoid(z, rtn_logdet=False))
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
      fn_dims = output_dims ** 2
      self._has_bias = kwargs.get("bias", False)
      if self._has_bias:
        fn_dims += output_dims
      # self.config.set_sizes(input_size=input_dims, output_size=mat_size)
      # self._param_net = torch_models.MultiLayerNN(self.config)
    elif func_type == "scale_shift":
      fn_dims = output_dims * 2
      # hidden_sizes = kwargs.get("hidden_sizes", None)
      # fixed_dims = kwargs.get("fixed_dims", None)
      # if hidden_sizes is None:
      #   raise ModelException("Need hidden sizes for scale-shift function.")
      # if fixed_dims is None:
      #   raise ModelException("Need fixed dim size for scale-shift function.")
      # self.hidden_sizes = hidden_sizes
      # self.fixed_dims = fixed_dims
      # activations = kwargs.get("activations", torch_models._IDENTITY)
      # activations = kwargs.get("activations", nn.Tanh())
      # Either single activation function or a list of activations of the same
      # size as len(hidden_sizes) + 1
      # self.activations = activations
    self.config.set_sizes(input_size=input_dims, output_size=fn_dims)
    self._param_net = torch_models.MultiLayerNN(self.config)
      # self.config.set_sizes(input_size=input_dims)
      # # The output dim is doubled, one set for (scale) and one set for (shift)
      # # all_sizes = [fixed_dims] + hidden_sizes + [output_dims * 2]
      # # param_sizes = [
      # #     all_sizes[i] * all_sizes[i+1] for i in range(len(all_sizes) - 1)]
      # self._param_net = torch_models.MultiOutputMLNN(self.config, param_sizes)

  def _get_lin_params(self, x):
    # output: n_pts x mat_size x mat_size
    lin_params = self._param_net(x)
    bias_dim = 1 if self._has_bias else 0
    lin_params = lin_params.view(
        -1, self.output_dims, self.output_dims + bias_dim)
    return lin_params

  def _get_ss_params(self, x):
    # try:
    ss_params = self._param_net(x)
    ss_params = ss_params.view(-1, self.output_dims, 2)
    return ss_params
    # except Exception as e:
    #   IPython.embed()
    #   raise(e)
    # all_sizes = (
    #     [self.fixed_dims] + self.hidden_sizes + [self.output_dims * 2])
    # ss_params = [
    #     torch.reshape(param, (-1, all_sizes[i + 1], all_sizes[i]))
    #     for i, param in enumerate(ss_params)
    # ]
    # return ss_params, self.activations

  def get_params(self, x):
    if self.func_type == "linear":
      return self._get_lin_params(x)
    elif self.func_type == "scale_shift":
      return self._get_ss_params(x)

  def forward(self, x):
    return self.get_params(x)


class ConditionalLinearTransformation(ConditionalInvertibleTransform):
  # Model linear transform as L U matrix decomposition where L is lower
  # triangular with unit diagonal and U is upper triangular with arbitrary
  # non-zero diagonal elements.
  def __init__(self, config):
    super(ConditionalLinearTransformation, self).__init__(config)

  def initialize(
      self, view_id, view_sizes, dev, nn_config, *args, **kwargs):

    super(ConditionalLinearTransformation, self).initialize(
        view_id, view_sizes, dev)
    self._view_sizes_obs = {
        vi: vdim for vi, vdim in self.view_sizes.items() if vi != self.view_id}

    self._param_net = FunctionParamNet(nn_config)

    # Need to account for bit flags
    input_dims = sum(self._view_sizes_obs.values()) * 2
    output_dims = self._dim

    self._param_net.initialize(
        func_type="linear", input_dims=input_dims, output_dims=output_dims,
        bias=self.config.has_bias)
    if dev:
      self._param_net.cuda(dev)
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
      t = lin_params[:, :, -1]
      lin_params = lin_params[:, :, :-1]
    else:
      t = 0

    L, U = torch_utils.LU_split(lin_params, dev=self._dev)
    return L, U, t

  def _impute_mv_data(self, x, x_o, b_o):
    # x_vs = x_o
    npts = x.shape[0]
    if b_o is None:
      b_o = {
          vi: (torch.ones(npts, device=self._dev) if vi in x_o else
               torch.zeros(npts, device=self._dev))
          for vi in self._view_sizes_obs
      }
    if self.view_id in x_o:
      x_o = {vi: xvi for vi, xvi in x_o.items() if vi != self.view_id}
      b_o = {vi: bvi for vi, bvi in b_o.items() if vi != self.view_id}
    # x_o[self.view_id] = torch.zeros_like(x)
    # b_o[self.view_id] = torch.zeros(npts)

    imputed_data = MVZeroImpute(
        x_o, b_o, self._view_sizes_obs, expand_b=True)

    return imputed_data

  def forward(self, x, x_o, b_o=None, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    x_o = torch_utils.dict_numpy_to_torch(x_o)
    if b_o is not None:
      b_o = torch_utils.dict_numpy_to_torch(b_o)

    x_imputed = self._impute_mv_data(x, x_o, b_o)

    L, U, t = self._get_params(x_imputed)

    try:
      # x_ = x.transpose(0, 1) if len(x.shape) > 1 else x.view(-1, 1)
      x_ = x.view(x.shape[0], -1, 1)
      z = L.matmul(U.matmul(x_)) + t.view(t.shape[0], -1, 1)
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

  def inverse(self, z, x_o, b_o=None, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x_o = torch_utils.dict_numpy_to_torch(x_o)
    if b_o is not None:
      b_o = torch_utils.dict_numpy_to_torch(b_o)

    x_imputed = self._impute_mv_data(z, x_o, b_o)
    L, U, t = self._get_params(x_imputed)

    try:
      z_ = z.view(z.shape[0], -1, 1)
      z_t = z_ - t.view(t.shape[0], -1, 1)
      # z_b_t = z_b.transpose(0, 1) if len(z_b.shape) > 1 else z_b
      # Torch solver for triangular system of equations
      # IPython.embed()
      sol_L = torch.triangular_solve(z_t, L, upper=False, unitriangular=True)[0]
      x = torch.triangular_solve(sol_L, U, upper=True)[0]
      # trtrs always returns 2-D output, even if input is 1-D. So we do this:
      # x = x_t.transpose(0, 1) if len(z.shape) > 1 else x_t.squeeze()
      x = x.squeeze()
      x = x if rtn_torch else torch_utils.torch_to_numpy(x)
    except Exception as e:
      IPython.embed()
      raise(e)

    return x


class ConditionalSSCTransform(ConditionalInvertibleTransform):
  def __init__(self, config, *args, **kwargs):
    super(ConditionalSSCTransform, self).__init__(config)
    # self.config = config

  def set_fixed_inds(self, index_mask):
    # Converting mask to boolean:
    index_mask = np.where(index_mask, 1, 0).astype(int)
    self._fixed_inds = np.nonzero(index_mask)[0]
    self._tfm_inds = np.nonzero(index_mask == 0)[0]

    self._fixed_dim = self._fixed_inds.shape[0]
    self._output_dim = self._tfm_inds.shape[0]

  def initialize(
      self, view_id, view_sizes, dev, index_mask, nn_config, *args, **kwargs):

    super(ConditionalSSCTransform, self).initialize(view_id, view_sizes, dev)

    self.set_fixed_inds(index_mask)
    self._view_sizes_obs = {
        vi: vdim for vi, vdim in self.view_sizes.items() if vi != self.view_id}
    self._view_sizes_obs[self.view_id] = self._fixed_dim

    self._param_net = FunctionParamNet(nn_config)
    # Ignore bit flags for main view.
    input_dims = sum(self._view_sizes_obs.values()) * 2 - self._fixed_dim
    output_dims = self._output_dim

    self._param_net.initialize(
        func_type="scale_shift", input_dims=input_dims, output_dims=output_dims)

  def _get_params(self, x_o):
    ss_params = self._param_net.get_params(x_o)
    log_scale, shift = ss_params[:, :, 0], ss_params[:, :, 1]
    return log_scale, shift

  def _impute_mv_data(self, x, x_o, b_o):
    # x_vs = x_o
    npts = x.shape[0]
    if b_o is None:
      b_o = {
          vi: (torch.ones(npts, device=self._dev) if vi in x_o else
               torch.zeros(npts, device=self._dev))
          for vi in self._view_sizes_obs
      }

    x_o[self.view_id] = x[:, self._fixed_inds]
    b_o[self.view_id] = torch.ones(npts, device=self._dev)

    ignored_bitflags = [self.view_id]
    imputed_data = MVZeroImpute(
        x_o, b_o, self._view_sizes_obs, ignored_bitflags, expand_b=True)

    return imputed_data

  def forward(self, x, x_o, b_o=None, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    x_o = torch_utils.dict_numpy_to_torch(x_o)
    if b_o is not None:
      b_o = torch_utils.dict_numpy_to_torch(b_o)

    x_imputed = self._impute_mv_data(x, x_o, b_o)

    log_scale, shift = self._get_params(x_imputed)
    scale = torch.exp(log_scale)

    x_fixed = x[:, self._fixed_inds]
    x_tfm = x[:, self._tfm_inds]

    z = torch.zeros_like(x, device=self._dev)
    z[:, self._fixed_inds] = x_fixed
    z[:, self._tfm_inds] = scale * x_tfm  + shift

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)

    if rtn_logdet:
      jac_logdet = log_scale.sum(1)
      return z, jac_logdet
    return z

  def inverse(self, z, x_o, b_o=None, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x_o = torch_utils.dict_numpy_to_torch(x_o)
    if b_o is not None:
      b_o = torch_utils.dict_numpy_to_torch(b_o)

    x_imputed = self._impute_mv_data(z, x_o, b_o)

    log_scale, shift = self._get_params(x_imputed)
    scale = torch.exp(-log_scale)

    z_fixed = z[:, self._fixed_inds]
    z_tfm = z[:, self._tfm_inds]

    x = torch.zeros_like(z, device=self._dev)
    x[:, self._fixed_inds] = z_fixed
    x[:, self._tfm_inds] = scale * (z_tfm - shift)

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


class CompositionConditionalTransform(ConditionalInvertibleTransform):
  def __init__(self, config):
    super(CompositionConditionalTransform, self).__init__(config)

  def _set_transform_ordered_list(self, tfm_list):
    self._tfm_list = nn.ModuleList(tfm_list)

  def _set_dim(self):
    # Check dims:
    dims = [tfm._dim for tfm in self._tfm_list if tfm._dim is not None]
    if len(dims) > 0:
      # print(dims)
      if any([dim != dims[0] for dim in dims]):
        raise ModelException("Not all transforms have the same dimension.")
      self._dim = dims[0]

  def initialize(
      self, view_id, view_sizes, tfm_list, init_args=None, dev=None, *args, **kwargs):
    super(CompositionConditionalTransform, self).initialize(
        view_id, view_sizes, dev)

    if tfm_list:
      self._set_transform_ordered_list(tfm_list)
    if init_args:
      for tfm, arg in zip(self._tfm_list, init_args):
        if isinstance(arg, tuple) or isinstance(arg, list):
          tfm.initialize(view_id, view_sizes, dev, *arg)
        elif arg is not None:
          tfm.initialize(view_id, view_sizes, dev, arg)
        else:
          tfm.initialize(view_id, view_sizes, dev)
    self._set_dim()

  def forward(self, x, x_o, b_o=None, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    x_o = torch_utils.dict_numpy_to_torch(x_o)
    if b_o is not None:
      b_o = torch_utils.dict_numpy_to_torch(b_o)

    z = x
    if rtn_logdet:
      jac_logdet = 0
    for tfm in self._tfm_list:
      z = tfm(z, x_o, b_o, rtn_torch=True, rtn_logdet=rtn_logdet)
      try:
        if rtn_logdet:
          z, tfm_jlogdet = z
          jac_logdet += tfm_jlogdet
      except Exception as e:
        IPython.embed()
        raise(e)

    z = z if rtn_torch else torch_utils.torch_to_numpy(z)
    return (z, jac_logdet) if rtn_logdet else z

  def inverse(self, z, x_o, b_o=None, rtn_torch=True):
    z = torch_utils.numpy_to_torch(z)
    x_o = torch_utils.dict_numpy_to_torch(x_o)
    if b_o is not None:
      b_o = torch_utils.dict_numpy_to_torch(b_o)

    x = z
    for tfm in self._tfm_list[::-1]:
      try:
        x = tfm.inverse(x, x_o, b_o)
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
    "scale_shift_coupling": ConditionalSSCTransform,
    "scale_shift": ConditionalSSCTransform,
    "fixed_linear": ConditionalLinearTransformation,
    "linear": ConditionalLinearTransformation,
}
def make_transform(
    configs, view_id, view_sizes, init_args=None, comp_config=None, dev=None):
  if not isinstance(configs, list):
    configs = [configs]

  tfm_list = []
  for cfg in configs:
    if cfg.tfm_type not in _TFM_TYPES:
      raise TypeError(
          "%s not a valid transform. Available transforms: %s" %
          (cfg.tfm_type, list(_TFM_TYPES.keys())))
    tfm_list.append(_TFM_TYPES[cfg.tfm_type](cfg))

  comp_config = (
      CTfmConfig("composition") if comp_config is None else comp_config)
  comp_tfm = CompositionConditionalTransform(comp_config)
  comp_tfm.initialize(view_id, view_sizes, tfm_list, init_args, dev)
  return comp_tfm

  # if config.tfm_type not in _TFM_TYPES:
  #   raise TypeError(
  #       "%s not a valid transform. Available transforms: %s" %
  #       (config.tfm_type, list(_TFM_TYPES.keys())))

  # tfm = _TFM_TYPES[config.tfm_type](config)
  # if init_args is not None:
  #   print(config.tfm_type, init_args)
  #   tfm.initialize(view_id, view_sizes, *init_args)
  # else:
  #   tfm.initialize(view_id, view_sizes)
  # return tfm

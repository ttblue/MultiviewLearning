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
from utils import math_utils, torch_utils


import IPython


# Options for designing conditional transforms:
# 1. F_o(x_o) gives parameters \theta of function F_u(x_u):
#    -    z | x_o = F_u(x_u; F_o(x_o))
# 2. F(x_u, x_o) takes both parameters:
#    -    z | x_o = F(x_u, x_o)

# Utils:
def MVZeroImpute(xvs, v_dims, expand_b=True):
  # expand_b -- Blow up binary flag to size of views if True, keep single bit
  # for each view otherwise
  tot_dim = np.sum([dim for _, dim in v_dims.items()])
  n_pts = xvs[utils.get_any_key(xvs)].shape[0]

  b_vals = np.empty((n_pts, 0))
  imputed_vals = np.empty((n_pts, 0))
  for vi, dim in v_dims.items():
    vi_vals = xvs[vi] if vi in xvs else np.zeros((npts, dim))
    imputed_vals = np.concatenate([imputed_vals, vi_vals], axis=1)

    vi_b = np.ones((n_pts,)) * (vi in xvs)
    if expand_b:
      vi_b = np.tile(vi_b, (1, dim))
    b_vals = np.concatenate([b_vals, vi_b], axis=1)

  output = np.c_[imputed_vals, b_vals]
  return output

################################################################################


# Functions of the form q(x_u | x_o)
class ConditionalInvertibleTransform(flow_transforms.InvertibleTransform):
  def __init__(self, config):
    super(InvertibleTransform, self).__init__()
    self.config = config
    self._dim = None

  def initialize(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def _get_params(self, x_o, rtn_torch=True):
    raise NotImplementedError("Abstract class method")

  def forward(self, x_u, x_o, rtn_torch=True, rtn_logdet=False):
    raise NotImplementedError("Abstract class method")

  def inverse(self, z_u, x_o):
    raise NotImplementedError("Abstract class method")

  def log_prob(self, x_u, x_o):
    z, log_det = self(x, rtn_torch=True, rtn_logdet=True)
    return self.base_log_prob(z) + log_det
 
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


# Class for function parameters for unobserved covariates given observed
# covariates
_FUNC_TYPES = ["linear", "scale_shift"]
class FunctionParamNet(torch_models.MultiLayerNN):
  def __init__(self, nn_config, *args, **kwargs):
    super(FunctionParamNet, self).__init__(nn_config, *args, **kwargs)

  def initialize(self, func_type, output_dims, **kwargs):
    if func_type not in _FUNC_TYPES:
      raise ModelException("Unknown function type %s." % func_type)

    self.func_type = func_type
    self.output_dims = output_dims

    if self.config.use_vae:
      print("Warning: Disabling VAE for param network.")
    self.config.use_vae = False

    if func_type == "linear":
      mat_size = output_dims ** 2
      self.config.set_sizes(output_size=mat_size)
      self._param_net = torch_models.MultiLayerNN(self.config)
    elif func_type == "scale_shift":
      hidden_sizes = kwargs.get("hidden_sizes", None)
      activation = kwargs.get("activation", torch_models._IDENTITY)
      self.hidden_sizes = hidden_sizes
      self.activation = activation
      if hidden_sizes is None:
        raise ModelException("Need hidden sizes for scale-shift function.")

      all_sizes = [self.config.input_size] + hidden_sizes + [output_dims]
      param_sizes = [
          all_sizes[i] * all_sizes[i+1] for i in range(len(all_sizes) - 1)]
      self._param_net = torch_models.MultiOutputMLNN(self.config, param_sizes)

  def _get_lin_params(self, x):
    # output: n_pts x mat_size x mat_size
    lin_params = self._param_net(x_o)
    lin_params = torch.view(lin_params, (self.output_dims, self.output_dims))
    return lin_params

  def _get_ss_params(self, x):
    ss_params = self._param_net(x_o)
    all_sizes = (
        [self.config.input_size] + self.hidden_sizes + [self.output_dims])
    ss_params = [
        torch.view(param, (all_sizes[i], all_sizes[i + 1]))
        for i, param in enumerate(ss_params)
    ]
    return ss_params, self.activation

  def get_params(self, x):
    if self.func_type == "linear":
      return self._get_lin_params(x)
    elif self.func_type == "scale_shift":
      return self._get_ss_params(x)

class ConditionalLinearTransformation(ConditionalInvertibleTransform):
  # Model linear transform as L U matrix decomposition where L is lower
  # triangular with unit diagonal and U is upper triangular with arbitrary
  # non-zero diagonal elements.
  def __init__(self, config):
    super(ConditionalLinearTransformation, self).__init__(config)

  def initialize(
      self, dim, init_lin_param=None, init_bias_param=None, *args, **kwargs):
    # init_lin_param: Tensor of shape dim x dim of initial values, or None.
    self._dim = dim

    init_lin_param = (
        torch.eye(dim) if init_lin_param is None else
        torch_utils.numpy_to_torch(init_lin_param))
    self._lin_param = torch.nn.Parameter(init_lin_param)

    if self.config.has_bias:
      init_bias_param = (
          torch.zeros(dim) if init_bias_param is None else
          torch_utils.numpy_to_torch(init_bias_param))
      self._b = torch.nn.Parameter(init_bias_param)
    else:
      self._b = torch.zeros(dim)

  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)

    L, U = torch_utils.LU_split(self._lin_param)

    x_ = x.transpose(0, 1) if len(x.shape) > 1 else x.view(-1, 1)
    y = L.matmul(U.matmul(x_)) + self._b.view(-1, 1)
    # Undoing dimension changes to be consistent with input
    y = y.transpose(0, 1) if len(x.shape) > 1 else y.squeeze()
    y = y if rtn_torch else torch_utils.torch_to_numpy(y)

    if rtn_logdet:
      # Determinant of triangular jacobian is product of the diagonals.
      # Since both L and U are triangular and L has unit diagonal,
      # this is just the product of diagonal elements of U.
      jac_logdet = (torch.ones(x.shape[0]) *
                    torch.sum(torch.log(torch.abs(torch.diag(U)))))
      return y, jac_logdet
    return y

  def inverse(self, y, rtn_torch=True):
    y = torch_utils.numpy_to_torch(y)

    L, U = torch_utils.LU_split(self._lin_param)
    
    y_b = y - self._b
    y_b_t = y_b.transpose(0, 1) if len(y_b.shape) > 1 else y_b
    # Torch solver for triangular system of equations
    # IPython.embed()
    sol_L = torch.triangular_solve(y_b_t, L, upper=False, unitriangular=True)[0]
    x_t = torch.triangular_solve(sol_L, U, upper=True)[0]
    # trtrs always returns 2-D output, even if input is 1-D. So we do this:
    x = x_t.transpose(0, 1) if len(y.shape) > 1 else x_t.squeeze()
    x = x if rtn_torch else torch_utils.torch_to_numpy(x)

    return x


# class ScaleShiftCouplingTransform(ConditionalInvertibleTransform):
#   def __init__(self, config, index_mask=None)
#     super(ScaleShiftCouplingTransform, self).__init__()
#     self.config = config
#     self._dim = None

#     if index_mask is not None:
#       self.set_fixed_indices(index_mask)

#   def set_fixed_inds(self, index_mask):
#     # Converting mask to boolean:
#     index_mask = np.where(index_mask, 1, 0).astype(int)
#     self._fixed_inds = np.nonzero(index_mask)[0]
#     self._tfm_inds = np.nonzero(index_mask == 0)[0]

#     self._dim = index_mask.shape[0]
#     self._fixed_dim = self._fixed_inds.shape[0]
#     self._output_dim = self._tfm_inds.shape[0]

#   def initialize(self, *args, **kwargs):
#     raise NotImplementedError("Abstract class method")

#   def _get_params(self, x_o, rtn_torch=True):
#     raise NotImplementedError("Abstract class method")

#   def forward(self, x_u, x_o, rtn_torch=True, rtn_logdet=False):
#     raise NotImplementedError("Abstract class method")

#   def inverse(self, z_u, x_o):
#     raise NotImplementedError("Abstract class method")

#   def log_prob(self, x_u, x_o):
#     z, log_det = self(x, rtn_torch=True, rtn_logdet=True)
#     return self.base_log_prob(z) + log_det

#   def _impute_function(self, x_o):
#     # x_o should be a dict with available/observable views as keys
#     pass

#   def _scale_shift_params(self, x_o):
#     x_o_imputed = self._impute_function(x_o)

#     scale_params = self._scale_transform(x_o)
#     shift_params = self._shift_transform(x_o)

#     return scale_params, shift_params

#   def _transform(self, x_u, x_o):
#     # S, T = self._shift_transform(x_o)
#     # x_u_tfm = S * x_u + 

#     x = torch_utils.numpy_to_torch(x)
#     y = torch.zeros_like(x)
#     x_fixed = x[:, self._fixed_inds]

#     scale = torch.exp(self._scale_tfm(x_fixed))
#     shift = self._shift_tfm(x_fixed)

#     y[:, self._fixed_inds] = x_fixed
#     y[:, self._tfm_inds] = scale * x[:, self._tfm_inds] + shift

#     y = y if rtn_torch else torch_utils.torch_to_numpy(y)

#     if rtn_logdet:
#       jac_logdet = self._scale_tfm(x_fixed).sum(1)
#       return y, jac_logdet
#     return y
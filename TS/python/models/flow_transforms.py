import numpy as np
import scipy
import torch
from torch import nn, optim
import time

from models.model_base import ModelException, BaseConfig
from utils import math_utils, torch_utils


import IPython


################################################################################
# Invertible coupling transforms:

# Current transforms --
# 1. Reverse
# 2. LeakyReLU
# 3. Coupled scale + shift
# 4. Fixed linear
# 5. [INCOMPLETE] Adaptable linear -- need to check on this
# 6. Composition

# Todo:
# 1. Recurrent
# 2. Recurrent Scale
# 3. RNNCoupling

class TfmConfig(BaseConfig):
  # General config object for all transforms
  def __init__(
      self, tfm_type="scale_shift_coupling", neg_slope=0.01, scale_config=None,
      shift_config=None, shared_wts=False, ltfm_config=None, bias_config=None,
      has_bias=True, lr=1e-3, batch_size=50, max_iters=1000, reg_coeff=.1,
      *args, **kwargs):
    super(TfmConfig, self).__init__(*args, **kwargs)

    self.tfm_type = tfm_type.lower()
    # LeakyReLU params:
    self.neg_slope = neg_slope

    # Shift Scale params:
    self.scale_config = scale_config
    self.shift_config = shift_config
    self.shared_wts = shared_wts

    # Adaptable Linear Transform params:
    self.linear_config = ltfm_config
    self.bias_config = bias_config

    # Fixed Linear Transform params:
    self.has_bias = has_bias

    # Some misc. parameters if training transforms only
    self.lr = lr
    self.batch_size = batch_size
    self.max_iters = max_iters

    self.reg_coeff = reg_coeff


class InvertibleTransform(nn.Module):
  def __init__(self, config):
    super(InvertibleTransform, self).__init__()
    self.config = config

  def initialize(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    raise NotImplementedError("Abstract class method")

  def inverse(self, y):
    raise NotImplementedError("Abstract class method")

  def loss(self, x_tfm, y, log_jac_det):
    recon_error = self.recon_criterion(x_tfm, y)
    logdet_reg = -torch.mean(log_jac_det)
    loss_val = recon_error + self.config.reg_coeff * logdet_reg
    return loss_val

  def _train_loop(self):
    shuffle_inds = np.random.permutation(self._npts)
    x, y = self._x[shuffle_inds], self._y[shuffle_inds]
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      x_batch, y_batch = x[b_start:b_end], y[b_start:b_end]
      x_tfm_batch, jac_logdet = self.forward(
          x_batch, rtn_torch=True, rtn_logdet=True)
      # keep_subsets = next(self._view_subset_shuffler)
      # xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets}
      self.opt.zero_grad()
      loss_val = self.loss(x_tfm_batch, y_batch, jac_logdet)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self, x, y):
    # Simple fitting procedure for transforming x to y
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._x, self._y = map(torch_utils.numpy_to_torch, [x, y])
    self._npts = self._x.shape[0]

    self.recon_criterion = nn.MSELoss(reduction="mean")
    self.opt = optim.Adam(self.parameters(), self.config.lr)
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))

    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
        self._train_loop()

        if self.config.verbose:
          itr_diff_time = time.time() - itr_start_time
          loss_val = float(self.itr_loss.detach())
          print("\nIteration %i out of %i (in %.2fs). Loss: %.5f" %
                (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
                end='\r')
      if self.config.verbose:
        print("\nIteration %i out of %i (in %.2fs). Loss: %.5f" %
              (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
              end='\r')
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
    return self


class ReverseTransform(InvertibleTransform):
  def __init__(self, config):
    super(ReverseTransform, self).__init__(config)

  def initialize(self, *args, **kwargs):
    pass

  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    reverse_idx = torch.arange(x.size(-1) -1, -1, -1).long()
    y = x.index_select(-1, reverse_idx)

    y = y if rtn_torch else torch_utils.torch_to_numpy(y)

    # Determinant of Jacobian reverse transform is 1.
    return (y, 0.) if rtn_logdet else y

  def inverse(self, y, rtn_torch=True):
    return self(y, rtn_torch=rtn_torch, rtn_logdet=False)


class LeakyReLUTransform(InvertibleTransform):
  def __init__(self, config):
    super(LeakyReLUTransform, self).__init__(config)

  def initialize(self, *args, **kwargs):
    neg_slope = self.config.neg_slope
    self._relu_func = torch.nn.LeakyReLU(negative_slope=neg_slope)
    self._inv_func = torch.nn.LeakyReLU(negative_slope=(1. / neg_slope))
    self._log_slope = np.log(neg_slope)

  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    y = self._relu_func(x)

    y = y if rtn_torch else torch_utils.torch_to_numpy(y)
    if rtn_logdet:
      jac_logdet = (x < 0).sum(1) * self._log_slope
      return y, jac_logdet
    return y

  def inverse(self, y, rtn_torch=True):
    y = torch_utils.numpy_to_torch(y)
    x = self._inv_func(y)

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


class ScaleShiftCouplingTransform(InvertibleTransform):
  def __init__(self, config, index_mask=None):
    super(ScaleShiftCouplingTransform, self).__init__(config)

    if index_mask is not None:
      self._set_fixed_inds(index_mask)

  def set_fixed_inds(self, index_mask):
    # index_mask -- d-length array with k ones. d is the length of the overall 
    #     input and k is the length of fixed elements which remained unchanged
    #     in the coupling; i.e. the input dimension of the scale and shift NNs.
    #     Assuming this is 1-d for now.

    # Converting mask to boolean:
    index_mask = np.where(index_mask, 1, 0).astype(int)
    self._fixed_inds = np.nonzero(index_mask)[0]
    self._tfm_inds = np.nonzero(index_mask == 0)[0]

    self._dim = index_mask.shape[0]
    self._fixed_dim = self._fixed_inds.shape[0]
    self._output_dim = self._tfm_inds.shape[0]

  def initialize(self, index_mask=None, *args, **kwargs):
    if self.config.shared_wts:
      raise NotImplementedError("Not yet implemented shared weights.")

    if index_mask is not None:
      self.set_fixed_inds(index_mask)

    self.config.scale_config.set_sizes(
        input_size=self._fixed_dim, output_size=self._output_dim)
    shift_config = (
        self.config.scale_config.copy()
        if self.config.shift_config is None else
        self.config.shift_config)
    shift_config.set_sizes(
        input_size=self._fixed_dim, output_size=self._output_dim)
    self._scale_tfm = torch_utils.MultiLayerNN(self.config.scale_config)
    self._shift_tfm = torch_utils.MultiLayerNN(shift_config)
    
  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    y = torch.zeros_like(x)
    x_fixed = x[:, self._fixed_inds]

    scale = torch.exp(self._scale_tfm(x_fixed))
    shift = self._shift_tfm(x_fixed)

    y[:, self._fixed_inds] = x_fixed
    y[:, self._tfm_inds] = scale * x[:, self._tfm_inds] + shift

    y = y if rtn_torch else torch_utils.torch_to_numpy(y)

    if rtn_logdet:
      jac_logdet = self._scale_tfm(x_fixed).sum(1)
      return y, jac_logdet
    return y

  def inverse(self, y, rtn_torch=True):
    y = torch_utils.numpy_to_torch(y)
    x = torch.zeros_like(y)
    y_fixed = y[self._fixed_inds]
    scale = torch.exp(-self._scale_tfm(y_fixed))
    shift = -self._shift_tfm(y_fixed)

    x[self._fixed_inds] = y_fixed
    x[self._tfm_inds] = scale * y[self._tfm_inds] + shift

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


class FixedLinearTransformation(InvertibleTransform):
  # Model linear transform as L U matrix decomposition where L is lower
  # triangular with unit diagonal and U is upper triangular with arbitrary
  # non-zero diagonal elements.
  def __init__(self, config):
    super(FixedLinearTransformation, self).__init__(config)

  def initialize(
      self, dim, init_lin_param=None, init_bias_param=None, *args, **kwargs):
    # init_lin_param: Tensor of shape dim x dim of initial values, or None.
    self._dim = dim

    init_lin_param = (
        torch.eye(dim) if init_lin_param is None else
        torch_utils.numpy_to_torch(init_lin_param))
    self._lin_param = torch.nn.Parameter(init_lin_param)
    self._L = torch.eye(self._dim) + torch.tril(self._lin_param, diagonal=-1)
    self._U = torch.triu(self._lin_param, diagonal=0)

    if self.config.has_bias:
      init_bias_param = (
          torch.zeros(dim) if init_bias_param is None else
          torch_utils.numpy_to_torch(init_bias_param))
      self._b = torch.nn.Parameter(init_bias_param)
    else:
      self._b = torch.zeros(dim)

  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    # This is to store L and U for current forward pass, so that it can be used
    # for the inverse.
    x_ = x.transpose(0, 1) if len(x.shape) > 1 else x.view(-1, 1)
    y = self._L.matmul(self._U.matmul(x_)) + self._b.view(-1, 1)
    # Undoing dimension changes to be consistent with input
    y = y.transpose(0, 1) if len(x.shape) > 1 else y.squeeze()
    y = y if rtn_torch else torch_utils.torch_to_numpy(y)

    if rtn_logdet:
      # Determinant of triangular jacobian is product of the diagonals.
      # Since both L and U are triangular and L has unit diagonal,
      # this is just the product of diagonal elements of U.
      jac_logdet = torch.sum(torch.log(torch.abs(torch.diag(self._U))))
      return y, jac_logdet
    return y

  def inverse(self, y, rtn_torch=True):
    y = torch_utils.numpy_to_torch(y)
    # Lt = self._L.transpose(0, 1)
    # Ut = self._U.transpose(0, 1)
    
    y_b = y - self._b
    y_b_t = y_b.transpose(0, 1) if len(y_b.shape) > 1 else y_b
    # Torch solver for triangular system of equations
    # IPython.embed()
    sol_L = torch.trtrs(y_b, self._L, upper=False, unitriangular=True)[0]
    x = torch.trtrs(sol_L, self._U, upper=True)[0]
    # trtrs always returns 2-D output, even if input is 1-D. So we do this:
    x = x.transpose(0, 1) if len(y.shape) > 1 else x.squeeze()
    x = x if rtn_torch else torch_utils.torch_to_numpy(x)

    return x


class AdaptiveLinearTransformation(InvertibleTransform):
  # Model linear transform as L U matrix decomposition where L is lower
  # triangular with unit diagonal and U is upper triangular with arbitrary
  # non-zero diagonal elements.
  # TODO: Incomplete
  def __init__(self, config):
    raise NotImplementedError("Not implemented yet.")
    super(LinearTransformation, self).__init__(config)
    self._L = None
    self._U = None

  def initialize(self, dim, *args, **kwargs):
    self._dim = dim

    bias_config = (
        self.config.linear_config.copy()
        if self.config.bias_config is None else
        self.config.linear_config)

    self.config.linear_config.set_sizes(input_size=dim, output_size=(dim ** 2))
    bias_config.set_sizes(input_size=dim, output_size=dim)

    self._lin_func = torch_utils.MultiLayerNN(self.config.linear_config)
    self._bias_func = torch_utils.MultiLayerNN(bias_config)

  def _get_LU(self, x):
    A_vals = self._lin_func(x)
    L = torch.eye(self._dim) + torch.tril(A_vals, diagonal=-1)
    U = torch.triu(A_vals, diagonal=0)
    return L, U

  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    # This is to store L and U for current forward pass, so that it can be used
    # for the inverse.
    self._L, self._U = self._get_LU(x)
    b = self._bias_func(x)

    y = self._L.dot(self._U.dot(x)) + b
    y = y if rtn_torch else torch_utils.torch_to_numpy(y)

    if rtn_logdet:
      # Determinant of triangular jacobian is product of the diagonals.
      # Since both L and U are triangular and L has unit diagonal,
      # this is just the product of diagonal elements of U.
      jac_logdet = torch.sum(torch.log(torch.abs(torch.diag(self._U))))
      return y, jac_logdet
    return y

  def inverse(self, y):
    pass
    # Ut = tf.transpose(U)
    # Lt = tf.transpose(L)
    # yt = tf.transpose(y)
    # sol = tf.matrix_triangular_solve(Ut, yt-tf.expand_dims(b, -1))
    # x = tf.transpose(
    #     tf.matrix_triangular_solve(Lt, sol, lower=False)
    # )
    # return x


class CompositionTransform(InvertibleTransform):
  def __init__(self, config, tfm_list=[], init_args=None):
    super(CompositionTransform, self).__init__(config)
    if tfm_list or init_args:
      self.initialize(tfm_list, init_args)

  def _set_transform_ordered_list(self, tfm_list):
    self._tfm_list = tfm_list
    for i, tfm in enumerate(tfm_list):
      self.add_module("tfm_%i" % i, tfm)

  def initialize(self, tfm_list, init_args=None, *args, **kwargs):
    if tfm_list:
      self._set_transform_ordered_list(tfm_list)
    if init_args:
      for tfm, arg in zip(self._tfm_list, init_args):
        if isinstance(arg, tuple):
          tfm.initialize(*arg)
        else:
          tfm.initialize(arg)

  def forward(self, x, rtn_torch=True, rtn_logdet=False):
    x = torch_utils.numpy_to_torch(x)
    y = x
    if rtn_logdet:
      jac_logdet = 0
    for tfm in self._tfm_list:
      y = tfm(y, rtn_torch=True, rtn_logdet=rtn_logdet)
      if rtn_logdet:
        y, tfm_jlogdet = y
        jac_logdet += tfm_jlogdet

    y = y if rtn_torch else torch_utils.torch_to_numpy(y)
    return (y, jac_logdet) if rtn_logdet else y

  def inverse(self, y, rtn_torch=True):
    y = torch_utils.numpy_to_torch(y)
    x = y
    for tfm in self._tfm_list[::-1]:
      x = tfm.inverse(x)

    return x if rtn_torch else torch_utils.torch_to_numpy(x)


_TFM_TYPES = {
    "reverse": ReverseTransform,
    "leaky_relu": LeakyReLUTransform,
    "scale_shift_coupling": ScaleShiftCouplingTransform,
    "fixed_linear": FixedLinearTransformation,
}
def make_transform(config, init_args=None):
  if isinstance(config, list):
    tfm_list = [make_transform(cfg) for cfg in config]
    return CompositionTransform(TfmConfig("composition"), tfm_list, init_args)

  if config.tfm_type not in _TFM_TYPES:
    raise TypeError(
        "%s not a valid transform. Available transforms: %s" %
        (config.tfm_type, list(_TFM_TYPES.keys())))

  tfm = _TFM_TYPES[config.tfm_type](config)
  if init_args is not None:
    tfm.initialize(init_args)
  return tfm


if __name__ == "__main__":
  dim = 10
  flts = []
  for i in range(10):
    mat = math_utils.random_unitary_matrix(dim)
    P, L, U = scipy.linalg.lu(mat)
    init_vals = L + U - np.eye(dim)
    flt = FixedLinearTransformation(TfmConfig())
    flt.initialize(dim, init_vals)
    flts.append(flt)

  ctf = CompositionTransform(TfmConfig(), flts)
  x = np.random.rand(10)
  y = ctf(x)
  x2 = ctf.inverse(y)
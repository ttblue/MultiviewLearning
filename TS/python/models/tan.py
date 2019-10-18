import multiprocessing as mp
import numpy as np
import torch
from torch import nn
import time

from models.model_base import ModelException, BaseConfig
from utils import torch_utils


import IPython


################################################################################
# Invertible coupling transforms:

class TfmConfig(BaseConfig):
  # General config object for all transforms
  def __init(
      self, scale_config=None, shift_config=None, shared_wts=False, *args,
      **kwargs):
    super(TfmConfig, self).__init__(*args, **kwargs)

    # Shift Scale params:
    self.scale_config = scale_config
    self.shift_config = shift_config
    self.shared_wts = shared_wts


class CouplingTransform(nn.Module):
  def __init__(self, config, index_mask=None):
    super(CouplingTransform, self).__init__()
    self.config = config

  def initialize(self):
    raise NotImplementedError("Abstract class method")

  def apply(self, x):
    raise NotImplementedError("Abstract class method")

  def invert(self, y):
    raise NotImplementedError("Abstract class method")

  def jacobian_determinant(self):
    raise NotImplementedError("Abstract class method")

  def forward(self, x):
    return self.apply(x, rtn_torch=True)


class ScaleShiftCouplingTransform(nn.CouplingTransform):
  def __init__(self, config, index_mask=None):
    super(CouplingTransform, self).__init__(config, index_mask)

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

  def initialize(self, index_mask=None):
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
    self._scale_tfm = torch_utils.MultiLayerNN(self.config.scale_config)
    self._shift_tfm = torch_utils.MultiLayerNN(shift_config)
    
  def apply(self, x, rtn_torch=True):
    if not isinstance(x, torch.Tensor):
      x = torch.from_numpy(x).type(torch_utils._DTYPE).requires_grad_(False)

    y = torch.zeros_like(x)
    x_fixed = x[self._fixed_inds]
    scale = torch.exp(self._scale_tfm(x_fixed))
    shift = self._shift_tfm(x_fixed)

    y[self._fixed_inds] = x_fixed
    y[self._tfm_inds] = scale * x[self._tfm_inds] + shift

    return y if rtn_torch else y.detach().numpy()

  def invert(self, y, rtn_torch=True):
    if not isinstance(y, torch.Tensor):
      y = torch.from_numpy(y).type(torch_utils._DTYPE).requires_grad_(False)

    x = torch.zeros_like(y)
    y_fixed = y[self._fixed_inds]
    scale = torch.exp(-self._scale_tfm(y_fixed))
    shift = -self._shift_tfm(y_fixed)

    x[self._fixed_inds] = y_fixed
    x[self._tfm_inds] = scale * y[self._tfm_inds] + shift

    return x if rtn_torch else x.detach().numpy()

  def jacobian_determinant(self, x):
    if not isinstance(x, torch.Tensor):
      x = torch.from_numpy(x).type(torch_utils._DTYPE).requires_grad_(False)

    x_fixed = x[self._fixed_inds]
    jac_det = torch.exp(torch.sum(self._scale_tfm(x_fixed)))


class ComposedCouplingTransform(nn.CouplingTransform):
  def __init__(self, config, tfm_list=[]):
    super(ComposedCouplingTransform, self).__init__(config)
    if tfm_list:
      self.set_transform_ordered_list(tfm_list)

  def set_transform_ordered_list(self, tfm_list):
    self._tfm_list = tfm_list
    for i, tfm in enumerate(tfm_list):
      self.add_module("coupling_tfm_%i" % i, tfm)

  def apply(self, x, rtn_torch=True):
    if not isinstance(x, torch.Tensor):
      x = torch.from_numpy(x).type(torch_utils._DTYPE).requires_grad_(False)

    y = x
    for tfm in self._tfm_list:
      y = tfm(y)

    return y if rtn_torch else y.detach().numpy()

  def invert(self, y, rtn_torch=True):
    if not isinstance(y, torch.Tensor):
      y = torch.from_numpy(y).type(torch_utils._DTYPE).requires_grad_(False)

    x = y
    for tfm in self._tfm_list[::-1]:
      x = tfm.invert(x)

    return x if rtn_torch else x.detach().numpy()

  def jacobian_determinant(self, x):
    if not isinstance(x, torch.Tensor):
      x = torch.from_numpy(x).type(torch_utils._DTYPE).requires_grad_(False)

    y = x
    for tfm in self._tfm_list:
      y = tfm(y)

    return y if rtn_torch else y.detach().numpy()
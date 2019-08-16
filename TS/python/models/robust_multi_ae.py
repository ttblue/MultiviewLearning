# (Variational) Autoencoder for multi-view + synchronous
import itertools
import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
import time

from models import multi_ae
import utils.torch_utils as tu
from utils.torch_utils import _DTYPE, _TENSOR_FUNC

import IPython


class RMAEConfig(multi_ae.MAEConfig):
  def __init__(self, vdrop_prob, zero_at_input=True, *args, **kwargs):
    super(RMAEConfig, self).__init__(*args, **kwargs)
    self.vdrop_prob = vdrop_prob
    self.zero_at_input = zero_at_input  # Zero at input vs. code


class RobustMultiAutoEncoder(multi_ae.MultiAutoEncoder):
  def __init__(self, config):
    super(RobustMultiAutoEncoder, self).__init__(config)
    # TODO:
    if self.config.use_var:
      raise NotImplementedError("Variational MAE not yet implemented.")
  #   self._initialize_dims_for_dropout()

  # def _initialize_dims_for_dropout(self):
  #   zero_dim_func = (
  #       (lambda vi: self.config.v_sizes[vi]) if self.config.zero_at_input else
  #       (lambda vi: self.config.encoder_params[vi].output_size)
  #   )
  #   self._zero_dims = {vi: zero_dim_func(vi) for vi in range(self._nviews)}
  #   self._zero_dim_func = zero_dim_func

  def _encode_view(self, xv, vi):
    if not isinstance(xv, torch.Tensor):
      xv = torch.from_numpy(xv).type(_DTYPE).requires_grad_(False)
    return self._en_layers[vi](xv)

  def _encode_missing_view(self, vi, npts):
    if self.config.zero_at_input:
      vi_zeros = torch.zeros(npts, self.config.v_sizes[vi])
      vi_code = self._encode_view(vi_zeros)
    else:
      vi_code = torch.zeros(npts, self.config.encoder_params[vi].output_size)
    return vi_code

  def _joint_code(self, view_codes):
    npts = view_codes[view_codes.keys()[0]].shape[0]
    view_codes = [
        (view_codes[vi] if vi in view_codes else
         self._encode_missing_view(vi, npts))
        for vi in range(self._nviews)
    ]
    joint_code = torch.cat(view_codes, axis=1)
    return joint_code

  def encode(self, xvs, include_missing=True, return_joint=True):
    view_codes = {}
    npts = xvs[xvs.keys()[0]].shape[0]
    for vi in range(self._nviews):
      if vi in xvs:
        view_codes[vi] = self._encode_view(xvs[vi], vi)
      elif include_missing:
        view_codes[vi] = self._encode_missing_view(vi, npts)

    return self._joint_code(view_codes) if return_joint else view_codes

  def forward(self, xvs):
    # Solve for alpha
    # if not isinstance(x, torch.Tensor):
    #   # Assuming it is numpy arry
    #   x = torch.from_numpy(x)
    # x.requires_grad_(False)
    zs = self.encode(xvs, include_missing=True, return_joint=True)

    # sampled_zs = [self._sample_codes(*z) for z in zs]

    # This is, for every encoded view, the reconstruction of every view
    recons = {}
    for vi in range(self._n_views):
      recons[vi] = self.decode(sampled_zs[vi])

    return zs, recons
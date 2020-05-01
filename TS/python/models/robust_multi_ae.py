# (Variational) Autoencoder for multi-view + synchronous
import itertools
import numpy as np
import torch
from torch import nn
from torch import optim
import time

from models import multi_ae, torch_models
from models.model_base import ModelException
from utils import utils, torch_utils
from utils.torch_utils import _DTYPE, _TENSOR_FUNC

import IPython


class RMAEConfig(multi_ae.MAEConfig):
  def __init__(
      self, joint_coder_params, drop_scale=False, zero_at_input=True,
      *args, **kwargs):
    super(RMAEConfig, self).__init__(*args, **kwargs)

    self.joint_coder_params = joint_coder_params
    self.drop_scale = drop_scale
    self.zero_at_input = zero_at_input  # Zero at input vs. code


class RobustMultiAutoEncoder(multi_ae.MultiAutoEncoder):
  def __init__(self, config):
    # TODO:
    # if config.use_vae:
    #   raise NotImplementedError("Variational MAE not yet implemented.")
    super(RobustMultiAutoEncoder, self).__init__(config)
    self._setup_joint_layer()
  #   self._initialize_dims_for_dropout()

  # def _initialize_dims_for_dropout(self):
  #   zero_dim_func = (
  #       (lambda vi: self.config.v_sizes[vi]) if self.config.zero_at_input else
  #       (lambda vi: self.config.encoder_params[vi].output_size)
  #   )
  #   self._zero_dims = {vi: zero_dim_func(vi) for vi in range(self._nviews)}
  #   self._zero_dim_func = zero_dim_func

  def _setup_joint_layer(self):
    input_size = np.sum(
        [self.config.encoder_params[vi].output_size
         for vi in range(self._nviews)])
    self.config.joint_coder_params.input_size = input_size
    self.config.joint_coder_params.output_size = self.config.code_size
    self._joint_coder = torch_models.MultiLayerNN(
        self.config.joint_coder_params)

  def _encode_missing_view(self, vi, npts):
    if self.config.zero_at_input:
      vi_zeros = torch.zeros(npts, self.config.v_sizes[vi])
      vi_code, _ = self._encode_view(vi_zeros, vi)
    else:
      vi_code = torch.zeros(npts, self.config.encoder_params[vi].output_size)
    return vi_code

  def _encode_view(self, xv, vi):
    npts = len(xv)
    valid_inds = [i for i in range(npts) if xv[i] is not None]

    IPython.embed()
    xv_valid = np.array([xv[i] for i in valid_inds])
    xv_valid = torch_utils.numpy_to_torch(xv_valid)
    valid_code = self._en_layers["E%i"%vi](xv_valid)

    code = self._encode_missing_view(vi, npts)
    code[valid_inds] = valid_code

    return code, valid_inds

  def _encode_joint(self, view_codes):
    npts = view_codes[utils.get_any_key(view_codes)].shape[0]
    view_codes = [
        (view_codes[vi] if vi in view_codes else
         self._encode_missing_view(vi, npts))
        for vi in range(self._nviews)
    ]
    joint_code_input = torch.cat(view_codes, dim=1)
    joint_code = self._joint_coder(joint_code_input)
    return joint_code

  def encode(self, xvs, include_missing=True, return_joint=True):
    view_codes = {}
    npts = len(xvs[utils.get_any_key(xvs)])
    
    # for scaling:
    n_available_views = np.ones(npts) * len(xvs)
    # scaling = (self._nviews / len(xvs)) if self.config.drop_scale else 1.0
    for vi in range(self._nviews):
      if vi in xvs:
        view_codes[vi], valid_inds = self._encode_view(xvs[vi], vi)

        invalid_inds = np.ones(npts)
        invalid_inds[valid_inds] = 0
        n_available_views[invalid_inds] -= 1
      elif include_missing:
        view_codes[vi] = self._encode_missing_view(vi, npts)

    scaling = (
        self._nviews / n_available_views if self.config.drop_scale else 1.0)
    view_codes *= scaling
    return self._encode_joint(view_codes) if return_joint else view_codes

  def _decode_view(self, z, vi):
    return self._de_layers["D%i"%vi](z)

  def decode(self, z, vi_out=None):
    # Not assuming tied weights yet
    vi_out = range(self._nviews) if vi_out is None else vi_out
    # Check if it's a single view
    if isinstance(vi_out, int):
      return self._decode_view(z, vi_out)

    recons = {vi:self._decode_view(z, vi) for vi in vi_out}
    return recons

  def forward(self, xvs):
    # Solve for alpha
    # if not isinstance(x, torch.Tensor):
    #   # Assuming it is numpy arry
    #   x = torch.from_numpy(x)
    # x.requires_grad_(False)
    zs = self.encode(xvs, include_missing=True, return_joint=True)
    # sampled_zs = [self._sample_codes(*z) for z in zs]
    # This is, for every encoded view, the reconstruction of every view
    recons = self.decode(zs)
    # recons = {}
    # for vi in range(self._n_views):
    #   recons[vi] = self.decode(sampled_zs[vi])
    return zs, recons

  def loss(self, xvs, recons, zs):
    obj = 0.
    npts = len(xvs[utils.get_any_key(xvs)])
    common_views = [vi for vi in recons if vi in xvs]
    for vi in common_views:
      xvi = xvs[vi]
      valid_inds = [i for i in range(npts) if xvi[i] is not None]
      xv_valid = np.array([xvi[i] for i in valid_inds])
      xv_valid = torch_utils.numpy_to_torch(xv_valid)
      obj += self.recon_criterion(xv_valid, recons[vi][valid_inds])

    # Additional loss based on the encoding:
    # Maybe explicitly force the encodings to be similar
    # KLD penalty
    return obj

  def _make_view_subset_shuffler(self):
    # Function to produce shuffled list of all subsets of views in a cycle
    view_subsets = []
    view_range = list(range(self._nviews))
    for nv in view_range:
      view_subsets.extend(list(itertools.combinations(view_range, nv + 1)))

    n_subsets = len(view_subsets)
    while True:
      shuffle_inds = np.random.permutation(n_subsets)
      for sidx in shuffle_inds:
        yield view_subsets[sidx]

  def _shuffle(self, xvs):
    npts = len(xvs[utils.get_any_key(xvs)])
    r_inds = np.random.permutation(npts)
    return {
        vi:[xv[i] for i in r_inds] for vi, xv in xvs.items()
    }

  def _train_loop(self):
    # Loop for epoch
    xvs = self._shuffle(self._view_data)
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      xvs_batch = {vi:xv[b_start:b_end] for vi, xv in xvs.items()}
      keep_subsets = next(self._view_subset_shuffler)
      xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets}

      self.opt.zero_grad()
      zs, recons = self.forward(xvs_dropped_batch)
      loss_val = self.loss(xvs_batch, recons, zs)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self, view_data):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._view_data = view_data
    self._npts = len(view_data[utils.get_any_key(view_data)])#.shape[0]
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))
    self._view_subset_shuffler = self._make_view_subset_shuffler()

    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("\nIteration %i out of %i." % (itr + 1, self.config.max_iters))
        self._train_loop()

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          print("Loss: %.5f" % float(self.itr_loss.detach()))
          print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    self._trained = True
    print("Training finished in %0.2f s." % (time.time() - all_start_time))

  def predict(self, xvs, vi_out=None, rtn_torch=False):
    if not self._trained:
      raise ModelException("Model not yet trained!")
    if vi_out is not None and not isinstance(vi_out, list):
      vi_out = [vi_out]

    # if not isinstance(xvs[utils.get_any_key(xvs)], torch.Tensor):
    #   xvs = {
    #       vi: torch.from_numpy(xv).type(_DTYPE).requires_grad_(False)
    #       for vi, xv in xvs.items()
    #   }

    zs = self.encode(xvs, include_missing=True, return_joint=True)
    preds = self.decode(zs, vi_out)
    if not rtn_torch:
      preds = {vo: p.detach().numpy() for vo, p in preds.items()}

    return preds
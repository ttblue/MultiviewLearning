# (Variational) Autoencoder for multi-view + synchronous
import itertools
import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
import time

from models import torch_models
from utils import torch_utils, utils
from utils.torch_utils import _DTYPE, _TENSOR_FUNC

import IPython


class MAEConfig(object):
  def __init__(
      self, v_sizes, code_size, encoder_params, decoder_params,
      code_sample_noise_var, max_iters, batch_size, lr, verbose):
    self.code_size = code_size
    self.v_sizes = v_sizes
    self.encoder_params = encoder_params
    self.decoder_params = decoder_params
    self.code_sample_noise_var = code_sample_noise_var

    self.max_iters = max_iters
    self.batch_size = batch_size
    self.lr = lr

    self.verbose = verbose


def default_MAE_config(v_sizes, dropout_p=0.5, code_size=10):
  n_views = len(v_sizes)

  # Default Encoder config:
  code_sample_noise_var = 0.0
  output_size = code_size
  layer_units = [256, 128]
  use_vae = False
  activation = nn.ReLU
  last_activation = torch_models.Identity
  encoder_params = {}
  for i in range(n_views):
    input_size = v_sizes[i]
    layer_types, layer_args = torch_utils.generate_linear_types_args(
        input_size, layer_units, output_size)
    encoder_params[i] = torch_models.MNNConfig(
        input_size=input_size, output_size=output_size, layer_types=layer_types,
        layer_args=layer_args, activation=activation,
        last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)

  input_size = code_size
  layer_units = [128, 256]
  use_vae = False
  last_activation = nn.Sigmoid
  decoder_params = {}
  for i in range(n_views):
    output_size = v_sizes[i]
    layer_types, layer_args = torch_utils.generate_linear_types_args(
        input_size, layer_units, output_size)
    decoder_params[i] = torch_models.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)

  max_iters = 1000
  batch_size = 50
  lr = 1e-3
  verbose = True
  config = MAEConfig(
      v_sizes=v_sizes,
      code_size=code_size,
      encoder_params=encoder_params,
      decoder_params=decoder_params,
      code_sample_noise_var=code_sample_noise_var,
      max_iters=max_iters,
      batch_size=batch_size,
      lr=lr,
      verbose=verbose)

  return config


class MultiAutoEncoder(nn.Module):
  # Auto encoder with multi-views
  def __init__(self, config):
    super(MultiAutoEncoder, self).__init__()
    self.config = config

    self._nviews = len(config.v_sizes)
    self._v_inds = np.cumsum(config.v_sizes)[:-1]

    self._initialize_layers()
    self._setup_optimizer()

    self.recon_criterion = nn.MSELoss(reduction="elementwise_mean")

    self._trained = False

  def _initialize_layers(self):
    # Encoder and decoder params
    self._en_layers = nn.ModuleDict()
    self._de_layers = nn.ModuleDict()

    for vi in range(self._nviews):
      self._en_layers["E%i"%vi] = torch_models.MultiLayerNN(
          self.config.encoder_params[vi])
      self._de_layers["D%i"%vi] = torch_models.MultiLayerNN(
          self.config.decoder_params[vi])

  # def _split_views(self, x, rtn_torch=True):
  #   if isinstance(x, torch.Tensor):
  #     # Assuming it is numpy arry
  #     x = np.atleast_2d(torch.numpy(x))

  #   x_v = np.array_split(x, self._v_inds, axis=1)
  #   if rtn_torch:
  #     x_v = [
  #         torch.from_numpy(v).type(_DTYPE).requires_grad_(False)
  #         for v in x_v
  #     ]
  #   return x_v

  def _get_valid(self, xvs):
    npts = len(xvs[utils.get_any_key(xvs)])
    valid_inds = {}
    xvs_valid = {}
    for vi in xvs:
      xvi = xvs[vi]
      valid_inds[vi] = [i for i in range(npts) if xvi[i] is not None]
      # IPython.embed()
      xvs_valid[vi] = torch_utils.numpy_to_torch(
          np.array([xvi[i] for i in valid_inds[vi]]))
    return xvs_valid, valid_inds

  def _encode_view(self, xv, vi):
    return self._en_layers["E%i"%vi](xv)

  def encode(self, xvs, aggregate=None, *args, **kwargs):
    xvs_valid, valid_inds = self._get_valid(xvs)
    codes = {vi:self._encode_view(xv, vi) for vi, xv in xvs_valid.items()}
    if aggregate == "mean":
      npts = len(xvs[utils.get_any_key(xvs)])
      navailable = torch.zeros(npts, 1)
      agg_codes = torch.zeros(npts, self.config.code_size)
      for vi in range(self._nviews):
        view_codes = torch.zeros(npts, self.config.code_size)
        view_codes[valid_inds[vi]] = codes[vi]
        navailable[valid_inds[vi]] += 1
        agg_codes += view_codes
      agg_codes /= navailable
      return agg_codes, valid_inds
    return codes, valid_inds

  # def _sample_codes(self, mu, logvar=None, noise_coeff=None):
  #   if logvar is None:
  #     return mu
  #   # Add noise to code formed for robustness of reconstruction
  #   noise_coeff = (
  #       self.config.code_sample_noise_var if noise_coeff is None else
  #       noise_coeff
  #   )
  #   err = _TENSOR_FUNC(logvar.size()).normal_()
  #   codes = torch.autograd.Variable(err)
  #   var = (noise_coeff * logvar).exp_()
  #   return codes.mul(var).add_(mu)

  def _decode_view(self, z, vi):
    return self._de_layers["D%i"%vi](z)

  def decode(self, zs, vi_out=None):
    # Not assuming tied weights yet
    vi_out = range(self._nviews) if vi_out is None else vi_out
    # Check if it's a single view 
    if isinstance(vi_out, int):
      return self._decode_view(zs, vi_out)

    # IPython.embed()
    # print(zs)
    recons = {vi:self._decode_view(zs, vi) for vi in vi_out}
    return recons

  def forward(self, xvs):
    # Solve for alpha
    # if not isinstance(x, torch.Tensor):
    #   # Assuming it is numpy arry
    #   x = torch.from_numpy(x)
    # x.requires_grad_(False)
    # zs = self.encode(xvs_valid)
    zs, valid_inds = self.encode(xvs)

    # This is, for every encoded view, the reconstruction of every view
    recons = {}
    for vi in range(self._nviews):
      recons[vi] = self.decode(zs[vi])

    return zs, recons, valid_inds

  def loss(self, xvs, recons, zs, valid_inds):
    # xv = self._split_views(x, rtn_torch=True)

    # obj = 0.
    # for vi in recons:
    #   for ridx in range(self._nviews):
    #     obj += self.recon_criterion(xvs[ridx], recons[vi][ridx])

    obj = 0.
    npts = len(xvs[utils.get_any_key(xvs)])

    # valid_inds = {}
    xvs_valid = {}
    for vi in xvs:
      xvi = xvs[vi]
      # valid_inds[vi] = [i for i in range(npts) if xvs[vi] is not None]
      xvs_valid[vi] = torch_utils.numpy_to_torch(
          np.array([xvi[i] for i in valid_inds[vi]]))

    for vi_r in recons:
      # For reconstruction of all views from vi_r
      vi_recons = recons[vi_r]
      vi_r_inds = valid_inds[vi_r]
      common_views = [vi for vi in vi_recons if vi in xvs_valid]
      for vi in common_views:
        xvi = xvs_valid[vi]
        v_inds = valid_inds[vi]
        _, vi_r_locs, vi_locs = np.intersect1d(
            vi_r_inds, v_inds, assume_unique=True, return_indices=True)
        try:
         obj += self.recon_criterion(xvi[vi_locs], vi_recons[vi][vi_r_locs])
        except:
          IPython.embed()

    # Additional loss based on the encoding:
    # Maybe explicitly force the encodings to be similar
    # KLD penalty
    return obj

  def _setup_optimizer(self):
    self.opt = optim.Adam(self.parameters(), self.config.lr)

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

      self.opt.zero_grad()
      zs, recons, valid_inds = self.forward(xvs_batch)
      loss_val = self.loss(xvs_batch, recons, zs, valid_inds)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self, view_data):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._view_data = view_data
    self._npts = len(view_data[utils.get_any_key(view_data)])
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))
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
    if vi_out is None:
      vi_out = np.arange(self._nviews).tolist()
    elif vi_out is not None and not isinstance(vi_out, list):
      vi_out = [vi_out]

    # if not isinstance(xv, torch.Tensor):
    #   xv = [
    #       torch.from_numpy(v).type(_DTYPE).requires_grad_(False)
    #       for v in xv
    #   ]

    codes, valid_inds = self.encode(xvs, aggregate="mean")
    preds = self.decode(codes, vi_out)
    if not rtn_torch:
      preds = {vi: vpred.detach().numpy() for vi, vpred in preds.items()}

    return preds


if __name__=="__main__":
  v_sizes = [3, 3, 3]
  config = default_MAE_Config(v_sizes)
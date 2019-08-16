# (Variational) Autoencoder for multi-view + synchronous
import itertools
import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
import time

from utils import torch_utils as tu
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


def default_RMAE_config(v_sizes):
  n_views = len(v_sizes)
  code_size = 10

  # Default Encoder config:
  output_size = code_size
  layer_units = [50, 20]
  use_var = True
  activation = nn.functional.relu
  last_activation = None
  encoder_params = {
      i: tu.MNNConfig(
          input_size=v_sizes[i],
          output_size=output_size,
          layer_units=layer_units,
          activation=activation,
          last_activation=last_activation,
          use_var=use_var)
      for i in range(n_views)
  }

  input_size = code_size
  use_var = False
  last_activation = nn.sigmoid
  decoder_params = {
      i: tu.MNNConfig(
          input_size=input_size,
          output_size=v_sizes[i],
          layer_units=layer_units,
          activation=activation,
          last_activation=last_activation,
          use_var=use_var)
      for i in range(n_views)
  }

  max_iters = 1000
  batch_size = 50
  lr = 1e-3
  verbose = True
  config = MAEConfig(
      v_sizes=v_sizes,
      code_size=code_size,
      encoder_params=encoder_params,
      decoder_params=decoder_params,
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
    self._en_layers = {}
    self._de_layers = {}

    # Need to call "add_module" so the parameters are found.
    def set_value(vdict, key, name, value):
      self.add_module(name, value)
      vdict[key] = value

    for vi in range(self._nviews):
      set_value(self._en_layers, vi, "en_%i" % vi,
                tu.MultiLayerNN(self.config.encoder_params[vi]))
      set_value(self._de_layers, vi, "de_%i" % vi,
                tu.MultiLayerNN(self.config.decoder_params[vi]))

  def _split_views(self, x, rtn_torch=True):
    if isinstance(x, torch.Tensor):
      # Assuming it is numpy arry
      x = np.atleast_2d(torch.numpy(x))

    x_v = np.array_split(x, self._v_inds, axis=1)
    if rtn_torch:
      x_v = [
          torch.from_numpy(v).type(_DTYPE).requires_grad_(False)
          for v in x_v
      ]
    return x_v

  def _encode_view(self, xv, vi):
    return self._en_layers[vi](xv)

  def encode(self, x):
    xvs = self._split_views(x, rtn_torch=True)
    codes = [self._encode_view(xv, vi) for vi, xv in enumerate(xvs)]
    return codes

  def _sample_codes(self, mu, logvar=None, noise_coeff=None):
    if logvar is None:
      return mu
    # Add noise to code formed for robustness of reconstruction
    noise_coeff = (
        self.config.code_sample_noise_var if noise_coeff is None else
        noise_coeff
    )
    err = _TENSOR_FUNC(logvar.size()).normal_()
    codes = torch.autograd.Variable(err)
    var = (noise_coeff * logvar).exp_()
    return codes.mul(var).add_(mu)

  def _decode_view(self, z, vi):
    return self._de_layers[vi](z)

  def decode(self, z, vi_out=None):
    # Not assuming tied weights yet
    vi_out = range(self._nviews) if vi_out is None else vi_out
    # Check if it's a single view 
    if isinstance(vi_out, int):
      return self._decode_view(z, vi_out)

    recons = [self._decode_view(z, vi) for vi in vi_out]
    return recons

  def forward(self, x):
    # Solve for alpha
    # if not isinstance(x, torch.Tensor):
    #   # Assuming it is numpy arry
    #   x = torch.from_numpy(x)
    # x.requires_grad_(False)
    zs = self.encode(x)
    sampled_zs = [self._sample_codes(*z) for z in zs]

    # This is, for every encoded view, the reconstruction of every view
    recons = {}
    for vi in range(self._nviews):
      recons[vi] = self.decode(sampled_zs[vi])

    return zs, recons

  def loss(self, x, recons, zs):
    xv = self._split_views(x, rtn_torch=True)

    obj = 0.
    for vi in recons:
      for ridx in range(self._nviews):
        obj += self.recon_criterion(xv[ridx], recons[vi][ridx])

    # Additional loss based on the encoding:
    # Maybe explicitly force the encodings to be similar
    # KLD penalty
    return obj

  def _setup_optimizer(self):
    self.opt = optim.Adam(self.parameters(), self.config.lr)

  def _shuffle(self, x):
    npts = x.shape[0]
    r_inds = np.random.permutation(npts)
    return x[r_inds]

  def _train_loop(self, x):
    x = self._shuffle(x)
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      x_batch = x[b_start:b_end]

      self.opt.zero_grad()
      zs, recons = self.forward(x_batch)
      loss_val = self.loss(x_batch, recons, zs)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self, x):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._n_batches = int(np.ceil(x.shape[0] / self.config.batch_size))
    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("\nIteration %i out of %i." % (itr + 1, self.config.max_iters))
        self._train_loop(x)

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          print("Loss: %.5f" % float(self.itr_loss.detach()))
          print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    print("Training finished in %0.2f s." % (time.time() - all_start_time))

  def predict(self, xv, vi_in, vi_out=None, rtn_torch=False):
    if vi_out is None:
      vi_out = np.arange(self._nviews).tolist()

    if not isinstance(vi_in, list):
      vi_in = [vi_in]
      xv = [xv]
    if not isinstance(vi_out, list):
      vi_out = [vi_out]

    if not isinstance(xv, torch.Tensor):
      xv = [
          torch.from_numpy(v).type(_DTYPE).requires_grad_(False)
          for v in xv
      ]

    codes = [self._encode_view(xv[i], vi)[0] for i, vi in enumerate(vi_in)]
    codes = torch.stack(codes, dim=0)
    z = torch.mean(codes, dim=0)

    preds = self.decode(z, vi_out)
    if not rtn_torch:
      preds = [p.detach().numpy() for p in preds]

    return preds[0] if len(vi_out) == 1 else preds


if __name__=="__main__":
  v_sizes = [3, 3, 3]
  config = default_MAE_Config(v_sizes)
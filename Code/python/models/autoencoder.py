# Simple autoencoder: to-do
import itertools
import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
import time

from models.model_base import BaseConfig
from models import torch_models
from utils import torch_utils, utils
from utils.torch_utils import _DTYPE, _TENSOR_FUNC

import IPython


class AEConfig(BaseConfig):
  def __init__(
      self, input_size, code_size, encoder_config, decoder_config, lm,
      dropout_p, max_iters, batch_size, lr, verbose,
      *args, **kwargs):

    super(AEConfig, self).__init__(*args, **kwargs)

    self.input_size = input_size
    self.code_size = code_size
    self.encoder_config = encoder_config
    self.decoder_config = decoder_config
    self.lm = lm
    # self.code_sample_noise_var = code_sample_noise_var

    self.dropout_p = dropout_p

    self.max_iters = max_iters
    self.batch_size = batch_size
    self.lr = lr

    self.verbose = verbose


class AutoEncoder(nn.Module):
  def __init__(self, config):
    super(AutoEncoder, self).__init__()
    self.config = config
    # self.initialize()

  def initialize(self):
    # Set up encoders and decoder
    encoder_config = self.config.encoder_config
    decoder_config = self.config.decoder_config
    encoder_config.dropout_p = self.config.dropout_p
    decoder_config.dropout_p = self.config.dropout_p

    encoder_config.set_sizes(
        input_size=self.config.input_size, output_size=self.config.code_size)
    decoder_config.set_sizes(
        input_size=self.config.code_size, output_size=self.config.input_size)

    self._encoder = torch_models.MultiLayerNN(encoder_config)
    self._decoder = torch_models.MultiLayerNN(decoder_config)
    self.recon_criterion = nn.MSELoss(reduction="mean")

    # Set up optimizer
    self.opt = optim.Adam(self.parameters(), self.config.lr)

    self._trained = False

  def loss(self, x, x_recon, z):
    err = self.recon_criterion(x, x_recon)
    return err

  def _encode(self, x):
    return self._encoder(x)

  def _decode(self, z):
    return self._decoder(z)

  def forward(self, x):
    z = self._encoder(x)
    return z

  def _shuffle(self, xs):
    npts = xs.shape[0]
    r_inds = np.random.permutation(npts)
    return xs[r_inds]

  def _train_loop(self):
    # Loop for epoch
    xs = self._shuffle(self._xs)
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      xs_batch = xs[b_start:b_end]

      self.opt.zero_grad()
      zs_batch = self.forward(xs_batch)
      xs_recon_batch = self._decode(zs_batch)
      loss_val = self.loss(xs_batch, xs_recon_batch, zs_batch)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self, xs, dev=None):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._xs = torch_utils.numpy_to_torch(xs)
    self._dev = dev
    if dev is not None:
      self._view_data.to(dev)

    self._npts = xs.shape[0]
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))
    self._loss_history = []

    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
        #   print(
        #       "\nIteration %i out of %i." % (itr + 1, self.config.max_iters),
        #       end='\r')
        self._train_loop()
        loss_val = float(self.itr_loss.detach())
        self._loss_history.append(loss_val)

        if self.config.verbose:
          # itr_duration = time.time() - itr_start_time
          # print("Loss: %.5f" % float(self.itr_loss.detach()))
          # print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))
          itr_diff_time = time.time() - itr_start_time
          print("Iteration %i out of %i (in %.2fs). Loss: %.5f. " %
                (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
                end='\r')
      if self.config.verbose and itr >= 0:
        # itr_duration = time.time() - itr_start_time
        # print("Loss: %.5f" % float(self.itr_loss.detach()))
        # print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))
        itr_diff_time = time.time() - itr_start_time
        loss_val = float(self.itr_loss.detach())
        print("Iteration %i out of %i (in %.2fs). Loss: %.5f. " %
              (itr + 1, self.config.max_iters, itr_diff_time, loss_val))

    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    self.eval()
    self._trained = True
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
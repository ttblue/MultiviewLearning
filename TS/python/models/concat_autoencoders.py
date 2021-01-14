# Baseline autoencoders

import numpy as np

import torch
from torch import nn

from models.model_base import BaseConfig
from models import multi_ae
from utils import utils


class MCAConfig(BaseConfig):
  def __init__(
      self, encoder_params, decoder_params, code_size,# view_dropout,
      max_iters, batch_size, lr, verbose,
      *args, **kwargs):

    super(MCAConfig, self).__init__(*args, **kwargs)

    self.encoder_params = encoder_params
    self.decoder_params = decoder_params
    self.code_size = code_size

    # self.view_dropout = view_dropout

    self.max_iters = max_iters
    self.batch_size = batch_size
    self.lr = lr

    self.verbose = verbose


# class MultiviewConcatAutoEncoder(multi_ae.MultiAutoEncoder):
class MultiviewConcatAutoEncoder(nn.Module):
  def __init__(self, config):
    super(MultiviewConcatAutoEncoder, self).__init__()
    self.config = config

  def _initialize_layers(self):
    # Encoder and decoder params
    self._encoder_net = torch_models.MultiLayerNN(
          self.config.encoder_params)
    self._decoder_net = torch_models.MultiLayerNN(
          self.config.decoder_params)

  def _concatenate_multiview(self, xvs, filler=None, rtn_torch=True):
    # Assumes numpy arrays
    npts = len(xvs[utils.get_any_key(xvs)])

    if filler is None:
      filler = {vi: np.zeros(vs) for vi, vs in self._v_sizes.items()}

    xvs_filled = []
    for vi in range(self._n_views):
      if vi in xvs:
        xvi = xvs[vi]
        xvi_filled = np.array(
            [(filler[vi] if ft is None else np.array(ft)) for ft in xvi])
      else:
        xvi_filled = np.tile(filler[vi], (npts, 1))
      xvs_filled.append(xvi_filled)
    cat_xvs = np.concatenate(xvi_filled, axis=1)

    return torch_utils.numpy_to_torch(cat_xvs) if rtn_torch else cat_xvs

  def _split_cat_views(self, xvs_cat, rtn_torch=True):
    # Assuming torch as input
    split_sizes = [self._v_sizes[vi] for vi in range(self._n_views)]
    xvs_split = torch.split(xvs_cat, split_sizes, dim=1)

    xvs = {vi: xvi for vi, xvi in enumerate(xvs_split)}
    return xvs if rtn_torch else torch_utils.dict_torch_to_numpy(xvs)

  def encode(self, xvs):
    xvs_cat = torch_utils.numpy_to_torch(self._concatenate_multiview(xvs))
    code = self._encoder_net(xvs_cat)
    return code

  def decode(self, code):
    xvs_cat_recon = self._decoder_net(code)
    xvs_recon = self._split_cat_views(xvs_cat_recon, rtn_torch=True)
    return xvs_recon

  def forward(self, xvs):
    code = self.encode(xvs)
    xvs_recon = self.decode(code)

    return xvs_recon, code

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
      recon, code, valid_inds = self.forward(xvs_batch)
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


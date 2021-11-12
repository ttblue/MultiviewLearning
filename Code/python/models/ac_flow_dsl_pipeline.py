# Loss function format:
# obj = loss_func(model, mv_data, label_data, zs, ls, lds)
# model is an instance of MACFlowDSLTrainer.
# zs -- representation encodings.

import copy
import itertools
import numpy as np
import scipy
import torch
from torch import nn, optim
import time

from models import autoencoder, conditional_flow_transforms,\
                   conditional_flow_transforms_mp, flow_likelihood,\
                   flow_transforms
from models.model_base import ModelException, BaseConfig
from models.conditional_flow_transforms import MVZeroImpute
from models.ac_flow_pipeline import MultiviewACFlowTrainer, MACFTConfig
from utils import math_utils, torch_utils, utils


import IPython


class MACFDTConfig(BaseConfig):
  def __init__(
      self, expand_b=True, use_pre_view_ae=False, no_view_tfm=False,
      likelihood_config=None, base_dist="gaussian", dsl_coeff=1.0,
      batch_size=50, lr=1e-3, max_iters=1000,
      verbose=True, *args, **kwargs):

    self.dsl_coeff = dsl_coeff
    super(MACFDTConfig, self).__init__(
        expand_b=expand_b, use_pre_view_ae=use_pre_view_ae,
        no_view_tfm=no_view_tfm, likelihood_config=likelihood_config,
        base_dist=base_dist, dsl_coeff=dsl_coeff, batch_size=batch_size,
        lr=lr, max_iters=max_iters, verbose=verbose,
        *args, **kwargs)


class MACFlowDSLTrainer(MultiviewACFlowTrainer):
  def __init__(self, config):
    super(MACFlowDSLTrainer, self).__init__(config)

  def set_ds_loss(self, loss_func):
    self._ds_loss = loss_func

  def _nll(self, l_vs, ld_vs, aggregate=True):
    # Give the log likelihood under transformed latent state
    x_nll = {}
    # IPython.embed()
    for vi, lvi in l_vs.items():
      x_nll[vi] = self._cond_lhoods["v_%i" % vi].nll(lvi) - ld_vs[vi]

    if aggregate:
      mean_ll = 0
      for vi, vi_nll in x_nll.items():
        mean_ll += torch.mean(vi_nll)
      return mean_ll
    return x_nll

  def loss(self, x_vs, ys, z_vs, l_vs, ld_vs, *args, **kwargs):
    loss_val = self._nll(l_vs, ld_vs, aggregate=True)
    if self.config.dsl_coeff > 0:
      # Assuming it isn't negative.
      ds_loss = self._ds_loss(
           self, x_vs, ys, z_vs, l_vs, ld_vs, *args, **kwargs)
      loss_val += self.config.dsl_coeff * ds_loss
    return loss_val

  def _shuffle(self, x_vs, ys):
    npts = x_vs[utils.get_any_key(x_vs)].shape[0]
    r_inds = np.random.permutation(npts)
    x_vs_shuffled = {vi:xv[r_inds] for vi, xv in x_vs.items()}
    ys_shuffled = ys[r_inds]
    return x_vs_shuffled, ys_shuffled

  def _train_loop(self):
    x_vs, ys = self._shuffle(self._view_data, self._ys)
    self.itr_loss = 0.
    for batch_idx in range(self._n_batches):
      batch_start = batch_idx * self.config.batch_size
      batch_end = min(batch_start + self.config.batch_size, self._npts)

      self._batch_npts = batch_end - batch_start
      xvs_batch = {vi:xv[batch_start:batch_end] for vi, xv in x_vs.items()}
      ys_batch = ys[batch_start:batch_end]
      available_views = next(self._view_subset_shuffler)
      b_o_batch = {
          vi:(torch.ones(self._batch_npts) if vi in available_views else
              torch.zeros(self._batch_npts))
          for vi in xvs_batch
      }
      # if self.config.verbose:
      #   print("  View subset selected: %s" % (available_views, ))
      # globals().update(locals())
      l_batch, z_batch, ld_batch = self._transform(
          xvs_batch, b_o_batch, rtn_logdet=True)

      # IPython.embed()
      # available_views = next(self._view_subset_shuffler)
      # xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets}
      self.opt.zero_grad()
      loss_val = self.loss(
          xvs_batch, ys_batch, z_batch, l_batch, ld_batch)
      if torch.isnan(loss_val):
        print("nan loss value. Exiting training.")
        self._training = False
        return
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

      if available_views not in self._view_subset_counts:
        self._view_subset_counts[available_views] = 0
      self._view_subset_counts[available_views] += 1

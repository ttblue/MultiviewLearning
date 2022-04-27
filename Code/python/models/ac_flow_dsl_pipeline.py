# Loss function format:
# obj = loss_func(mv_data, label_data, zs, ls, lds)
# model is an instance of MACFlowDSLTrainer.
# zs -- representation encodings.

import copy
import itertools
import numpy as np
import scipy
import torch
from torch import nn, optim
from torch.nn.utils import _stateless
import time

from models import autoencoder, conditional_flow_transforms,\
                   conditional_flow_transforms_mp, flow_likelihood,\
                   flow_transforms
from models.model_base import ModelException, BaseConfig
from models.conditional_flow_transforms import MVZeroImpute
from models.ac_flow_pipeline import MultiviewACFlowTrainer, MACFTConfig
from utils import math_utils, torch_utils, utils


import IPython


class MACFDTConfig(MACFTConfig):
  def __init__(
      self, expand_b=True, use_pre_view_ae=False, no_view_tfm=False,
      likelihood_config=None, base_dist="gaussian", dsl_coeff=1.0,
      batch_size=50, lr=1e-3, max_iters=1000, grad_clip=5.,
      verbose=True, *args, **kwargs):

    self.dsl_coeff = dsl_coeff
    super(MACFDTConfig, self).__init__(
        expand_b=expand_b, use_pre_view_ae=use_pre_view_ae,
        no_view_tfm=no_view_tfm, likelihood_config=likelihood_config,
        base_dist=base_dist, dsl_coeff=dsl_coeff, batch_size=batch_size,
        lr=lr, max_iters=max_iters, grad_clip=grad_clip, verbose=verbose,
        *args, **kwargs)


class MACFlowDSLTrainer(MultiviewACFlowTrainer):
  def __init__(self, config):
    super(MACFlowDSLTrainer, self).__init__(config)
    self._ds_loss = None

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
        mean_ll = mean_ll + torch.mean(vi_nll)
      return mean_ll
    return x_nll

  def loss(self, x_vs_recon, ys, z_vs, l_vs, ld_vs, *args, **kwargs):
    loss_val = self._nll(l_vs, ld_vs, aggregate=True)
    self._curr_nll_loss = loss_val.detach().numpy()
    self._curr_dsl_loss = 0.

    if self.config.dsl_coeff > 0 and self._ds_loss is not None:
      # Assuming it isn't negative.
      ds_loss = self._ds_loss(
           x_vs_recon, ys, z_vs, l_vs, ld_vs, *args, **kwargs)
      loss_val = loss_val + self.config.dsl_coeff * ds_loss
      self._curr_dsl_loss = self.config.dsl_coeff * ds_loss.detach().numpy()
      # print(self._curr_dsl_loss)
    return loss_val

  def forward(self, x_vs, b_o, ys, available_views, *args, **kwargs):
    l_vs, z_vs, ld_vs = self._transform(x_vs, b_o, rtn_logdet=True)
    sampled_views = [i for i in range(self._n_views) if i not in available_views]
    samples = {
        vi: self._sample_view(vi, x_vs, b_o)
        for vi in sampled_views
    }
    xvs_recon = {
        vi:(samples[vi] if vi in sampled_views else xvi)
        for vi, xvi in x_vs.items()
    }
    return xvs_recon, z_vs, l_vs, ld_vs

  def _make_view_subset_shuffler(self):
    # Function to produce shuffled list of all subsets of views in a cycle
    view_subsets = []
    view_range = list(range(self._nviews))
    for nv in range(1, self._nviews):
      view_subsets.extend(list(itertools.combinations(view_range, nv)))

    n_subsets = len(view_subsets)
    while True:
      shuffle_inds = np.random.permutation(n_subsets)
      for sidx in shuffle_inds:
        yield view_subsets[sidx]

  def _shuffle(self, x_vs, ys):
    npts = x_vs[utils.get_any_key(x_vs)].shape[0]
    r_inds = np.random.permutation(npts)
    x_vs_shuffled = {vi:xv[r_inds] for vi, xv in x_vs.items()}
    ys_shuffled = ys[r_inds]
    return x_vs_shuffled, ys_shuffled

  def _get_imputed(self, samples_vs, x_vs, b_vs):
    imputed_vs = {
        vi: (x_vi * b_vs[vi] + (1 - b_vs[vi]) * sample_vs[vi])
        for vi, x_vi in x_vs.items()
    }
    return imputed_vs

  def _get_cloned_parameters(self):
    cloned_parameters = {
        k: v.clone() for k, v in dict(self.named_parameters()).items()}
    return cloned_parameters

  def _train_loop(self):
    torch.autograd.set_detect_anomaly(True)
    x_vs, ys = self._shuffle(self._view_data, self._ys)
    self.itr_loss = 0.
    self.itr_nll_loss = 0.
    self.itr_dsl_loss = 0.
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

      # args = (xvs_batch, b_o_batch, ys_batch, available_views)
      # # IPython.embed()
      # xvs_recon_batch, z_vs_batch, l_vs_batch, ld_vs_batch = (
      #     _stateless.functional_call(self, self._get_cloned_parameters(), args))
      # loss_val = self.loss(
      #     xvs_recon_batch, ys_batch, z_vs_batch, l_vs_batch, ld_vs_batch)
      # if self.config.verbose:
      #   print("  View subset selected: %s" % (available_views, ))
      # globals().update(locals())
      l_batch, z_batch, ld_batch = self._transform(
          xvs_batch, b_o_batch, rtn_logdet=True)
      # xvs_batch_av = {vi:xv for vi, xv in xvs_batch.items() if vi in available_views}
      v_samp = [i for i in range(self._n_views) if i not in available_views]

      # samp_batch1 = self.sample(xvs_batch, b_o_batch, v_samp)
      samp_batch = {
          vi: self._sample_view(vi, xvs_batch, b_o_batch)
          for vi in v_samp
      }
      # print(samp_batch.keys())
      # xvs_batch = {vi:xvi.detach() for vi, xvi in xvs_batch.items()}
      xvs_batch_recon = {
          vi:(samp_batch[vi] if vi in v_samp else xvi)
          for vi, xvi in xvs_batch.items()
      }
      # IPython.embed()
      # available_views = next(self._view_subset_shuffler)
      # xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets
      loss_val = self.loss(
          xvs_batch_recon, ys_batch, z_batch, l_batch, ld_batch)
      if torch.isnan(loss_val):
        print("nan loss value. Exiting training.")
        self._training = False
        return
      # print(v_samp)
      self.opt.zero_grad()
      try:
        loss_val.backward() #retain_graph=True)
      except Exception as e:
        IPython.embed()
        raise(e)

      nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
      self.opt.step()

      self.itr_loss = self.itr_loss + loss_val.detach().numpy()
      self.itr_nll_loss = self.itr_nll_loss + self._curr_nll_loss
      self.itr_dsl_loss = self.itr_dsl_loss + self._curr_dsl_loss
      # print(self.itr_loss, self.itr_nll_loss, self.itr_dsl_loss)
      # print("ABCD")

      if available_views not in self._view_subset_counts:
        self._view_subset_counts[available_views] = 0
      self._view_subset_counts[available_views] += 1

  def _get_trainable_parameters(self):
    trainable_parameters = [p for p in self.parameters() if p.requires_grad_]
    return trainable_parameters

  def fit(self, x_vs, ys, b_o=None, loss_func=None, dev=None):
    # Currently not using input b_o
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._training = True
    # For convenience
    self._npts = x_vs[utils.get_any_key(x_vs)].shape[0]
    self._view_data = torch_utils.dict_numpy_to_torch(x_vs)
    self._n_views = len(x_vs)
    self._ys = torch_utils.numpy_to_torch(ys)
    if dev is not None:
      self._view_data.to(dev)
      self._ys.to(dev)

    if loss_func:
      self.set_ds_loss(loss_func)
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))

    self._view_subset_shuffler = self._make_view_subset_shuffler()
    self._view_subset_counts = {}
    # Set up optimizer
    self.opt = optim.Adam(self._get_trainable_parameters(), self.config.lr)

    self._loss_history = []
    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_diff_time = 0.
          itr_start_time = time.time()
        self._train_loop()
        if not self._training:
          break

        loss_val = self.itr_loss
        loss_nll = self.itr_nll_loss
        loss_dsl = self.itr_dsl_loss
        self._loss_history.append((loss_val, loss_nll, loss_dsl))
        if self.config.verbose:
          itr_diff_time = time.time() - itr_start_time
          print("  Iteration %i/%i (in %.2fs). Loss: %.5f. NLL: %.5f. DSL: %.5f" %
                (itr + 1, self.config.max_iters, itr_diff_time, loss_val, loss_nll, loss_dsl),
                end='\r')
      if self.config.verbose:
        print("  Iteration %i/%i (in %.2fs). Loss: %.5f. NLL: %.5f. DSL: %.5f" %
              (itr + 1, self.config.max_iters, itr_diff_time, loss_val, loss_nll, loss_dsl),
              end='\r')
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")

    self._training = False
    self._loss_history = np.array(self._loss_history)
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
    return self
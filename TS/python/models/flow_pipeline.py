import numpy as np
import scipy
import torch
from torch import nn, optim
import time

from models.model_base import ModelException, BaseConfig
from models import flow_likelihood, flow_transforms
from utils import math_utils, torch_utils, utils


import IPython


class MFTConfig(BaseConfig):
  def __init__(
      self, shared_tfm_config_list=[], view_tfm_config_lists={},
      likelihood_config=None, base_dist="gaussian",
      batch_size=50, lr=1e-3, max_iters=1000,
      *args, **kwargs):
    super(MFTConfig, self).__init__(*args, **kwargs)

    self.shared_tfm_config_list = shared_tfm_config_list
    self.view_tfm_config_lists = view_tfm_config_lists
    self.likelihood_config = likelihood_config
    self.base_dist = base_dist

    self.batch_size = batch_size
    self.lr = lr
    self.max_iters = max_iters


class MultiviewFlowTrainer(nn.Module):
  # Class for basic flow training without arbitrary conditioning.
  def __init__(self, config):
    self.config = config
    super(MultiviewFlowTrainer, self).__init__()

  def initialize(self, shared_tfm_init_args, view_tfm_init_args):
    # Initialize transforms, mm model, optimizer, etc.

    # Shared and view transform initialization
    self._shared_tfm = flow_transforms.make_transform(
        self.config.shared_tfm_config_list, shared_tfm_init_args)
    self._nviews = len(self.config.view_tfm_config_lists)
    self._view_tfms = nn.ModuleDict()
    self._dim = 0
    self._view_dims = {}
    for vi, cfg_list in self.config.view_tfm_config_lists.items():
      init_args = view_tfm_init_args[vi]
      tfm = flow_transforms.make_transform(cfg_list, init_args)
      self._view_tfms["v_%i"%vi] = tfm
      self._view_dims[vi] = tfm._dim
      self._dim += tfm._dim  # Assume there are no "None" dims

    if self.config.likelihood_config is None:
      if self.config.base_dist not in flow_likelihood.BASE_DISTS:
        raise NotImplementedError(
            "Base dist. type %s not implemented and likelihood model"
            " not provided." % self.config.base_dist)
      else:
        self._likelihood_model = flow_likelihood.make_base_distribution(
            self.config.base_dist, self._dim)
    else:
      self._likelihood_model = flow_likelihood.make_likelihood_model(
          self.config.likelihood_config, self._dim)

  def _cat_views(self, zvs):
    zvs_cat = torch.cat([zvs[vi] for vi in range(self._nviews)], dim=1)
    return zvs_cat

  def _split_views(self, zvs_cat):
    split_sizes = [self._view_dims[vi] for vi in range(self._nviews)]
    zvs_split = {
        vi:zvi for vi, zvi in enumerate(torch.split(zvs_cat, split_sizes, 1))}
    return zvs_split

  def _transform_views(self, xvs, rtn_logdet=True):
    z_vs = {
        vi: self._view_tfms["v_%i"%vi](
            xvi, rtn_torch=True, rtn_logdet=rtn_logdet)
        for vi, xvi in xvs.items()
    }
    if rtn_logdet:
      log_jac_det = torch.sum(
          torch.stack([z_vi[1] for z_vi in z_vs.values()], 1), 1)
      z_vs = {vi: zvi[0] for vi, zvi in z_vs.items()}
    zvs_cat = self._cat_views(z_vs)

    return (zvs_cat, log_jac_det) if rtn_logdet else zvs_cat

  def _transform(self, xvs, rtn_logdet=True):
    # Transform input covariates using invertible transforms.
    zvs_cat = self._transform_views(xvs, rtn_logdet=rtn_logdet)
    if rtn_logdet:
      zvs_cat, log_jac_det_vs = zvs_cat
      z, log_jac_det_shared = self._shared_tfm(
          zvs_cat, rtn_torch=True, rtn_logdet=True)
      # Separate view transforms into common space can be considered as a single 
      # transform with block-diagonal Jacobian. Then, determinant is product
      # of determinants of diagonal blocks, which  are individual transform
      # jacobian determinants.
      log_jac_det = log_jac_det_vs + log_jac_det_shared
      return z, log_jac_det

    z = self._shared_tfm(zvs_cat, rtn_torch=True, rtn_logdet=False)
    return z

  def _nll(self, z):
    # Give the log likelihood under transformed z
    z_nll = self._likelihood_model.nll(z)
    return z_nll

  def forward(self, xvs, rtn_logdet=True):
    return self._transform(xvs, rtn_logdet)

  def loss(self, z_nll, log_jac_det):
    nll_loss = -torch.sum(log_jac_det) + torch.sum(z_nll)
    return nll_loss

  def _shuffle(self, xvs):
    npts = xvs[utils.get_any_key(xvs)].shape[0]
    r_inds = np.random.permutation(npts)
    return {vi:xv[r_inds] for vi, xv in xvs.items()}

  def _train_loop(self):
    xvs = self._shuffle(self._view_data)
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      xvs_batch = {vi:xv[b_start:b_end] for vi, xv in xvs.items()}
      globals().update(locals())
      z_batch, log_jac_det = self._transform(xvs_batch, rtn_logdet=True)
      z_batch_nll = self._nll(z_batch)

      # keep_subsets = next(self._view_subset_shuffler)
      # xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets}
      self.opt.zero_grad()
      loss_val = self.loss(z_batch_nll, log_jac_det)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def invert(self, z, rtn_torch=True):
    z_inv = self._shared_tfm.inverse(z)
    z_views = self._split_views(z_inv)
    x_views = {
        vi:self._view_tfms["v_%i"%vi].inverse(z_vi, rtn_torch)
        for vi, z_vi in z_views.items()
    }
    return x_views

  def fit(self, xvs):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    # For convenience
    npts = xvs[utils.get_any_key(xvs)].shape[0]
    self._view_data = torch_utils.dict_numpy_to_torch(xvs)
    self._n_batches = int(np.ceil(npts / self.config.batch_size))

    # Set up optimizer
    self.opt = optim.Adam(self.parameters(), self.config.lr)

    self._loss_history = []
    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
        self._train_loop()

        loss_val = float(self.itr_loss.detach())
        self._loss_history.append(loss_val)
        if self.config.verbose:
          itr_diff_time = time.time() - itr_start_time
          print("\nIteration %i out of %i (in %.2fs). Loss: %.5f" %
                (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
                end='\r')
      if self.config.verbose:
        print("\nIteration %i out of %i (in %.2fs). Loss: %.5f" %
              (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
              end='\r')
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")

    self._loss_history = np.array(self._loss_history)
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
    return self

  def sample(self, n, rtn_torch=True):
    z_samples = self._likelihood_model.sample((n,))
    return self.invert(z_samples, rtn_torch)
    # raise NotImplementedError("Implement this!")

  def log_likelihood(self, x):
    z, log_jac_det = self._transform(x, rtn_logdet=True)
    ll = self._likelihood_model.log_prob(z) + log_jac_det
    return float(ll.sum())
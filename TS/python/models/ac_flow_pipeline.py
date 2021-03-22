import copy
import numpy as np
import scipy
import torch
from torch import nn, optim
import time

from models.model_base import ModelException, BaseConfig
from models import conditional_flow_transforms, flow_likelihood, flow_transforms
from utils import math_utils, torch_utils, utils


import IPython


# Some utilities:
def MVZeroImpute(xvs, v_dims, ignored_view=None, expand_b=True):
  # @expand_b: Blow up binary flag to size of views if True, keep single bit
  #     for each view otherwise.
  # @ignored_view: If not None, view to ignore while zero-imputing and
  #     concatenating.
  tot_dim = np.sum([dim for vi, dim in v_dims.items() if vi != ignored_view])
  n_pts = xvs[utils.get_any_key(xvs)].shape[0]

  b_vals = np.empty((n_pts, 0))
  imputed_vals = np.empty((n_pts, 0))
  vi_idx = 0
  for vi, dim in v_dims.items():
    if vi == ignore_view:
      continue
    vi_vals = xvs[vi] if vi in xvs else np.zeros((npts, dim))
    imputed_vals = np.concatenate([imputed_vals, vi_vals], axis=1)

    vi_b = np.ones((n_pts,)) * (vi in xvs)
    if expand_b:
      vi_b = np.tile(vi_b, (1, dim))
    b_vals = np.concatenate([b_vals, vi_b], axis=1)

  output = np.c_[imputed_vals, b_vals]
  return output
#


class MACFTConfig(BaseConfig):
  def __init__(
      expand_b=True, likelihood_config=None, base_dist="gaussian",
      batch_size=50, lr=1e-3, max_iters=1000, verbose=True,
      *args, **kwargs):
    super(MFTConfig, self).__init__(*args, **kwargs)

    self.expand_b = expand_b
    self.likelihood_config = likelihood_config
    self.base_dist = base_dist

    self.batch_size = batch_size
    self.lr = lr
    self.max_iters = max_iters

    self.verbose = verbose


class MultiviewACFlowTrainer(nn.Module):
  # Class for basic flow training without arbitrary conditioning.
  def __init__(self, config):
    self.config = config
    super(MultiviewACFlowTrainer, self).__init__()

  def initialize(
      self, view_tfm_config_lists, view_tfm_init_args,
      cond_tfm_config_lists, cond_tfm_init_args):
    # Initialize transforms, mm model, optimizer, etc.

    self._view_tfm_config_lists = view_tfm_config_lists
    self._cond_tfm_config_lists = cond_tfm_config_lists
    # View encoders:
    self._nviews = len(view_tfm_config_lists)
    self._view_tfms = nn.ModuleDict()
    self._dim = 0
    self._view_dims = {}
    for vi, cfg_list in view_tfm_config_lists.items():
      init_args = view_tfm_init_args[vi]
      tfm = flow_transforms.make_transform(cfg_list, init_args)
      self._view_tfms["v_%i"%vi] = tfm
      self._view_dims[vi] = tfm._dim
      self._dim += tfm._dim  # Assume there are no "None" dims

    # Conditional transforms
    # self._shared_tfm = conditional_flow_transforms.make_transform(
    #     self.config.shared_tfm_config_list, shared_tfm_init_args)
    self._cond_tfms = nn.ModuleDict()
    for vi, cfg_list in cond_tfm_config_lists.items():
      init_args = cond_tfm_init_args[vi]
      tfm = conditional_flow_transforms.make_transform(cfg_list, init_args)
      self._cond_tfms["v_%i"%vi] = tfm

    # if self.config.likelihood_config is None:
    #   if self.config.base_dist not in flow_likelihood.BASE_DISTS:
    #     raise NotImplementedError(
    #         "Base dist. type %s not implemented and likelihood model"
    #         " not provided." % self.config.base_dist)
    #   else:
    #     self._likelihood_model = flow_likelihood.make_base_distribution(
    #         self.config.base_dist, self._dim)
    # else:
    #   self._likelihood_model = flow_likelihood.make_likelihood_model(
    #       self.config.likelihood_config, self._dim)
    if self.config.likelihood_config is None:
      if self.config.base_dist not in flow_likelihood.BASE_DISTS:
        raise NotImplementedError(
            "Base dist. type %s not implemented and likelihood model"
            " not provided." % self.config.base_dist)
      else:
        self._cond_lhoods = nn.ModuleDict()
        for vi, vdim in self._view_dims.items():
          self._cond_lhoods["v_%i"%vi] = flow_likelihood.make_base_distribution(
              self.config.base_dist, vdim)
    else:
      self._cond_lhoods = nn.ModuleDict()
      for vi, vdim in self._view_dims.items():
        self._cond_lhoods["v_%i"%vi] = flow_likelihood.make_base_distribution(
            self.config.likelihood_config, vdim)
      # self._likelihood_model = flow_likelihood.make_likelihood_model(
      #     self.config.likelihood_config, self._dim)

  def _cat_views(self, zvs):
    zvs_cat = torch.cat([zvs[vi] for vi in range(self._nviews)], dim=1)
    return zvs_cat

  def _split_views(self, zvs_cat):
    split_sizes = [self._view_dims[vi] for vi in range(self._nviews)]
    zvs_split = {
        vi:zvi for vi, zvi in enumerate(torch.split(zvs_cat, split_sizes, 1))}
    return zvs_split

  def _encode_views(self, xvs, rtn_logdet=True):
    z_vs = {
        vi: self._view_tfms["v_%i"%vi](
            xvi, rtn_torch=True, rtn_logdet=rtn_logdet)
        for vi, xvi in xvs.items()
    }
    if rtn_logdet:
      # log_jac_det = torch.sum(
      #     torch.stack([z_vi[1] for z_vi in z_vs.values()], 1), 1)
      log_jac_det = {vi: z_vi[1] for vi, z_vi in z_vs.items()}
      z_vs = {vi: z_vi[0] for vi, z_vi in z_vs.items()}
    # zvs_cat = self._cat_views(z_vs)

    return (z_vs, log_jac_det) if rtn_logdet else z_vs

  def _transform_views_cond(self, z_vs, available_views=None, rtn_logdet=True):
    z_available = (
        z_vs if available_views is None else
        {vi: z_vs[vi] for vi in available_views}
    )

    expand_b = self.config.expand_b
    l_vs = {}
    logdet_vs = {}
    for vi, z_vi in z_vs.items():
      z_cat = MVZeroImpute(
          z_available, self._view_dims, ignored_view=vi, expand_b=expand_b)
      l_vi = self._cond_tfms["v_%i" % vi](z_vi, z_cat, rtn_logdet=rtn_logdet)
      if rtn_logdet:
        l_vi, logdet_vs[vi] = l_vi
      l_vs[vi] = l_vi

    return (l_vs, logdet_vs) if rtn_logdet else l_vs

  def _transform(self, xvs, available_views, rtn_logdet=True):
    # Transform input covariates using invertible transforms.
    z_vs = self._encode_views(xvs, rtn_logdet=rtn_logdet)
    if rtn_logdet:
      z_vs, logdet_vs = z_vs
      # z, log_jac_det_shared = self._shared_tfm(
      #     zvs_cat, rtn_torch=True, rtn_logdet=True)
      # Separate view transforms into common space can be considered as a single 
      # transform with block-diagonal Jacobian. Then, determinant is product
      # of determinants of diagonal blocks, which  are individual transform
      # jacobian determinants.
      # log_jac_det = log_jac_det_vs + log_jac_det_shared
      # return z, log_jac_det

    l_vs = self._transform_views_cond(
        z_vs, available_views, rtn_torch=True, rtn_logdet=rtn_logdet)
    if rtn_logdet:
      l_vs, l_ld_vs = l_vs
      logdet_vs = {
          vi: logdet_vs[vi] + l_ld_vs[vi] for vi in l_ld_vs
      }
      return l_vs, z_vs, logdet_vs

    return l_vs, z_vs

  def _nll(self, l_vs):
    # Give the log likelihood under transformed latent state
    l_nll = {}
    for vi, lvi in l_vs.items():
      l_nll[vi] = self._cond_lhoods["v_%i" % vi](lvi)
    return l_nll

  def forward(self, xvs, rtn_logdet=True):
    zl_vs = self._transform(xvs, rtn_logdet)
    if rtn_logdet:
      z_vs, l_vs, ld_vs = zl_vs
      output = (l_vs, ld_vs)
    else:
      z_vs, l_vs = zl_vs
      output = l_vs

    return l_vs

  def loss(self, l_nll, ld_vs, aggregate="sum", *args, **kwargs):
    nll_loss = {}
    total_loss = 0.
    for vi in l_nll:
      l_nll_vi = l_nll[vi]
      ld_vi = ld_vs[vi]
      nll_loss[vi] = -torch.sum(ld_vi) + torch.sum(l_nll_vi)
      total_loss += nll_loss

    if aggregate == "sum":
      return total_loss
    return nll_loss

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
      available_views = next(self._view_subset_shuffler)
      # globals().update(locals())
      l_batch, z_batch, ld_batch = self._transform(
          xvs_batch, available_views, rtn_logdet=True)
      l_batch_nll = self._nll(l_batch)

      # available_views = next(self._view_subset_shuffler)
      # xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets}
      self.opt.zero_grad()
      loss_val = self.loss(l_batch_nll, ld_batch, aggregate="sum")
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def invert(self, l_vs, x_o, rtn_torch=True):
    z_o = self._encode_views(x_o, rtn_logdet=False)
    x_vs = {}
    for vi, lvi in l_vs.items():
      z_cat = MVZeroImpute(
          z_o, self._view_dims, ignored_view=vi, expand_b=expand_b)

      l_inv = self._cond_tfm["v_%i"%vi].inverse(z, z_cat)
      x_vs[vi] = self._view_tfms["v_%i"%vi].inverse(l_inv)

    return x_vs

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

  def sample(self, n, x_o, rtn_torch=True):
    sampling_views = [vi for vi in range(self._nviews) if vi not in x_o]
    samples = {}
    l_samples = {vi:self._cond_lhoods[vi].sample((n,)) for vi in sampling_views}
    return self.invert(l_samples, x_o, rtn_torch=rtn_torch)
    # raise NotImplementedError("Implement this!")

  # def log_likelihood(self, x):
  #   z, log_jac_det = self._transform(x, rtn_logdet=True)
  #   ll = self._likelihood_model.log_prob(z) + log_jac_det
  #   return float(ll.sum())

    # raise NotImplementedError("Implement this!")


# class ACFlowTrainer(FlowTrainer):
#   # Class for AC (arbitrary conditioning) flow training.
#   def __init__(self, config):
#     super(ACFlowTrainer, self).__init__(config)

#   def initialize(self):
#     pass

#   def _train_loop(self):
#     pass

#   def fit(self, x):
#     raise NotImplementedError("Implement this!")

#   def sample(self, n):
#     raise NotImplementedError("Implement this!")

#   def log_likelihood(self, x):
#     raise NotImplementedError("Implement this!")


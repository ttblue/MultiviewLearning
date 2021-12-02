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
from utils import math_utils, torch_utils, utils


import IPython

_NP_DTYPE = np.float64


class MACFTConfig(BaseConfig):
  def __init__(
      self, expand_b=True, use_pre_view_ae=False, no_view_tfm=False,
      likelihood_config=None, base_dist="gaussian",
      batch_size=50, lr=1e-3, max_iters=1000,
      verbose=True, *args, **kwargs):
    super(MACFTConfig, self).__init__(*args, **kwargs)

    self.expand_b = expand_b
    self.no_view_tfm = no_view_tfm
    self.use_pre_view_ae = use_pre_view_ae

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
      self, view_sizes, view_tfm_configs, view_tfm_init_args,
      cond_tfm_configs, cond_tfm_init_args, view_ae_configs,
      view_ae_model_files=None, dev=None):
    # Initialize transforms, mm model, optimizer, etc.

    # IPython.embed()
    self._view_dims = view_sizes
    self._nviews = len(view_sizes)
    self._view_tfm_configs = view_tfm_configs
    self._cond_tfm_configs = cond_tfm_configs

    self._dev = dev

    # View encoders:
    self._use_pre_view_ae = (
        False if view_ae_configs is None else
        self.config.use_pre_view_ae)
    self._view_aes = nn.ModuleDict() if self._use_pre_view_ae else None
    self._view_tfms = nn.ModuleDict()
    self._dim = 0
    for vi, cfg_list in view_tfm_configs.items():
      init_args = view_tfm_init_args[vi]
      tfm = flow_transforms.make_transform(cfg_list, init_args)
      self._view_tfms["v_%i"%vi] = tfm
      # self._view_dims[vi] = tfm._dim
      if self._use_pre_view_ae:
        vi_ae_config = view_ae_configs[vi]
        self._view_aes["v_%i"%vi] = autoencoder.AutoEncoder(vi_ae_config)
        view_sizes[vi] = vi_ae_config.code_size

      self._dim += view_sizes[vi]  # Assume there are no "None" dims

    if view_ae_model_files:
      self.load_ae_from_files(view_ae_model_files)

    # Conditional transforms
    # self._shared_tfm = conditional_flow_transforms.make_transform(
    #     self.config.shared_tfm_config_list, shared_tfm_init_args)
    self._cond_tfms = nn.ModuleDict()
    for vi, cfg_list in cond_tfm_configs.items():
      init_args = cond_tfm_init_args[vi]
      tfm = conditional_flow_transforms.make_transform(
        cfg_list, vi, view_sizes, init_args)
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
        # for vi, vdim in self._view_dims.items():
        for vi, vdim in view_sizes.items():
          self._cond_lhoods["v_%i"%vi] = flow_likelihood.make_base_distribution(
              self.config.base_dist, vdim)
    else:
      self._cond_lhoods = nn.ModuleDict()
      # for vi, vdim in self._view_dims.items():
      for vi, vdim in view_sizes.items():
        self._cond_lhoods["v_%i"%vi] = flow_likelihood.make_base_distribution(
            self.config.likelihood_config, vdim)
      # self._likelihood_model = flow_likelihood.make_likelihood_model(
      #     self.config.likelihood_config, self._dim)

  # def _cat_views(self, zvs):
  #   zvs_cat = torch.cat([zvs[vi] for vi in range(self._nviews)], dim=1)
  #   return zvs_cat

  # def _split_views(self, zvs_cat):
  #   split_sizes = [self._view_dims[vi] for vi in range(self._nviews)]
  #   zvs_split = {
  #       vi:zvi for vi, zvi in enumerate(torch.split(zvs_cat, split_sizes, 1))}
  #   return zvs_split
  def load_ae_from_files(self, view_ae_model_files):
    for vi, vfile in view_ae_model_files.items():
      self._view_aes["v_%i"%vi].load_state_dict(torch.load(vfile))
      self._view_aes["v_%i"%vi].requires_grad = False
      self._view_aes["v_%i"%vi].eval()

  def _encode_views(self, x_vs, b_o, rtn_logdet=True):
    if self._use_pre_view_ae:
      x_vs = {
          vi: self._view_aes["v_%i"%vi](x_vi)
          for vi, x_vi in x_vs.items()
      }

    if self.config.no_view_tfm:
      z_vs = torch_utils.dict_numpy_to_torch(x_vs)
      n_pts = x_vs[utils.get_any_key(x_vs)].shape[0]
      if rtn_logdet:
        log_jac_det = {vi: torch.zeros((n_pts, 1)) for vi in z_vs}
    else:
      z_vs = {
        vi:self._view_tfms["v_%i"%vi](
              x_vi, rtn_torch=True, rtn_logdet=rtn_logdet)
          for vi, x_vi in x_vs.items()
      }
      z_vs = {
          vi:self._view_tfms["v_%i"%vi](
              x_vi, rtn_torch=True, rtn_logdet=rtn_logdet)
          for vi, x_vi in x_vs.items()
      }
      if rtn_logdet:
        log_jac_det = {vi:z_vi[1] for vi, z_vi in z_vs.items()}
        z_vs = {vi:z_vi[0] for vi, z_vi in z_vs.items()}
      # for vi, x_vi in x_vs.items():
      #   z_vi = self._view_tfms["v_%i"%vi](
      #       x_vi, rtn_torch=True, rtn_logdet=rtn_logdet)
      #   if rtn_logdet:

        # b_vi = b_o[vi]
        # z_vi = torch.zeros_like(x_vi)
        # if rtn_logdet:
        #   ljd_vi = torch.zeros_like(b_vi)

        # available_inds = torch.nonzero(b_vi).squeeze()
        # if len(available_inds) > 0:
        #   z_vi_tfm = self._view_tfms["v_%i"%vi](
        #     x_vi[available_inds], rtn_torch=True, rtn_logdet=rtn_logdet)
        #   if rtn_logdet:
        #     z_vi_tfm, ljd_vi[available_inds] = z_vi_tfm        
        #   z_vi[available_inds] = z_vi_tfm

        # z_vs[vi] = z_vi
        # if rtn_logdet:
        #   log_jac_det[vi] = ljd_vi
      # zvs_cat = self._cat_views(z_vs)
    return (z_vs, log_jac_det) if rtn_logdet else z_vs

  def _transform_views_cond(self, z_vs, b_o, rtn_logdet=True):
    # IPython.embed()
    # expand_b = self.config.expand_b
    l_vs = {}
    if rtn_logdet:
      logdet_vs = {}
    for vi, z_vi in z_vs.items():
        b_vi = b_o[vi]
        z_vi_o = {vo:z_vo for vo, z_vo in z_vs.items() if vo != vi}
        b_vi_o = {vo:b_vo for vo, b_vo in b_o.items() if vo != vi}
      # z_cat = MVZeroImpute(
      #     z_available, self._view_dims, ignored_view=vi, expand_b=expand_b)
        l_vi = self._cond_tfms["v_%i" % vi](
            z_vi, z_vi_o, b_vi_o, rtn_logdet=rtn_logdet)
        if rtn_logdet:
          l_vi, logdet_vs[vi] = l_vi
        l_vs[vi] = l_vi

    return (l_vs, logdet_vs) if rtn_logdet else l_vs

  def _transform(self, x_vs, b_o, rtn_logdet=True):
    # Transform input covariates using invertible transforms.
    z_vs = self._encode_views(x_vs, b_o, rtn_logdet=rtn_logdet)
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

    l_vs = self._transform_views_cond(z_vs, b_o, rtn_logdet=rtn_logdet)
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
    # IPython.embed()
    for vi, lvi in l_vs.items():
      l_nll[vi] = self._cond_lhoods["v_%i" % vi].nll(lvi)
    return l_nll

  def forward(self, x_vs, b_o, rtn_logdet=True):
    zl_vs = self._transform(x_vs, b_o, rtn_logdet)
    if rtn_logdet:
      z_vs, l_vs, ld_vs = zl_vs
      output = (l_vs, ld_vs)
    else:
      z_vs, l_vs = zl_vs
      output = l_vs

    return output

  def loss(self, l_nll, ld_vs, aggregate=True, *args, **kwargs):
    nll_loss = {}
    total_loss = 0.
    for vi in l_nll:
      l_nll_vi = l_nll[vi]
      ld_vi = ld_vs[vi]
      nll_loss[vi] = -torch.mean(ld_vi) + torch.mean(l_nll_vi)
      total_loss += nll_loss[vi]

    if aggregate:
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

  def _shuffle(self, x_vs):
    npts = x_vs[utils.get_any_key(x_vs)].shape[0]
    r_inds = np.random.permutation(npts)
    return {vi:xv[r_inds] for vi, xv in x_vs.items()}

  def _train_loop(self):
    x_vs = self._shuffle(self._view_data)
    self.itr_loss = 0.
    for batch_idx in range(self._n_batches):
      batch_start = batch_idx * self.config.batch_size
      batch_end = min(batch_start + self.config.batch_size, self._npts)

      self._batch_npts = batch_end - batch_start
      xvs_batch = {vi:xv[batch_start:batch_end] for vi, xv in x_vs.items()}
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
      l_batch_nll = self._nll(l_batch)

      # IPython.embed()
      # available_views = next(self._view_subset_shuffler)
      # xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets}
      self.opt.zero_grad()
      loss_val = self.loss(l_batch_nll, ld_batch, aggregate="sum")
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

  def _invert_view(self, vi, l_vi, x_o, b_o):
    l_inv = self._cond_tfms["v_%i"%vi].inverse(l_vi, x_o, b_o)
    if self.config.no_view_tfm:
      x_inv = l_inv
    else:
      x_inv = self._view_tfms["v_%i"%vi].inverse(l_inv)

    if self._use_pre_view_ae:
      x_inv = {
          vi: self._view_aes["v_%i"%vi]._decode(x_vi)
          for vi, x_vi in x_inv.items()
      }

    return x_inv

  def invert(self, l_vs, x_o, b_o, rtn_torch=True, batch_size=None):
    # Batch size to sample in smaller chunks for reduced memory usage
    if batch_size is None:
      z_o = self._encode_views(x_o, b_o, rtn_logdet=False)
      x_vs = {}
      for vi, lvi in l_vs.items():
        # z_cat = MVZeroImpute(
        #     z_o, self._view_dims, ignored_view=vi, expand_b=self.config.expand_b)
        x_vs[vi] = self._invert_view(vi, lvi, x_o, b_o)

    else:
      n_pts = l_vs[utils.get_any_key(l_vs)].shape[0]
      x_vs = {vi: [] for vi in l_vs}
      for start_idx in np.arange(n_pts, step=batch_size):
        end_idx = start_idx + batch_size
        l_vs_batch = {
            vi: lvi[start_idx:end_idx]
            for vi, lvi in l_vs.items()
        }
        x_o_batch = {
            vi: xo_vi[start_idx:end_idx]
            for vi, xo_vi in x_o.items()   
        }
        b_o_batch = {
            vi: bo_vi[start_idx:end_idx]
            for vi, bo_vi in b_o.items()   
        }
        # x_s_batch = self.invert(
        #     l_vs_batch, x_o_batch, rtn_torch=True, batch_size=None)
        for vi, lvi in l_vs_batch.items():
          x_vs[vi].append(self._invert_view(vi, lvi, x_o_batch, b_o_batch))
          # x_vs[vi].append(xsb_vi)

      x_vs = {vi:torch.cat(xvi, dim=0) for vi, xvi in x_vs.items()}

    if not rtn_torch:
      x_vs = torch_utils.dict_torch_to_numpy(x_vs)
    return x_vs

  def fit(self, x_vs, b_o=None, dev=None):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._training = True
    # For convenience
    self._npts = x_vs[utils.get_any_key(x_vs)].shape[0]
    self._view_data = torch_utils.dict_numpy_to_torch(x_vs)
    if dev is not None:
      self._view_data.to(dev)
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))

    self._view_subset_shuffler = self._make_view_subset_shuffler()
    self._view_subset_counts = {}
    # Set up optimizer
    self.opt = optim.Adam(self.parameters(), self.config.lr)

    self._loss_history = []
    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_diff_time = 0.
          itr_start_time = time.time()
        self._train_loop()
        if not self._training:
          break

        loss_val = float(self.itr_loss.detach())
        self._loss_history.append(loss_val)
        if self.config.verbose:
          itr_diff_time = time.time() - itr_start_time
          print("  Iteration %i out of %i (in %.2fs). Loss: %.5f" %
                (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
                end='\r')
      if self.config.verbose:
        print("  Iteration %i out of %i (in %.2fs). Loss: %.5f" %
              (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
              end='\r')
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")

    self._training = False
    self._loss_history = np.array(self._loss_history)
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
    return self

  def _pad_incomplete_data(self, x_vs, b_vs):
    n_pts = x_vs[utils.get_any_key(x_vs)].shape[0]
    x_vs_padded = {
        vi: (x_vs[vi] if vi in x_vs else
             torch.zeros((n_pts, self._view_dims[vi])))
        for vi in range(self._nviews)
    }
    b_vs = (b_vs or {})
    b_vs_padded = {
        vi: (b_vs[vi] if vi in b_vs else torch.zeros(n_pts))
        for vi in range(self._nviews)
    }
    return x_vs_padded, b_vs_padded

  def _sample_view(self, view_id, x_o, b_o, batch_size=None):
    sample_inds = (b_o[view_id] == 0).nonzero().squeeze()
    # print("SAMPLE INDS: %s" % (sample_inds,))
    n_samples = sample_inds.shape[0]
    if n_samples == 0:
      return x_o[view_id]

    x_o_subset = {vi: x_o_vi[sample_inds] for vi, x_o_vi in x_o.items()}
    b_o_subset = {vi: b_o_vi[sample_inds] for vi, b_o_vi in b_o.items()}
    l_samples = self._cond_lhoods["v_%i"%view_id].sample((n_samples,))

    # print(l_samples.shape)
    if batch_size is None or n_samples <= batch_size:
      # IPython.embed()
      x_view_samples = self._invert_view(
          view_id, l_samples, x_o_subset, b_o_subset)
    else:
      x_view_samples = []
      for start_idx in np.arange(n_samples, step=batch_size):
        end_idx = start_idx + batch_size
        l_batch = l_samples[start_idx:end_idx]
        x_o_batch = {
            vi: xo_vi[start_idx:end_idx]
            for vi, xo_vi in x_o_subset.items()   
        }
        b_o_batch = {
            vi: bo_vi[start_idx:end_idx]
            for vi, bo_vi in b_o_subset.items()   
        }
        x_view_samples.append(
            self._invert_view(view_id, l_batch, x_o_batch, b_o_batch))

      x_view_samples = torch.cat(x_view_samples, dim=0)
        # # x_s_batch = self.invert(
        # #     l_vs_batch, x_o_batch, rtn_torch=True, batch_size=None)
        # for vi, lvi in l_vs_batch.items():
        #   x_vs[vi].append(self._invert_view(vi, lvi, x_o_batch, b_o_batch))
    x_view = x_o[view_id]
    x_view[sample_inds] = x_view_samples
    return x_view

  def sample(
      self, x_o, b_o=None, sampled_views=None, batch_size=None, rtn_torch=True):
    n_pts = x_o[utils.get_any_key(x_o)].shape[0]
    # sampling_views = [vi for vi in range(self._nviews) if vi not in x_o]
    x_o = torch_utils.dict_numpy_to_torch(x_o)
    b_o = (
        torch_utils.dict_numpy_to_torch(b_o) if b_o else
        {
            vi: (torch.zeros(n_pts)
                 if vi in sampled_views else torch.ones(n_pts))
            for vi in range(self._nviews)
        }
    )
    x_o, b_o = self._pad_incomplete_data(x_o, b_o)
    sampled_views = sampled_views or list(range(self._nviews))
    samples = {
        vi: self._sample_view(vi, x_o, b_o, batch_size)
        for vi in sampled_views
    }
    return (samples if rtn_torch else torch_utils.dict_torch_to_numpy(samples))
    # samples = {}
    # l_samples = {
    #     vi:self._cond_lhoods["v_%i"%vi].sample((n_samples,))
    #     for vi in sampling_views}
    # return self.invert(
    #     l_samples, x_o, b_o, rtn_torch=rtn_torch, batch_size=batch_size)
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


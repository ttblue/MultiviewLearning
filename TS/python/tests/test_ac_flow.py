#!/usr/bin/env python
import itertools
import numpy as np
import os
import scipy
from sklearn import manifold
import torch
from torch import nn
import umap

from models import ac_flow_pipeline, conditional_flow_transforms,\
    flow_likelihood, flow_pipeline, flow_transforms, torch_models
from synthetic import flow_toy_data, multimodal_systems
from utils import math_utils, torch_utils, utils

from tests.test_flow import make_default_tfm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import IPython


###
class SimpleArgs:
  def __init__(self, options):
    # Default values
    arg_vals = {v[0]: v[-1] for v in options}
    self.__dict__.update(arg_vals)
###


def convert_numpy_to_float32(data):
  if isinstance(data, np.ndarray):
    data = data.astype("float32")
  if isinstance(data, dict):
    data = {i:idat.astype("float32") for i, idat in data.items()}
  if isinstance(data, list):
    data = [idat.astype("float32") for idat in data]
  return data


def make_default_nn_config():
  input_size = 10  # Computed online
  output_size = 10  # Computed online
  layer_units = [32, 64]
  use_vae = False
  activation = nn.ReLU  # nn.functional.relu
  last_activation = torch_models.Identity  # functional.sigmoid
  # layer_types = None
  # layer_args = None
  bias = True
  dropout_p = 0.
  layer_types, layer_args = torch_utils.generate_linear_types_args(
        input_size, layer_units, output_size, bias)
  nn_config = torch_models.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)
  return nn_config


def make_default_tfm_config(tfm_type="shift_scale_coupling"):
  neg_slope = 0.1
  scale_config = make_default_nn_config()
  scale_config.last_activation = torch.nn.Tanh
  shift_config = make_default_nn_config()
  shift_config.last_activation = torch.nn.Sigmoid
  shared_wts = False

  ltfm_config = make_default_nn_config()
  bias_config = make_default_nn_config()

  has_bias = True

  base_dist = "gaussian"
  reg_coeff = 0.
  lr = 1e-3
  batch_size = 50
  max_iters = 1000

  stopping_eps = 1e-5
  num_stopping_iter = 1000

  grad_clip = 2.

  verbose = True

  config = flow_transforms.TfmConfig(
      tfm_type=tfm_type, neg_slope=neg_slope, scale_config=scale_config,
      shift_config=shift_config, shared_wts=shared_wts, ltfm_config=ltfm_config,
      bias_config=bias_config, has_bias=has_bias, reg_coeff=reg_coeff,
      base_dist=base_dist, lr=lr, batch_size=batch_size, max_iters=max_iters,
      stopping_eps=stopping_eps, num_stopping_iter=num_stopping_iter,
      grad_clip=grad_clip, verbose=verbose)
  return config


def make_default_cond_tfm_config(tfm_type="shift_scale_coupling"):
  neg_slope = 0.1
  func_nn_config = make_default_nn_config()
  func_nn_config.last_activation = torch.nn.Tanh
  has_bias = True

  ltfm_config = make_default_nn_config()
  bias_config = make_default_nn_config()

  has_bias = True

  base_dist = "gaussian"
  reg_coeff = 0.
  lr = 1e-3
  batch_size = 50
  max_iters = 1000

  stopping_eps = 1e-5
  num_stopping_iter = 1000

  grad_clip = 2.

  verbose = True

  config = conditional_flow_transforms.CTfmConfig(
      tfm_type=tfm_type, neg_slope=neg_slope, func_nn_config=func_nn_config,
      has_bias=has_bias, reg_coeff=reg_coeff,
      base_dist=base_dist, lr=lr, batch_size=batch_size, max_iters=max_iters,
      stopping_eps=stopping_eps, num_stopping_iter=num_stopping_iter,
      grad_clip=grad_clip, verbose=verbose)
  return config


class ArgsCopy:
  def __init__(self, args):
    self.__dict__.update(args.__dict__)
    # self.num_ss_tfm = args.num_ss_tfm
    # self.num_lin_tfm = args.num_lin_tfm
    # self.use_leaky_relu = args.use_leaky_relu
    # self.use_reverse = args.use_reverse


def make_default_likelihood_config(args):
  model_type = "linear_arm"
  n_components = args.n_components
  dist_type = "gaussian" if args.dist_type == "mv_gaussian" else args.dist_type

  hidden_size = 32
  theta_nn_config = make_default_nn_config()
  theta_nn_config.last_activation = torch.nn.Tanh
  cell_type = "LSTM"  # not needed for linear_arm

  verbose = True

  config = flow_likelihood.ARMMConfig(
      model_type=model_type, dist_type=dist_type, n_components=n_components,
      hidden_size=hidden_size, theta_nn_config=theta_nn_config,
      cell_type=cell_type, verbose=verbose)

  return config


def make_default_data(args, split=False):
  tfm_types = [ "linear"]
  if args.dtype == "single_dim_copy":
    Z = np.concatenate([np.random.randn(args.npts, 1)] * args.ndim, axis=1)
    scale = 5.
    scale_mat = np.eye(args.ndim) * scale
    X = Z.dot(scale_mat)
    tfm_args = []
  else:
    Z, X, tfm_args = flow_toy_data.simple_transform_data(
        args.npts, args.ndim, tfm_types)

  if split:
    split_frac = [0.8, 0.2]
    (tr_Z, te_Z), inds = utils.split_data(Z, split_frac, get_inds=True)
    tr_X, te_X = [X[idx] for idx in inds]

    return (tr_Z, te_Z), (tr_X, te_X), tfm_args
  return Z, X, tfm_args


def make_default_data_X(args, split=False, normalize_scale=None):
  # Z, X, tfm_args = flow_toy_data.simple_transform_data(
  #     args.npts, args.ndim, tfm_types)  
  X = np.random.randn(args.npts, args.ndim)
  if normalize_scale is not None:
    X_norm = np.linalg.norm(X, axis=1).reshape(-1, 1)
    X = X / X_norm * normalize_scale

  if split:
    split_frac = [0.8, 0.2]
    tr_X, te_X = utils.split_data(X, split_frac, get_inds=False)
    return tr_X, te_X
  return X


def make_default_overlapping_data(args):
  npts = args.npts
  nviews = args.nviews
  ndim = args.ndim
  perturb_eps = args.peps

  scale = 1
  centered = True
  overlap = True
  gen_D_alpha = False

  data, ptfms = multimodal_systems.generate_redundant_multiview_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      overlap=overlap, gen_D_alpha=gen_D_alpha, perturb_eps=perturb_eps)

  return data, ptfms


def make_default_overlapping_data2(args):
  npts = args.npts
  nviews = args.nviews
  ndim = args.ndim
  perturb_eps = args.peps

  scale = 1
  centered = True
  overlap = True

  data, ptfms = multimodal_systems.generate_local_overlap_multiview_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      perturb_eps=perturb_eps)

  return data, ptfms


def rotate_and_shift_data(data, ndim, scale):
  R = math_utils.random_unitary_matrix(ndim)
  t = np.random.randn(ndim) * scale
  data = data.dot(R) + t
  ptfm = R, t

  return data, ptfm


def make_default_independent_data(args, rotate_and_shift=True):
  npts = args.npts
  nviews = args.nviews
  ndim = args.ndim
  perturb_eps = args.peps
  scale = args.scale

  centered = True

  if nviews > ndim:
    raise ValueError(
        "Number of views (%i) should be <= number of dims (%i)."%
        (nviews, ndim))

  cat_data = np.random.randn(npts, ndim) * scale
  ptfm = None
  if rotate_and_shift:
    cat_data, ptfm = rotate_and_shift_data(cat_data, ndim, scale)

  data = {
    vi:vdat for vi, vdat in enumerate(np.array_split(cat_data, nviews, 1))}
  return data, ptfm


def make_default_shape_data(args):
  npts = args.npts
  nviews = args.nviews
  ndim = args.ndim
  noise_eps = args.peps
  scale = args.scale
  shape = args.shape

  centered = True

  if nviews > ndim:
    raise ValueError(
        "Number of views (%i) should be <= number of dims (%i)."%
        (nviews, ndim))

  view_dim = ndim // nviews

  data, ptfms = flow_toy_data.multiview_lifted_3d_manifold(
      npts, scale=scale, nviews=nviews, view_dim=view_dim, shape=shape,
      noise_eps=noise_eps)

  return data, ptfm


def make_default_cond_tfms(
    args, view_sizes, tfm_args=[], double_tot_dim=False, rtn_args=False):
  # dim = args.ndim
  num_ss_tfm = args.num_ss_tfm
  num_lin_tfm = args.num_lin_tfm
  use_leaky_relu = args.use_leaky_relu
  use_reverse = args.use_reverse

  # Generate config list:
  tfm_configs = {}
  tfm_inits = {}
  default_hidden_sizes = [64, 128]
  default_activation = nn.Tanh
  default_nn_config = make_default_nn_config()

  tot_dim = sum(view_sizes.values())
  if double_tot_dim:
    tot_dim = tot_dim * 2
  #################################################
  # Bit-mask couple transform
  tfm_idx = 0
  idx_args = tfm_args[tfm_idx] if tfm_idx < len(tfm_args) else None
  bit_mask = None
  if idx_args is not None and idx_args[0] == "scaleshift":
    bit_mask = idx_args[1]

  for vi, vdim in view_sizes.items():
    vi_tfm_configs = []
    vi_tfm_inits = []
    obs_dim = tot_dim - vdim
    if double_tot_dim:
      obs_dim -= vdim
    dim = unobs_dim = vdim
    hidden_sizes = default_hidden_sizes
    for i in range(num_ss_tfm):
      scale_shift_tfm_config = make_default_cond_tfm_config("scale_shift")
      vi_tfm_configs.append(scale_shift_tfm_config)

      if bit_mask is not None:
        bit_mask = 1 - bit_mask
      else:
        bit_mask = np.zeros(dim)
        bit_mask[np.random.permutation(dim)[:dim//2]] = 1

      vi_tfm_inits.append(
          (obs_dim, bit_mask, default_hidden_sizes, default_activation,
           default_nn_config))

    # Fixed linear transform
    tfm_idx = 1
    # L, U = tfm_args[tfm_idx][1:]
    # _, Li, Ui = scipy.linalg.lu(np.linalg.inv(L.dot(U)))
    # init_mat = np.tril(Li, -1) + np.triu(Ui, 0)
    # eps = 1e-1
    # noise = np.random.randn(*init_mat.shape) * eps
    for i in range(num_lin_tfm):
      linear_tfm_config = make_default_cond_tfm_config("linear")
      linear_tfm_config.has_bias = True
      vi_tfm_configs.append(linear_tfm_config)
      vi_tfm_inits.append((obs_dim, unobs_dim, default_nn_config))# init_mat))

    # # Leaky ReLU
    tfm_idx = 2
    if use_leaky_relu:
      leaky_relu_config = make_default_cond_tfm_config("leaky_relu")
      leaky_relu_config.neg_slope = 0.1
      vi_tfm_configs.append(leaky_relu_config)
      vi_tfm_inits.append(None)

    # Reverse
    tfm_idx = 3
    if use_reverse:
      reverse_config = make_default_cond_tfm_config("reverse")
      vi_tfm_configs.append(reverse_config)
      vi_tfm_inits.append(None)

    tfm_configs[vi] = vi_tfm_configs
    tfm_inits[vi] = vi_tfm_inits
    #################################################

  if rtn_args:
    return tfm_configs, tfm_inits

  comp_config = make_default_tfm_config("composition")
  models = {
      vi: conditional_flow_transforms.make_transform(
          tfm_configs[vi], tfm_inits[vi], comp_config)
      for vi in tfm_configs
  }

  return models


def make_default_pipeline_config(args, view_sizes={}):
  tot_dim = sum(view_sizes.values())
  # print(args.__dict__.keys())
  all_view_args = ArgsCopy(args)
  # all_view_args.ndim = tot_dim
  cond_config_lists, cond_inits_lists = make_default_cond_tfms(
      all_view_args, view_sizes, double_tot_dim=True, rtn_args=True)

  view_tfm_config_lists = {}
  view_tfm_init_lists = {}
  for vi, vdim in view_sizes.items():
    vi_args = ArgsCopy(args)
    vi_args.ndim = vdim
    vi_cfg_list, vi_init = make_default_tfm(vi_args, rtn_args=True)

    view_tfm_config_lists[vi] = vi_cfg_list
    view_tfm_init_lists[vi] = vi_init

  likelihood_config = make_default_likelihood_config(args) if args.use_ar else None
  base_dist = "mv_gaussian"

  expand_b = True

  batch_size = 50
  lr = 1e-3
  max_iters = args.max_iters

  verbose = True

  # IPython.embed()
  config = ac_flow_pipeline.MACFTConfig(
      expand_b=expand_b, likelihood_config=likelihood_config,
      base_dist=base_dist, batch_size=batch_size, lr=lr, max_iters=max_iters,
      verbose=verbose)

  return config, (view_tfm_config_lists, view_tfm_init_lists),\
      (cond_config_lists, cond_inits_lists)


def make_default_likelihood_model(args):
  if not args.use_ar:
    return None
  config = make_default_likelihood_config(args)
  model = flow_likelihood.make_likelihood_model(config)
  model.initialize(args.ndim)

  return model


def concat_with_binary_flags(data, main_view, b_available, expand_b=True):
  zero_imputed_data = {}
  b_cat = {}
  output = np.empty((data[0].shape[0], 0))
  for vi in data:
    if vi == main_view:
      continue
    b = b_available[vi]
    b = b.reshape(-1, 1)
    zero_imputed_data[vi] = data[vi] * b
    if expand_b:
      b_cat[vi] = np.tile(b, (1, data[vi].shape[1]))
    else:
      b_cat[vi] = b

    output = np.concatenate([output, zero_imputed_data[vi], b_cat[vi]], axis=1)

  return output, zero_imputed_data, b_cat


def make_missing_dset(data, main_view, include_nv_0=True, separate_nv=False):
  vc_dim = data[main_view].shape[1]
  npts =  data[main_view].shape[0]
  tot_dim = np.sum([vdat.shape[1] for vi, vdat in data.items() if vi != main_view])
  if separate_nv:
    x = {}
    x_o = {}
  else:
    x = np.empty((0, vc_dim))
    x_o = np.empty((0, tot_dim * 2))

  mv_x = data[main_view]
  obs_views = [vi for vi in data if vi != main_view]

  tot_views = len(obs_views)
  start_nv = 0 if include_nv_0 else 1
  for nv in range(start_nv, tot_views + 1):
    if separate_nv:
      x[nv] = np.empty((0, vc_dim))
      x_o[nv] = np.empty((0, tot_dim * 2))
    for perm in itertools.combinations(obs_views, nv):
      b_available = {
          vi: np.ones(npts) * int(vi in perm)
          for vi in obs_views

      }
      output, zero_imputed_data, b_cat = concat_with_binary_flags(
          data, main_view, b_available)
      if separate_nv:
        x[nv] = np.concatenate([x[nv], mv_x], axis=0)
        x_o[nv] = np.concatenate([x_o[nv], output], axis=0)
      else:
        x = np.concatenate([x, mv_x], axis=0)
        x_o = np.concatenate([x_o, output], axis=0)

  return x, x_o


def simple_test_cond_tfms(args):
  # (tr_Z, te_Z), (tr_X, te_X), tfm_args = make_default_data(args, split=True)
  data, ptfms = make_default_overlapping_data(args)
  n_tr = int(0.8 * args.npts)
  n_te = args.npts - n_tr

  tr_data = {vi:vdat[:n_tr] for vi, vdat in data.items()}
  te_data = {vi:vdat[n_tr:] for vi, vdat in data.items()}

  view_sizes = {vi: vdat.shape[1] for vi, vdat in data.items()}
  main_view = 0
  models = make_default_cond_tfms(args, view_sizes)
  model = models[main_view]

  config = model.config
  config.batch_size = 1000
  config.lr = 1e-4
  config.reg_coeff = 0.1
  config.max_iters = args.max_iters
  config.stopping_eps = 1e-8

  x_tr = tr_data[main_view].astype(np.float32)
  x_o_tr = np.concatenate(
      [tr_data[vi] for vi in range(args.nviews) if vi != main_view],
      axis=1).astype(np.float32)
  x_te = te_data[main_view].astype(np.float32)
  x_o_te = np.concatenate(
      [te_data[vi] for vi in range(args.nviews) if vi != main_view],
      axis=1).astype(np.float32)

  # IPython.embed()
  model.fit(x_tr, x_o_tr)

  x_tr = torch.from_numpy(x_tr)
  x_te = torch.from_numpy(x_te)

  z_tr = model(x_tr, x_o_tr, rtn_torch=True)
  zi_tr = model.inverse(z_tr, x_o_tr, rtn_torch=True)
  z_te = model(x_te, x_o_te, rtn_torch=True)
  zi_te = model.inverse(z_te, x_o_te, rtn_torch=True)
  print("Ready")
  IPython.embed()


def test_cond_missing_tfms(args):
  # (tr_Z, te_Z), (tr_X, te_X), tfm_args = make_default_data(args, split=True)
  data, ptfms = make_default_overlapping_data(args)
  n_tr = int(0.8 * args.npts)
  n_te = args.npts - n_tr

  tr_data = {vi:vdat[:n_tr] for vi, vdat in data.items()}
  te_data = {vi:vdat[n_tr:] for vi, vdat in data.items()}

  view_sizes = {vi: vdat.shape[1] for vi, vdat in data.items()}
  main_view = 0
  models = make_default_cond_tfms(args, view_sizes, double_tot_dim=True)
  model = models[main_view]

  config = model.config
  config.batch_size = 1000
  config.lr = 1e-4
  config.reg_coeff = 0.1
  config.max_iters = args.max_iters
  config.stopping_eps = 1e-8


  x_tr, x_o_tr = make_missing_dset(tr_data, main_view, separate_nv=False)
  x_te, x_o_te = make_missing_dset(te_data, main_view, separate_nv=True)

  # x_tr = tr_data[main_view].astype(np.float32)
  # x_o_tr = np.concatenate(
  #     [tr_data[vi] for vi in range(args.nviews) if vi != main_view],
  #     axis=1).astype(np.float32)
  # x_te = te_data[main_view].astype(np.float32)
  # x_o_te = np.concatenate(
  #     [te_data[vi] for vi in range(args.nviews) if vi != main_view],
  #     axis=1).astype(np.float32)

  # IPython.embed()
  model.fit(x_tr, x_o_tr)

  # x_tr = torch.from_numpy(x_tr)
  # x_te = {nv: torch.from_numpy(x_nv) for nv, x_nv in x_te.items()}

  z_tr = model(x_tr, x_o_tr, rtn_torch=True)
  zi_tr = model.inverse(z_tr, x_o_tr, rtn_torch=True)


  z_te = {nv: model(x_te[nv], x_o_te[nv], rtn_torch=True) for nv in x_te}
  zi_te = {nv: model.inverse(z_te[nv], x_o_te[nv], rtn_torch=True) for nv in x_te}
  # zi_te = model.inverse(z_te, x_o_te, rtn_torch=True)
  print("Ready")
  IPython.embed()
  # n_test_samples = args.npts // 2
  # samples = model.sample(n_test_samples, inverted=False, rtn_torch=False)
  # # samples = np.concatenate(
  # #     [np.random.randn(n_test_samples, 1)] * args.ndim, axis=1)
  # X_samples = model.inverse(samples, rtn_torch=False)
  # X_all = np.r_[tr_X, te_X, X_samples]
  # Z_all = np.r_[tr_Zinv_pred, te_Zinv_pred, samples]

  # IPython.embed()

  # tsne = umap.UMAP(2)
  # y_x = tsne.fit_transform(X_all)
  # y_z = tsne.fit_transform(Z_all)
  # plot_data = {"x": y_x, "z": y_z}
  # pdtype = "z"
  # y = plot_data[pdtype]

  # plt.scatter(y[:args.npts, 0], y[:args.npts, 1], color="b")
  # plt.scatter(y[args.npts:, 0], y[args.npts:, 1], color="r")
  # plt.show()


def test_ptbxl(args):
  


_MV_DATAFUNCS = {
    "o1": make_default_overlapping_data,
    "o2": make_default_overlapping_data2,
    "ind": make_default_independent_data,
    "sh": make_default_shape_data,
}
def test_pipeline(args):
  data_func = _MV_DATAFUNCS.get(args.dtype, make_default_overlapping_data)
  data, ptfms = data_func(args)
  data = convert_numpy_to_float32(data)

  n_tr = int(0.8 * args.npts)
  n_te = args.npts - n_tr

  tr_data = {vi:vdat[:n_tr] for vi, vdat in data.items()}
  te_data = {vi:vdat[n_tr:] for vi, vdat in data.items()}

  view_sizes = {vi: vdat.shape[1] for vi, vdat in data.items()}

  # IPython.embed()
  # cond_tfm_config_lists, cond_tfm_init_args = make_default_cond_tfms(
  #       args, view_sizes, rtn_args=True)
  config, view_config_and_inits, cond_config_and_inits = \
      make_default_pipeline_config(args, view_sizes=view_sizes)
  view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

  IPython.embed()

  model = ac_flow_pipeline.MultiviewACFlowTrainer(config)
  model.initialize(
      view_tfm_config_lists, view_tfm_init_lists,
      cond_tfm_config_lists, cond_tfm_init_lists)

  model.fit(data)
  n_test_samples = n_te
  # sample_data = model.sample(n_test_samples, rtn_torch=False)

  IPython.embed()

  # tsne = umap.UMAP(n_components=2)
  # compare_dat = {}
  # for vi, tr_view in train_data.items():
  #   sample_view = sample_data[vi]
  #   all_view = np.r_[tr_view, sample_view]
  #   all_comp = tsne.fit_transform(all_view)

  #   compare_dat[vi] = {
  #       "true": all_comp[:args.npts], "sample": all_comp[args.npts:]}

  # train_z = torch_utils.torch_to_numpy(model(train_data, False)) #, rtn_torch=False)
  # sample_z = torch_utils.torch_to_numpy(model(sample_data, False)) #, rtn_torch=False)
  # all_z = np.r_[train_z, sample_z]
  # all_comp = tsne.fit_transform(all_z)
  # compare_dat["z"] = {
  #       "true": all_comp[:args.npts], "sample": all_comp[args.npts:]}

  # IPython.embed()
  
  # shape = "cube"
  # shape_data = flow_toy_data.shaped_3d_manifold(1000, shape=shape, scale=1.0)
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # ax.scatter(shape_data[:, 0], shape_data[:, 1], shape_data[:, 2])
  # plt.show()
  # ax.scatter(all_comp2[:args.npts, 0], all_comp2[:args.npts, 1], all_comp2[:args.npts,2])
  # ax.scatter(all_comp2[args.npts:, 0], all_comp2[args.npts:, 1], all_comp2[args.npts:,2])
  # plt.show()

  for vi, vdat in compare_dat.items():
    tr_dat = vdat["true"]
    sample_dat = vdat["sample"]
    plt.figure()
    plt.scatter(tr_dat[:, 0], tr_dat[:, 1], color="b", label="True")
    plt.scatter(sample_dat[:, 0], sample_dat[:, 1], color="r", label="Sampled")
    plt.title("Train vs. sampled data -- View %s"%vi)
    plt.legend()
    plt.show()
    plt.pause(1.0)

  # For 3D viewing:
  tsne = umap.UMAP(n_components=3)
  plt.scatter(all_comp2[:args.npts, 0], all_comp2[:args.npts, 1], color="b", label="True")
  plt.scatter(all_comp2[args.npts:, 0], all_comp2[args.npts:, 1], color="r", label="Samples")
  plt.legend()
  plt.show()

_TEST_FUNCS = {
    0: simple_test_cond_tfms,
    1: test_cond_missing_tfms,
    2: test_pipeline,
}


if __name__ == "__main__":
  np.set_printoptions(linewidth=1000, precision=3, suppress=True)
  torch.set_printoptions(precision=3)
  options = [
      ("etype", str, "Expt. type (gen/rec)", "gen"),
      ("dtype", str, "Data type (random/single_dim_copy/o1/o2/ind/sh)", "ind"),
      ("nviews", int, "Number of views", 3),
      ("npts", int, "Number of points", 1000),
      ("ndim", int, "Dimensions", 10),
      ("peps", float, "Perturb epsilon", 0.),
      ("scale", float, "Scale of the data.", 1.),
      ("shape", str, "Shape of toy data.", "cube"),
      ("max_iters", int, "Number of iters for opt.", 10000),
      ("batch_size", int, "Batch size for opt.", 100),
      ("num_ss_tfm", int, "Number of shift-scale tfms", 1),
      ("num_lin_tfm", int, "Number of linear tfms", 1),
      ("use_leaky_relu", bool, "Flag for using leaky relu tfm", False),
      ("use_reverse", bool, "Flag for using reverse tfm", False),
      ("dist_type", str, "Base dist. type ([mv_]gaussian/laplace/logistic)",
       "mv_gaussian"),
      ("use_ar", bool, "Flag for base dist being an AR model", False),
      ("n_components", int, "Number of components for likelihood MM", 5),
      ]
  args = utils.get_args(options)
  
  func = _TEST_FUNCS.get(args.expt, simple_test_cond_tfms)
  func(args)
 
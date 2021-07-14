#!/usr/bin/env python
import itertools
import numpy as np
import os
import scipy
from sklearn import manifold
import time
import torch
from torch import nn
import umap

from dataprocessing import ecg_data, split_single_view_dsets as ssvd,\
    multiview_datasets as mvd
from models import ac_flow_pipeline, conditional_flow_transforms,\
    flow_likelihood, flow_pipeline, flow_transforms, torch_models
from synthetic import flow_toy_data, multimodal_systems
from utils import math_utils, torch_utils, utils

from tests.test_flow import make_default_tfm
from tests.test_ac_flow import SimpleArgs, convert_numpy_to_float32,\
    make_default_nn_config, make_default_tfm_config, ArgsCopy,\
    make_default_likelihood_config, make_default_data, make_default_data_X,\
    make_default_overlapping_data, make_default_overlapping_data2,\
    rotate_and_shift_data, make_default_independent_data,\
    make_default_shape_data, make_missing_dset



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import IPython


def make_default_cond_tfm_config(tfm_type="shift_scale_coupling"):
  neg_slope = 0.1
  is_sigmoid = False
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
      tfm_type=tfm_type, neg_slope=neg_slope, is_sigmoid=is_sigmoid,
      func_nn_config=func_nn_config, has_bias=has_bias, reg_coeff=reg_coeff,
      base_dist=base_dist, lr=lr, batch_size=batch_size, max_iters=max_iters,
      stopping_eps=stopping_eps, num_stopping_iter=num_stopping_iter,
      grad_clip=grad_clip, verbose=verbose)
  return config


_AVAILABLE_TRANSFORMS = {
    "s": "scale_shift_couping",
    "l": "linear",
    "r": "rnn_couping",
    "v": "reverse",
    "k": "leaky_relu",
    "g": "logit",
}
def make_default_tfms(
    args, view_sizes, double_tot_dim=False, rtn_args=False, is_cond=True):
  # dim = args.ndim
  # reset_bit_mask = True
  # num_ss_tfm = args.num_cond_ss_tfm
  # num_lin_tfm = args.num_cond_lin_tfm

  if is_cond:
    start_tfm_set = args.cond_start_tfm_set
    repeat_tfm_set = args.cond_repeat_tfm_set
    num_tfm_sets = args.num_cond_tfm_sets
    end_tfm_set = args.cond_end_tfm_set
    num_ss_tfm = max(args.num_cond_ss_tfm, 3)  # Need at least 3 ss tfms
    config_func = make_default_cond_tfm_config
  else:
    start_tfm_set = args.start_tfm_set
    repeat_tfm_set = args.repeat_tfm_set
    num_tfm_sets = args.num_tfm_sets
    end_tfm_set = args.end_tfm_set
    num_ss_tfm = max(args.num_ss_tfm, 3)  # Need at least 3 ss tfms
    config_func = make_default_tfm_config

  for tf in start_tfm_set + repeat_tfm_set + end_tfm_set:
    if tf not in _AVAILABLE_TRANSFORMS:
      raise ValueError("Transformation %s not available." % tf)
    if tf == "r":
      raise ValueError("RNN Coupling not yet available.")

  all_tfm_set = start_tfm_set + repeat_tfm_set * num_tfm_sets + end_tfm_set

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
  # tfm_idx = 0
  # idx_args = tfm_args[tfm_idx] if tfm_idx < len(tfm_args) else None
  # init_bit_mask = None
  # if idx_args is not None and idx_args[0] == "scaleshift":
  #   init_bit_mask = idx_args[1]

  for vi, vdim in view_sizes.items():
    # bit_mask = init_bit_mask
    vi_tfm_configs = []
    vi_tfm_inits = []
    obs_dim = tot_dim - vdim
    if double_tot_dim:
      obs_dim -= vdim
    dim = unobs_dim = vdim
    hidden_sizes = default_hidden_sizes

    for tf in all_tfm_set:
      if tf == "s":
        bit_mask = np.zeros(dim)
        bit_mask[np.random.permutation(dim)[:dim//2]] = 1
        for i in range(num_ss_tfm):
          scale_shift_tfm_config = config_func("scale_shift")
          vi_tfm_configs.append(scale_shift_tfm_config)
          vi_tfm_inits.append(
              (bit_mask, default_nn_config))
          bit_mask = 1 - bit_mask

      elif tf == "l":
        linear_tfm_config = config_func("linear")
        linear_tfm_config.has_bias = True
        vi_tfm_configs.append(linear_tfm_config)
        vi_tfm_inits.append(default_nn_config)# init_mat))

      elif tf == "r":
        raise ValueError("RNN Coupling not yet available.")

      elif tf == "v":
        reverse_config = config_func("reverse")
        vi_tfm_configs.append(reverse_config)
        vi_tfm_inits.append(())

      elif tf == "k":
        leaky_relu_config = config_func("leaky_relu")
        leaky_relu_config.neg_slope = 0.1
        vi_tfm_configs.append(leaky_relu_config)
        vi_tfm_inits.append(())

      elif tf == "g":
        logit_config = config_func("logit")
        vi_tfm_configs.append(logit_config)
        vi_tfm_inits.append(())

      tfm_configs[vi] = vi_tfm_configs
      tfm_inits[vi] = vi_tfm_inits
    #################################################

  if rtn_args:
    return tfm_configs, tfm_inits

  comp_config = config_func("composition")
  if is_cond:
    models = {
        vi: conditional_flow_transforms.make_transform(
            tfm_configs[vi], vi, view_sizes, tfm_inits[vi], comp_config)
        for vi in tfm_configs
    }
  else:
    models = {
      vi: flow_transforms.make_transform(
            tfm_configs[vi], tfm_inits[vi], comp_config)
      for vi in tfm_configs
    }

  return models


def make_default_pipeline_config(
    args, view_sizes={}, no_view_tfm=False, start_logit=False):

  tot_dim = sum(view_sizes.values())
  # print(args.__dict__.keys())
  all_view_args = ArgsCopy(args)
  # all_view_args.ndim = tot_dim
  cond_config_lists, cond_inits_lists = make_default_tfm(
      all_view_args, view_sizes, double_tot_dim=True, rtn_args=True,
      is_cond=True)

  view_tfm_config_lists, view_tfm_init_lists = make_default_tfm(
      all_view_args, view_sizes, double_tot_dim=True, rtn_args=True,
      is_cond=False)
  # for vi, vdim in view_sizes.items():
  #   vi_args = ArgsCopy(args)
  #   vi_args.ndim = vdim
  #   vi_cfg_list, vi_init = make_default_tfm(
  #       vi_args, rtn_args=True, start_logit=start_logit)

  #   view_tfm_config_lists[vi] = vi_cfg_list
  #   view_tfm_init_lists[vi] = vi_init

  likelihood_config = make_default_likelihood_config(args) if args.use_ar else None
  base_dist = "mv_gaussian"

  expand_b = True

  batch_size = 50
  lr = 1e-3
  max_iters = args.max_iters

  verbose = True

  # IPython.embed()
  config = ac_flow_pipeline.MACFTConfig(
      expand_b=expand_b, no_view_tfm=no_view_tfm,
      likelihood_config=likelihood_config, base_dist=base_dist,
      batch_size=batch_size, lr=lr, max_iters=max_iters, verbose=verbose)

  return config, (view_tfm_config_lists, view_tfm_init_lists),\
      (cond_config_lists, cond_inits_lists)


def make_default_likelihood_model(args):
  if not args.use_ar:
    return None
  config = make_default_likelihood_config(args)
  model = flow_likelihood.make_likelihood_model(config)
  model.initialize(args.ndim)

  return model


def make_default_overlapping_data(args):
  npts = args.npts
  nviews = args.nviews
  ndim = args.ndim
  perturb_eps = args.peps
  scale = args.scale

  centered = False
  overlap = True
  gen_D_alpha = False

  data, ptfms = multimodal_systems.generate_redundant_multiview_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      overlap=overlap, gen_D_alpha=gen_D_alpha, perturb_eps=perturb_eps)

  return data, ptfms


def tfm_seq(model, x, b, x_o):
  x_o = torch_utils.dict_numpy_to_torch(x_o)
  x = torch_utils.numpy_to_torch(x)

  tfm_list = model._tfm_list
  x0 = x
  x_seq = [x]
  ld_seq = []
  params = []

  for tfm in tfm_list:
    x, ld = tfm(x, x_o, b, rtn_logdet=True)
    x_seq.append(torch_utils.torch_to_numpy(x))
    ld_seq.append(torch_utils.torch_to_numpy(ld))
    if hasattr (tfm, "_get_params"):
      ximputed = tfm._impute_mv_data(x, x_o, b)
      ps = [torch_utils.torch_to_numpy(p) for p in tfm._get_params(ximputed)]
      params.append(ps)
    else:
      params.append(None)

  _, total_ld = model(x0, x_o, b, rtn_logdet=True)
  total_ld = torch_utils.torch_to_numpy(total_ld)

  return x_seq, ld_seq, params, total_ld


def inv_tfm_seq(model, z, b, x_o):
  x_o = torch_utils.dict_numpy_to_torch(x_o)
  z = torch_utils.numpy_to_torch(z)

  tfm_list = model._tfm_list
  z0 = z
  z_seq = [z]
  ld_seq = []
  params = []

  for tfm in tfm_list[::-1]:
    z = tfm.inverse(z, x_o, b)
    z_seq.append(torch_utils.torch_to_numpy(z))
    if hasattr (tfm, "_get_params"):
      ximputed = tfm._impute_mv_data(z, x_o, b)
      ps = [torch_utils.torch_to_numpy(p) for p in tfm._get_params(ximputed)]
      params.append(ps)
    else:
      params.append(None)

  return z_seq, params


def simple_test_cond_tfms(args):
  print("Test: Simple conditional transforms.")
  # (tr_Z, te_Z), (tr_X, te_X), tfm_args = make_default_data(args, split=True)
  data, ptfms = make_default_overlapping_data(args)
  n_tr = int(0.8 * args.npts)
  n_te = args.npts - n_tr

  bias_const = 2.0
  data = {vi:(vdat + bias_const) for vi, vdat in data.items()}

  tr_data = {vi:vdat[:n_tr] for vi, vdat in data.items()}
  te_data = {vi:vdat[n_tr:] for vi, vdat in data.items()}

  view_sizes = {vi: vdat.shape[1] for vi, vdat in data.items()}
  main_view = 0
  models = make_default_tfms(args, view_sizes, is_cond=True)
  model = models[main_view]

  config = model.config
  config.batch_size = 1000
  config.lr = 1e-4
  config.reg_coeff = 0.1
  config.max_iters = args.max_iters
  config.stopping_eps = 1e-8

  b_o_tr = {vi: np.ones(n_tr) for vi in tr_data}
  b_o_te = {vi: np.ones(n_te) for vi in te_data}
  # x_tr = tr_data[main_view].astype(np.float32)
  # x_o_tr = np.concatenate(
  #     [tr_data[vi] for vi in range(args.nviews) if vi != main_view],
  #     axis=1).astype(np.float32)
  # x_te = te_data[main_view].astype(np.float32)
  # x_o_te = np.concatenate(
  #     [te_data[vi] for vi in range(args.nviews) if vi != main_view],
  #     axis=1).astype(np.float32)

  IPython.embed()
  model.fit(tr_data, b_o_tr)
  IPython.embed()
  x_tr = tr_data[main_view]
  x_te = te_data[main_view]
  z_tr, ld_tr = model.forward(x_tr, tr_data, rtn_logdet=True, rtn_torch=False)
  z_te, ld_te = model.forward(x_te, te_data, rtn_logdet=True, rtn_torch=False)
  ld_tr = torch_utils.torch_to_numpy(ld_tr)
  ld_te = torch_utils.torch_to_numpy(ld_te)
  i_tr = model.inverse(z_tr, tr_data, rtn_torch=False)
  i_te = model.inverse(z_te, te_data, rtn_torch=False)
  s_tr = model.sample(tr_data, use_mean=False)
  s_te = model.sample(te_data, use_mean=False)
  IPython.embed()

  # x_tr = torch.from_numpy(x_tr)
  # x_te = torch.from_numpy(x_te)

  # z_tr = model(x_tr, x_o_tr, rtn_torch=True)
  # zi_tr = model.inverse(z_tr, x_o_tr, rtn_torch=True)
  # z_te = model(x_te, x_o_te, rtn_torch=True)
  # zi_te = model.inverse(z_te, x_o_te, rtn_torch=True)
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

  # config.no_view_tfm = True
  # IPython.embed()

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



def pred_results_inliers(
    model, sub_dat, y, percentile=90, ignore_mult=3., use_valid_n=False):
  # Outliers are considered incorrect predictions
  abs_sub_dat = np.abs(sub_dat)
  sub_perc_vals = np.percentile(abs_sub_dat, axis=0, q=percentile)
  vinds = (abs_sub_dat < ignore_mult * sub_perc_vals).all(axis=1)
  print("Valid frac: %.2f" % vinds.sum()/vinds.shape[0])

  valid_x = sub_dat[vinds]
  valid_y = y[vinds]
  pred_y = model.predict(valid_x)

  if use_valid_n:
    n_pts = valid_y.shape[0]
  else:
    n_pts = y.shape[0]
  acc = (pred_y == valid_y).sum() / n_pts

  return acc


def get_ptbxl_results(base_model, sample_sets, y, set_labels):
  n = y.shape[0]
  results = {}
  for i, set_label in enumerate(set_labels):
    samples_i = sample_sets[set_label]
    results_i = {}
    for nv, nv_samples_i in samples_i.items():
      results_i = {}
      for subset, sub_dat in nv_samples_i.items():
        # pred_y = base_model.predict(sub_dat)
        # sub_acc = (pred_y == y)/n
        sub_acc = pred_results_inliers(base_model, sub_dat, y)
        results_i[subset] = sub_acc

    results[set_label] = results_i

  return results


def get_ptbxl_results_all_labels(base_models, sample_sets, y_tr, y_te, set_labels):
  results = {}
  y_lbl = {"tr": y_tr, "te": y_te}
  ptbxl_labels = ["NORM", "MI", "STTC", "CD", "HYP"]
  set_labels = list(sample_sets.keys())
  for dtype, y_d in y_lbl.items():
    results[dtype] = {}
    samples_d = {set_name: sample_sets[set_name][dtype] for set_name in sample_sets}
    for i, lbl in enumerate(ptbxl_labels):
      model_l = base_model[lbl]
      y_lbl = get_labels(y_d, [lbl])
      results_lbl = get_ptbxl_results(model_l, samples_d, y_lbl, set_labels)
      print(samples_d.keys())
      print(results_lbl.keys())
      results[dtype][lbl] = results_lbl
  return results


def convert_ptbxl_results_to_mats(results):
  acc_mats = {}
  num_labels = None
  label_list = None
  num_sets = None
  set_list = None
  num_subsets = None
  subset_list = None
  for dtype, results_d in results.items():
    acc_mats[dtype] = {}
    if num_labels is None:
      num_labels = len(results_d)
    if label_list is None:
      label_list = list(results_d.keys())
    for li, lbl in enumerate(label_list):
      results_ld = results_d[lbl]
      if num_sets is None:
        num_sets = len(results_ld)
      if set_list is None:
        set_list = list(results_ld.keys())
      for set_i, set_type in enumerate(set_list):
        results_sld = results_ld[set_type]
        if num_subsets is None:
          num_subsets = len(results_sld)
        if not acc_mats[dtype][set_type]:
          acc_mats[dtype][lbl] = np.zeros(num_subsets, num_sets)
        if subset_list is None:
          subset_list = list(results_sld.keys())
        for si, subset in enumerate(subset_list):
          acc_mats[dtype][lbl][si, set_i] = results_sld[subset]

  return acc_mats, label_list, set_list, subset_list


def plot_ptbxl_heatmaps(acc_mats, lbl_names, set_list, subset_list):
  dname = {"tr": "Training", "te": "Testing"}
  x_ticks = np.arange(0.5, len(subset_list))
  y_ticks = np.arange(0.5, len(set_list))
  for dtype, accs_d in acc_mats.items():
    for lbl, mat in accs_d.items():
      fig = plt.figure()
      hm = plt.imshow(mat)
      cbar = plt.colorbar(hm)
      plt.title("%s accuracy: %s" % (dname[dtype], set_name))
      plt.xtick_labels(x_ticks, subset_list)
      plt.ytick_labels(y_ticks, set_list)
      # for mind in msplit_inds:
      #   mind -= 0.5
      #   plt.axvline(x=mind, ls="--")
      #   plt.axhline(y=mind, ls="--")
      plt.show()


def test_ptbxl(args):
  local_data_file = "./ptbxl_data.npy"
  load_start_time = time.time()
  if os.path.exists(local_data_file):
    trX, trY, teX, teY = np.load(local_data_file, allow_pickle=True).tolist()
    ann = eeg_data.load_ptbxl_annotations()
  else:
    (trX, trY), (teX, teY), ann = eeg_data.load_ptbxl()

  print("Time taken to load PTB-XL: %.2fs" % (time.time() - load_start_time))
  IPython.embed()

  view_sizes = {vi:trX[vi].shape[1] for vi in trX}

  config, view_config_and_inits, cond_config_and_inits = \
      make_default_pipeline_config(args, view_sizes=view_sizes)
  view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits
  IPython.embed()

  model = ac_flow_pipeline.MultiviewACFlowTrainer(config)
  model.initialize(
      view_tfm_config_lists, view_tfm_init_lists,
      cond_tfm_config_lists, cond_tfm_init_lists)

  model.fit(trX)


def stratified_sample(view_data, labels, n_sampled, frac=None, get_inds=True):
  npts = labels.shape[0]
  if frac is not None:
    n_sampled = np.round(frac * npts).astype(int)

  ltypes, lcounts = np.unique(labels, return_counts=True)
  sampled_counts = (lcounts / npts * n_sampled).astype(int)
  sampled_counts[0] = n_sampled - sampled_counts[1:].sum()
  idx_set = []
  for i, l in enumerate(ltypes):
    li_count = sampled_counts[i]
    tot_li = lcounts[i]
    linds = (labels == l).nonzero()[0]
    idx_set += [linds[i] for i in np.random.permutation(tot_li)[:li_count]]

  np.random.shuffle(idx_set)
  labels_sampled = labels[idx_set]
  view_data_sampled = {vi: vdat[idx_set] for vi, vdat in view_data.items()}

  if get_inds:
    return view_data_sampled, labels_sampled, idx_set
  return view_data_sampled, labels_sampled


def get_single_digit(x, y, digit=0):
  digit_inds = (y == digit).nonzero()[0]
  x_digit = {vi:xvi[digit_inds] for vi, xvi in x.items()}
  y_digit = y[digit_inds]
  return x_digit, y_digit


def test_mnist(args):
  local_data_file = "./ptbxl_data.npy"
  load_start_time = time.time()
  n_views = 3
  tr_data, va_data, te_data = ssvd.load_split_mnist(n_views=n_views)
  (x_tr, y_tr) = tr_data
  (x_va, y_va) = va_data
  (x_te, y_te) = te_data
  print("Time taken to load MNIST: %.2fs" % (time.time() - load_start_time))
  IPython.embed()

  view_sizes = {vi:x_tr[vi].shape[1] for vi in x_tr}

  config, view_config_and_inits, cond_config_and_inits = \
      make_default_pipeline_config(args, view_sizes=view_sizes, start_logit=True)
  view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

  model = ac_flow_pipeline.MultiviewACFlowTrainer(config)
  model.initialize(
      view_tfm_config_lists, view_tfm_init_lists,
      cond_tfm_config_lists, cond_tfm_init_lists)
  IPython.embed()

  model.fit(x_tr)


_MV_DATAFUNCS = {
    "o1": make_default_overlapping_data,  # need any 2 views for all 
    "o2": make_default_overlapping_data2,  # need K-1 views for all
    "ind": make_default_independent_data,
    "sh": make_default_shape_data,
}

_TEST_FUNCS = {
    0: simple_test_cond_tfms,
    1: test_cond_missing_tfms,
    2: test_pipeline,
    3: test_ptbxl,
    4: test_mnist,
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
      ("init_tfm_set", str,
       "Initial sequence of s/l/v/k/r/g (3x scale coupling, linear, reverse,"
       "leaky_relu, rnn coupling, logit)", ""),
      ("repeat_tfm_set", str,
        "Repeated tfm set after @start_tfm_set (same tfms as before)", "slv"),
      ("num_tfm_sets", int, "Number of tfm sequences from @repeat_tfm_set", 3),
      ("end_tfm_set", str,
        "Final sequence of tfms (same tfms as before)", ""),
      ("cond_start_tfm_set", str,
       "Initial sequence for conditional tfms (same tfms as before)", ""),
      ("cond_repeat_tfm_set", str,
       "Repeated tfm set after @cond_start_tfm_set (same tfms as before)",
       "slv"),
      ("num_cond_tfm_sets", int,
       "Number of tfm sequences from @cond_repeat_tfm_set", 3),
      ("cond_end_tfm_set", str,
        "Final sequence of conditional tfms (same tfms as before)", ""),
      ("num_ss_tfm", int, "Number of ss tfms in a row (min 3)", 3),
      ("num_cond_ss_tfm", int,
       "Number of ss tfms in a row for conditional transform (min 3)", 3),
      ("num_lin_tfm", int, "DEPRECATED", 3),
      ("num_cond_lin_tfm", int, "DEPRECATED", 3),
      ("use_leaky_relu", bool, "DEPRECATED", False),
      ("use_reverse", bool, "DEPRECATED", False),
      ("dist_type", str, "Base dist. type ([mv_]gaussian/laplace/logistic)",
       "mv_gaussian"),
      ("use_ar", bool, "Flag for base dist being an AR model", False),
      ("n_components", int, "Number of components for likelihood MM", 5),
      ]
  args = utils.get_args(options)
  
  func = _TEST_FUNCS.get(args.expt, simple_test_cond_tfms)
  func(args)
 
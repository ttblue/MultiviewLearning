#!/usr/bin/env python
import itertools
import numpy as np
import os
import scipy
from sklearn import manifold, decomposition
import time
import torch
from torch import nn, functional
import umap

from dataprocessing import ecg_data, split_single_view_dsets as ssvd,\
    multiview_datasets as mvd
from models import ac_flow_pipeline, autoencoder, conditional_flow_transforms,\
    flow_likelihood, flow_pipeline, flow_transforms, torch_models
from synthetic import flow_toy_data, multimodal_systems
from utils import math_utils, torch_utils, utils

from tests.test_flow import make_default_tfm
from tests.test_ac_flow import SimpleArgs, convert_numpy_to_float32,\
    make_default_tfm_config, ArgsCopy, make_default_likelihood_config,\
    make_default_data, make_default_data_X, make_default_overlapping_data,\
    make_default_overlapping_data2, rotate_and_shift_data,\
    make_default_independent_data, make_default_shape_data

from matplotlib import pyplot as plt, patches
from mpl_toolkits.mplot3d import Axes3D

import IPython


def make_default_nn_config(
    bounded=False, layer_units=None, last_activation=None):
  input_size = 10  # Computed online
  output_size = 10  # Computed online
  layer_units = [32]#[32, 64] if layer_units is None else layer_units
  use_vae = False
  activation = nn.ReLU  # nn.functional.relu
  if bounded:
    last_activation = torch.nn.Tanh
  elif last_activation is None:
    last_activation = torch.nn.ReLU
  # functional.sigmoid

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
    args, view_sizes, double_tot_dim=False, rtn_args=False, is_cond=True,
    dev=None):
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
  default_hidden_sizes = [64]#, 128]
  default_activation = nn.Tanh
  default_nn_config = make_default_nn_config()
  default_bounded_nn_config = make_default_nn_config(bounded=True)

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
              (bit_mask, default_bounded_nn_config))
          bit_mask = 1 - bit_mask

      elif tf == "l":
        linear_tfm_config = config_func("linear")
        linear_tfm_config.has_bias = True
        if is_cond:
          vi_tfm_inits.append(default_nn_config)# init_mat))
          linear_tfm_config.func_nn_config.last_activation = torch.nn.Tanh
        else:
          vi_tfm_inits.append(vdim)# init_mat))
        vi_tfm_configs.append(linear_tfm_config)

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
            tfm_configs[vi], vi, view_sizes, tfm_inits[vi], comp_config, dev)
        for vi in tfm_configs
    }
  else:
    models = {
      vi: flow_transforms.make_transform(
            tfm_configs[vi], tfm_inits[vi], comp_config)
      for vi in tfm_configs
    }

  return models


def make_default_ae_config(args):
  # nn_config = make_default_nn_config()
  input_size = None  # set online
  code_size = args.ae_code_size
  dropout_p = 0.2
  lm = 0.
  max_iters = 10000
  batch_size = 100
  lr = 1e-3

  encoder_config = make_default_nn_config(layer_units=[128, 64])
  decoder_config = make_default_nn_config(layer_units=[64, 128], bounded=True)

  verbose = True
  ae_config = autoencoder.AEConfig(
      input_size=input_size, code_size=code_size,
      encoder_config=encoder_config, decoder_config=decoder_config,
      lm=lm, dropout_p=dropout_p, max_iters=max_iters, batch_size=batch_size,
      lr=lr, verbose=verbose)
  return ae_config


def make_default_pipeline_config(
    args, view_sizes={}, no_view_tfm=False, start_logit=False):

  tot_dim = sum(view_sizes.values())
  # print(args.__dict__.keys())
  all_view_args = ArgsCopy(args)
  # all_view_args.ndim = tot_dim
  cond_config_lists, cond_inits_lists = make_default_tfms(
      all_view_args, view_sizes, rtn_args=True, is_cond=True)

  view_tfm_config_lists, view_tfm_init_lists = make_default_tfms(
      all_view_args, view_sizes, rtn_args=True, is_cond=False)

  use_pre_view_ae = args.use_ae
  view_ae_configs = None
  if use_pre_view_ae:
    view_ae_configs = {vi:make_default_ae_config(args) for vi in view_sizes}
    view_ae_model_files = None
    if args.ae_model_file is not None:
      view_ae_model_files = {
          vi: (args.ae_model_file % vi) for vi in view_sizes
      }
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
      (cond_config_lists, cond_inits_lists), view_ae_configs


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


def make_missing_dset(data, main_view, include_nv_0=False, separate_nv=False):
  v_dims = {vi:vdat.shape[1] for vi, vdat in data.items()}
  obs_views = [vi for vi in data if vi not in main_view]
  main_dim = v_dims[main_view]
  npts =  data[main_view].shape[0]

  x = {} if separate_nv else []
  x_o = {} if separate_nv else {vi:[] for vi in obs_views}
  b_o = {} if separate_nv else {vi:[] for vi in obs_views}

  mv_x = data[main_view]
  obs_views = [vi for vi in data if vi != main_view]

  tot_views = len(obs_views)
  start_nv = 0 if include_nv_0 else 1
  for nv in range(start_nv, tot_views + 1):
    x_nv = []
    x_o_nv = {vi:[] for vi in obs_views}
    b_o_nv = {vi:[] for vi in obs_views}

    for perm in itertools.combinations(obs_views, nv):
      x_nv.append(mv_x)
      for vi in obs_views:
        x_o_nv[vi].append(data[vi])
        b_o_nv[vi].append((np.ones(npts) * int(vi in perm)))
      # output, zero_imputed_data, b_cat = concat_with_binary_flags(
      #     data, main_view, b_available)
    if separate_nv:
      x[nv] = np.concatenate(x_nv, axis=0)
      x_o[nv] = {
          vi:np.concatenate(x_o_vi, axis=0) for vi, x_o_vi in x_o_nv.items()}
      b_o[nv] = {
          vi:np.concatenate(b_o_vi, axis=0) for vi, b_o_vi in b_o_nv.items()}
    else:
      x.extend(x_nv)
      for vi in obs_views:
        x_o[vi].extend(x_o_nv[vi])
        b_o[vi].extend(b_o_nv[vi])

  if not separate_nv:
    x = np.concatenate(x, axis=0)
    x_o = {vi:np.concatenate(x_o_vi, axis=0) for vi, x_o_vi in x_o.items()}
    b_o = {vi:np.concatenate(b_o_vi, axis=0) for vi, b_o_vi in b_o.items()}

  return x, x_o, b_o


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
  s_tr = model.sample(tr_data, use_mean=True)
  s_te = model.sample(te_data, use_mean=True)
  IPython.embed()

  # x_tr = torch.from_numpy(x_tr)
  # x_te = torch.from_numpy(x_te)

  # z_tr = model(x_tr, x_o_tr, rtn_torch=True)
  # zi_tr = model.inverse(z_tr, x_o_tr, rtn_torch=True)
  # z_te = model(x_te, x_o_te, rtn_torch=True)
  # zi_te = model.inverse(z_te, x_o_te, rtn_torch=True)
  print("Ready")
  IPython.embed()


def get_sampled_cat(gen_vals, true_vals):
  all_vi = np.sort(np.unique(list(gen_vals.keys()) + list(true_vals.keys())))
  all_vals = {vi:(gen_vals[vi] if vi in gen_vals else true_vals[vi]) for vi in all_vi}
  cat_vals = np.concatenate([all_vals[vi] for vi in all_vi], axis=1)
  return cat_vals


def plot_digit(cat_fs, xy=(-1, 8.5), w=10, h=29):
  if not isinstance(cat_fs, list): cat_fs = [cat_fs]
  digit_plots = [cat_f.reshape(28, 28) for cat_f in cat_fs]
  if len(cat_fs) > 1:
    digit_plot = np.concatenate(digit_plots, axis=1)
  else:
    digit_plot = digit_plots[0]

  digit_plot[digit_plot < 0] = 0
  digit_plot[digit_plot >= 1] = 1

  fig, ax = plt.subplots()
  ax.imshow(digit_plot, cmap="gray")
#   rect1 = [0, 9, 28, 9]
#   rect2 = [28, 18, 28, 18]
#   rect3 = [56, 28, 28, 28]
  r1 = patches.Rectangle(xy, h, w, linewidth=1, edgecolor='g', facecolor='none')
  # r1 = patches.Rectangle((-1, -1), 29, 9.5, linewidth=1, edgecolor='g', facecolor='none')
  # r2 = patches.Rectangle((28, 0), 28, 17, linewidth=1, edgecolor='g', facecolor='none')
  # r3 = patches.Rectangle((56, 0), 27, 27, linewidth=1, edgecolor='g', facecolor='none')
  ax.add_patch(r1)
  # ax.add_patch(r2)
  # ax.add_patch(r3)
  plt.show()


def plot_many_digits(
    pred_digits, true_digits, first=None, grid_size=(10, 10),
    rect_args=((-1, 8.5), 10, 28.5), title=""):

  pred_digits = np.copy(pred_digits)
  # print(pred_digits.dtype)
  # pred_digits = [dig.reshape(28, 28) for dig in pred_digits]
  pred_digits = pred_digits.reshape(-1, 28, 28)
  if true_digits is not None:
    # true_digits = [dig.reshape(28, 28) for dig in true_digits]
    true_digits = true_digits.reshape(-1, 28, 28)
  ndigs = len(pred_digits)

  nrows, ncols = grid_size
  if true_digits is None:
    fig, axs = plt.subplots(
        nrows, ncols, gridspec_kw = {'wspace':0.05, 'hspace':0.05})
  else:
    fig, axs = plt.subplots(nrows, ncols)

  # IPython.embed()
  # plt.subplots_adjust(wspace=0, hspace=0)
  white_line = np.ones((pred_digits[0].shape[1], 1))
  xy, h, w = rect_args
  dig_idx = 0
  plotted_first = (first is None)
  for ri in range(nrows):
    if dig_idx >= ndigs:
      break

    for ci in range(ncols):
      if dig_idx >= ndigs:
        break

      ax = axs[ri, ci]
      if plotted_first:
        print("Plotting digit %i" % (dig_idx + 1), end="\r")
        # ax.set_aspect('equal')
        pred_dig = pred_digits[dig_idx]
        # IPython.embed()
        pred_dig[pred_dig < 0] = 0
        pred_dig[pred_dig > 1] = 1

        if true_digits is not None:
          true_dig = true_digits[dig_idx]
          plot_dig = np.concatenate([pred_dig, white_line, true_dig], axis=1)
        else:
          plot_dig = pred_dig
        dig_idx += 1

        ax.imshow(plot_dig, cmap="gray")
        rect = patches.Rectangle(
            xy, w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_visible(False)
        ax.yaxis.set_ticks([])
      else:
        plotted_first = True
        first = first.reshape(28, 28)
        ax.imshow(first, cmap="gray")
        rect = patches.Rectangle(
            (0, 0), 27, 27, linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_visible(False)
        ax.yaxis.set_ticks([])

  # IPython.embed()
  plt.tight_layout()
  # if true_digits is None:
  if title:
    fig.suptitle(title, fontsize=20)
  plt.show()
  # plt.pause(10.)
  # IPython.embed()


def batch_sample(model, x_o, b_o, use_mean, batch_size=100):
  npts = x_o[utils.get_any_key(x_o)].shape[0]
  samples = []

  n_batches = np.ceil(npts/batch_size).astype(int)
  for bidx in range(n_batches):
    print("Sampling batch %i out of %i." % (bidx + 1, n_batches), end='\r')
    start_ind = bidx * batch_size
    end_ind = start_ind + batch_size
    x_o_batch = {vi:xvi[start_ind:end_ind] for vi, xvi in x_o.items()}
    b_o_batch = (
        {vi:bvi[start_ind:end_ind] for vi, bvi in b_o.items()}
        if b_o is not None else None
    )
    samples.append(
        model.sample(x_o_batch, b_o_batch, use_mean=use_mean, rtn_torch=False))

  samples = np.concatenate(samples, axis=0)
  return samples


def trunc_svd_dim_red(X, p_s0=0.05):
  dim = math_utils.get_svd_frac_dim(X, p_s0=p_s0)
  svd_model = decomposition.TruncatedSVD(n_components=dim)


def test_mnist(args):
  load_start_time = time.time()
  n_views = 3
  all_tr_data, all_va_data, all_te_data = ssvd.load_split_mnist(n_views=n_views)
  (tr_data, y_tr) = all_tr_data
  (va_data, y_va) = all_va_data
  (te_data, y_te) = all_te_data
  print("Time taken to load MNIST: %.2fs" % (time.time() - load_start_time))

  dev = None
  if torch.cuda.is_available() and args.gpu_num >= 0:
    dev = torch.device("cuda:%i" % args.gpu_num)

  n_sampled_tr = args.npts
  n_sampled_te = args.npts // 2
  tr_data, y_tr, tr_idxs = stratified_sample(tr_new, y_tr, n_sampled=n_sampled_tr)
  te_data, y_te, te_idxs = stratified_sample(te_new, y_te, n_sampled=n_sampled_te)
  n_tr = tr_data[0].shape[0]
  n_te = te_data[0].shape[0]
  globals().update(locals())
  b_o_tr = {vi: np.ones(n_tr) for vi in tr_data}
  b_o_te = {vi: np.ones(n_te) for vi in te_data}
  # digit = 0
  # x_tr, y_tr = get_single_digit(x_tr, y_tr, digit=digit)
  view_sizes = {vi:xv.shape[1] for vi, xv in tr_data.items()}
  main_view = 1
  models = make_default_tfms(args, view_sizes, is_cond=True, dev=dev)
  model = models[main_view]

  config = model.config
  config.max_iters = args.max_iters
  config.batch_size = args.batch_size
  # config, view_config_and_inits, cond_config_and_inits = \
  #     make_default_pipeline_config(args, view_sizes=view_sizes, start_logit=True)
  # view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  # cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

  # model = ac_flow_pipeline.MultiviewACFlowTrainer(config)
  # model.initialize(
  #     view_tfm_config_lists, view_tfm_init_lists,
  #     cond_tfm_config_lists, cond_tfm_init_lists)

  # model.to(dev)
  IPython.embed()
  model.fit(tr_data, b_o_tr, dev=dev)
  IPython.embed()
  cpu_dev = torch.device("cpu")
  model.to(dev)
  globals().update(locals())
  x_tr = tr_data[main_view]
  x_te = te_data[main_view]
  z_tr, ld_tr = model.forward(x_tr, tr_data, rtn_logdet=True, rtn_torch=False)
  z_te, ld_te = model.forward(x_te, te_data, rtn_logdet=True, rtn_torch=False)
  ld_tr = torch_utils.torch_to_numpy(ld_tr)
  ld_te = torch_utils.torch_to_numpy(ld_te)
  globals().update(locals())
  i_tr = model.inverse(z_tr, tr_data, rtn_torch=False)
  i_te = model.inverse(z_te, te_data, rtn_torch=False)
  s_tr = model.sample(tr_data, use_mean=True)
  s_te = model.sample(te_data, use_mean=True)
  # IPython.embed()
  globals().update(locals())
  cat_tr = np.concatenate([tr_data[vi] for vi in range(len(tr_data))], axis=1)
  cat_va = np.concatenate([va_data[vi] for vi in range(len(va_data))], axis=1)
  cat_te = np.concatenate([te_data[vi] for vi in range(len(te_data))], axis=1)
  globals().update(locals())
  te_digits = get_sampled_cat({main_view:s_te}, te_data)
  tr_digits = get_sampled_cat({main_view:s_tr}, tr_data)
  # va_digits = get_sampled_cat({main_view:s_va}, va_data)

  didx = 10
  plt_type = "te"
  n_samples = 100
  base_data = tr_data if plt_type == "tr" else te_new
  globals().update(locals())
  sample_xo = {vi:xvi[([didx]*n_samples)] for vi, xvi in base_data.items()}
  didx_samples = model.sample(sample_xo, use_mean=False)
  didx_recon = torch_utils.torch_to_numpy(
      view_ae_models[main_view]._decode(didx_samples))
  if plt_type == "tr":
    sample_xo_all = {vi:vdat[tr_idxs][([didx]*n_samples)] for vi, vdat in all_tr_data[0].items()}
    lbl = all_tr_data[1][tr_idxs][didx]
    first = cat_tr[didx]
  elif plt_type == "te":
    sample_xo_all = {vi:vdat[([didx]*n_samples)] for vi, vdat in all_te_data[0].items()}
    lbl = all_te_data[1][didx]
    first = cat_te[didx]
  didx_digits = get_sampled_cat({main_view: didx_recon}, sample_xo_all)
  plot_many_digits(didx_digits, None, first=first, grid_size=(10, 10), title="Test digit: %i" % lbl)


def test_mnist_ae(args):
  load_start_time = time.time()
  n_views = 3
  all_tr_data, all_va_data, all_te_data = ssvd.load_split_mnist(n_views=n_views)
  (tr_data, y_tr) = all_tr_data
  (va_data, y_va) = all_va_data
  (te_data, y_te) = all_te_data
  print("Time taken to load MNIST: %.2fs" % (time.time() - load_start_time))

  view_sizes = {vi:xv.shape[1] for vi, xv in tr_data.items()}
  view_ae_config = make_default_ae_config(args)

  max_iters = 1000
  view_ae_models = {}
  for vi, vdim in view_sizes.items():
    view_ae_config = make_default_ae_config(args)
    view_ae_config.input_size = vdim
    view_ae_config.max_iters = max_iters

    vi_ae = autoencoder.AutoEncoder(view_ae_config)
    view_ae_models[vi] = vi_ae

  IPython.embed()
  for vi, vi_ae in view_ae_models.items():
    vi_xs = tr_data[vi]
    print("Training AE for view %i" % vi)
    print("Size of data: %s" % (vi_xs.shape,))

    vi_ae.initialize()
    vi_ae.fit(vi_xs)

  IPython.embed()


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
  config, view_config_and_inits, cond_config_and_inits, view_ae_configs = \
      make_default_pipeline_config(args, view_sizes=view_sizes)
  view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

  # config.no_view_tfm = True
  # IPython.embed()
  dev = None
  if torch.cuda.is_available() and args.gpu_num >= 0:
    dev = torch.device("cuda:%i" % args.gpu_num)

  model = ac_flow_pipeline.MultiviewACFlowTrainer(config)
  model.initialize(
      view_sizes, view_tfm_config_lists, view_tfm_init_lists,
      cond_tfm_config_lists, cond_tfm_init_lists, view_ae_configs)

  IPython.embed()

  model.fit(data)
  n_test_samples = n_te
  # sample_data = model.sample(n_test_samples, rtn_torch=False)

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


# def test_mnist(args):
#   local_data_file = "./ptbxl_data.npy"
#   load_start_time = time.time()
#   n_views = 3
#   tr_data, va_data, te_data = ssvd.load_split_mnist(n_views=n_views)
#   (x_tr, y_tr) = tr_data
#   (x_va, y_va) = va_data
#   (x_te, y_te) = te_data
#   print("Time taken to load MNIST: %.2fs" % (time.time() - load_start_time))
#   IPython.embed()

#   view_sizes = {vi:x_tr[vi].shape[1] for vi in x_tr}

#   config, view_config_and_inits, cond_config_and_inits = \
#       make_default_pipeline_config(args, view_sizes=view_sizes, start_logit=True)
#   view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
#   cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

#   model = ac_flow_pipeline.MultiviewACFlowTrainer(config)
#   model.initialize(
#       view_tfm_config_lists, view_tfm_init_lists,
#       cond_tfm_config_lists, cond_tfm_init_lists)
#   IPython.embed()

#   model.fit(x_tr)


_MV_DATAFUNCS = {
    "o1": make_default_overlapping_data,  # need any 2 views for all 
    "o2": make_default_overlapping_data2,  # need K-1 views for all
    "ind": make_default_independent_data,
    "sh": make_default_shape_data,
}

_TEST_FUNCS = {
    0: simple_test_cond_tfms,
    1: test_mnist,
    2: test_mnist_ae,
    3: test_pipeline,
    # 1: test_cond_missing_tfms,
    # 3: test_ptbxl,
}


if __name__ == "__main__":
  np.set_printoptions(linewidth=1000, precision=3, suppress=True)
  torch.set_printoptions(precision=3)
  options = [
      ("etype", str, "Expt. type (gen/rec)", "gen"),
      ("dtype", str, "Data type (random/single_dim_copy/o1/o2/ind/sh)", "ind"),
      ("gpu_num", int, "GPU ID if using GPU. -1 for CPU.", -1),
      ("nviews", int, "Number of views", 3),
      ("npts", int, "Number of points", 1000),
      ("ndim", int, "Dimensions", 10),
      ("peps", float, "Perturb epsilon", 0.),
      ("scale", float, "Scale of the data.", 1.),
      ("shape", str, "Shape of toy data.", "cube"),
      ("max_iters", int, "Number of iters for opt.", 10000),
      ("batch_size", int, "Batch size for opt.", 100),
      ("start_tfm_set", str,
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
      ("use_ae", bool, "Flag for using autoencoders before view tfm.", False),
      ("ae_code_size", int,
       "Code size for view pre-flow tfm AutoEncoders (-1 for no AE)", 20),
      ("ae_model_file", str,
        "If not None, should have a model for each view, with \%i in the "
        "filename for the view index", None),
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
  
  func = _TEST_FUNCS.get(args.expt, test_mnist)
  func(args)
 

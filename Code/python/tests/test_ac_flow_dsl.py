#!/usr/bin/env python
import itertools
import numpy as np
import os
import pickle
import scipy
from sklearn import manifold, decomposition
import time
import torch
from torch import nn, functional
import umap

from dataprocessing import ecg_data, split_single_view_dsets as ssvd,\
    mnist_utils, multiview_datasets as mvd, physionet
from models import ac_flow_pipeline, ac_flow_dsl_pipeline,\
    autoencoder, conditional_flow_transforms,\
    flow_likelihood, flow_pipeline, flow_transforms, torch_models
from synthetic import flow_toy_data, multimodal_systems
from utils import math_utils, torch_utils, utils

from tests.test_flow import make_default_tfm
from tests.test_ac_flow import SimpleArgs, convert_numpy_to_float32,\
    convert_numpy_to_float64, make_default_overlapping_data,\
    make_default_overlapping_data2, make_default_independent_data,\
    make_default_shape_data, rotate_and_shift_data
    # make_default_tfm_config, ArgsCopy, make_default_likelihood_config,\
    # make_default_data, make_default_data_X, 
    # make_default_overlapping_data2, rotate_and_shift_data,\
    # make_default_independent_data, make_default_shape_data
from tests.test_ac_flow_jp import make_default_nn_config, make_default_tfms,\
    make_default_cond_tfm_config, ArgsCopy, make_default_likelihood_config,\
    stratified_sample, get_sampled_cat, get_sampled_cat_grid
from utils import plot_utils

from matplotlib import pyplot as plt, patches
from mpl_toolkits.mplot3d import Axes3D


from sklearn import ensemble, kernel_ridge, linear_model, neural_network, svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV as gscv


import IPython
import importlib as imp


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

  dsl_coeff = 1.0

  batch_size = 50
  lr = 1e-4
  max_iters = args.max_iters
  grad_clip = 5.0

  verbose = True

  # IPython.embed()
  config = ac_flow_dsl_pipeline.MACFDTConfig(
      expand_b=expand_b, no_view_tfm=no_view_tfm,
      likelihood_config=likelihood_config, base_dist=base_dist,
      dsl_coeff=dsl_coeff, batch_size=batch_size, lr=lr, max_iters=max_iters,
      grad_clip=grad_clip, verbose=verbose)

  return config, (view_tfm_config_lists, view_tfm_init_lists),\
      (cond_config_lists, cond_inits_lists), view_ae_configs


def test_pipeline_dsl(args):
  data_func = _MV_DATAFUNCS.get(args.dtype, make_default_overlapping_data)
  data, ptfms = data_func(args)
  # data = convert_numpy_to_float32(data)

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


def test_mnist_dsl(args):
  load_start_time = time.time()
  n_views = 4
  split_shape = "grid"
  all_tr_data, all_va_data, all_te_data, split_inds = ssvd.load_split_mnist(
      n_views=n_views, shape=split_shape)
  (tr_data, y_tr) = all_tr_data
  (va_data, y_va) = all_va_data
  (te_data, y_te) = all_te_data
  print("Time taken to load MNIST: %.2fs" % (time.time() - load_start_time))

  torch.set_default_dtype(torch.float64)
  load_trunc_mnist_models = True
  p_s0 = 0.05
  svd_model_file = "./saved_models/mnist/tsvd_v%i/mdl.pkl" % n_views
  if load_trunc_mnist_models and os.path.exists(svd_model_file):
    with open(svd_model_file, "rb") as fh:
      svd_models = {vi:pickle.load(fh) for vi in range(n_views)}
      tr_data = {
          vi: smdl.transform(tr_data[vi]) for vi, smdl in svd_models.items()
      }
  else:
    tr_data, svd_models = trunc_svd_dim_red(tr_data, p_s0=p_s0)
    with open(svd_model_file, "wb") as fh:
      for vi in range(n_views):
        pickle.dump(svd_models[vi], fh)

  mnist8_onnx_file = os.path.join(
      os.getenv("HOME"), "Research/MultiviewLearning/Code/",
      "pretrained_models/mnist/mnist-8.onnx")
  mnist8 = mnist_utils.MNIST8(mnist8_onnx_file, n_views)
  mnist8.save_svd_models(svd_models)

  tr_data = convert_numpy_to_float64(tr_data)
  te_data = convert_numpy_to_float64({
      vi: smdl.transform(te_data[vi]) for vi, smdl in svd_models.items()
  })
  va_data = convert_numpy_to_float64({
      vi: smdl.transform(va_data[vi]) for vi, smdl in svd_models.items()
  })
  y_tr, y_te, y_va = convert_numpy_to_float64([y_tr, y_te, y_va])


  # IPython.embed()
  # dev = None
  # if torch.cuda.is_available() and args.gpu_num >= 0:
  #   dev = torch.device("cuda:%i" % args.gpu_num)

  n_sampled_tr = args.npts
  n_sampled_te = args.npts // 2
  tr_data, y_tr, tr_idxs = stratified_sample(tr_data, y_tr, n_sampled=n_sampled_tr)
  te_data, y_te, te_idxs = stratified_sample(te_data, y_te, n_sampled=n_sampled_te)
  n_tr = tr_data[0].shape[0]
  n_te = te_data[0].shape[0]
  globals().update(locals())
  b_o_tr = {vi: np.ones(n_tr) for vi in tr_data}
  b_o_te = {vi: np.ones(n_te) for vi in te_data}
  # digit = 0
  # x_tr, y_tr = get_single_digit(x_tr, y_tr, digit=digit)
  view_sizes = {vi:xv.shape[1] for vi, xv in tr_data.items()}

  config, view_config_and_inits, cond_config_and_inits, view_ae_configs = \
      make_default_pipeline_config(args, view_sizes=view_sizes)
  view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

  config.no_view_tfm = True
  # IPython.embed()
  dev = None
  if torch.cuda.is_available() and args.gpu_num >= 0:
    dev = torch.device("cuda:%i" % args.gpu_num)

  model = ac_flow_dsl_pipeline.MACFlowDSLTrainer(config)
  model.initialize(
      view_sizes, view_tfm_config_lists, view_tfm_init_lists,
      cond_tfm_config_lists, cond_tfm_init_lists, view_ae_configs)
  # model.set_ds_loss(mnist8)

  # IPython.embed()
  # main_view = 1
  # models = make_default_tfms(args, view_sizes, is_cond=True, dev=dev)
  # model = models[main_view]

  # config = model.config
  # config.max_iters = args.max_iters
  # config.batch_size = args.batch_size
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
  model.fit(tr_data, y_tr, b_o_tr, mnist8, dev=dev)
  IPython.embed()

  cpu_dev = torch.device("cpu")
  model.to(dev)

  view_subsets = []
  view_range = list(range(n_views))
  for nv in view_range:
    view_subsets.extend(list(itertools.combinations(view_range, nv + 1)))

  globals().update(locals())
  # n_te = 500
  te_data = {vi:xvi[:n_te] for vi, xvi in te_data.items()}
  globals().update(locals())

  # cat_tr = np.concatenate([tr_data[vi] for vi in range(len(tr_data))], axis=1)
  # # cat_va = np.concatenate([va_data[vi] for vi in range(len(va_data))], axis=1)
  # cat_te = np.concatenate(
  #     [te_data[vi] for vi in range(len(te_data))], axis=1)
  tr_base = {vi:all_tr_data[0][vi][tr_idxs] for vi in range(n_views)}
  te_base = {vi:all_te_data[0][vi][te_idxs] for vi in range(n_views)}
  n_tr = tr_base[0].shape[0]
  n_te = te_base[0].shape[0]
  true_tr_digits = get_sampled_cat_grid(tr_base, np.zeros((n_tr, 784)))
  true_te_digits = get_sampled_cat_grid(te_base, np.zeros((n_te, 784)))

  tr_digits = {}
  te_digits = {}
  for vsub in view_subsets:
    # x_o_tr = {vi:tr_data[vi] for vi in vsub}
    # x_o_te = {vi:te_data[vi] for vi in vsub}

    trd = model.sample(
        tr_data, b_o=None, sampled_views=vsub, batch_size=None, rtn_torch=False)
    ted = model.sample(
        te_data, b_o=None, sampled_views=vsub, batch_size=None, rtn_torch=False)

    trd2 = {
      vi:svd_models[vi].inverse_transform(vtrd) for vi, vtrd in trd.items()}
    ted2 = {
      vi:svd_models[vi].inverse_transform(vted) for vi, vted in ted.items()}

    tr_digits[vsub] = get_sampled_cat_grid(trd2, true_tr_digits)
    te_digits[vsub] = get_sampled_cat_grid(ted2, true_te_digits)


  good_dids = {}
  bad_dids = {}
  neutral_dids = {}

  good_dids[0,] = [6, 49, 90, 93]
  neutral_dids[0,] = [11, 64, 76, 88]
  bad_dids[0,] = [18, 43, 72, 98]

  good_dids[1,] = [1, 64]
  neutral_dids[1,] = [18, 29, 97]
  bad_dids[1,] = [2, 3, 17, 34]

  good_dids[2,] = [33, 55]
  neutral_dids[2,] = [1, 44, 68]
  bad_dids[2,] = [29, 47, 54, 71, 80]

  good_dids[3,] = [37, ]
  neutral_dids[3,] = [4, 90, 54]# 1, 3, 9]
  bad_dids[3,] = [69, 70]

  good_dids[0,1] = [9, 47, 68]
  neutral_dids[0,1] = [1, 5, 58, 96]
  bad_dids[0,1] = [13, 27, 52, 43]

  good_dids[0,2] = [84]
  neutral_dids[0,2] = [5, 66, 78, 120]
  bad_dids[0,2] = [4, 83, 97]

  good_dids[0,3] = [56, 58]
  neutral_dids[0,3] = [17, 32, 83]
  bad_dids[0,3] = [1, 4, 6]

  good_dids[1,2] = []
  neutral_dids[1,2] = [68, 10]
  bad_dids[1,2] = [78, 49]

  good_dids[1,3] = [23, 30, 64, 100]
  neutral_dids[1,3] = [17, 47]
  bad_dids[1,3] = [84, 111]

  good_dids[2,3] = [40, 51, 100]
  neutral_dids[2,3] = [34, 41, 56, 62]
  bad_dids[2,3] = [4, 7, 9, 54]

  good_dids[0,1,2] = [30, 38, 49, 59, 115]
  neutral_dids[0,1,2] = [3, 57, 83, 95, 53, 10]
  bad_dids[0,1,2] = [44, 72, 99, 105]

  good_dids[0,1,3] = [24, 80, 105, 116]
  neutral_dids[0,1,3] = [17, 89, 97]
  bad_dids[0,1,3] = [7, 34, 73, 107]

  good_dids[0,2,3] = [30, 53]
  neutral_dids[0,2,3] = [38, 91, 102]
  bad_dids[0,2,3] = [36, 51, 109]

  good_dids[1,2,3] = [49]
  neutral_dids[1,2,3] = [22, 92, 103]
  bad_dids[1,2,3] = [38, 100, 72]

def make_perm_digit_lists(ids, digit_sets, true_digits, nviews=1, totviews=4):
  perm_list = []
  digit_list = [[] for _ in digit_sets]
  true_digit_list = []

  flip_perm = (nviews == 3)

  for perm, idlist in ids.items():
    if len(perm) != nviews:
      continue
    # IPython.embed()
    for di, dset in enumerate(digit_sets):
      digit_list[di].append(dset[perm][idlist])
    true_digit_list.append(true_digits[idlist])
    if flip_perm:
      fperm = tuple([i for i in range(totviews) if i not in perm])
      perm_list.extend([fperm] * len(idlist))
    else:
      perm_list.extend([perm] * len(idlist))

  digit_list = [np.concatenate(dlist, axis=0) for dlist in digit_list]
  true_digit_list = np.concatenate(true_digit_list, axis=0)

  return digit_list, true_digit_list, perm_list


def complement_subset_keys(data, nviews):
  cdata = {}
  view_range = list(range(nviews))
  for perm, pdat in data.items():
    cperm = tuple([i for i in view_range if i not in perm])
    cdata[cperm] = pdat
  return cdata


def convert_to_numpy(val):
  if isinstance(val, dict):
    val = {k: convert_to_numpy(v) for k, v in val.items()}
    return val
  else:
    return np.array(val)


def train_classifier(base_x, base_y, hidden_sizes=(128, 64), n_estimators=10):
  classifier = ensemble.RandomForestClassifier(n_estimators=n_estimators)
  # classifier = neural_network.MLPClassifier(
  #     hidden_layer_sizes=hidden_sizes, max_iter=500)
  cat_x = get_sampled_cat({}, base_x)
  return classifier.fit(cat_x, base_y)


def evaluate_mv_performance(classifier, perm_xs, true_x, true_y):
  nviews = len(true_x)
  cat_x = get_sampled_cat({}, true_x)
  npts = cat_x.shape[0]

  base_preds = classifier.predict(cat_x)
  base_acc = (base_preds == true_y).sum() / npts

  all_view_subset = tuple(range(nviews))
  perm_accs = {all_view_subset: base_acc}
  perm_preds = {all_view_subset: base_preds}
  nv_accs = {vi: (0, 0) for vi in range(1, nviews)}

  for perm, px in perm_xs.items():
    cat_px = get_sampled_cat(px, true_x)
    perm_pred = classifier.predict(cat_px)
    pacc = (perm_pred == true_y).sum() / npts

    nv = len(perm)
    if nv == nviews:
      continue
    perm_accs[perm] = pacc
    perm_preds[perm] = perm_pred
    curr_nv_acc = nv_accs[nv]
    nv_accs[nv] = curr_nv_acc[0] + pacc, curr_nv_acc[1] + 1

  nv_accs = {nv: nvi[0] / nvi[1] for nv, nvi in nv_accs.items()}
  nv_accs[nviews] = base_acc

  return perm_accs, perm_preds, nv_accs


def test_mitbih(args):
  load_start_time = time.time()
  tr_frac = 0.8
  # tr_data, te_data = physionet.get_mv_mitbih_split(tr_frac)
  polysom_file = "./data/mitbih/polysom_fft_full.npy"
  tr_data, te_data = np.load(polysom_file, allow_pickle=True).tolist()
  (tr_x, tr_y, tr_ya, tr_ids) = tr_data
  (te_x, te_y, te_ya, te_ids) = te_data
  print("Time taken to load MITBIH: %.2fs" % (time.time() - load_start_time))
  sub_names = {0:"ECG", 1:"BP", 2:"EEG"}

  fft_size = 20
  remove_first_freq = True
  if remove_first_freq:
    f_inds = list(range(1, fft_size)) + list(range(fft_size + 1, 2 * fft_size))
    tr_x = {vi: xvi[:, f_inds] for vi, xvi in tr_x.items()}
    te_x = {vi: xvi[:, f_inds] for vi, xvi in te_x.items()}

  torch.set_default_dtype(torch.float64)
  view_sizes = {vi:xv.shape[1] for vi, xv in tr_x.items()}
  n_views = len(view_sizes)

  config, view_config_and_inits, cond_config_and_inits, view_ae_configs = \
      make_default_pipeline_config(args, view_sizes=view_sizes)
  view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

  config.no_view_tfm = True
  # IPython.embed()
  dev = None
  if torch.cuda.is_available() and args.gpu_num >= 0:
    dev = torch.device("cuda:%i" % args.gpu_num)

  model = ac_flow_dsl_pipeline.MACFlowDSLTrainer(config)
  model.initialize(
      view_sizes, view_tfm_config_lists, view_tfm_init_lists,
      cond_tfm_config_lists, cond_tfm_init_lists, view_ae_configs)

  n_sampled_tr = args.npts
  n_sampled_te = args.npts // 2
  # (full_tr_x, full_tr_y, full_tr_ya) = (tr_x, tr_y, tr_ya)
  tr_x, tr_y, tr_idxs = stratified_sample(tr_x, tr_y, n_sampled=n_sampled_tr)
  # te_data, y_te, te_idxs = stratified_sample(te_data, y_te, n_sampled=n_sampled_te)
  n_tr = tr_x[0].shape[0]
  tr_x = convert_numpy_to_float64(tr_x)
  tr_y = convert_numpy_to_float64(tr_y)

  IPython.embed()
  model.fit(tr_x, tr_y, None, None)
  IPython.embed()

  te_x = convert_numpy_to_float64(te_x)
  te_y = convert_numpy_to_float64(te_y)
  view_subsets = []
  view_range = list(range(n_views))
  for nv in range(1, n_views):
    view_subsets.extend(list(itertools.combinations(view_range, nv)))

  tr_samples = {}
  te_samples = {}
  for vsub in view_subsets:
    # x_o_tr = {vi:tr_data[vi] for vi in vsub}
    # x_o_te = {vi:te_data[vi] for vi in vsub}
    globals().update(locals())
    tr_x_sub = {vi: xvi for vi, xvi in tr_x.items() if vi not in vsub}
    te_x_sub = {vi: xvi for vi, xvi in te_x.items() if vi not in vsub}
    tr_samples[vsub] = model.sample(
        tr_x_sub, b_o=None, sampled_views=vsub, batch_size=None, rtn_torch=False)
    te_samples[vsub] = model.sample(
        te_x_sub, b_o=None, sampled_views=vsub, batch_size=None, rtn_torch=False)

    # tr_digits[vsub] = get_sampled_cat_grid(trd2, true_tr_digits)
    # te_digits[vsub] = get_sampled_cat_grid(ted2, true_te_digits)

  tr_samples = complement_subset_keys(tr_samples, n_views)
  te_samples = complement_subset_keys(te_samples, n_views)

  hidden_sizes = [128, 64]
  n_estimators = 10
  classifier = train_classifier(tr_x, tr_y, hidden_sizes, n_estimators)
  tr_perm_accs, tr_perm_preds, tr_nv_accs = evaluate_mv_performance(
      classifier, tr_samples, tr_x, tr_y)
  te_perm_accs, te_perm_preds, te_nv_accs = evaluate_mv_performance(
      classifier, te_samples, te_x, te_y)
  print(tr_perm_accs)
  print(te_perm_accs)

  IPython.embed()


def default_polysom_classifier_config(args, view_sizes):
  tot_dim = sum([v for v in view_sizes.values()])

  nn_config = make_default_nn_config()
  nn_config.set_sizes(input_size=tot_dim)
  nn_config.last_activation = torch_models.Identity

  lr = 1e-3
  batch_size = 50
  max_iters = 5000
  grad_clip = 5.
  verbose = True

  config = physionet.PolysomConfig(
      nn_config=nn_config, lr=lr, batch_size=batch_size, max_iters=max_iters,
      grad_clip=grad_clip, verbose=verbose)
  return config


def test_mitbih_dsl(args):
  load_start_time = time.time()
  tr_frac = 0.8
  # tr_data, te_data = physionet.get_mv_mitbih_split(tr_frac)
  polysom_file = "./data/mitbih/polysom_fft_full.npy"
  tr_data, te_data = np.load(polysom_file, allow_pickle=True).tolist()
  (tr_x, tr_y, tr_ya, tr_ids) = tr_data
  (te_x, te_y, te_ya, te_ids) = te_data
  print("Time taken to load MITBIH: %.2fs" % (time.time() - load_start_time))
  sub_names = {0:"ECG", 1:"BP", 2:"EEG"}

  fft_size = 20
  remove_first_freq = False
  if remove_first_freq:
    f_inds = list(range(1, fft_size)) + list(range(fft_size + 1, 2 * fft_size))
    tr_x = {vi: xvi[:, f_inds] for vi, xvi in tr_x.items()}
    te_x = {vi: xvi[:, f_inds] for vi, xvi in te_x.items()}

  torch.set_default_dtype(torch.float64)
  view_sizes = {vi:xv.shape[1] for vi, xv in tr_x.items()}
  n_views = len(view_sizes)

  config, view_config_and_inits, cond_config_and_inits, view_ae_configs = \
      make_default_pipeline_config(args, view_sizes=view_sizes)
  view_tfm_config_lists, view_tfm_init_lists = view_config_and_inits
  cond_tfm_config_lists, cond_tfm_init_lists = cond_config_and_inits

  config.no_view_tfm = True
  config.dsl_coeff = 1000.
  # IPython.embed()
  dev = None
  if torch.cuda.is_available() and args.gpu_num >= 0:
    dev = torch.device("cuda:%i" % args.gpu_num)

  model = ac_flow_dsl_pipeline.MACFlowDSLTrainer(config)
  model.initialize(
      view_sizes, view_tfm_config_lists, view_tfm_init_lists,
      cond_tfm_config_lists, cond_tfm_init_lists, view_ae_configs)

  n_sampled_tr = args.npts
  n_sampled_te = args.npts // 2
  # (full_tr_x, full_tr_y, full_tr_ya) = (tr_x, tr_y, tr_ya)
  tr_x, tr_y, tr_idxs = stratified_sample(tr_x, tr_y, n_sampled=n_sampled_tr)
  # te_data, y_te, te_idxs = stratified_sample(te_data, y_te, n_sampled=n_sampled_te)
  n_tr = tr_x[0].shape[0]
  tr_x = convert_numpy_to_float64(tr_x)
  tr_y = convert_numpy_to_float64(tr_y)

  cat_tr = get_sampled_cat({}, tr_x)
  cat_te = get_sampled_cat({}, te_x)

  IPython.embed()
  # DSL
  freeze_loss = True
  loss_config = default_polysom_classifier_config(args, view_sizes)
  loss_func = physionet.PolysomClassifier(loss_config)
  loss_func.pre_train(cat_tr, tr_y)

  if freeze_loss:
    loss_func.freeze()

  IPython.embed()
  model.fit(tr_x, tr_y, None, loss_func)
  IPython.embed()

  te_x = convert_numpy_to_float64(te_x)
  te_y = convert_numpy_to_float64(te_y)
  view_subsets = []
  view_range = list(range(n_views))
  for nv in range(1, n_views):
    view_subsets.extend(list(itertools.combinations(view_range, nv)))

  tr_samples = {}
  te_samples = {}
  for vsub in view_subsets:
    # x_o_tr = {vi:tr_data[vi] for vi in vsub}
    # x_o_te = {vi:te_data[vi] for vi in vsub}
    globals().update(locals())
    tr_x_sub = {vi: xvi for vi, xvi in tr_x.items() if vi not in vsub}
    te_x_sub = {vi: xvi for vi, xvi in te_x.items() if vi not in vsub}
    tr_samples[vsub] = model.sample(
        tr_x_sub, b_o=None, sampled_views=vsub, batch_size=None, rtn_torch=False)
    te_samples[vsub] = model.sample(
        te_x_sub, b_o=None, sampled_views=vsub, batch_size=None, rtn_torch=False)

    # tr_digits[vsub] = get_sampled_cat_grid(trd2, true_tr_digits)
    # te_digits[vsub] = get_sampled_cat_grid(ted2, true_te_digits)

  tr_samples = complement_subset_keys(tr_samples, n_views)
  te_samples = complement_subset_keys(te_samples, n_views)

  cat_tr = get_sampled_cat({}, tr_x)
  cat_te = get_sampled_cat({}, te_x)

  nn_pgrid = {"hidden_layer_sizes": [[256, 128], [100, 50], [50, 50], [128], [50]]}
  nn_classifier = gscv(neural_network.MLPClassifier(max_iter=500), nn_pgrid, verbose=100)
  nn_classifier.fit(cat_tr, tr_y)
  rf_pgrid = {"max_features": [20, 25, 30], "max_depth": [5, 10]}
  rf_classifier = gscv(ensemble.RandomForestClassifier(n_estimators=100, min_samples_leaf=2), rf_pgrid, verbose=100)
  rf_classifier.fit(cat_tr, tr_y)
  # hidden_sizes = [128, 64]
  # n_estimators = 10

  # classifier = train_classifier(tr_x, tr_y, hidden_sizes, n_estimators)
  tr_perm_accs, tr_perm_preds, tr_nv_accs = evaluate_mv_performance(
      classifier, tr_samples, tr_x, tr_y)
  te_perm_accs, te_perm_preds, te_nv_accs = evaluate_mv_performance(
      classifier, te_samples, te_x, te_y)
  print(tr_perm_accs)
  print(te_perm_accs)

  IPython.embed()


def interactive(): pass


_TEST_FUNCS = {
    0: test_pipeline_dsl,
    1: test_mnist_dsl,
    2: test_mitbih,
    3: test_mitbih_dsl,
    -1: interactive,
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
      ("dsl_coeff", float, "Coefficient for down-stream loss.", 1.),
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
  
  func = _TEST_FUNCS.get(args.expt, interactive)
  func(args)
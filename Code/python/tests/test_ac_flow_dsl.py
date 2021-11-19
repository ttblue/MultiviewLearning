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
    multiview_datasets as mvd
from models import ac_flow_pipeline, ac_flow_dsl_pipeline,\
    autoencoder, conditional_flow_transforms,\
    zzflow_likelihood, flow_pipeline, flow_transforms, torch_models
from synthetic import flow_toy_data, multimodal_systems
from utils import math_utils, torch_utils, utils

from tests.test_flow import make_default_tfm
from tests.test_ac_flow import SimpleArgs, convert_numpy_to_float32,\
    make_default_overlapping_data, make_default_overlapping_data2,\
    make_default_independent_data, make_default_shape_data,\
    rotate_and_shift_data
    # make_default_tfm_config, ArgsCopy, make_default_likelihood_config,\
    # make_default_data, make_default_data_X, 
    # make_default_overlapping_data2, rotate_and_shift_data,\
    # make_default_independent_data, make_default_shape_data
from tests.test_ac_flow_jp import make_default_nn_config, make_default_tfms,\
    make_default_cond_tfm_config, ArgsCopy, make_default_likelihood_config,\

from matplotlib import pyplot as plt, patches
from mpl_toolkits.mplot3d import Axes3D

import IPython


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
  lr = 1e-3
  max_iters = args.max_iters

  verbose = True

  # IPython.embed()
  config = ac_flow_pipeline_dsl.MACFDTConfig(
      expand_b=expand_b, no_view_tfm=no_view_tfm,
      likelihood_config=likelihood_config, base_dist=base_dist,
      dsl_coeff=dsl_coeff, batch_size=batch_size, lr=lr, max_iters=max_iters,
      verbose=verbose)

  return config, (view_tfm_config_lists, view_tfm_init_lists),\
      (cond_config_lists, cond_inits_lists), view_ae_configs


def test_pipeline_dsl(args):
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


def test_mnist_dsl(args):
  pass


_TEST_FUNCS = {
    0: test_pipeline_dsl,
    1: test_mnist_dsl,
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
  
  func = _TEST_FUNCS.get(args.expt, test_mnist)
  func(args)
 

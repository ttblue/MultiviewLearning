#!/usr/bin/env python

import numpy as np
import os
import scipy.linalg as slg
import torch
from torch import nn

from models import flow_transforms #, flow_likelihood, flow_pipeline
from models import torch_models
from synthetic import flow_toy_data
from utils import utils, torch_utils

import matplotlib.pyplot as plt

import IPython


def default_nn_config():
  input_size = 10  # Computed online
  output_size = 10  # Computed online
  layer_units = [32, 64]
  use_vae = False
  activation = nn.ReLU  # nn.functional.relu
  last_activation = nn.ReLU  #torch_models.Identity  # functional.sigmoid
  # layer_types = None
  # layer_args = None
  bias = False
  dropout_p = 0.
  layer_types, layer_args = torch_utils.generate_linear_types_args(
        input_size, layer_units, output_size, bias)
  nn_config = torch_models.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)
  return nn_config


def default_tfm_config(tfm_type="shift_scale_coupling"):
  neg_slope = 0.1
  scale_config = default_nn_config()
  shift_config = default_nn_config()
  shared_wts = False

  ltfm_config = default_nn_config()
  bias_config = default_nn_config()

  has_bias = True

  lr = 1e-3
  batch_size = 50
  max_iters = 1000
  reg_coeff = 0.

  verbose = True

  config = flow_transforms.TfmConfig(
      tfm_type=tfm_type, neg_slope=neg_slope, scale_config=scale_config,
      shift_config=shift_config, shared_wts=shared_wts, ltfm_config=ltfm_config,
      bias_config=bias_config, has_bias=has_bias, lr=lr, batch_size=batch_size,
      max_iters=max_iters, reg_coeff=reg_coeff, verbose=verbose)
  return config


def default_likelihood_config():
  config = flow_likelihood.ARMMConfig()
  return config


def default_pipeline_config():
  pass


def simple_test_tfms(args):
  tfm_types = ["scale_shift", "linear"]
  X, Y, tfm_args = flow_toy_data.simple_transform_data(
      args.npts, args.ndim, tfm_types)

  split_frac = [0.8, 0.2]
  (tr_X, te_X), inds = utils.split_data(X, split_frac, get_inds=True)
  tr_Y, te_Y = [Y[idx] for idx in inds]

  # Generate config list:
  tfm_configs = []
  tfm_inits = []

  #################################################
  # Then a fixed linear transform
  tfm_idx = 1
  num_lin_tfm = 1
  dim = X.shape[1]
  for i in range(num_lin_tfm):
    linear_tfm_config = default_tfm_config("fixed_linear")
    tfm_configs.append(linear_tfm_config)
    tfm_inits.append((dim,))

  # First transform is bit-mask.
  tfm_idx = 0 
  num_ss_tfm = 1
  idx_args = tfm_args[tfm_idx]
  if idx_args[0] == "scaleshift":
    bit_mask = idx_args[1]
  else:
    bit_mask = np.zeros(args.ndim)
    bit_mask[np.random.permutation(args.ndim)[:args.ndim//2]] = 1
  for i in range(num_ss_tfm):
    scale_shift_tfm_config = default_tfm_config("scale_shift_coupling")
    tfm_configs.append(scale_shift_tfm_config)
    tfm_inits.append((bit_mask,))
    bit_mask = 1 - bit_mask

  # # Leaky ReLU
  tfm_idx = 2
  use_leaky_relu = False
  if use_leaky_relu:
    leaky_relu_config = default_tfm_config("leaky_relu")
    leaky_relu_config.neg_slope = 0.1
    tfm_configs.append(leaky_relu_config)
    tfm_inits.append(None)

  # Reverse
  tfm_idx = 3
  use_reverse = False
  if use_reverse:
    reverse_config = default_tfm_config("reverse")
    tfm_configs.append(reverse_config)
    tfm_inits.append(None)
  #################################################

  comp_tfm = flow_transforms.make_transform(tfm_configs, tfm_inits)
  config = comp_tfm.config
  config.batch_size = 1000
  config.lr = 1e-3
  config.reg_coeff = 0.1
  config.max_iters = 1#50000
  IPython.embed()
  comp_tfm.fit(tr_X, tr_Y)

  # config = linear_tfm_config
  # config.batch_size = 100
  # config.max_iters = 10000
  # config.has_bias = False
  # linear_tfm = flow_transforms.make_transform(config)
  # L, U = tfm_args[0][1:]
  # # P, L, U = slg.lu(W)
  # init_mat = np.tril(L, -1) + np.triu(U, 0)
  # eps = 1e-1
  # noise = np.random.randn(*init_mat.shape) * eps
  # linear_tfm.initialize(dim)#, init_mat + noise)
  # comp_tfm.fit(tr_X, tr_Y)

  IPython.embed()


def simple_test_likelihood(args):
  pass


def test_pipeline(args):
  pass


_TEST_FUNCS = {
    0: simple_test_tfms,
    # 1: simple_test_likelihood,
    # 2: test_pipeline

}
if __name__ == "__main__":
  np.set_printoptions(linewidth=1000, precision=3, suppress=True)
  torch.set_printoptions(precision=3)
  options = [
      ("npts", int, "Number of points", 1000),
      ("ndim", int, "Dimensions", 10)
      ]
  args = utils.get_args(options)
  
  func = _TEST_FUNCS.get(args.expt, simple_test_tfms)
  func(args)

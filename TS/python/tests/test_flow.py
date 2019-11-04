import numpy as np
import os
import torch
from torch import nn

from models import flow_transforms#, flow_likelihood, flow_pipeline
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
  last_activation = nn.ReLU  #torch_utils.Identity  # functional.sigmoid
  # layer_types = None
  # layer_args = None
  bias = False
  dropout_p = 0.
  layer_types, layer_args = torch_utils.generate_linear_types_args(
        input_size, layer_units, output_size, bias)
  nn_config = torch_utils.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)
  return nn_config


def default_tfm_config(tfm_type="shift_scale_coupling"):
  neg_slope = 0.1
  scale_config = default_nn_config()
  shift_config = default_nn_config()
  shared_wts = False

  linear_config = default_nn_config()
  bias_config = default_nn_config()

  has_bias = True

  verbose = True

  config = flow_transforms.TfmConfig(
      tfm_type=tfm_type, neg_slope=neg_slope, scale_config=scale_config,
      shift_config=shift_config, shared_wts=shared_wts, ltfm_config=ltfm_config,
      bias_config=bias_config, has_bias=has_bias, verbose=verbose)
  return config


def default_likelihood_config():
  config = flow_likelihood.ARMMConfig()
  return config


def default_pipeline_config():
  pass


def simple_test_tfms(args):
  X, Y, tfm_args = flow_toy_data.simple_transform_data(args.npts, args.ndim)

  split_frac = [0.8, 0.2]
  (tr_X, te_X), inds = utils.split_data(X, split_frac, get_inds=True)
  tr_Y, te_Y = [Y[idx] for idx in inds]


  linear_tfm_config = default_tfm_config("fixed_linear")
  linear_tfm = flow_transforms.make_transform(linear_tfm_config)
  linear_tfm.initialize(X.shape[1])

  linear_tfm.fit(tr_X, tr_Y)

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
  options = [
      ("npts", int, "Number of points", 1000),
      ("ndim", int, "Dimensions", 10)
      ]
  args = utils.get_args(options)
  
  func = _TEST_FUNCS.get(args.expt, simple_test_tfms)
  func(args)

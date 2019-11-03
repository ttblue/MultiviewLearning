import numpy as np
import os
import torch
from torch import nn

from models import flow_transforms, flow_likelihood, flow_pipeline
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


def default_tfm_config(tfm_type="shift_scale_coupling"):
  neg_slope = 0.1

  config = flow_transforms.TfmConfig()
  return config


def default_likelihood_config():
  config = flow_likelihood.TfmConfig()
  return config


def default_pipeline_config():
  pass


def simple_test_tfms(args):
  X, Y, tfm_args = flow_toy_data.simple_transform_data(args.npts, args.ndim)



def simple_test_likelihood(args):
  pass


def test_pipeline(args):
  pass


_TEST_FUNCS = {
    0: simple_test_tfms,
    1: simple_test_likelihood,
    2: test_pipeline

}
if __name__ == "__main__":
  options = [
      ("npts", int, "Number of points", 1000),
      ("ndim", int, "Dimensions", 10)
      ]
  args = utils.default_parser(options)
  
  func = _TEST_FUNCS.get(args.expt, simple_test_tfms)
  func(args)

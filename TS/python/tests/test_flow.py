import numpy as np
import os
import torch
from torch import nn

from models import flow_transforms, flow_likelihood, flow_pipeline
from utils import utils

import matplotlib.pyplot as plt

import IPython


def default_tfm_config():
  config = flow_transforms.TfmConfig()
  return config


def default_likelihood_config():
  config = flow_likelihood.TfmConfig()
  return config


def default_pipeline_config():
  pass


def simple_test_tfms():
  pass


def simple_test_likelihood():
  pass


if __name__ == "__main__":
  args = utils.default_parser()

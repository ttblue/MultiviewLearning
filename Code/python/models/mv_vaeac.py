import numpy as np
import scipy
import torch
from torch import nn, optim
import time

from models.model_base import ModelException, BaseConfig
from models import flow_likelihood, flow_transforms
from utils import math_utils, torch_utils, utils


import IPython


class MVVAEACConfig(BaseConfig):
  def __init__(
      self, shared_tfm_config_list=[], view_tfm_config_lists={},
      likelihood_config=None, base_dist="gaussian",
      batch_size=50, lr=1e-3, max_iters=1000,
      *args, **kwargs):
    super(MVVAEACConfig, self).__init__(*args, **kwargs)

    self.shared_tfm_config_list = shared_tfm_config_list
    self.view_tfm_config_lists = view_tfm_config_lists
    self.likelihood_config = likelihood_config
    self.base_dist = base_dist

    self.batch_size = batch_size
    self.lr = lr
    self.max_iters = max_iters


class MVVAEAC(nn.module):
  def __init__(self, config):
    super(MVVAEAC, self).__init__()
    self.config = config

  def initialize(self):
    pass

  def fit(self):
    pass

  def sample(self):
    pass
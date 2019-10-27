import numpy as np
import scipy
import torch
from torch import nn
import time

from models.model_base import ModelException, BaseConfig
from models import flow_likelihood, flow_transforms
from utils import math_utils, torch_utils


import IPython


class FTConfig(BaseConfig):
  def __init__(
    self, tfm_config_list=[],  likelihood_config=None, *args, **kwargs):
    super(FTConfig, self).__init__(*args, **kwargs)

    self.tfm_config_list = tfm_config_list
    self.likelihood_config = likelihood_config


class FlowTrainer(nn.Module):
  # Class for basic flow training without arbitrary conditioning.
  def __init__(self, config):
    self.config = config
    super(FlowTrainer, self).__init__()

  def initialize(self, dim, tfm_init_args):
    # Initialize transforms, mm model, optimizer, etc.
    tfm_list = []
    for config, init_args in zip(self.config.tfm_config_list, tfm_init_args):
      tfm = flow_transforms.make_transform(config)
      tfm.initialize(init_args)
      tfm_list.append(tfm)

    comp_config = flow_transforms.TfmConfig("composition")
    self._comp_tfm = flow_transforms.CompositionTransform(comp_config, tfm_list)

    


  def _transform(self, x):
    # Transform input covariates using invertible transforms.
    raise NotImplementedError("Implement this!")
    return z, log_jac_det

  def _nll(self, z):
    # Give the log likelihood under transformed z
    raise NotImplementedError("Implement this!")
    return z_nll 

  def loss(self, z_nll, log_jac_det):
    nll_loss = -torch.sum(log_jac_det) + torch.sum(z_nll)
    return nll_loss

  def _train_loop(self):
    pass

  def fit(self, x):
    

  def sample(self, n):
    raise NotImplementedError("Implement this!")

  def log_likelihood(self, x):
    raise NotImplementedError("Implement this!")




class ACFlowTrainer(FlowTrainer):
  # Class for AC (arbitrary conditioning) flow training.
  def __init__(self, config):
    super(ACFlowTrainer, self).__init__(config)

  def initialize(self):
    pass

  def _train_loop(self):
    pass

  def fit(self, x):
    raise NotImplementedError("Implement this!")

  def sample(self, n):
    raise NotImplementedError("Implement this!")

  def log_likelihood(self, x):
    raise NotImplementedError("Implement this!")


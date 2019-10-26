import numpy as np
import scipy
import torch
from torch import nn
import time

from models.model_base import ModelException, BaseConfig
from utils import math_utils, torch_utils


import IPython


class LikelihoodConfig(BaseConfig):
  # General config for likelihood models
  def __init__(self, dist_type="gaussian", n_components=5, *arg, **kwargs):
    super(LikelihoodConfig, self).__init__(*args, **kwargs)

    # Mixture model params:
    self.dist_type = dist_type.lower()
    self.n_components = n_components


class AbstractLikelihood(object):
  # Abstract class for likelihood representation.
  def __init__(self, config):
    self.config = config

  def initialize(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def sample(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def log_likelihood(self, x, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def nll(self, x, *args, **kwargs):
    lls = self.log_likelihood(x, *args, **kwargs)
    return -torch.mean(lls)


################################################################################
# Base distributions





################################################################################
# Auto-regressive models
_DIST_TYPES = ["gaussian", "laplace", "logistic"]
class ARMixtureModel(AbstractLikelihood):
  # Mixture model -- can be gaussian, laplace or logistic
  def __init__(self, config):
    super(ARMixtureModel, self).__init__(config)
    if config.dist_type not in _DIST_TYPES:
      raise ModelException(
          "Base dist. type %s not implemented." % config.dist_type)

  def compute_mm_params(self, x):
    raise NotImplementedError("Abstract class method")

  def _normalize_wts(self, wts):
    return torch.softmax(wts, -1)

  def _compute_dist_logterms(self, x, mus, lsigmas):
    sigmas = torch.exp(sigmas)
    diff = (x - means) / sigmas

    if self.config.dist_type == "gaussian":
      log_norm_consts = -lsigmas - 0.5 * np.log(2.0 * np.pi)
      log_kernel = -0.5 * (diff ** 2)
    elif self.config.dist_type == "laplace":
      log_norm_consts = -lsigmas - np.log(2.0)
      log_kernel = -torch.abs(diff)
    elif self.config.dist_type == "logistic":
      log_norm_consts = -lsigmas
      log_kernel = -tf.nn.softplus(diff) - tf.nn.softplus(-diff)

    return log_norm_consts, log_kernel

  def log_likelihood(self, x):
    wts, mus, lsigmas = self.compute_mm_params(x)
    wts = self._normalize_wts(wts)

    log_norm_consts, log_kernel = self._compute_dist_logterms(x, mus, lsigmas)
    log_exp_terms = log_kernel + log_norm_consts + wts
    lls = torch.logsumexp(log_exp_terms, -1)

    return log_likelihoods



class LinearARM(AbstractLikelihood):
  def __init__(self, config):
    super(LinearARM, self).__init__(config)

  def initialize(self, dim):
    

class RecurrentARM(AbstractLikelihood):
  def __init__(self, config):
    super(RecurrentARM, self).__init__(config)

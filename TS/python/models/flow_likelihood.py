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
  def __init__(
    self, model_type="linear_arm", dist_type="gaussian", n_components=5,
    *arg, **kwargs):
    super(LikelihoodConfig, self).__init__(*args, **kwargs)

    self.model_type = model_type.lower()
    # Mixture model params:
    self.dist_type = dist_type.lower()
    self.n_components = n_components


class AbstractLikelihood(nn.Module):
  # Abstract class for likelihood representation.
  def __init__(self, config):
    self.config = config
    super(AbstractLikelihood, self).__init__()

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
# Auto-regressive models
class ARMMConfig(LikelihoodConfig):
  # General config for likelihood models
  def __init__(
      self, ar_type="linear", dist_type="gaussian", n_components=5,
      hidden_size=32, theta_nn_config=None, cell_type="LSTM", *arg, **kwargs):
    super(LikelihoodConfig, self).__init__(
        dist_type, n_components, *args, **kwargs)

    # AR model mm-parameter function params
    self.ar_type = ar_type
    self.hidden_size = hidden_size
    self.theta_nn_config = theta_nn_config

    # RNN AR params
    self.cell_type = cell_type


_DIST_TYPES = ["gaussian", "laplace", "logistic"]
class ARMixtureModel(AbstractLikelihood):
  # Using the general framework as given in TAN
  # Mixture model -- can be gaussian, laplace or logistic
  def __init__(self, config):
    if config.dist_type not in _DIST_TYPES:
      raise ModelException(
          "Base dist. type %s not implemented." % config.dist_type)
    super(ARMixtureModel, self).__init__(config)

  def initialize(self, dim, *args, **kwargs):
    self._dim = dim

    # The MM parameters are wts, mus, lsigmas (3 sets)
    self.config.theta_nn_config.set_sizes(
        input_size=self.config.hidden_size, output_size=(self.config.n_components * 3))
    # Fully connected NN to map hidden state to MM params
    self._theta_nn = torch_utils.MultiLayerNN(self.config.theta_nn_config)

  def g_func(self, x_i, h_prev, *args, **kwargs):
    raise NotImplementedError("Abstract class methods")

  def theta_func(self, h, *args, **kwargs):
    mm_params = self._theta_nn(h)
    wts, mus, lsigmas = torch.split(mm_params, 3, 1)  # TODO: check syntax
    return wts, mus, lsigmas

  def compute_mm_params(self, x):
    # x: N x d dataset of features
    # Output:
    # wts: N x d x ncomp of (unnormalized) weights of components
    # mus: N x d x ncomp of component mus
    # lsigmas: N x d x ncomp of component lsigmas
    N = x.shape[0]
    wts = []
    mus = []
    lsigmas = []
    h_i = None
    for i in range(self._dim):
      x_i = x[:, :i]
      # Computing next state from covariates till ith dim and previous state
      h_i = self.g_func(x_i, h_i)
      # Computing MM parameters as a function of hidden state
      wts_i, mus_i, lsigmas_i = self.theta_func(h_i)
      wts.append(wts_i)
      mus.append(mus_i)
      lsigmas.append(lsigmas_i)

    wts = torch.concatenate(wts, 1)
    mus = torch.concatenate(mus, 1)
    lsigmas = torch.concatenate(lsigmas, 1)

    return wts, mus, lsigmas

  def _normalize_wts(self, wts):
    return torch.softmax(wts, -1)

  def _compute_dist_logterms(self, x, mus, lsigmas):
    sigmas = torch.exp(lsigmas)
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

  def initialize(self, dim, *args, **kwargs):
    super(LinearARM, self).initialize(dim, *args, **kwargs)
    # The W_0 has no corresponding dimension from the data so is just 0
    # for convenience
    self._Ws = [0.]
    for i in range(1, dim):
      init_W_i = torch.zeros(self.config.hidden_size, i)
      W_i = torch.nn.Parameter(init_W_i)
      self._Ws.append(W_i)
      self.add_module("W_%i" % i, W_i)  ## TODO: self.add_parameter?

    init_b = torch.zeros(self.config.hidden_size)
    self._b = torch.nn.Parameter(init_b)

  def g_func(self, x_i, h_prev, *args, **kwargs):
    i = x_i.shape[1]
    Wx_i = torch.matmul(self._Ws[i], x_i.transpose(0, 1))
    h_i = Wx_i.transpose(0, 1) + self._b
    return h_i


class RecurrentARM(AbstractLikelihood):
  def __init__(self, config):
    super(RecurrentARM, self).__init__(config)
    raise NotImplementedError("Not implemented yet.")

  def initialize(self, dim, *args, **kwargs):
    super(LinearARM, self).initialize(dim, *args, **kwargs)

    # TODO: maybe just define RNN cell instead of using torch_utils?
    self._rnn_cell = None


  def g_func(self, x_i, h_prev, *args, **kwargs):
    if h_prev is None:

    i = x_i.shape[1]
    Wx_i = torch.matmul(self._Ws[i], x_i.transpose(0, 1))
    h_i = Wx_i.transpose(0, 1) + self._b
    return h_i


_LIKELIHOOD_TYPES = {
    "linear_arm": LinearARM,
    "recurrent_arm": RecurrentARM,
}
def make_likelihood_model(config):
  if config.model_type not in _LIKELIHOOD_TYPES:
    raise TypeError(
        "%s not a valid likelihood model. Available models: %s" %
        (config.model_type, _LIKELIHOOD_TYPES))

  return _LIKELIHOOD_TYPES[config.model_type](config)
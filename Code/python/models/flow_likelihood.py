import numpy as np
import scipy
import torch
from torch import nn
import time

from models.model_base import ModelException, BaseConfig
from models import torch_models
from utils import math_utils, torch_utils


import IPython


class LikelihoodConfig(BaseConfig):
  # General config for likelihood models
  def __init__(
    self, model_type="linear_arm", dist_type="gaussian", *args, **kwargs):
    super(LikelihoodConfig, self).__init__(*args, **kwargs)

    self.model_type = model_type.lower()
    # Mixture model params:
    self.dist_type = dist_type.lower()


class AbstractLikelihood(nn.Module):
  # Abstract class for likelihood representation.
  def __init__(self, config):
    self.config = config
    super(AbstractLikelihood, self).__init__()

  def initialize(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def mean(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def sample(self, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def log_prob(self, x, *args, **kwargs):
    raise NotImplementedError("Abstract class method")

  def nll(self, x, aggregate=None, *args, **kwargs):
    x_nll = -self.log_prob(x, *args, **kwargs)
    if aggregate == "sum":
      return x_nll.sum()
    elif aggregate == "mean":
      return x_nll.mean()
    return x_nll


BASE_DISTS = ["mv_gaussian"] #, "laplace", "logistic"]
_TORCH_DISTRIBUTIONS_MAP = {
    "mv_gaussian": torch.distributions.MultivariateNormal,
    "gaussian": torch.distributions.Normal,
    "laplace": torch.distributions.Laplace,
    "logistic": torch.distributions.LogisticNormal,
}
_TORCH_DISTRIBUTIONS = list(_TORCH_DISTRIBUTIONS_MAP.values())

class BaseDistribution(AbstractLikelihood):
  # Simple likelihoods
  def __init__(self, config):
    super(BaseDistribution, self).__init__(config)

  def initialize(self, dim, loc=None, scale=None):
    self._dim = dim
    if self.config.dist_type not in BASE_DISTS:
      raise ModelException(
          "Base dist. type %s not available." % config.dist_type)

    loc = torch.zeros(dim) if loc is None else loc
    if scale is None:
      scale = (
          torch.eye(dim) if self.config.dist_type == "mv_gaussian" else
          torch.ones(dim))
    self._base_dist = _TORCH_DISTRIBUTIONS_MAP[self.config.dist_type](loc, scale)

  def sample(self, shape, rtn_torch=True):
    # IPython.embed()
    if isinstance(shape, int):
      shape = (shape,)
    # elif len(shape) == 1 and self._dim > 1:
    #   shape = (shape[0], self._dim)
    samples = self._base_dist.sample(shape)
    return samples if rtn_torch else torch_utils.torch_to_numpy(samples)

  def log_prob(self, x, rtn_torch=True):
    log_prob = self._base_dist.log_prob(x)
    return log_prob if rtn_torch else torch_utils.torch_to_numpy(log_prob)


def make_base_distribution(dist_type, dim, loc=None, scale=None):
  config = LikelihoodConfig(model_type="base_dist", dist_type=dist_type)
  base_dist = BaseDistribution(config)
  base_dist.initialize(dim, loc, scale)
  return base_dist


################################################################################
# Auto-regressive models
class ARMMConfig(LikelihoodConfig):
  # General config for likelihood models
  def __init__(
      self, model_type="linear_arm", dist_type="gaussian", n_components=5,
      hidden_size=32, theta_nn_config=None, cell_type="LSTM", *args, **kwargs):
    super(ARMMConfig, self).__init__(
        model_type=model_type, dist_type=dist_type, arg=args, kwargs=kwargs)

    self.n_components = n_components

    # AR model mm-parameter function params
    self.hidden_size = hidden_size
    self.theta_nn_config = theta_nn_config

    # RNN AR params
    self.cell_type = cell_type


class ARMixtureModel(AbstractLikelihood):
  # Using the general framework as given in TAN
  # Mixture model -- can be gaussian, laplace or logistic
  def __init__(self, config):
    super(ARMixtureModel, self).__init__(config)

  def initialize(self, dim, *args, **kwargs):
    if self.config.dist_type not in _TORCH_DISTRIBUTIONS_MAP:
      raise ModelException(
          "Base dist. type %s not available." % config.dist_type)

    self._dim = dim
    self._torch_dist = _TORCH_DISTRIBUTIONS_MAP[self.config.dist_type]
    # The MM parameters are wts, mus, lsigmas (3 sets)
    self.config.theta_nn_config.set_sizes(
        input_size=self.config.hidden_size,
        output_size=(self.config.n_components * 3))
    # Fully connected NN to map hidden state to MM params
    self._theta_nn = torch_models.MultiLayerNN(self.config.theta_nn_config)

  def g_func(self, x_i, h_prev, *args, **kwargs):
    raise NotImplementedError("Abstract class methods")

  def theta_func(self, h, *args, **kwargs):
    mm_params = self._theta_nn(h)
    wts, mus, lsigmas = torch.split(mm_params, self.config.n_components, 1)
    return wts, mus, lsigmas

  def compute_mm_params(self, x):
    # x: N x d dataset of features
    # Output:
    # wts: N x d x ncomp of (unnormalized) weights of components
    # mus: N x d x ncomp of component mus
    # lsigmas: N x d x ncomp of component lsigmas
    x = torch_utils.numpy_to_torch(x)
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

    wts = torch.stack(wts, 2)
    mus = torch.stack(mus, 2)
    lsigmas = torch.stack(lsigmas, 2)
    return wts, mus, lsigmas

  def _normalize_wts(self, wts):
    return torch.softmax(wts, -1)

  def _compute_dist_logterms(self, x, mus, lsigmas):
    x = x.unsqueeze(1).repeat(1, self.config.n_components, 1)
    sigmas = torch.exp(lsigmas)
    diff = (x - mus) / sigmas

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

  def log_prob(self, x, rtn_torch=True):
    wts, mus, lsigmas = self.compute_mm_params(x)
    wts = self._normalize_wts(wts)
    log_norm_consts, log_kernel = self._compute_dist_logterms(x, mus, lsigmas)
    log_exp_terms = log_kernel + log_norm_consts + wts
    log_probs = torch.logsumexp(log_exp_terms, 1).sum(1)

    return log_probs if rtn_torch else torch_utils.torch_to_numpy(log_probs)

  def _sample_MM(self, wts, mus, sigmas):
    samples = []
    for wts_c, mus_c, sigmas_c in zip(wts, mus, sigmas):
      # First sample mm component
      comp_idx = torch.distributions.Categorical(wts_c).sample()
      # Then sample from component
      mu, sigma = mus_c[comp_idx], sigmas_c[comp_idx]
      samples.append(self._torch_dist(mu, sigma).sample())
    return torch.tensor(samples).view(-1, 1)

  def sample(self, n, rtn_torch=True):
    dim = self._dim
    if isinstance(n, tuple):
      if len(n) > 1:
        dim = min(self._dim, n[1])
      n = n[0]

    h_i = None
    z_samples = torch.empty((n, 0))
    for i in range(dim):
      # Computing next state from covariates till ith dim and previous state
      h_i = self.g_func(z_samples, h_i)

      # Computing MM parameters as a function of hidden state
      wts_i, mus_i, lsigmas_i = self.theta_func(h_i)
      wts_i = self._normalize_wts(wts_i)
      sigmas_i = torch.exp(lsigmas_i)

      samples_c = self._sample_MM(wts_i, mus_i, sigmas_i)
      z_samples = torch.cat([z_samples, samples_c], dim=1)

    return z_samples if rtn_torch else torch_utils.torch_to_numpy(z_samples)


class LinearARM(ARMixtureModel):
  def __init__(self, config):
    super(LinearARM, self).__init__(config)

  def initialize(self, dim, *args, **kwargs):
    super(LinearARM, self).initialize(dim, *args, **kwargs)
    # The W_0 has no corresponding dimension from the data so is just 0
    # for convenience
    self._Ws = nn.ParameterDict()
    for i in range(1, dim):
      init_W_i = torch.zeros(self.config.hidden_size, i)
      W_i = torch.nn.Parameter(init_W_i)
      self._Ws["W_%i"%i] = W_i

    init_b = torch.zeros(self.config.hidden_size)
    self._b = torch.nn.Parameter(init_b)

  def g_func(self, x_i, *args, **kwargs):
    i = x_i.shape[1]
    if i == 0:
      h_i = self._b.view(1, -1).repeat(x_i.shape[0], 1)
      return h_i

    Wx_i = torch.matmul(self._Ws["W_%i"%i], x_i.transpose(0, 1))
    h_i = Wx_i.transpose(0, 1) + self._b
    return h_i


class RecurrentARM(ARMixtureModel):
  def __init__(self, config):
    super(RecurrentARM, self).__init__(config)
    raise NotImplementedError("Not implemented yet.")

  def initialize(self, dim, *args, **kwargs):
    super(LinearARM, self).initialize(dim, *args, **kwargs)

    # TODO: maybe just define RNN cell instead of using torch_models?
    self._rnn_cell = None

  def g_func(self, x_i, h_prev, *args, **kwargs):
    if h_prev is None:
      pass

    i = x_i.shape[1]
    Wx_i = torch.matmul(self._Ws[i], x_i.transpose(0, 1))
    h_i = Wx_i.transpose(0, 1) + self._b
    return h_i


_LIKELIHOOD_TYPES = {
    "linear_arm": LinearARM,
    "recurrent_arm": RecurrentARM,
}
def make_likelihood_model(config, dim=None):
  if config.model_type not in _LIKELIHOOD_TYPES:
    raise TypeError(
        "%s not a valid likelihood model. Available models: %s" %
        (config.model_type, _LIKELIHOOD_TYPES))

  lhood_model = _LIKELIHOOD_TYPES[config.model_type](config)
  if dim is not None:
    lhood_model.initialize(dim)
  return lhood_model


## Mean functions
def get_mean(dist, n_samples, *args, **kwargs):
  if type(dist) in _TORCH_DISTRIBUTIONS:
    mu = dist.mean.view(1, -1)
    mu = torch.tile(mu, (n_samples, 1))
    return mu
  else:
    raise ValueError("Mean function for %s not implemented." % dist)
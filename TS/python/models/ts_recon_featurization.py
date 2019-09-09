# (Variational) Autoencoder for multi-view + synchronous
import itertools
import numpy as np
import torch
from torch import nn
from torch import optim
import time

from models.model_base import ModelException, BaseConfig
from utils import utils
import utils.torch_utils as tu
from utils.torch_utils import _DTYPE, _TENSOR_FUNC

import IPython


class TSRFConfig(BaseConfig):
  def __init__(
      self, encoder_params, decoder_params, time_delay_tau=10,
      time_delay_ndim=3, fixed_len=True,
      *args, **kwargs):

    self.encoder_params = encoder_params
    self.decoder_params = decoder_params

    self.time_delay_tau = time_delay_tau
    self.time_delay_ndim = time_delay_ndim 
    self.fixed_len = fixed_len

    super(TSRFConfig, self).__init__(*args, **kwargs)


class TimeSeriesReconFeaturization(nn.Model):
  def __init__(self, config):
    super(TimeSeriesReconFeaturization, self).__init__()
    self.config = config

  def _initialize(self):
    # Create RN encoder
    _td_dim = self._dim * max(self.config.time_delay_ndim, 1)
    self.e

    # Create RN decoder

  def _td_embedding(self, ts):
    # Assume ts is big enough.
    if self.config.time_delay_ndim <= 1:
      return ts

    nts = ts.shape[0]
    tau = self.config.time_delay_tau
    ndim = self.config.time_delay_ndim
    td_embedding = np.concatenate(
        [ts[i*tau: nts-(ndim-i-1)*tau] for i in range(ndim)], axis=1)

    return td_embedding

  def _rnn_output(self, config): 

  def encode(self, ts):
    tde = self._td_embedding(ts)


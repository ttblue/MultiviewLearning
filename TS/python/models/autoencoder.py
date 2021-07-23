# Simple autoencoder: to-do
import itertools
import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
import time

from models.model_base import BaseConfig
from models import torch_models
from utils import torch_utils, utils
from utils.torch_utils import _DTYPE, _TENSOR_FUNC

import IPython


class AEConfig(BaseConfig):
  def __init__(
      self, v_sizes, code_size, encoder_params, decoder_params, lm,
      code_sample_noise_var, max_iters, batch_size, lr, verbose,
      *args, **kwargs):

    super(MAEConfig, self).__init__(*args, **kwargs)

    self.code_size = code_size
    self.v_sizes = v_sizes
    self.encoder_params = encoder_params
    self.decoder_params = decoder_params
    self.lm = lm
    self.code_sample_noise_var = code_sample_noise_var

    # self.view_dropout = view_dropout

    self.max_iters = max_iters
    self.batch_size = batch_size
    self.lr = lr

    self.verbose = verbose
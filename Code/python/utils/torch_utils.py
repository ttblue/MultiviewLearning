# Basic utilities for pytorch
import numpy as np
import torch
from torch import nn

from models.model_base import BaseConfig
from utils import utils

import IPython


_DTYPE = torch.float64
_NP_DTYPE = np.float64
_TENSOR_FUNC = torch.FloatTensor
torch.set_default_dtype(_DTYPE)
# For RNNs.
_BATCH_FIRST = False


################################################################################
## Type converters
def numpy_to_torch(var, copy=False, dev=None):
  if not isinstance(var, torch.Tensor):
    if copy: var = var.copy()
    var = torch.from_numpy(var).type(_DTYPE).requires_grad_(False)
    if dev:
      var = var.cuda(dev)
  return var


def torch_to_numpy(var, copy=False):
  if isinstance(var, torch.Tensor):
    var = var.detach().numpy().astype(_NP_DTYPE)
    if copy: var = var.copy()    
  return var


def dict_numpy_to_torch(data, copy=False, dev=None):
  return {i: numpy_to_torch(x, copy, dev) for i, x in data.items()}


def dict_torch_to_numpy(data, copy=False):
  return {i: torch_to_numpy(x, copy) for i, x in data.items()}

################################################################################
# Misc. utils

# NN module wrappers:
def generate_linear_types_args(
      input_size, layer_units, output_size, bias=True):
  all_sizes = [input_size] + layer_units + [output_size]
  ltypes = [torch.nn.Linear] * (len(all_sizes) - 1)
  largs = [(l1, l2, bias) for l1, l2 in zip(all_sizes[:-1], all_sizes[1:])]
  return ltypes, largs


def LU_split(W, dev=None):
  # Note: this is NOT an LU decomp. This just splits a tensor variable
  # into lower triangular L with unit diagonal and upper triangular U.
  # The corresponding triangular elements of L and U are the same as W
  # and in the same position as well.
  # This is for storing the matrix itself as the LU decomposition.
  dim = W.shape[-1]
  module = torch if isinstance(W, torch.Tensor) else np
  L = module.eye(dim, device=dev) + module.tril(W, -1)
  U = module.triu(W, 0)

  return L, U


def LU_join(L, U):
  # Again, this is just to put together L and U.
  dim = L.shape[-1]
  module = torch if isinstance(L, torch.Tensor) else np
  W = module.tril(L, -1) + module.triu(U, 0)

  return W
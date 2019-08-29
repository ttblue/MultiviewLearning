# Basic utilities for pytorch
import numpy as np
import torch
from torch import nn

from models.model_base import BaseConfig
from utils import utils

import IPython


_DTYPE = torch.float32
_TENSOR_FUNC = torch.FloatTensor
torch.set_default_dtype(_DTYPE)


################################################################################
## Type converters
def dict_numpy_to_torch(data):
  if not isinstance(data[utils.get_any_key(data)], torch.Tensor):
    data = {
        i: torch.from_numpy(x).type(_DTYPE).requires_grad_(False)
        for i, x in data.items()
    }
  return data


def dict_torch_to_numpy(data):
  if isinstance(data[utils.get_any_key(data)], torch.Tensor):
    data = {i: x.detach().numpy() for i, x in data.items()}
  return data

################################################################################

# NN module wrappers:
def generate_linear_types_args(
      input_size, layer_units, output_size, bias=True):
  all_sizes = [input_size] + layer_units + [output_size]
  ltypes = [torch.nn.Linear] * (len(all_sizes) - 1)
  largs = [(l1, l2, bias) for l1, l2 in zip(all_sizes[:-1], all_sizes[1:])]
  return ltypes, largs


# TODO: Maybe keep layer_types/args params independent of output/input_size
# Right now, the first element corresponds to the input and the last
# element corresponds to the output.  
class MNNConfig(BaseConfig):
  def __init__(
      self, input_size, output_size, layer_types, layer_args, activation,
      last_activation, use_vae, *args, **kwargs):
    """
    layer_types: List of types of layers. Eg, Linear, Conv1d, etc.
    """
    super(MNNConfig, self).__init__(*args, **kwargs)

    self.input_size = input_size
    self.output_size = output_size
    self.layer_types = layer_types
    self.layer_args = layer_args
    self.activation = activation
    self.last_activation = last_activation
    self.use_vae = use_vae

  def set_sizes(
      self, input_size=None, output_size=None, layer_units=None, bias=True):
    if input_size is not None:
      largs = self.layer_args[0]
      self.input_size = input_size
      self.layer_args[0] = (input_size, largs[1], largs[2])
    if output_size is not None:
      largs = self.layer_args[-1]
      self.output_size = output_size
      self.layer_args[-1] = (largs[0], output_size, largs[2])
    if layer_units is not None:
      self.layer_types, self.layer_args = generate_linear_types_args(
          self.input_size, layer_units, self.output_size, bias)

# Identity singleton
class Identity(nn.Module):
  def forward(self, x):
    return x
_IDENTITY = Identity()


# Zeros singleton
class Zeros(nn.Module): 
  def forward(self, x): 
    return x.mul(0.)
_ZEROS = Zeros()


class MultiLayerNN(nn.Module):
  """
  Note: If config.layer_args is an empty list, this will just function as an
  identity function. In the case of encoding, this will return the input for mu
  and zeros_like(input) for logvar.
  """
  def __init__(self, config):
    super(MultiLayerNN, self).__init__()
    self.config = config
    self._setup_layers()

  def _setup_layers(self):
    # Default value of logvars
    self._logvar = _ZEROS
    num_layers = len(self.config.layer_args)
    self._layer_indices = []
    if num_layers == 0:
      self._layer_op = _IDENTITY
      self._mu = _IDENTITY
    else:
      _activation = self.config.activation()
      all_ops = []
      lidx = 0
      for i, (ltype, largs) in enumerate(
          zip(self.config.layer_types[:-1], self.config.layer_args[:-1])):
        # Interleave linear operations with activations
        all_ops.append(ltype(*largs))
        self._layer_indices.append(lidx)
        lidx += 2
        # If last layer, use "last_activation". Else, use "activation."
        # all_ops.append(
        #     self._activation() if i < num_layers - 1 else
        #     self._last_activation())
        all_ops.append(_activation)

      # If list is empty, this defaults to identity.
      self._layer_op = nn.Sequential(*all_ops)  # Pre-last-layer ops
      ltype = self.config.layer_types[-1]
      largs = list(self.config.layer_args[-1])
      largs[1] = self.config.output_size  # Just in case
      self._mu = ltype(*largs)
      self._last_activation = self.config.last_activation()

      if self.config.use_vae:
        # self._logvar = nn.Linear(largs[1], self.config.output_size)
        self._logvar = ltype(*largs)

  def forward(self, x):
    if not isinstance(x, torch.Tensor):
      x = torch.from_numpy(x.astype(np.float32))
    x = self._layer_op(x)
    mu_x = self._last_activation(self._mu(x))
    return (mu_x, self._logvar(x)) if self.config.use_vae else mu_x

  def get_layer_params(self, lidx):
    if lidx > len(self._layer_indices) or lidx == -1:
      if self.config.use_vae:
        return self._mu, self._logvar
      return self._mu
    lidx = self._layer_indices[lidx]
    return self._layer_op[lidx]


class RNNConfig(object):
  def __init__(
      self, input_size, hidden_size, num_layers, cell_type, return_only_hidden,
      return_only_final):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.cell_type = cell_type
    self.return_only_hidden = return_only_hidden
    self.return_only_final = return_only_final


class RNNWrapper(nn.Module):
  def __init__(self, config):
    super(RNNWrapper, self).__init__()
    self.config = config
    self._setup_cells()

  def _setup_cells(self):
    cell_type = self.config.cell_type
    self.cell = cell_type(
        input_size=self.config.input_size, hidden_size=self.config.hidden_size,
        num_layers=self.config.num_layers)

  # def _init_hidden(self):
  #   # Initialize hidden state and cell state for RNN
  #   # Note: not sure if we need this.
  #   return (
  #       torch.zeros(self.config.num_layers, self.config.batch_size,
  #                   self.config.hidden_size),
  #       torch.zeros(self.config.num_layers, self.config.batch_size,
  #                   self.config.hidden_size))

  def forward(self, ts_batch, hc_0=None):
    # ts_batch: time_steps x batch_size x input_size
    if not isinstance(ts_batch, torch.Tensor):
      ts_batch = torch.from_numpy(ts_batch.astype(np.float32))

    # For now, no attention mechanism
    # IPython.embed()
    output, (hn, cn) = self.cell(ts_batch, hc_0)
    if self.config.return_only_final:
      return hn
    if self.config.return_only_hidden:
      return output
    return output, (hn, cn)

# Some silly tests for RNN with torch
# opt1, (hn1, cn1) = ll(ipt)
# op1, hn1, cn1 = [v.detach().numpy() for v in (opt1, hn1, cn1)]

# opt2 = []
# hidden = None
# for i in range(nt):
#   pti = torch.FloatTensor(ip[i]).view(1, -1).unsqueeze(1)
#   opi, hidden = ll(pti, hidden)
#   opt2.append(opi.detach().numpy())
# op2 = np.concatenate(opt2, axis=0)
# hn2, cn2 = [hv.detach().numpy() for hv in hidden]
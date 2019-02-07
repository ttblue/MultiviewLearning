# Basic utilities for pytorch
import numpy as np

import torch
from torch import nn

import IPython


_DTYPE = torch.float32
_TENSOR_FUNC = torch.FloatTensor
torch.set_default_dtype(_DTYPE)


def generate_layer_types_args(input_size, layer_units, output_size):
  all_sizes = [input_size] + layer_units + [output_size]
  ltypes = [torch.nn.Linear] * (len(all_sizes) - 1)
  largs = [(l1, l2) for l1, l2 in zip(all_sizes[:-1], all_sizes[1:])]
  return ltypes, largs


class MNNConfig(object):
  def __init__(
      self, input_size, output_size, layer_types, layer_args, activation,
      last_activation, use_vae):
    """
    layer_types: List of types of layers. Eg, Linear, Conv1d, etc.
    """
    self.input_size = input_size
    self.output_size = output_size
    self.layer_types = layer_types
    self.layer_args = layer_args
    self.activation = activation
    self.last_activation = last_activation
    self.use_vae = use_vae


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

    self._activation = self.config.activation
    self._last_activation = self.config.last_activation
    self._setup_layers()

  def _setup_layers(self):
    # Default value of logvars
    self._logvar = _ZEROS
    num_layers = len(self.config.layer_args)
    if num_layers == 0:
      self._layer_op = _IDENTITY
      self._mu = _IDENTITY
    else:
      all_ops = []
      for i, (ltype, largs) in enumerate(
          zip(self.config.layer_types, self.config.layer_args)):
        # Interleave linear operations with activations
        all_ops.append(ltype(*largs))
        # If last layer, use "last_activation". Else, use "activation."
        all_ops.append(
            self._activation() if i < num_layers - 1 else
            self._last_activation())

      self._layer_op = nn.Sequential(*all_ops)
      self._mu = nn.Linear(largs[-1], self.config.output_size)
      if self.config.use_vae:
        self._logvar = nn.Linear(largs[-1], self.config.output_size)

  def forward(self, x):
    if not isinstance(x, torch.Tensor):
      x = torch.from_numpy(x.astype(np.float32))
    x = self._layer_op(x)
    if self.config.use_vae:
      return self._mu(x), self._logvar(x)
    return self._mu(x)


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

  def forward(self, ts_batch, h_0=None, c_0=None):
    # ts_batch: time_steps x batch_size x input_size
    if not isinstance(ts_batch, torch.Tensor):
      ts_batch = torch.from_numpy(ts_batch.astype(np.float32))

    # For now, no attention mechanism
    # IPython.embed()
    hc_0 = None if h_0 is None else (h_0, c_0)
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

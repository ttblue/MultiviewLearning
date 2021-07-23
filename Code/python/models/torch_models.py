# Basic models using pytorch
import numpy as np
import torch
from torch import nn

from models.model_base import BaseConfig
from utils import utils, torch_utils


import IPython


################################################################################
## Feedforward NN

# TODO: Maybe keep layer_types/args params independent of output/input_size
# Right now, the first element corresponds to the input and the last
# element corresponds to the output.  
class MNNConfig(BaseConfig):
  def __init__(
      self, input_size, output_size, layer_types, layer_args, activation,
      last_activation, dropout_p, use_vae, *args, **kwargs):
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
    self.dropout_p = dropout_p
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
      self.layer_types, self.layer_args = (
          torch_utils.generate_linear_types_args(
              self.input_size, layer_units, self.output_size, bias))


# Identity singleton
class Identity(nn.Module):
  def forward(self, x, *args):
    return x
_IDENTITY = Identity()


# Zeros singleton
class Zeros(nn.Module): 
  def forward(self, x, *args):
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
    self._setup_output()

  def _setup_layers(self):
    self.dropout = nn.Dropout(self.config.dropout_p)
    # Default value of logvars
    self._logvar = _ZEROS
    num_layers = len(self.config.layer_args)
    self._layer_indices = []
    if num_layers == 0:
      self._layer_op = _IDENTITY
    else:
      _activation = self.config.activation() if num_layers > 1 else None
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

  def _setup_output(self):
    num_layers = len(self.config.layer_args)
    if num_layers == 0:
      self._mu = _IDENTITY
      if self.config.use_vae:
        self._logvar = _IDENTITY
    else:
      ltype = self.config.layer_types[-1]
      largs = list(self.config.layer_args[-1])
      largs[1] = self.config.output_size  # Just in case
      self._mu = ltype(*largs)
      self._last_activation = self.config.last_activation()

      if self.config.use_vae:
        # self._logvar = nn.Linear(largs[1], self.config.output_size)
        self._logvar = ltype(*largs)

  def forward(self, x, disable_logvar=False):
    x = torch_utils.numpy_to_torch(x)
    x = self.dropout(x)
    x = self._layer_op(x)
    mu_x = self._last_activation(self._mu(x))
    if not self.training or not self.config.use_vae or disable_logvar:
      return mu_x
    return (mu_x, self._logvar(x))

  def get_layer_params(self, lidx):
    if lidx > len(self._layer_indices) or lidx == -1:
      if self.config.use_vae:
        return self._mu, self._logvar
      return self._mu
    lidx = self._layer_indices[lidx]
    return self._layer_op[lidx]


class MultiOutputMLNN(MultiLayerNN):
  def __init__(self, config, output_sizes, *args, **kwargs):
    self.output_sizes = {i: os_i for i, os_i in enumerate(output_sizes)}
    self._n_outputs = len(self.output_sizes)
    super(MultiOutputMLNN, self).__init__(config)

  def _setup_output(self):
    num_layers = len(self.config.layer_args)
    num_outputs = len(self.output_sizes)
    if num_layers == 0:
      self._mu = [_IDENTITY] * num_outputs
      if self.config.use_vae:
        self._logvar = [_IDENTITY] * num_outputs
    else:
      self._mu = torch.nn.ModuleDict()
      if self.config.use_vae:
        self._logvar = torch.nn.ModuleDict()
      ltype = self.config.layer_types[-1]
      largs = list(self.config.layer_args[-1])
      for i, output_size in self.output_sizes.items():
        largs[1] = output_size
        self._mu["op_%i"%i] = ltype(*largs)
        if self.config.use_vae:
          self._logvar["op_%i"%i] = ltype(*largs)

      self._last_activation = self.config.last_activation()

  def forward(self, x, disable_logvar=False):
    x = torch_utils.numpy_to_torch(x)
    x = self.dropout(x)
    x = self._layer_op(x)

    mu_x = [self._last_activation(self._mu["op_%i"%i](x))
            for i in range(self._n_outputs)]
    if not self.training or not self.config.use_vae or disable_logvar:
      return mu_x

    logvar_x = [self._last_activation(self._logvar["op_%i"%i](x))
                for i in range(self._n_outputs)]
    return (mu_x, logvar_x)

################################################################################
## Recurrent nets

_CELL_TYPES = {
    "lstm": nn.LSTM,
    "rnn": nn.RNN,
    "gru": nn.GRU,
}

def get_cell_type(cell_type):
  if isinstance(cell_type, str) and cell_type in _CELL_TYPES:
    return _CELL_TYPES[cell_type]
  return cell_type


class RNNConfig(BaseConfig):
  def __init__(
      self, input_size, hidden_size, num_layers, bias, cell_type, output_len,
      dropout_p, return_only_hidden, return_only_final, *args, **kwargs):

    super(RNNConfig, self).__init__(*args, **kwargs)

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.cell_type = cell_type
    self.output_len = output_len

    self.dropout_p = dropout_p

    self.return_only_hidden = return_only_hidden
    self.return_only_final  = return_only_final

  def set_sizes(self, input_size=None, hidden_size=None, bias=True):
    if input_size is not None:
      self.input_size = input_size
    if hidden_size is not None:
      self.hidden_size = hidden_size
    if bias is not None:
      self.bias = bias


class RNNWrapper(nn.Module):
  def __init__(self, config):
    super(RNNWrapper, self).__init__()
    self.config = config
    self._setup_cells()

  def _setup_cells(self):
    cell_type = get_cell_type(self.config.cell_type)
    self.cell = cell_type(
        input_size=self.config.input_size, hidden_size=self.config.hidden_size,
        num_layers=self.config.num_layers, bias=self.config.bias,
        dropout=self.config.dropout_p, batch_first=_BATCH_FIRST)

  # def _init_hidden(self):
  #   # Initialize hidden state and cell state for RNN
  #   # Note: not sure if we need this.
  #   return (
  #       torch.zeros(self.config.num_layers, self.config.batch_size,
  #                   self.config.hidden_size),
  #       torch.zeros(self.config.num_layers, self.config.batch_size,
  #                   self.config.hidden_size))
  def _forecast(self, init_step, hc_0, output_len=None):
    hstate = hc_0[0]
    cstate = hc_0[1]
    output_len = self.config.output_len if output_len is None else output_len

    if init_step is None:
      batch_size = 1 if hstate is None else hstate.shape[1]
      init_step = torch.zeros(1, batch_size, self.config.input_size)
    else:
      init_step = torch_utils.numpy_to_torch(init_step)
      if len(init_step.shape) < 3:
        # Make into sequences
        init_step = init_step.unsqueeze(0)
      else:
        init_step = init_step.transpose(0, 1)

    output = init_step
    outputs = []
    hstates = []
    cstates = []
    for t in range(output_len):
      output, (hstate, cstate) = self.cell(output, (hstate, cstate))
      outputs.append(output)
      hstates.append(hstate)
      cstates.append(cstate)
    outputs = torch.cat(outputs, dim=0)  # _BATCH_FIRST is False
    # These two have batches as the second dim
    hstates = torch.cat(hstates, dim=0)
    cstates = torch.cat(cstates, dim=0)

    # Transpose outputs back:
    outputs = outputs.transpose(0, 1)

    if self.config.return_only_final:
      return hstate
    if self.config.return_only_hidden or not self.training:
      return outputs
    return outputs, (hstates, cstates)

  def forward(self, ts, hc_0=None, forecast=False, output_len=None):
    # ts: batch_size x time_steps x input_size
    if forecast:
      return self._forecast(ts, hc_0, output_len)

    ts = torch_utils.numpy_to_torch(ts)
    ts = ts.transpose(0, 1)
    # For now, no attention mechanism
    output, (hn, cn) = self.cell(ts, hc_0)
    output = output.transpose(0, 1)
    if self.config.return_only_final:
      return hn
    if self.config.return_only_hidden or not self.training:
      return output
    return output, (hn, cn)
# Some silly tests for RNN with torch
# opt1, (hn1, cn1) = ll(ipt)
# op1, hn1, cn1 = [torch_utils.torch_to_numpy(v) for v in (opt1, hn1, cn1)]

# opt2 = []
# hidden = None
# for i in range(nt):
#   pti = torch.FloatTensor(ip[i]).view(1, -1).unsqueeze(1)
#   opi, hidden = ll(pti, hidden)
#   opt2.append(torch_utils.torch_to_numpy(opi))
# op2 = np.concatenate(opt2, axis=0)
# hn2, cn2 = [torch_utils.torch_to_numpy(hv) for hv in hidden]
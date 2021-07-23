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
from utils.torch_utils import _DTYPE, _TENSOR_FUNC, _IDENTITY

import IPython


################################################################################
# Weight initialization:
def xavier_init(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)
  elif isinstance(m, nn.LSTM):
    for layer_weights in m.all_weights:
      for wt in layer_weights:
        if len(wt.shape) > 1:
          nn.init.xavier_uniform_(wt)
        else:
          wt.data.fill_(0.01)


def zero_init(m):
  if isinstance(m, nn.Linear):
    m.weight.data.fill_(0.)
    m.bias.data.fill_(0.)
  elif isinstance(m, nn.LSTM):
    for layer_weights in m.all_weights:
      for wt in layer_weights:
        wt.data.fill_(0.)


def initialize_weights(model, init_type="xavier"):
  wfunc = xavier_init if init_type == "xavier" else zero_init
  model.apply(wfunc)


_DEFAULT_INIT = "xavier"

################################################################################
# Encoder/Decoder class
class FRFWConfig(BaseConfig):
  def __init__(
      self, input_size, output_size, pre_ff_config, rnn_config, post_ff_config,
      output_len, return_all_outputs, *args, **kwargs):

    super(FRFWConfig, self).__init__(*args, **kwargs)

    self.input_size = input_size
    self.output_size = output_size

    self.pre_ff_config = pre_ff_config
    self.rnn_config = rnn_config
    self.post_ff_config = post_ff_config

    self.output_len = output_len  # For decoding from single instance

    self.return_all_outputs = return_all_outputs

  def set_sizes(self, input_size=None, output_size=None):
    if input_size is not None:
      self.input_size = input_size
    if output_size is not None:
      self.output_size = output_size


class FRFWrapper(nn.Module):
  # Wrapper for an overall network which has a feed-foward NN, followed by a
  # recurrent NN followed by a feed forward NN.
  def __init__(self, config):
    super(FRFWrapper, self).__init__()
    self.config = config
    self._setup_layers()

  def _setup_layers(self):
    if self.config.pre_ff_config is None:
      self._pre_net = _IDENTITY
      rnn_input_size = self.config.input_size
    else:
      input_size = self.config.input_size
      self.config.pre_ff_config.set_sizes(input_size=input_size)
      self._pre_net = tu.MultiLayerNN(self.config.pre_ff_config)
      rnn_input_size = self.config.pre_ff_config.output_size

    if self.config.post_ff_config is None:
      self._post_net = _IDENTITY
      rnn_hidden_size = self.config.output_size
    else:
      input_size = (
          rnn_input_size if self.config.rnn_config is None else
          self.config.rnn_config.hidden_size
      )
      output_size = self.config.output_size
      self.config.post_ff_config.set_sizes(
          input_size=input_size, output_size=output_size)
      self._post_net = tu.MultiLayerNN(self.config.post_ff_config)
      rnn_hidden_size = input_size

    if self.config.rnn_config is None:
      self._rec_net = _IDENTITY
    else:
      self.config.rnn_config.set_sizes(
          input_size=rnn_input_size, hidden_size=rnn_hidden_size)
      self._rec_net = tu.RNNWrapper(self.config.rnn_config)

  def forward(
        self, ts, hc_0=None, forecast=False, output_len=None):
    # forecast: Can be "state", "input" or None/False. If not None/False, the
    # pre_ff output is fed in as the initial state or initial input depending
    # on forecast.

    # ts: batch_size x time_steps x input_size
    if not isinstance(ts, torch.Tensor):
      ts = torch.from_numpy(ts).type(_DTYPE).requires_grad_(False)

    pre_config = self.config.pre_ff_config
    rnn_config = self.config.rnn_config
    post_config = self.config.post_ff_config

    # Apply the initial feedforward net
    pre_output = self._pre_net(ts)
    # Select the appropriate subset of this output for input to the RNN
    rnn_input = (
        pre_output[0] if (
            pre_config is not None and
            pre_config.use_vae and
            self.training
        ) else
        pre_output
    )
    # Apply the recurrent net
    if forecast:
      if forecast == "state":
        # The input state/cell need to have batch and seq len flipped
        h_0 = rnn_input.transpose(0, 1)
        c_0 = torch.zeros_like(h_0)
        hc_0 = (h_0, c_0)
        rnn_input = None

      rnn_output = self._rec_net(
          rnn_input, hc_0, forecast=True, output_len=output_len)
    else:
      rnn_output = self._rec_net(rnn_input, hc_0, forecast=False)
    # Select the appropriate subset of this output for input to the post net
    post_input = (
        rnn_output if (
            rnn_config is None or
            rnn_config.return_only_final or
            rnn_config.return_only_hidden or
            not self.training
        ) else
        rnn_output[0]
    )
    # Apply the final feed forward net
    post_output = self._post_net(post_input)

    if self.config.return_all_outputs:
      return pre_output, rnn_output, post_output

    output = (
        post_output[0] if (
            post_config is not None and
            post_config.use_vae and
            self.training
        ) else
        post_output
    )

    # Hack
    if not forecast:
      # Output is final hidden state which has batch and seq switched
      output = output.transpose(0, 1)
    return output


class TSRFConfig(BaseConfig):
  def __init__(
      self, encoder_config, decoder_config, hidden_size, time_delay_tau=10,
      time_delay_ndim=3, batch_size=50, lr=1e-4, max_iters=10000,
      *args, **kwargs):

    self.encoder_config = encoder_config
    self.decoder_config = decoder_config
    self.hidden_size = hidden_size

    self.time_delay_tau = time_delay_tau
    self.time_delay_ndim = time_delay_ndim 
    # self.output_len = output_len  # Inferred from data

    self.batch_size = batch_size
    self.lr = lr
    self.max_iters = max_iters

    super(TSRFConfig, self).__init__(*args, **kwargs)


class TimeSeriesReconFeaturization(nn.Module):
  def __init__(self, config):
    super(TimeSeriesReconFeaturization, self).__init__()
    self.config = config

  def _initialize(self):
    _td_dim = self._dim * max(self.config.time_delay_ndim, 1)
    # Create RN encoder
    self.config.encoder_config.set_sizes(
        input_size=_td_dim, output_size=self.config.hidden_size)
    # Can change these as needed:
    self.config.encoder_config.return_all_outputs = False
    self.config.encoder_config.rnn_config.return_only_hidden = False
    self.config.encoder_config.rnn_config.return_only_final = True
    self.encoder = FRFWrapper(self.config.encoder_config)

    # Create RN decoder
    # Need to make sure decoder uses the latent rep as initial state, not input.
    dconfig = self.config.decoder_config
    dconfig.set_sizes(
        input_size=self.config.hidden_size, output_size=self._dim)
    dconfig.return_all_outputs = False
    rnn_input_size = self._dim  # Not sure what to do here.
    rnn_hidden_size = (
        dconfig.input_size if dconfig.pre_ff_config is None else
        dconfig.pre_ff_config.output_size
    )
    dconfig.rnn_config.set_sizes(
          input_size=rnn_input_size, hidden_size=rnn_hidden_size)
    # Can change these as needed:
    dconfig.rnn_config.return_only_hidden = True
    dconfig.rnn_config.return_only_final = False

    # If output is different size than latent state (when using td embedding)
    # then need extra layer to make RNN output be of right dimension.
    if _td_dim != self._dim:
      if dconfig.post_ff_config is None:
        input_size = rnn_hidden_size
        output_size = dconfig.output_size
        ltypes, largs = tu.generate_linear_types_args(
            input_size, [], output_size)
        activation = tu.Identity
        last_activation = tu.Identity
        dropout_p = 0.
        use_vae = False

        dconfig.post_ff_config = tu.MNNConfig(
            input_size=input_size, output_size=output_size, layer_types=ltypes,
            layer_args=largs, activation=activation,
            last_activation=last_activation, dropout_p=dropout_p,
            use_vae=use_vae)

    self.decoder = FRFWrapper(dconfig)

    self.recon_criterion = nn.MSELoss(reduction="elementwise_mean")
    self.opt = optim.Adam(self.parameters(), self.config.lr)

  def _td_embedding(self, ts):
    # Assume ts is big enough.
    if self.config.time_delay_ndim <= 1:
      return ts

    seq_len = ts.shape[1]
    tau = self.config.time_delay_tau
    ndim = self.config.time_delay_ndim
    td_embedding = np.concatenate(
        [ts[:, i*tau: seq_len-(ndim-i-1)*tau] for i in range(ndim)], axis=2)

    return td_embedding

  def encode(self, ts):
    tde = self._td_embedding(ts)
    encoding = self.encoder(tde, forecast=False)
    return encoding

  def decode(self, encoding, output_len=None):
    output = self.decoder(encoding, forecast="state", output_len=output_len)
    return output

  def forward(self, ts):
    # ts: batch_size x seq length x input_dim
    encoding = self.encode(ts)
    output = self.decode(encoding, output_len=self._seq_len)
    return encoding, output

  def loss(self, ts, encoding, output):
    try:
      obj = self.recon_criterion(ts, output)
    except:
      IPython.embed()
    # Additional loss based on the encoding:
    # KLD penalty? Sparsity?
    return obj

  def _shuffle(self, ts_dset):
    npts = ts_dset.shape[0]
    r_inds = np.random.permutation(npts)
    return ts_dset[r_inds]

  def _train_loop(self):
    ts_dset = self._shuffle(self._ts_dset)
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      ts_batch = ts_dset[b_start:b_end]

      self.opt.zero_grad()
      batch_encoding, batch_output = self.forward(ts_batch)
      loss_val = self.loss(ts_batch, batch_encoding, batch_output)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val
    self._loss_history.append(float(self.itr_loss.detach()))

  def fit(self, ts_dset):
    # ts_dset: num_ts x seq_len x input_dim
    if self.config.verbose:
      all_start_time = time.time()
    self.train()
    # Convert to numpy then torch
    ts_dset = np.array(ts_dset)
    self._ts_dset = torch.from_numpy(ts_dset).type(_DTYPE).requires_grad_(False)

    # self._ts_dset = ts_dset
    self._npts = ts_dset.shape[0]
    self._seq_len = ts_dset.shape[1]
    self._dim = ts_dset.shape[2]
    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))

    self._initialize()
    initialize_weights(self, _DEFAULT_INIT)
    self._loss_history = []
    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("\nIteration %i out of %i." % (itr + 1, self.config.max_iters))
        self._train_loop()

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          print("Loss: %.5f" % float(self.itr_loss.detach()))
          print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))

  def reconstruct(self, te_dset, rtn_numpy=True):
    encoding = self.encode(te_dset)
    recon = self.decode(encoding, output_len=te_dset.shape[1])
    if rtn_numpy:
      return recon.detach().numpy()
    return recon

  def save_to_file(self, fname):
    torch.save(self.state_dict(), fname)

  def load_from_file(self, fname):
    self.load_state_dict(fname)
    self.eval()
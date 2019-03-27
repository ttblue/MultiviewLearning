# (Variational) Autoencoder for multi-view + synchronous using RNNs (LSTM)
# Based on the approach: https://arxiv.org/pdf/1707.07961.pdf

# Input/encoding pipeline for each view:
# Input --> Encoder: Featurized input --> RNN: Hidden state -->
#     EncoderL: Latent state

# Output/decoding pipeline for each view:
# Latent state --> DecoderL: Hidden state --> RNN: Featurized output -->
#     Decoder: Output


import itertools
import numpy as np
import time
import torch
from torch import autograd
from torch import nn
from torch import optim

import torch_utils as tu
from torch_utils import _DTYPE, _TENSOR_FUNC

import IPython


class MRNNAEConfig(object):

  def __init__(
      self, n_views, encoder_params, decoder_params, use_vae, max_iters,
      batch_size, lr, verbose):
    self.n_views = n_views
    self.encoder_params = encoder_params
    self.decoder_params = decoder_params
    self.use_vae = use_vae

    self.max_iters = max_iters
    self.batch_size = batch_size
    self.lr = lr

    self.verbose = verbose


# # TODO: Modify this
# def default_MRNNAE_Config(v_sizes):
#   n_views = len(v_sizes)
#   code_size = 10

#   # Default Encoder config:
#   output_size = code_size
#   layer_units = [50, 20]
#   is_encoder = True
#   activation = nn.functional.relu
#   last_activation = None
#   encoder_params = {
#       i: tu.MNNConfig(
#           input_size=v_sizes[i],
#           output_size=output_size,
#           layer_units=layer_units,
#           is_encoder=is_encoder,
#           activation=activation,
#           last_activation=last_activation)
#       for i in range(n_views)
#   }

#   input_size = code_size
#   is_encoder = False
#   last_activation = nn.sigmoid
#   decoder_params = {
#       i: tu.MNNConfig(
#           input_size=input_size,
#           output_size=v_sizes[i],
#           layer_units=layer_units,
#           is_encoder=is_encoder,
#           activation=activation,
#           last_activation=last_activation)
#       for i in range(n_views)
#   }

#   max_iters = 1000
#   batch_size = 50
#   lr = 1e-3
#   verbose = True
#   config = MAEConfig(
#       v_sizes=v_sizes,
#       code_size=code_size,
#       encoder_params=encoder_params,
#       decoder_params=decoder_params,
#       max_iters=max_iters,
#       batch_size=batch_size,
#       lr=lr,
#       verbose=verbose)

#   return config


class MultiRNNAutoEncoder(nn.Module):
  # Auto encoder with multi-views
  def __init__(self, config):
    super(MultiRNNAutoEncoder, self).__init__()
    self.config = config

    self._n_views = config.n_views

    self._initialize_layers()
    self._setup_optimizer()

    self.recon_criterion = nn.MSELoss(reduction="elementwise_mean")

    self.trained = False

  def _initialize_layers(self):
    # Initialize encoder:
    self._encoder = {}
    self._decoder = {}
    module_dict = {"enc":self._encoder, "dec":self._decoder}
    for vi in range(self._n_views):
      params = {
          "enc": self.config.encoder_params[vi],
          "dec": self.config.decoder_params[vi],
      }
      for ptype, pconfig in params.items():
        layer_funcs = pconfig["layer_funcs"]
        layer_configs = pconfig["layer_config"]
        all_ops = []
        for lfunc, lconfig in zip(layer_funcs, layer_configs):
          layer = lfunc() if lconfig is None else lfunc(lconfig)
          all_ops.append(layer)

        module_op = nn.Sequential(*all_ops)
        self.add_module("%s_%i" % (ptype, vi), module_op)
        module_dict[ptype][vi] = module_op

  # def _initialize_layers(self):
  #   # For encoding and decoding each view, need to initialize:
  #   # 1. RNN layers
  #   # 2. Pre-RNN feedforward NN layers
  #   # 3. Post-RNN feedforward NN layers
  #   self._pre_en_layers = {}
  #   self._en_rnn = {}
  #   self._post_en_layers = {}

  #   self._pre_de_layers = {}
  #   self._de_rnn = {}
  #   self._post_de_layers = {}

  #   # Need to call "add_module" so the parameters are found.
  #   def set_value(vdict, key, name, value):
  #     self.add_module(name, value)
  #     vdict[key] = value

  #   for vi in range(self._n_views):
  #     # TODO: For now, we don't want the pre-encoding/decoding stage to have
  #     #       sampling. Same for post-decoding stage.

  #     self.config.pre_en_params[vi].use_vae = False
  #     self.config.pre_de_params[vi].use_vae = False
  #     self.config.post_de_params[vi].use_vae = False

  #     # Over-ride individual layer info if not using vae
  #     self.config.post_en_params[vi].use_vae = self.config.use_vae

  #     set_value(self._pre_en_layers, vi, "pre_en_%i" % vi,
  #               tu.MultiLayerNN(self.config.pre_en_params[vi]))
  #     set_value(self._en_rnn, vi, "en_rnn_%i" % vi,
  #               tu.RNNWrapper(self.config.en_rnn_params[vi]))
  #     set_value(self._post_en_layers, vi, "post_en_%i" % vi,
  #               tu.MultiLayerNN(self.config.post_en_params[vi]))

  #     set_value(self._pre_de_layers, vi, "pre_de_%i" % vi,
  #               tu.MultiLayerNN(self.config.pre_de_params[vi]))
  #     set_value(self._de_rnn, vi, "de_rnn_%i" % vi,
  #               tu.RNNWrapper(self.config.de_rnn_params[vi]))
  #     set_value(self._post_de_layers, vi, "post_de_%i" % vi,
  #               tu.MultiLayerNN(self.config.post_de_params[vi]))

  def _setup_optimizer(self):
    self.opt = optim.Adam(self.parameters(), self.config.lr)

  def _encode_view(self, xv, vi):
    # IPython.embed()
    return self._encoder[vi](xv)
    # pre_layer = self._pre_en_layers[vi]
    # rnn_layer = self._en_rnn[vi]
    # post_layer = self._post_en_layers[vi]

    # h_0 = c_0 = None
    # # We're forcing no sampling going while constructing pre-layers.
    # code = post_layer(rnn_layer(pre_layer(xv), h_0=h_0, c_0=c_0)[0])
    # return code

  def encode(self, txs):
    # txs:  [n_ts x n_batch x n_features(vi) for vi in views]
    codes = {
        vi:self._encode_view(xv, vi) for vi, xv in txs.items()
    }
    return codes

  def _sample_codes(self, mu, logvar, noise_coeff=0.5):
    # Add noise to code formed for robustness of reconstruction
    err = _TENSOR_FUNC(logvar.size()).normal_()
    codes = torch.autograd.Variable(err)
    std = (noise_coeff * logvar).exp_()
    return codes.mul(std).add_(mu)

  def _decode_view(self, code, vi):
    return self._decoder[vi](code)
    # h_0 = c_0 = None
    # recon = post_layer(rnn_layer(pre_layer(z), h_0=h_0, c_0=c_0)[0])
    # return recon

  def decode(self, code, vi_out=None):
    # Not assuming tied weights yet
    vi_out = range(self._n_views) if vi_out is None else vi_out
    recons = {vi:self._decode_view(code, vi) for vi in vi_out}
    return recons

  def forward(self, txs):
    # Solve for alpha
    v_check = list(txs.keys())[0]
    if not isinstance(txs[v_check], torch.Tensor):
      txs = {
          vi:torch.from_numpy(np.asarray(txs[vi])).type(_DTYPE)
          for vi in txs
    }
    # x.requires_grad_(False)
    zs = self.encode(txs)
    if self.config.use_vae:
      zs = {vi:self._sample_codes(zs[vi]) for vi in zs}

    # This is, for every encoded view, the reconstruction of every view
    recons = {}
    for vi in range(self._n_views):
      recons[vi] = self.decode(zs[vi])

    return zs, recons

  def loss(self, txs, recons, zs):
    obj = 0.
    for vi in recons:
      for ridx in range(self._n_views):
        obj += self.recon_criterion(txs[ridx], recons[vi][ridx])

    # Additional loss based on the encoding:
    # Maybe explicitly force the encodings to be similar
    # KLD penalty
    return obj

  def _train_loop(self, dset):
    dset.shuffle_data()
    self.itr_loss = 0.
    for tx_batch in dset.get_ts_batches(
        self.config.batch_size, permutation=(1, 0, 2)):
      # Rearranging dimensions -- no need to do this now
      # x_batch = [tx.permute((1, 0, 2)) for tx in txs]

      self.opt.zero_grad()
      zs, recons = self.forward(tx_batch)
      loss_val = self.loss(tx_batch, recons, zs)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self, dset):
    # dset: dataset.MultimodalTimeSeriesDataset
    # Assuming synchronized
    if not dset.synced:
      raise ValueError("Dataset must be synchronous.")

    # txs = [
    #     torch.from_numpy(np.asarray(tx, dtype="float32").transpose(1, 0, 2))
    #     for tx in txs]
    self._n_ts = dset.n_ts
    dset.convert_to_torch()
    # IPython.embed()

    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("Epoch %i out of %i." % (itr + 1, self.config.max_iters))
        self._train_loop(dset)

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          print("Loss: %.5f" % float(self.itr_loss.detach()))
          print("Epoch %i took %0.2fs.\n" % (itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")

    self.trained = True
    print("Training finished in %0.2fs." % (time.time() - all_start_time))

  def predict(self, txs, vi_in, vi_out=None, rtn_torch=False):
    # txs: List of size n_views with n_ts x n_steps x n_features(view)
    # TODO: Fix this to work with multi-view input.
    if isinstance(vi_in, list):
      raise Exception("Not yet ready for multiview input.")

    if vi_out is None:
      vi_out = np.arange(self._n_views).tolist()

    if not isinstance(txs, torch.Tensor):
      txs = torch.from_numpy(np.asarray(txs)).type(_DTYPE).requires_grad_(False)
    txs = txs.permute(1, 0, 2)
    # if not isinstance(vi_in, list):
    #   vi_in = [vi_in]
    #   txs = [txs]
    # if not isinstance(vi_out, list):
    #   vi_out = [vi_out]

    # if not isinstance(txs, torch.Tensor):
    #   txs = torch.from_numpy(np.asarray(txs)).type(_DTYPE).requires_grad_(False)

    # # Transpose dimensions
    # txs = txs.permute(1, 0, 2)

    # codes = [self._encode_view(txs[i], vi) for i, vi in enumerate(vi_in)]
    # if self.config.use_vae:
    #   codes = [code[0] for code in codes]

    # codes = torch.stack(codes, dim=0)
    # z = torch.mean(codes, dim=0)

    # # Retranspose dimensions after decoding
    # preds = [pr.permute(1, 0, 2) for pr in self.decode(z, vi_out)]
    # if not rtn_torch:
    #   preds = [p.detach().numpy() for p in preds]

    # return preds[0] if len(vi_out) == 1 else preds
    z = self._encode_view(txs, vi_in)
    if self.config.use_vae:
      z = z[0]
    # codes = torch.stack(codes, dim=0)
    # z = torch.mean(codes, dim=0)

    # Retranspose dimensions after decoding
    preds = {
        vi: pr.permute(1, 0, 2) for vi, pr in self.decode(z, vi_out).items()
    }
    if not rtn_torch:
      preds = {
        vi: pr.detach().numpy() for vi, pr in preds.items()
    }

    return preds


if __name__=="__main__":
  v_sizes = [3, 3, 3]
  # config = default_MAE_Config(v_sizes)
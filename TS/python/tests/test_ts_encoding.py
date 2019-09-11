# Tests for some CCA stuff on pig data.
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import ts_recon_featurization
from synthetic import multimodal_systems as ms
from utils import torch_utils as tu, utils


try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


np.set_printoptions(precision=5, suppress=True)


def default_FFNN_config(input_size, output_size):
  layer_units = [16, 16]  #[32] # [32, 64]
  use_vae = False
  activation = nn.ReLU  # nn.functional.relu
  last_activation = nn.Sigmoid  # functional.sigmoid

  dropout_p = 0.

  bias = True
  layer_types, layer_args = tu.generate_linear_types_args(
        input_size, layer_units, output_size, bias)

  nn_config = tu.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)
  return nn_config


def default_RNN_config(input_size, hidden_size):
  num_layers = 1
  bias = True
  cell_type = nn.LSTM
  output_len = 50
  dropout_p = 0.

  return_only_hidden = False
  return_only_final = False

  rnn_config = tu.RNNConfig(
      input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
      bias=bias, cell_type=cell_type, output_len=output_len,
      dropout_p=dropout_p, return_only_hidden=return_only_hidden,
      return_only_final=return_only_final)
  return rnn_config


def default_FRFW_config(use_pre=False, use_post=False):
  # Encoder config:
  model_input_size = 3   # Temporary -- overwritten by data
  pre_output_size = 8
  rnn_hidden_size = 8
  model_output_size = 8

  if use_pre:
    input_size = model_input_size
    output_size = pre_output_size
    pre_ff_config = default_FFNN_config(input_size, output_size)
    rnn_input_size = output_size
  else:
    pre_ff_config = None
    rnn_input_size = model_input_size

  if use_post:
    input_size = rnn_hidden_size
    output_size = model_output_size
    post_ff_config = default_FFNN_config(input_size, output_size)
  else:
    post_ff_config = None
    rnn_hidden_size = model_output_size

  rnn_config = default_RNN_config(rnn_input_size, rnn_hidden_size)

  input_size = model_input_size  # Temporary
  output_size = model_output_size

  output_len = 50
  return_all_outputs = False

  frfw_config = ts_recon_featurization.FRFWConfig(
    input_size=input_size, output_size=output_size, pre_ff_config=pre_ff_config,
    rnn_config=rnn_config, post_ff_config=post_ff_config, output_len=output_len,
    return_all_outputs=return_all_outputs
  )
  return frfw_config


def default_TSRF_config(use_pre=[], use_post=[]):
  use_pre_enc = ("encoder" in use_pre)
  use_post_enc = ("encoder" in use_post)
  latent_size = 8
  encoder_config = default_FRFW_config(use_pre_enc, use_post_enc)
  encoder_config.output_size = latent_size

  use_pre_dec = ("decoder" in use_pre)
  use_post_dec = ("decoder" in use_post)
  decoder_config = default_FRFW_config(use_pre_dec, use_post_dec)
  decoder_config.input_size = latent_size

  hidden_size = latent_size

  time_delay_tau = 10
  time_delay_ndim = 3

  batch_size = 100
  lr = 1e-4
  max_iters = 10000
  verbose = True

  config = ts_recon_featurization.TSRFConfig(
      encoder_config=encoder_config, decoder_config=decoder_config,
      hidden_size=hidden_size, time_delay_tau=time_delay_tau,
      time_delay_ndim=time_delay_ndim, batch_size=batch_size, lr=lr,
      max_iters=max_iters, verbose=verbose)

  return config


def load_pig_data(num_pigs=-1, channels=None, valid_labels=None):
  pig_list = pig_videos.FFILE_PNUMS
  if num_pigs > 0:
    pig_list = pig_list[:num_pigs]

  ds = 1
  ds_factor = 25
  view_feature_sets = None
  pig_data = pig_videos.load_pig_features_and_labels(
      pig_list=pig_list, ds=ds, ds_factor=ds_factor, feature_columns=channels,
      view_feature_sets=view_feature_sets, save_new=False,
      valid_labels=valid_labels)

  return pig_data


def split_ts_into_windows(ts, window_size, ignore_rest=False, shuffle=True):
  # round_func = np.floor if ignore_rest else np.ceil
  n_win = int(np.ceil(ts.shape[0] / window_size))
  split_inds = np.arange(1, n_win).astype(int) * window_size
  split_data = np.split(ts, split_inds, axis=0)
  last_ts = split_data[-1]
  n_overlap = window_size - last_ts.shape[0]
  if n_overlap > 0:
    if ignore_rest:
      split_data = split_data[:-1]
    else:
      last_ts = np.r_[split_data[-2][-n_overlap:], last_ts]
      split_data[-1] = last_ts
  windows = np.array(split_data)

  if shuffle:
    r_inds = np.random.permutation(windows.shape[0])
    windows = windows[r_inds]

  return windows


def aggregate_multipig_data(key_data, window_size=100, n=1000, shuffle=True):
  ignore_rest = False

  nviews = len(key_data[utils.get_any_key(key_data)]["features"])
  data = {i:[] for i in range(nviews)}

  for pnum in key_data:
    vfeats = key_data[pnum]["features"]
    for i, vf in enumerate(vfeats):
      vf_windows = split_ts_into_windows(
          vf, window_size, ignore_rest, shuffle=False)  # Shuffle at the end
      data[i].append(vf_windows)

  for i in data:
    data[i] = np.concatenate(data[i], axis=0)

  if shuffle:
    npts = data[0].shape[0]
    shuffle_inds = np.random.permutation(npts)
    data = {i: data[i][shuffle_inds] for i in data}

  if n > 0:
    data = {i: data[i][:n] for i in data}
  #IPython.embed()

  return data


def rescale(data, noise_std=1e-3):
  mins = {i: data[i].min(axis=0) for i in data}
  maxs = {i: data[i].max(axis=0) for i in data}
  diffs = {i: (maxs[i] - mins[i]) for i in mins}
  diffs = {i: np.where(diff, diff, 1) for i, diff in diffs.items()}

  noise = {
      i: np.random.randn(*dat.shape) * noise_std for i, dat in data.items()
  }
  data = {i: ((data[i] - mins[i]) / diffs[i] + noise[i]) for i in data}
  # IPython.embed()
  return data


def split_data(xvs, n=10, split_inds=None):
  xvs = {vi:np.array(xv) for vi, xv in xvs.items()}
  npts = xvs[utils.get_any_key(xvs)].shape[0]
  if split_inds is None:
    split_inds = np.linspace(0, npts, n + 1).astype(int)
  else:
    split_inds = np.array(split_inds)
  start = split_inds[:-1]
  end = split_inds[1:]

  split_xvs = [
      {vi: xv[idx[0]:idx[1]] for vi, xv in xvs.items()}
      for idx in zip(start, end)
  ]
  return split_xvs, split_inds


def test_ts_encoding(num_pigs=3, channel=0, phase=None):
  channels = None
  pig_data = load_pig_data(num_pigs, channels, valid_labels=[phase])

  window_size = 100
  n = -1
  window_data = aggregate_multipig_data(pig_data, window_size=window_size, n=n)
  npts = window_data[utils.get_any_key(window_data)].shape[0]

  tr_frac = 0.8
  split_inds = [0, int(tr_frac * npts), npts]
  (tr_data, te_data), _ = split_data(window_data, split_inds=split_inds)

  config = default_TSRF_config()
  config.max_iters = 5000
  config.lr = 1e-4

  model = ts_recon_featurization.TimeSeriesReconFeaturization(config)
  tr_data_channel = tr_data[0][:, :, [channel]]
  # IPython.embed()
  model.fit(tr_data_channel)
  IPython.embed()


if __name__ == "__main__":
  import sys
  channel = int(sys.argv[1]) if len(sys.argv) > 1 else 0
  phase = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  num_pigs = 2
  test_ts_encoding(num_pigs, channel, phase)
  # IPython.embed()
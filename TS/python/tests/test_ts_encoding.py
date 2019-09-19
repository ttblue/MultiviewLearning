# Tests for some CCA stuff on pig data.
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import ts_recon_featurization
from synthetic import multimodal_systems as ms
from utils import time_series_utils as tsu, torch_utils as tu, utils


try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


np.set_printoptions(precision=5, suppress=True)


_PIG_DATA_FREQUENCY = 255.
_TAU_IN_S = 0.2
_WINDOW_SIZE_IN_S = 3.
_WINDOW_SPLIT_THRESH_S = 5.

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
  pre_output_size = 64
  rnn_hidden_size = 64
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
  latent_size = 64
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


def load_pig_data(num_pigs=-1, channels=None, ds_factor=25, valid_labels=None):
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


def plot_windows(truevals, output, ndisp, ph_name, ch_name, nwin, wsize):
  tv_plot = truevals.reshape(-1, truevals.shape[-1])
  op_plot = output.reshape(-1, truevals.shape[-1])

  if nwin is not None and nwin > 0:
    ndisp = nwin * wsize
  if ndisp > 0:
    tv_plot = tv_plot[:ndisp]
    op_plot = op_plot[:ndisp]
  ntsteps = tv_plot.shape[0]

  plt.plot(tv_plot, color='b', label="Ground Truth")
  plt.plot(op_plot, color='r', label="Predicted")
  plt.legend(fontsize=15)
  plt.title("Phase: %s -- Vital: %s" % (ph_name, ch_name), fontsize=30)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  # if nwin is not None and nwin > 0:

  win_x = wsize
  while win_x < ntsteps:
    plt.axvline(x = win_x, ls="--")
    win_x += wsize

  mng = plt.get_current_fig_manager()
  mng.window.showMaximized()
  plt.show()


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
  channels = pig_videos.ALL_FEATURE_COLUMNS
  ds_factor = 25

  valid_labels = None if phase is None else [phase]
  pig_data = load_pig_data(
      num_pigs, channels=channels, ds_factor=ds_factor,
      valid_labels=valid_labels)

  noise_coeff = 0.
  data_frequency = int(_PIG_DATA_FREQUENCY / ds_factor)
  window_size = int(_WINDOW_SIZE_IN_S * data_frequency)
  n = -1
  window_tstamps, window_data, window_labels = tsu.convert_data_into_windows(
      pig_data, window_size=window_size, n=n, smooth=True, scale=True,
      noise_std=0)
  npts = window_data[utils.get_any_key(window_data)].shape[0]
  # scaled_data = rescale(window_data, noise_coeff)
  tr_frac = 0.8
  split_inds = [0, int(tr_frac * npts), npts]
  (tr_data, te_data), _ = split_data(window_data, split_inds=split_inds)

  config = default_TSRF_config(use_pre=["encoder"], use_post=["encoder"])
  config.time_delay_tau = int(_TAU_IN_S * data_frequency)
  config.max_iters = 5000
  config.lr = 1e-4

  model = ts_recon_featurization.TimeSeriesReconFeaturization(config)
  tr_data_channel = tr_data[0][:, :, [channel]]
  # IPython.embed()
  model.fit(tr_data_channel)
  IPython.embed()

  # For plotting:
  te_data_channel = te_data[0][:, :, [channel]]
  wsize = te_data_channel.shape[1]
  output = model.decode(model.encode(te_data_channel), output_len=wsize)
  output = output.detach().numpy()
  ndisp = -1
  nwin = 10
  ph_name = pig_videos.PHASE_MAP.get(phase, str(phase))
  ch = channels[channel + 1]  # first channel is tstamp
  ch_name = pig_videos.VS_MAP.get(ch, str(ch))

  plot_windows(te_data_channel, output, ndisp, ph_name, ch_name, nwin, wsize)

  # On the servers:
  # encoding = model.encode(te_data[0][:, :, [channel]])
  # output = model.decode(encoding, te_data[0].shape[1])
  # np.save("out_ph%i_ch%i.npy" % (phase, channel), (te_data[0][:,:,[channel]], output.detach().numpy()))


if __name__ == "__main__":
  import sys
  channel = int(sys.argv[1]) if len(sys.argv) > 1 else 0
  phase = int(sys.argv[2]) if len(sys.argv) > 2 else None
  num_pigs = 2
  test_ts_encoding(num_pigs, channel, phase)
  # IPython.embed()

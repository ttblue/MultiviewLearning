# Tests for some CCA stuff on pig data.
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import ts_recon_featurization, ts_fourier_featurization
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
_WINDOW_SIZE_IN_S = 5.
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


def load_pig_data(num_pigs=-1, ds_factor=25, valid_labels=None):
  pig_list = pig_videos.FFILE_PNUMS
  if num_pigs > 0:
    pig_list = pig_list[:num_pigs]
  channels = pig_videos.ALL_FEATURE_COLUMNS

  ds = 1
  view_feature_sets = None
  pig_data = pig_videos.load_pig_features_and_labels(
      pig_list=pig_list, ds=ds, ds_factor=ds_factor, feature_columns=channels,
      view_feature_sets=view_feature_sets, save_new=False,
      valid_labels=valid_labels)

  return pig_data


def plot_windows(tvals, labels, ndisp, ph_name, ch_name, nwin, wsize, ax=None):
  plot_ts = []
  for win in tvals:
    plot_ts.append(win.reshape(-1, win.shape[-1]))

  if nwin is not None and nwin > 0:
    ndisp = nwin * wsize
  if ndisp > 0:
    plot_ts = [win[:ndisp] for win in plot_ts]

  ntsteps = plot_ts[0].shape[0]

  ax = plt if ax is None else ax

  for win, lbl in zip(plot_ts, labels):
    ax.plot(win, label=lbl)    
  # ax.plot(tv_plot, color='b', label="Ground Truth")
  # ax.plot(op_plot, color='r', label="Predicted")
  try:
    ax.title("Phase: %s -- Vital: %s" % (ph_name, ch_name), fontsize=30)
    ax.xticks(fontsize=15)
    ax.yticks(fontsize=15)
    ax.legend(fontsize=15)
  except TypeError:
    ax.set_title("Phase: %s -- Vital: %s" % (ph_name, ch_name), fontsize=10)
    ax.legend()
    # ax.set_xticks(fontsize=5)
    # ax.set_yticks(fontsize=5)
  # if nwin is not None and nwin > 0:

  win_x = wsize
  while win_x < ntsteps:
    ax.axvline(x = win_x, ls="--")
    win_x += wsize


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


def split_pigs_into_train_test(pig_data, tr_frac=0.8, window_size=100):
  num_pigs = len(pig_data)

  tr_num = int(tr_frac * num_pigs)
  if tr_num == 0:
    tr_num = 1
  if tr_num == num_pigs and num_pigs > 1:
    tr_num -= 1

  pig_keys = list(pig_data.keys())
  np.random.shuffle(pig_keys)

  print("Training pigs: %s" % pig_keys[:tr_num])
  print("Testing pigs: %s" % pig_keys[tr_num:])
  tr_pig_data = {key: pig_data[key] for key in pig_keys[:tr_num]}
  te_pig_data = {key: pig_data[key] for key in pig_keys[tr_num:]}

  noise_coeff = 0.
  n = -1
  tr_all_data = tsu.convert_data_into_windows(
      tr_pig_data, window_size=window_size, n=n, smooth=True, scale=True,
      noise_std=noise_coeff)
  te_all_data = tsu.convert_data_into_windows(
      te_pig_data, window_size=window_size, n=n, smooth=True, scale=True,
      noise_std=0)

  return tr_all_data, te_all_data


def test_ts_encoding(num_pigs=3, channel=0, phase=None):
  ds_factor = 25

  valid_labels = None if phase is None else [phase]
  pig_data = load_pig_data(
      num_pigs, ds_factor=ds_factor, valid_labels=valid_labels)

  # noise_coeff = 0.
  data_frequency = int(_PIG_DATA_FREQUENCY / ds_factor)
  window_size = int(_WINDOW_SIZE_IN_S * data_frequency)
  # n = -1
  # window_tstamps, window_data, window_labels = tsu.convert_data_into_windows(
  #     pig_data, window_size=window_size, n=n, smooth=True, scale=True,
  #     noise_std=0)
  tr_frac = 0.8
  tr_all_data, te_all_data = split_pigs_into_train_test(
      pig_data, tr_frac=tr_frac, window_size=window_size)
  tr_wtstamps, tr_wdata, tr_wlabels = tr_all_data
  te_wtstamps, te_wdata, te_wlabels = te_all_data

  # npts = window_data[utils.get_any_key(window_data)].shape[0]
  # # scaled_data = rescale(window_data, noise_coeff)
  # tr_frac = 0.8
  # split_inds = [0, int(tr_frac * npts), npts]
  # (tr_data, te_data), _ = split_data(window_data, split_inds=split_inds)

  config = default_TSRF_config(use_pre=["encoder"], use_post=["encoder"])
  config.time_delay_tau = int(_TAU_IN_S * data_frequency)
  config.max_iters = 1
  config.lr = 1e-4

  model = ts_recon_featurization.TimeSeriesReconFeaturization(config)
  tr_data_channel = tr_wdata[0][:, :, [channel]]
  # IPython.embed()
  model.fit(tr_data_channel)
  IPython.embed()

  # For plotting:
  te_data_channel = te_wdata[0][:, :, [channel]]
  wsize = te_data_channel.shape[1]
  output = model.decode(model.encode(te_data_channel), output_len=wsize)
  output = output.detach().numpy()
  ndisp = -1
  nwin = 10
  ph_name = pig_videos.PHASE_MAP.get(phase, str(phase))
  ch = pig_videos.ALL_FEATURE_COLUMNS[channel + 1] - 2# first channel is tstamp
  ch_name = pig_videos.VS_MAP.get(ch, str(ch))

  plot_windows(te_data_channel, output, ndisp, ph_name, ch_name, nwin, wsize)


def default_FF_config():
  ndim = 10
  use_imag = True
  config = ts_fourier_featurization.FFConfig(ndim)
  return config


def test_fourier_ts_encoding(num_pigs=3, phase=None):
  ds_factor = 25
  valid_labels = None if phase is None else [phase]
  pig_data = load_pig_data(
      num_pigs, ds_factor=ds_factor, valid_labels=valid_labels)

  data_frequency = int(_PIG_DATA_FREQUENCY / ds_factor)
  window_size = int(_WINDOW_SIZE_IN_S * data_frequency)
  tr_frac = 0.8
  tr_all_data, te_all_data = split_pigs_into_train_test(
      pig_data, tr_frac=tr_frac, window_size=window_size)
  tr_wtstamps, tr_wdata, tr_wlabels = tr_all_data
  te_wtstamps, te_wdata, te_wlabels = te_all_data

  config = default_FF_config()
  config.ndim = 30
  config.use_imag = False

  model = ts_fourier_featurization.TimeSeriesFourierFeaturizer(config)
  tr_data_channel = tr_wdata[0]
  te_data_channel = te_wdata[0]
  # IPython.embed()
  model.fit(tr_data_channel, None)
  IPython.embed()
  ch_models = model.split_channels_into_models()
  dsets = {"Train": tr_data_channel, "Test": te_data_channel}

  # Yes, the model save file extensions are ".fart".
  # No, it does not stand for anything relevant.
  # Please forgive the author for their immature file-naming tendencies.
  save_file_base_name = "fft_feats_ws%i.fart" % window_size
  model_save_file = os.path.join(
      os.getenv("RESEARCH_DIR"), "tests/saved_models", save_file_base_name)
  # For plotting:
  nrows = 2
  ncols = 3

  nwin = 5
  channels = pig_videos.ALL_FEATURE_COLUMNS
  wsize = tr_data_channel.shape[1]
  ndisp = -1
  all_labels = ["Ground Truth", "Real Freq. Recon", "Imag Freq. Recon"]
  ph_name = pig_videos.PHASE_MAP.get(phase, str(phase))

  for dset_used in dsets:
    fig, axs = plt.subplots(nrows, ncols)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt_data_channel = dsets[dset_used]
    for i, ch in enumerate(channels[1:]):
      row, col = i // ncols, (i % ncols)
      ax = axs[row, col]
      true_vals = plt_data_channel[:, :, [i]]
      codes = model.encode(plt_data_channel, use_imag=True)
      recon_tvals_im = model.decode(codes, use_imag=True)[:, :, [i]]
      codes = model.encode(plt_data_channel, use_imag=False)
      recon_tvals_re = model.decode(codes, use_imag=False)[:, :, [i]]

      tvals = [true_vals, recon_tvals_re, recon_tvals_im][:-1]
      labels = all_labels[:-1]
      ch_key = ch - 2
      ch_name = pig_videos.VS_MAP.get(ch_key, str(ch_key))
      plot_windows(tvals, labels, ndisp, ph_name, ch_name, nwin, wsize, ax)
    fig.suptitle("Dataset: %s" % dset_used)

  # On the servers:
  # encoding = model.encode(te_data[0][:, :, [channel]])
  # output = model.decode(encoding, te_data[0].shape[1])
  # np.save("out_ph%i_ch%i.npy" % (phase, channel), (te_data[0][:,:,[channel]], output.detach().numpy()))


if __name__ == "__main__":
  import sys
  expt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  channel = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  phase = int(sys.argv[3]) if len(sys.argv) > 3 else 0
  num_pigs = 10
  if expt == 0:
    test_ts_encoding(num_pigs, channel, phase)
  else:
    test_fourier_ts_encoding(num_pigs, phase)
  # IPython.embed()

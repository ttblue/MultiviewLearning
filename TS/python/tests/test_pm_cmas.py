# Tests for some CCA stuff on pig data.
import itertools
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import predictive_maintenance_datasets
from models import \
    embeddings, greedy_multi_view_rl, greedy_single_view_rl,\
    naive_multi_view_rl, naive_single_view_rl, ovr_mcca_embeddings,\
    robust_multi_ae, torch_models, ts_fourier_featurization
from synthetic import multimodal_systems as ms
from tests.test_greedy_mvrl import default_GMVRL_config
from tests.test_mv_pig_data import \
    plot_heatmap, default_NGSRL_config, default_RMAE_config,\
    aggregate_multipig_data, rescale, split_data, make_subset_list, error_func,\
    all_subset_accuracy
from utils import torch_utils, utils, time_series_utils

try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


# Plotting funcs:
def plot_windows(
    tvals, labels, ndisp, title, nwin, wsize, ax=None, shuffle=True):
  plot_ts = []
  for win in tvals:
    plot_ts.append(win.reshape(-1, win.shape[-1]))

  if nwin is not None and nwin > 0:
    ndisp = nwin * wsize

  if shuffle and ndisp > 0:
    shuffle_inds = np.random.permutation(plot_ts[0].shape[0])[:ndisp]
    plot_ts = [win[shuffle_inds] for win in plot_ts]

  if ndisp > 0:
    plot_ts = [win[:ndisp] for win in plot_ts]

  ntsteps = plot_ts[0].shape[0]

  ax = plt if ax is None else ax

  for win, lbl in zip(plot_ts, labels):
    ax.plot(win, label=lbl)
  # ax.plot(tv_plot, color='b', label="Ground Truth")
  # ax.plot(op_plot, color='r', label="Predicted")
  if title:
    try:
      ax.title(title, fontsize=30)
      ax.xticks(fontsize=15)
      ax.yticks(fontsize=15)
      ax.legend(fontsize=15)
    except TypeError:
      ax.set_title(title, fontsize=10)
      ax.legend()
      # ax.set_xticks(fontsize=5)
      # ax.set_yticks(fontsize=5)
    # if nwin is not None and nwin > 0:

  win_x = wsize
  while win_x < ntsteps:
    ax.axvline(x = win_x, ls="--")
    win_x += wsize


def split_into_windows(data, window_size, shuffle=True):
  ids, ts, feats, ys = data["ids"], data["ts"], data["features"], data["y"]

  window_data = {key: [] for key in data}
  # Go over the t-series of each unit
  idx = 0
  for u_id, u_ts, u_ft, u_y in zip(ids, ts, feats, ys):
    if u_ft.shape[0] < window_size:
      print("Skipping %s. Not enough data for a window." % (u_id,))
      continue
    u_ts_ft = np.c_[u_ts.reshape(-1, 1), u_ft]
    try:
      w_ts_ft = time_series_utils.split_ts_into_windows(
          u_ts_ft, window_size, ignore_rest=False, shuffle=shuffle)
    except Exception as e:
      IPython.embed()
      raise e
    idx += 1
    w_ts, w_ft = w_ts_ft[:, :, 0], w_ts_ft[:, :, 1:]
    n_win = w_ft.shape[0]
    w_ids = [u_id] * n_win
    w_ys = [u_y] * n_win

    window_data["ids"].extend(w_ids)
    window_data["y"].extend(w_ys)
    window_data["ts"].append(w_ts)
    window_data["features"].append(w_ft)

  for key in ["ts", "features"]:
    window_data[key] = np.concatenate(window_data[key], axis=0)

  return window_data


_FFT_DIM = 30
_FFT_MODEL = None
_CH_MODELS = None
# Featurization depends only on training data. So this should be called first
# on training data.
def fft_featurize_data(window_data):
  global _FFT_MODEL, _CH_MODELS

  ts, feat = window_data["ts"], window_data["features"]
  window_size = feat.shape[1]

  if _FFT_MODEL is None:
    fft_model_file = os.path.join(
        os.getenv("RESEARCH_DIR"), "tests/saved_models",
        "cmas_fft_feats_ws%i.fart" % window_size)
    config = ts_fourier_featurization.FFConfig(
        ndim=_FFT_DIM, use_imag=False, verbose=True)
    _FFT_MODEL = ts_fourier_featurization.TimeSeriesFourierFeaturizer(config)
    if os.path.exists(fft_model_file):
      print("Loading fft model.")
      _FFT_MODEL.load_from_file(fft_model_file)
      _FFT_MODEL.config.ndim = _FFT_DIM
    else:
      print("Training and saving fft model.")
      _FFT_MODEL.fit(feat, ts)
      _FFT_MODEL.save_to_file(fft_model_file)

    _CH_MODELS = _FFT_MODEL.split_channels_into_models()

  mv_data = {}
  for i, ch_model in enumerate(_CH_MODELS):
    ch_windows = feat[:, :, [i]]
    mv_data[i] = np.squeeze(ch_model.encode(ch_windows))

  return mv_data


def load_cmas_data(window_size=20):
  normalize = True
  dset_type = "all"
  data, misc = predictive_maintenance_datasets.load_cmas(dset_type, normalize)

  # Just need to make sure fft feats are done on train data first.
  window_data = {}
  fft_window_data = {}
  for dset_type in ["train", "test"]:
    wdata = split_into_windows(data[dset_type], window_size)
    fft_window_data[dset_type] = fft_featurize_data(wdata)
    window_data[dset_type] = wdata

  return window_data, fft_window_data


def reconstruct_ts(codes, output_len):
  output_ts = {}
  for k, ch_code in codes.items():
    output_ts[k] = _CH_MODELS[k].decode(ch_code, output_len)

  return output_ts


def test_nn(args):
  window_size = args.wsize
  npts = args.npts
  win_data, cmas_data = load_cmas_data(window_size=window_size)

  # Fit model.
  tr_w_ffts = cmas_data["train"]
  te_w_ffts = cmas_data["test"]
  tr_wdata = win_data["train"]["features"]
  te_wdata = win_data["test"]["features"]

  dsets = {"Train": tr_wdata, "Test": te_wdata}

  config = default_NGSRL_config(sv_type="nn")
  config.njobs = None if args.njobs == -1 else args.njobs
  if npts > 0:
    tr_w_ffts = {vi: d[:npts] for vi, d in tr_w_ffts.items()}

  # IPython.embed()
  config.single_view_config.lambda_global = 1e-3
  config.single_view_config.lambda_group = 0 # 1e-1
  config.single_view_config.sp_eps = 5e-5
  config.single_view_config.max_iters = args.max_iters

  model = naive_multi_view_rl.NaiveBlockSparseMVRL(config)
  # try:
  model.fit(tr_w_ffts)
  # except Exception as e:
  #   print("Something went wrong with training.")
  #   IPython.embed()
  IPython.embed()
  globals().update(locals())
  vlens = [tr_w_ffts[vi].shape[1] for vi in range(len(tr_w_ffts))]
  msplit_inds = [] # np.cumsum(vlens)[:-1]
  view_subset = msplit_names = []

  tr_preds = model.predict(tr_w_ffts)
  te_preds = model.predict(te_w_ffts)

  tr_preds_ts = reconstruct_ts(tr_preds, window_size)
  te_preds_ts = reconstruct_ts(te_preds, window_size)

  dsets_ts = {"Train": tr_wdata, "Test": te_wdata}
  dsets_pred_ts = {"Train": tr_preds_ts, "Test": te_preds_ts}
  # IPython.embed()

  # For plotting:
  globals().update(locals())
  nrows = 6
  ncols = 4

  n_channels = dsets_ts["Train"].shape[2]
  nwin = 5
  wsize = window_size
  ndisp = -1
  labels = ["Ground Truth", "Recon."]
  title = ""
  for dset_used in dsets_ts:
    fig, axs = plt.subplots(nrows, ncols)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt_ts = dsets_ts[dset_used]
    plt_pred_ts = dsets_pred_ts[dset_used]
    for i in range(n_channels):
      if i >= nrows * ncols:
        break
      row, col = i // ncols, (i % ncols)
      if nrows == 1:
        ax = axs[col]
      elif ncols == 1:
        ax = axs[row]
      else:
        ax = axs[row, col]

      tvals = [plt_ts[:, :, [i]], plt_pred_ts[i]]

      plot_windows(tvals, labels, ndisp, title, nwin, wsize, ax)
    fig.suptitle("Dataset: %s" % dset_used)

  # lnum = valid_labels[0]
  # fl = "nmat_lbl_%i_opt.npy" % lnum
  # idx = 0 
  # while os.path.exists(fl):
  #    fl = "nmat_lbl_%i_opt%i.npy" % (lnum, idx)
  #    idx += 1
  # np.save(fl, model.nullspace_matrix()) 

  plot_heatmap(model.nullspace_matrix(), msplit_inds, msplit_names)


def test_rmae(args):
  drop_scale = True
  zero_at_input = True

  window_size = args.wsize
  npts = args.npts
  win_data, cmas_data = load_cmas_data(window_size=window_size)

  # Fit model.
  tr_w_ffts = cmas_data["train"]
  te_w_ffts = cmas_data["test"]
  tr_wdata = win_data["train"]["features"]
  te_wdata = win_data["test"]["features"]

  dsets = {"Train": tr_wdata, "Test": te_wdata}

  if npts > 0:
    tr_w_ffts = {vi: d[:npts] for vi, d in tr_w_ffts.items()}

  # IPython.embed()
  v_sizes = [tr_w_ffts[vi].shape[1] for vi in tr_w_ffts]
  config = default_RMAE_config(v_sizes)
  config.drop_scale = drop_scale
  config.zero_at_input = zero_at_input
  config.max_iters = args.max_iters

  model = robust_multi_ae.RobustMultiAutoEncoder(config)
  # try:
  model.fit(tr_w_ffts)
  # except Exception as e:
  #   print("Something went wrong with training.")
  #   IPython.embed()
  IPython.embed()
  globals().update(locals())
  vlens = [tr_w_ffts[vi].shape[1] for vi in range(len(tr_w_ffts))]
  msplit_inds = np.cumsum(vlens)[:-1]
  view_subset = msplit_names = []

  tr_preds = model.predict(tr_w_ffts)
  te_preds = model.predict(te_w_ffts)

  tr_preds_ts = reconstruct_ts(tr_preds, window_size)
  te_preds_ts = reconstruct_ts(te_preds, window_size)

  dsets_ts = {"Train": tr_wdata, "Test": te_wdata}
  dsets_pred_ts = {"Train": tr_preds_ts, "Test": te_preds_ts}
  # IPython.embed()

  # For plotting:
  globals().update(locals())
  nrows = 6
  ncols = 4

  n_channels = len(tr_w_ffts)
  nwin = 5
  wsize = window_size
  ndisp = -1
  labels = ["Ground Truth", "Recon."]
  title = ""
  for dset_used in dsets_ts:
    fig, axs = plt.subplots(nrows, ncols)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt_ts = dsets_ts[dset_used]
    plt_pred_ts = dsets_pred_ts[dset_used]
    for i in range(n_channels):
      if i >= nrows * ncols:
        break
      row, col = i // ncols, (i % ncols)
      if nrows == 1:
        ax = axs[col]
      elif ncols == 1:
        ax = axs[row]
      else:
        ax = axs[row, col]

      tvals = [plt_ts[:, :, [i]], plt_pred_ts[i]]

      plot_windows(tvals, labels, ndisp, title, nwin, wsize, ax)
    fig.suptitle("Dataset: %s" % dset_used)

  # lnum = valid_labels[0]
  # fl = "nmat_lbl_%i_opt.npy" % lnum
  # idx = 0 
  # while os.path.exists(fl):
  #    fl = "nmat_lbl_%i_opt%i.npy" % (lnum, idx)
  #    idx += 1
  # np.save(fl, model.nullspace_matrix()) 

  # plot_heatmap(model.nullspace_matrix(), msplit_inds, msplit_names)


def test_greedy(args):
  window_size = args.wsize
  npts = args.npts
  win_data, cmas_data = load_cmas_data(window_size=window_size)

  # Fit model.
  tr_w_ffts = cmas_data["train"]
  te_w_ffts = cmas_data["test"]
  tr_wdata = win_data["train"]["features"]
  te_wdata = win_data["test"]["features"]
  # globals().update(locals())
  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}
  config = default_GMVRL_config(sv_type="nn")
  config.single_view_config.lambda_reg = 1e-2
  config.single_view_config.regularizer = "L1"
  config.single_view_config.max_iters = args.max_iters

  config.parallel = False
  config.single_view_config.parallel = True
  # config.lambda_global = 0  #1e-1
  # config.lambda_group = 0 #0.5  #1e-1
  # config.sp_eps = 5e-5
  # config.n_solves = 1

  model = greedy_multi_view_rl.GreedyMVRL(config)
  IPython.embed()
  model.fit(tr_w_ffts)
  IPython.embed()
  # globals().update(locals())
  vlens = [tr_w_ffts[vi].shape[1] for vi in range(len(data))]
  msplit_inds = np.cumsum(vlens)[:-1]

  # model.compute_projections(ng)
  # projections = model.view_projections
  # for i in projections: 
  #   for j in projections[i]: 
  #     projections[i][j] = projections[i][j].T
  ng = 1
  for ng in range(1, 4):
    plot_heatmap(model.nullspace_matrix(None, ng), msplit_inds, "%i greedy views" % ng)
    plt.show()
  IPython.embed()
  # plot_heatmap(model.nullspace_matrix(), msplit_inds)


_TEST_FUNCS = {
    0: test_nn,
    1: test_rmae,
    2: test_greedy,
}
if __name__ == "__main__":
  np.set_printoptions(linewidth=1000, precision=3, suppress=True)
  torch.set_printoptions(precision=3)
  options = [
      ("npts", int, "Number of points", 1000),
      ("wsize", int, "Number of t-steps per window", 20),
      ("njobs", int, "Number of processes to run in parallel", 3),
      ("max_iters", int, "Number of iterations of training", 1000),
      ]
  args = utils.get_args(options)
  
  func = _TEST_FUNCS.get(args.expt, test_nn)
  func(args)

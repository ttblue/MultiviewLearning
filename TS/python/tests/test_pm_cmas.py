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
  fft_window_data = {}
  for dset_type in ["train", "test"]:
    window_data = split_into_windows(data[dset_type], window_size)
    fft_window_data[dset_type] = fft_featurize_data(window_data)

  return fft_window_data


def reconstruct_ts(codes, output_len):
  output_ts = {}
  for k, ch_code in codes.items():
    output_ts[k] = _CH_MODELS[k].decode(ch_code, output_len)

  return output_ts


def test_nn(args):
  window_size = args.wsize
  npts = args.npts
  cmas_data = load_cmas_data(window_size=window_size)

  # Fit model.
  tr_w_ffts = cmas_data["train"]
  te_w_ffts = cmas_data["test"]

  dsets = {"Train": tr_w_ffts, "Test": te_w_ffts}

  config = default_NGSRL_config(sv_type="nn")
  config.n_jobs = None if args.n_jobs == -1 else args.n_jobs
  if npts > 0:
    tr_w_ffts = {vi: d[:npts] for vi, d in tr_w_ffts.items()}

  # IPython.embed()
  config.single_view_config.lambda_global = 1e-3
  config.single_view_config.lambda_group = 0 # 1e-1
  config.single_view_config.sp_eps = 5e-5
  # config.single_view_config.max_iters = 1

  model = naive_multi_view_rl.NaiveBlockSparseMVRL(config)
  # try:
  model.fit(tr_w_ffts)
  # except Exception as e:
  #   print("Something went wrong with training.")
  #   IPython.embed()
  IPython.embed()
  vlens = [tr_w_ffts[vi].shape[1] for vi in range(len(tr_w_ffts))]
  msplit_inds = np.cumsum(vlens)[:-1]
  view_subset = [1, 2, 3, 4, 5, 6, 10] #None
  msplit_names = [pig_videos.VS_MAP[vidx] for vidx in view_subset]
  IPython.embed()

  tr_preds = model.predict(tr_w_ffts)
  te_preds = model.predict(te_w_ffts)

  tr_preds_ts = reconstruct_ts(tr_preds, window_size)
  te_preds_ts = reconstruct_ts(te_preds, window_size)

  dsets_ts = {"Train": tr_wdata[0], "Test": te_wdata[0]}
  dsets_pred_ts = {"Train": tr_preds_ts, "Test": te_preds_ts}


  # For plotting:
  nrows = 2
  ncols = 3

  nwin = 5
  channels = pig_videos.ALL_FEATURE_COLUMNS
  wsize = window_size
  ndisp = -1
  labels = ["Ground Truth", "Recon."]

  for dset_used in dsets_ts:
    fig, axs = plt.subplots(nrows, ncols)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt_ts = dsets_ts[dset_used]
    plt_pred_ts = dsets_pred_ts[dset_used]
    for i, ch in enumerate(channels[1:]):
      row, col = i // ncols, (i % ncols)
      ax = axs[row, col]

      tvals = [plt_ts[:, :, [i]], plt_pred_ts[i]]

      ch_key = ch - 2
      ch_name = pig_videos.VS_MAP.get(ch_key, str(ch_key))
      plot_windows(tvals, labels, ndisp, ph_name, ch_name, nwin, wsize, ax)
    fig.suptitle("Dataset: %s" % dset_used)

  # lnum = valid_labels[0]
  # fl = "nmat_lbl_%i_opt.npy" % lnum
  # idx = 0 
  # while os.path.exists(fl):
  #    fl = "nmat_lbl_%i_opt%i.npy" % (lnum, idx)
  #    idx += 1
  # np.save(fl, model.nullspace_matrix()) 

  plot_heatmap(model.nullspace_matrix(), msplit_inds, msplit_names)


def test_vitals_only_rmae(num_pigs=-1, npts=1000, phase=None):
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

  tr_w_ffts = fft_featurize_pig_data(tr_wdata[0])
  te_w_ffts = fft_featurize_pig_data(te_wdata[0])

  drop_scale = True
  zero_at_input = True
  v_sizes = [tr_w_ffts[vi].shape[1] for vi in tr_w_ffts]
  config = default_RMAE_config(v_sizes)

  config.drop_scale = drop_scale
  config.zero_at_input = zero_at_input
  config.max_iters = 10000

  # IPython.embed()
  model = robust_multi_ae.RobustMultiAutoEncoder(config)
  model.fit(tr_w_ffts)
  # vlens = [data[vi].shape[1] for vi in range(len(data))]
  # msplit_inds = np.cumsum(vlens)[:-1]
  # preds = model.predict(te_w_ffts)
  # pred_ts = reconstruct_ts(preds, output_len=window_size)
  IPython.embed()
  vlens = [tr_w_ffts[vi].shape[1] for vi in range(len(tr_w_ffts))]
  msplit_inds = np.cumsum(vlens)[:-1]
  view_subset = [1, 2, 3, 4, 5, 6, 10] #None
  msplit_names = [pig_videos.VS_MAP[vidx] for vidx in view_subset]

  tr_preds = model.predict(tr_w_ffts)
  te_preds = model.predict(te_w_ffts)

  tr_preds_ts = reconstruct_ts(tr_preds, window_size)
  te_preds_ts = reconstruct_ts(te_preds, window_size)

  dsets_ts = {"Train": tr_wdata[0], "Test": te_wdata[0]}
  dsets_pred_ts = {"Train": tr_preds_ts, "Test": te_preds_ts}

  # plot_heatmap(model.nullspace_matrix(), msplit_inds


_TEST_FUNCS = {
    0: test_nn,
}
if __name__ == "__main__":
  np.set_printoptions(linewidth=1000, precision=3, suppress=True)
  torch.set_printoptions(precision=3)
  options = [
      ("npts", int, "Number of points", 1000),
      ("wsize", int, "Number of t-steps per window", 20),
      ("n_jobs", int, "Number of processes to run in parallel", 3),
      ]
  args = utils.get_args(options)
  
  func = _TEST_FUNCS.get(args.expt, test_nn)
  func(args)

# Tests for some CCA stuff on pig data.
import itertools
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import \
    embeddings, greedy_multi_view_rl, greedy_single_view_rl,\
    naive_multi_view_rl, naive_single_view_rl, ovr_mcca_embeddings,\
    robust_multi_ae, torch_models, ts_fourier_featurization
from synthetic import multimodal_systems as ms
from utils import torch_utils as tu, utils

# testing utilities from other script
from tests.test_mv_pig_data import \
    plot_heatmap, default_NGSRL_config, default_RMAE_config,\
    aggregate_multipig_data, rescale, split_data, make_subset_list, error_func,\
    all_subset_accuracy
from tests.test_greedy_mvrl import default_GMVRL_config
from tests.test_ts_encoding import \
    default_FFNN_config, default_RNN_config, default_FRFW_config,\
    default_FF_config, default_TSRF_config, load_pig_data, plot_windows,\
    split_pigs_into_train_test

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


_FFT_DIM = 30
_WINDOW_SIZE = 50
_FFT_MODEL = None
_CH_MODELS = None
def fft_featurize_pig_data(window_data):
  global _FFT_MODEL, _CH_MODELS

  if _FFT_MODEL is None:
    fft_model_file = os.path.join(
        os.getenv("RESEARCH_DIR"), "tests/saved_models",
        "fft_feats_ws%i.fart" % _WINDOW_SIZE)
    _FFT_MODEL = ts_fourier_featurization.TimeSeriesFourierFeaturizer(None)
    _FFT_MODEL.load_from_file(fft_model_file)
    _FFT_MODEL.config.ndim = _FFT_DIM
    _CH_MODELS = _FFT_MODEL.split_channels_into_models()

  mv_data = {}
  for i, ch_model in enumerate(_CH_MODELS):
    ch_windows = window_data[:, :, [i]]
    mv_data[i] = np.squeeze(ch_model.encode(ch_windows))

  return mv_data


def reconstruct_ts(codes, output_len):
  output_ts = {}
  for k, ch_code in codes.items():
    output_ts[k] = _CH_MODELS[k].decode(ch_code, output_len)

  return output_ts


def test_vitals_only_nn(num_pigs=-1, npts=1000, phase=None):
  ds_factor = 25
  valid_labels = None if phase is None else [phase]
  pig_data = load_pig_data(
      num_pigs, ds_factor=ds_factor, valid_labels=valid_labels)

  ph_name = pig_videos.PHASE_MAP.get(phase, "all")
  print("Using phase: %s" % ph_name)

  data_frequency = int(_PIG_DATA_FREQUENCY / ds_factor)
  window_size = int(_WINDOW_SIZE_IN_S * data_frequency)
  tr_frac = 0.8
  tr_all_data, te_all_data = split_pigs_into_train_test(
      pig_data, tr_frac=tr_frac, window_size=window_size)
  tr_wtstamps, tr_wdata, tr_wlabels = tr_all_data
  te_wtstamps, te_wdata, te_wlabels = te_all_data

  tr_mv_feats = fft_featurize_pig_data(tr_wdata[0])
  te_mv_feats = fft_featurize_pig_data(te_wdata[0])
  dsets = {"Train": tr_mv_feats, "Test": te_mv_feats}

  config = default_NGSRL_config(sv_type="nn")
  if npts > 0:
    tr_mv_feats = {vi: d[:npts] for vi, d in tr_mv_feats.items()}

  # IPython.embed()
  config.single_view_config.lambda_global = 1e-3
  config.single_view_config.lambda_group = 0 # 1e-1
  config.single_view_config.sp_eps = 5e-5
  # config.single_view_config.max_iters = 1

  model = naive_multi_view_rl.NaiveBlockSparseMVRL(config)
  try:
    model.fit(tr_mv_feats)
  except Exception as e:
    print("Something went wrong with training.")
    IPython.embed()
  IPython.embed()
  vlens = [tr_mv_feats[vi].shape[1] for vi in range(len(tr_mv_feats))]
  msplit_inds = np.cumsum(vlens)[:-1]
  view_subset = [1, 2, 3, 4, 5, 6, 10] #None
  msplit_names = [pig_videos.VS_MAP[vidx] for vidx in view_subset]
  IPython.embed()

  tr_preds = model.predict(tr_mv_feats)
  te_preds = model.predict(te_mv_feats)

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

  tr_mv_feats = fft_featurize_pig_data(tr_wdata[0])
  te_mv_feats = fft_featurize_pig_data(te_wdata[0])

  drop_scale = True
  zero_at_input = True
  v_sizes = [tr_mv_feats[vi].shape[1] for vi in tr_mv_feats]
  config = default_RMAE_config(v_sizes)

  config.drop_scale = drop_scale
  config.zero_at_input = zero_at_input
  config.max_iters = 10000

  # IPython.embed()
  model = robust_multi_ae.RobustMultiAutoEncoder(config)
  model.fit(tr_mv_feats)
  # vlens = [data[vi].shape[1] for vi in range(len(data))]
  # msplit_inds = np.cumsum(vlens)[:-1]
  # preds = model.predict(te_mv_feats)
  # pred_ts = reconstruct_ts(preds, output_len=window_size)
  IPython.embed()
  vlens = [tr_mv_feats[vi].shape[1] for vi in range(len(tr_mv_feats))]
  msplit_inds = np.cumsum(vlens)[:-1]
  view_subset = [1, 2, 3, 4, 5, 6, 10] #None
  msplit_names = [pig_videos.VS_MAP[vidx] for vidx in view_subset]

  tr_preds = model.predict(tr_mv_feats)
  te_preds = model.predict(te_mv_feats)

  tr_preds_ts = reconstruct_ts(tr_preds, window_size)
  te_preds_ts = reconstruct_ts(te_preds, window_size)

  dsets_ts = {"Train": tr_wdata[0], "Test": te_wdata[0]}
  dsets_pred_ts = {"Train": tr_preds_ts, "Test": te_preds_ts}

  # plot_heatmap(model.nullspace_matrix(), msplit_inds


def test_pff_NGSRL_NN(num_pigs=-1, npts=1000, phase=None):
  # fname = "./data/mv_dim_%i_data.npy" % nviews
  # if not os.path.exists(fname):
  #   data, ptfms = default_data(nviews=nviews, ndim=dim)
  #   np.save(fname, [data, ptfms])
  # else:
  #   data, ptfms = np.load(fname)
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

  tr_mv_feats = fft_featurize_pig_data(tr_wdata[0])
  te_mv_feats = fft_featurize_pig_data(te_wdata[0])

  config = default_GMVRL_config(sv_type="nn")

  config.single_view_config.lambda_reg = 1e-2
  config.single_view_config.regularizer = "L1"
  config.single_view_config.max_iters = 200

  # IPython.embed()
  config.parallel = False
  config.n_jobs = 2
  config.single_view_config.parallel = True
  # config.lambda_global = 0  #1e-1
  # config.lambda_group = 0 #0.5  #1e-1
  # config.sp_eps = 5e-5
  # config.n_solves = 1

  model = greedy_multi_view_rl.GreedyMVRL(config)
  model.fit(tr_mv_feats)

  IPython.embed()
  preds = model.predict(te_mv_feats)
  pred_ts = reconstruct_ts(preds, output_len=window_size)
  # plot_heatmap(model.nullspace_matrix(), msplit_inds)


if __name__ == "__main__":
  import sys

  param_num = 1
  expt = int(sys.argv[param_num]) if len(sys.argv) > param_num else 2; param_num += 1
  phase = int(sys.argv[param_num]) if len(sys.argv) > param_num else 0; param_num += 1
  # channel = int(sys.argv[param_num]) if len(sys.argv) > param_num else 0; param_num += 1
  num_pigs = 5
  npts = 1000

  if expt == 0:
    test_vitals_only_nn(num_pigs=num_pigs, npts=npts, phase=phase)
  elif expt == 1:
    test_vitals_only_rmae(num_pigs=num_pigs, npts=npts, phase=phase)
  else:
    test_pff_NGSRL_NN(num_pigs=num_pigs, npts=npts, phase=phase)
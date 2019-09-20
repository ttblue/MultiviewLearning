# Tests for some CCA stuff on pig data.
import itertools
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import \
    embeddings, ovr_mcca_embeddings, naive_multi_view_rl, naive_single_view_rl,\
    robust_multi_ae
from synthetic import multimodal_systems as ms
from utils import torch_utils as tu, utils

# testing utilities from other script
from tests.testing_mv_pig_data import \
    plot_heatmap, default_NGSRL_config, default_RMAE_config,\
    aggregate_multipig_data, rescale, split_data, make_subset_list, error_func,\
    all_subset_accuracy

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


def test_vitals_only_fft_feat_opt(num_pigs=-1, npts=1000, lnun=0):
  pnums = pig_videos.FFILE_PNUMS
  if num_pigs > 0:
    pnums = pnums[:num_pigs]

  config = default_NGSRL_config(sv_type="opt")

  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}

  # IPython.embed()
  config.single_view_config.lambda_global = 1e-3
  config.single_view_config.lambda_group = 0 # 1e-1
  config.single_view_config.sp_eps = 5e-5

  # config.solve_joint = False

  config.single_view_config.n_solves = 1#5
  config.single_view_config.lambda_group_init = 1e-5
  config.single_view_config.lambda_group_beta = 3

  config.single_view_config.resolve_change_thresh = 0.05
  config.single_view_config.n_resolve_attempts = 15

  config.parallel = True
  config.n_jobs = 4
  # config.lambda_global = 0  #1e-1
  # config.lambda_group = 0 #0.5  #1e-1
  # config.sp_eps = 5e-5
  # config.n_solves = 1

  model = naive_multi_view_rl.NaiveBlockSparseMVRL(config)
  model.fit(data)

  vlens = [data[vi].shape[1] for vi in range(len(data))]
  msplit_inds = np.cumsum(vlens)[:-1]
  msplit_names = [pig_videos.VS_MAP[vidx] for vidx in view_subset]
  IPython.embed()

  lnum = valid_labels[0]
  fl = "nmat_lbl_%i_opt.npy" % lnum 
  idx = 0 
  while os.path.exists(fl):
     fl = "nmat_lbl_%i_opt%i.npy" % (lnum, idx)
     idx += 1
  np.save(fl, model.nullspace_matrix()) 

  plot_heatmap(model.nullspace_matrix(), msplit_inds, msplit_names)
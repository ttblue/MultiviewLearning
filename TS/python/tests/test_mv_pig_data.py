# Tests for some CCA stuff on pig data.
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import embeddings, ovr_mcca_embeddings, naive_multi_view_rl, \
                   naive_single_view_rl
from synthetic import multimodal_systems as ms
from utils import torch_utils as tu, utils


try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


np.set_printoptions(precision=5, suppress=True)


def plot_heatmap(mat, msplit_inds=[], misc_title=""):
  fig = plt.figure()
  hm = plt.imshow(mat)
  plt.title("Redundancy Matrix: %s" % misc_title)
  cbar = plt.colorbar(hm)
  for mind in msplit_inds:
    mind -= 0.5
    plt.axvline(x=mind, ls="--")
    plt.axhline(y=mind, ls="--")
  plt.show(block=True)


def compute_cov(fls, ds_factor=25):
  nf = len(fls)
  x2 = 0.
  mu = 0.
  n = 0
  std = 0.

  print("Initial computations:")
  for i, fl in enumerate(fls):
    print("%i/%i" % (i + 1, nf), end='\r')
    x = np.load(fl)[::ds_factor, 1:]
    mu += x.sum(0)
    x2 += (x ** 2).sum(0)
    n += x.shape[0]

    del x

  mu /= n
  x2 /= n
  print("%i/%i" % (i + 1, nf))
  std = np.sqrt(x2 - mu **2)
  print(std)

  # IPython.embed()

  xtx = 0.
  print("Second pass:")
  for i, fl in enumerate(fls):
    print("%i/%i" % (i + 1, nf), end='\r')
    x = (np.load(fl)[::ds_factor, 1:] - mu) / std
    xtx += x.T.dot(x)

    del x
  print("%i/%i" % (i + 1, nf))

  xtx /= n

  return xtx


def default_NGSRL_config(sv_type="opt", as_dict=False):
  sp_eps = 1e-5
  verbose = True

  group_regularizer = "inf"
  global_regularizer = "L1"
  lambda_group = 1e-1
  lambda_global = 1e-1

  if sv_type == "opt":
    n_solves = 5
    lambda_group_init = 1e-5
    lambda_group_beta = 10

    resolve_change_thresh = 0.05
    n_resolve_attempts = 3

    single_view_config = naive_single_view_rl.SVOSConfig(
        group_regularizer=group_regularizer,
        global_regularizer=global_regularizer, lambda_group=lambda_group,
        lambda_global=lambda_global, n_solves=n_solves,
        lambda_group_init=lambda_group_init,
        lambda_group_beta=lambda_group_beta,
        resolve_change_thresh=resolve_change_thresh,
        n_resolve_attempts=n_resolve_attempts, sp_eps=sp_eps, verbose=verbose)
    if as_dict: single_view_config = single_view_config.__dict__
  elif sv_type == "nn":
    input_size = 10  # Computed online
    # Default Encoder config:
    output_size = 10  # Computed online
    layer_units = []  #[32] # [32, 64]
    use_vae = False
    activation = nn.ReLU  # nn.functional.relu
    last_activation = tu.Identity  #nn.Sigmoid  # functional.sigmoid
    # layer_types = None
    # layer_args = None
    bias = False
    layer_types, layer_args = tu.generate_linear_types_args(
          input_size, layer_units, output_size, bias)
    nn_config = tu.MNNConfig(
        input_size=input_size, output_size=output_size, layer_types=layer_types,
        layer_args=layer_args, activation=activation,
        last_activation=last_activation, use_vae=use_vae)

    batch_size = 32
    lr = 1e-3
    max_iters = 1000
    single_view_config = naive_single_view_rl.SVNNSConfig(
      nn_config=nn_config, group_regularizer=group_regularizer,
        global_regularizer=global_regularizer, lambda_group=lambda_group,
        lambda_global=lambda_global, batch_size=batch_size, lr=lr,
        max_iters=max_iters)
  else:
    raise ValueError("Single view type %s not implemented." % sv_type)

  single_view_solver_type = sv_type

  # solve_joint = False
  parallel = True
  n_jobs = None

  config = naive_multi_view_rl.NBSMVRLConfig(
      single_view_solver_type=single_view_solver_type,
      single_view_config=single_view_config, parallel=parallel, n_jobs=n_jobs,
      verbose=verbose)

  return config.__dict__ if as_dict else config

# Feature names:
# 0:  time (absolute time w.r.t. start of experiment)
# 1:  x-value (relative time w.r.t. first vital sign reading)
# 2:  EKG
# 3:  Art_pressure_MILLAR
# 4:  Art_pressure_Fluid_Filled
# 5:  Pulmonary_pressure
# 6:  CVP (central venous pressure)
# 7:  Plethysmograph
# 8:  CCO (continuous cardiac output)
# 9:  SVO2 (Mixed venous oxygen saturation)
# 10: SPO2 (peripheral capillary oxygen saturation)
# 11: Airway_pressure
# 12: Vigeleo_SVV (Stroke Volume Variation)

# Using 3, 4, 5, 6, 7, 11

def aggregate_multipig_data(pig_data, shuffle=True):
  nviews = len(pig_data[utils.get_any_key(pig_data)]["features"])
  data = {i:[] for i in range(nviews)}

  for pnum in pig_data:
    vfeats = pig_data[pnum]["features"]
    for i, vf in enumerate(vfeats):
      data[i].append(vfeats[i])

  for i in data:
    data[i] = np.concatenate(daata[i], axis=0)

  if shuffle:
    npts = data[0].shape[0]
    shuffle_inds = np.permutation(npts)
    data = {i: data[i][shuffle_inds] for i in data}

  return data


def test_vitals_only_opt(num_pigs=3):
  pnums = pig_videos.COMMON_PNUMS
  if num_pigs > 0:
    pnums = pnums[:num_pigs]

  ds = 5
  ws = 30
  nfeats = 3
  view_subset = None
  valid_labels = None
  pig_data = pig_videos.load_tdPCA_featurized_slow_pigs(
      pig_list=pnums, ds=ds, ws=ws, nfeats=nfeats, view_subset=view_subset,
      valid_labels=valid_labels)
  data = aggregate_multipig_data(pig_data)

  config = default_NGSRL_config(sv_type="opt")

  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}

  # IPython.embed()
  config.single_view_config.lambda_global = 0 #1e-3
  config.single_view_config.lambda_group = 0 # 1e-1
  config.single_view_config.sp_eps = 5e-5

  # config.solve_joint = False

  config.single_view_config.n_solves = 5
  config.single_view_config.lambda_group_init = 1e-5
  config.single_view_config.lambda_group_beta = 3

  config.single_view_config.resolve_change_thresh = 0.05
  config.single_view_config.n_resolve_attempts = 15

  config.parallel = True
  # config.lambda_global = 0  #1e-1
  # config.lambda_group = 0 #0.5  #1e-1
  # config.sp_eps = 5e-5
  # config.n_solves = 1

  model = naive_multi_view_rl.NaiveBlockSparseMVRL(config)
  model.fit(data)

  vlens = [data[vi].shape[1] for vi in range(len(data))]
  msplit_inds = np.cumsum(vlens)[:-1]
  IPython.embed()
  plot_heatmap(model.nullspace_matrix(), msplit_inds)


if __name__ == "__main__":
  # dirname = "/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/waveform/slow/numpy_arrays"
  # cols = "[0, 3, 4, 5, 6, 7, 11]"
  # ds_factor = 25
  # fls = [os.path.join(dirname, fl) for fl in os.listdir(dirname) if cols in fl]
  # sigma = compute_cov(fls, ds_factor=ds_factor)
  # plot_heatmap(sigma)


  # IPython.embed()
  num_pigs = 3
  test_vitals_only_opt
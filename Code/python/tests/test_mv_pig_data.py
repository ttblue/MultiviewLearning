# Tests for some CCA stuff on pig data.
import itertools
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import embeddings, ovr_mcca_embeddings, naive_multi_view_rl, \
                   naive_single_view_rl, robust_multi_ae, torch_models
from synthetic import multimodal_systems as ms
from utils import torch_utils as tu, utils


try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


np.set_printoptions(precision=5, suppress=True)


def plot_heatmap(mat, msplit_inds=[], msplit_names=[], misc_title=""):
  fig = plt.figure()
  hm = plt.imshow(mat)
  plt.title("Redundancy Matrix: %s" % misc_title)
  cbar = plt.colorbar(hm)
  for mind in msplit_inds:
    mind -= 0.5
    plt.axvline(x=mind, ls="--")
    plt.axhline(y=mind, ls="--")
  if msplit_names:
    n_names = len(msplit_names)
    endpts = np.r_[0, msplit_inds, mat.shape[0]]
    midpts = (endpts[:-1] + endpts[1:]) / 2
    plt.xticks(midpts[:n_names], msplit_names, rotation=45)
    plt.yticks(midpts[:n_names], msplit_names, rotation=45)
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

  if sv_type in ["opt", "naive_opt"]:
    sv_type = "naive_opt"
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
  elif sv_type in ["nn", "naive_nn"]:
    sv_type = "naive_nn"
    input_size = 10  # Computed online
    # Default Encoder config:
    output_size = 10  # Computed online
    layer_units = [32, 64]
    use_vae = False
    activation = nn.ReLU  # nn.functional.relu
    last_activation = torch_models.Identity  #nn.Sigmoid  # functional.sigmoid
    dropout_p = 0.
    # layer_types = None
    # layer_args = None
    bias = False
    layer_types, layer_args = tu.generate_linear_types_args(
          input_size, layer_units, output_size, bias)
    nn_config = torch_models.MNNConfig(
        input_size=input_size, output_size=output_size, layer_types=layer_types,
        layer_args=layer_args, activation=activation, dropout_p=dropout_p,
        last_activation=last_activation, use_vae=use_vae)

    batch_size = 32
    lr = 1e-4
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


def default_RMAE_config(v_sizes):
  n_views = len(v_sizes)
  hidden_size = 16
  joint_code_size = 32

  dropout_p = 0.0

  # Default Encoder config:
  output_size = hidden_size
  layer_units = [32] # [32, 64]
  use_vae = False
  activation = nn.ReLU  # nn.functional.relu
  last_activation = nn.Sigmoid  # functional.sigmoid
  encoder_params = {}
  for i in range(n_views):
    input_size = v_sizes[i]
    layer_types, layer_args = tu.generate_linear_types_args(
        input_size, layer_units, output_size)
    encoder_params[i] = torch_models.MNNConfig(
        input_size=input_size, output_size=output_size, layer_types=layer_types,
        layer_args=layer_args, activation=activation,
        last_activation=last_activation, use_vae=use_vae, dropout_p=dropout_p)

  input_size = joint_code_size
  layer_units = [32]  #[64, 32]
  use_vae = False
  last_activation = torch_models.Identity
  decoder_params = {}
  for i in range(n_views):
    output_size = v_sizes[i]
    layer_types, layer_args = tu.generate_linear_types_args(
        input_size, layer_units, output_size)
    decoder_params[i] = torch_models.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, use_vae=use_vae, dropout_p=dropout_p)

  input_size = hidden_size * len(v_sizes)
  output_size = joint_code_size
  layer_units = [64]  #[64, 64]
  layer_types, layer_args = tu.generate_linear_types_args(
      input_size, layer_units, output_size)
  use_vae = False
  joint_coder_params = torch_models.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, use_vae=use_vae, dropout_p=dropout_p)

  drop_scale = True
  zero_at_input = True

  code_sample_noise_var = 0.
  max_iters = 1000
  batch_size = 50
  lr = 1e-3
  verbose = True
  config = robust_multi_ae.RMAEConfig(
      joint_coder_params=joint_coder_params, drop_scale=drop_scale,
      zero_at_input=zero_at_input, v_sizes=v_sizes, code_size=joint_code_size,
      encoder_params=encoder_params, decoder_params=decoder_params,
      code_sample_noise_var=code_sample_noise_var, max_iters=max_iters,
      batch_size=batch_size, lr=lr, verbose=verbose)

  return config

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

def aggregate_multipig_data(key_data, shuffle=True, n=1000):
  nviews = len(key_data[utils.get_any_key(key_data)]["features"])
  data = {i:[] for i in range(nviews)}

  for pnum in key_data:
    vfeats = key_data[pnum]["features"]
    for i, vf in enumerate(vfeats):
      data[i].append(vfeats[i])

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


def test_vitals_only_opt(num_pigs=-1, npts=1000, lnun=0):
  pnums = pig_videos.FFILE_PNUMS
  if num_pigs > 0:
    pnums = pnums[:num_pigs]

  ds = 5
  ws = 30
  nfeats = 6
  view_subset = [1, 2, 3, 4, 5, 6, 10] #None
  valid_labels = [lnum]
  pig_data = pig_videos.load_tdPCA_featurized_slow_pigs(
      pig_list=pnums, ds=ds, ws=ws, nfeats=nfeats, view_subset=view_subset,
      valid_labels=valid_labels)
  data = aggregate_multipig_data(pig_data, n=npts)
  data = rescale(data)

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


def make_subset_list(nviews):
    view_subsets = []
    view_range = list(range(nviews))
    for nv in view_range:
      view_subsets.extend(list(itertools.combinations(view_range, nv + 1)))


def error_func(true_data, pred):
  return np.sum([np.linalg.norm(true_data[vi] - pred[vi]) for vi in pred])


def all_subset_accuracy(model, data, max_subs=1000):
  n_views = len(data)
  view_range = np.arange(n_views)
  all_errors = {}
  subset_errors = {}
  for nv in view_range:
    stime = time.time()
    s_error = []
    subsets = list(itertools.combinations(view_range, nv + 1))
    random_inds = np.random.permutation(len(subsets))[:max_subs]
    for rind in random_inds:
      subset = subsets[rind]
      input_data = {vi:data[vi] for vi in subset}
      pred = model.predict(input_data)
      err = error_func(data, pred)
      s_error.append(err)
      all_errors[subset] = err
    subset_errors[(nv + 1)] = np.mean(s_error)
    print("%i/%i in %.2fs" % (nv + 1, n_views, time.time() - stime), end="\r")
  return subset_errors, all_errors


def test_vitals_only_rmae(
    num_pigs=-1, npts=-1, lnum=0, drop_scale=True, zero_at_input=True):
  pnums = pig_videos.FFILE_PNUMS
  if num_pigs > 0:
    pnums = pnums[:num_pigs]

  ds = 5
  ws = 30
  nfeats = 6
  view_subset = [1, 2, 3, 4, 5, 6, 10] #None
  valid_labels = [lnum]
  pig_data = pig_videos.load_tdPCA_featurized_slow_pigs(
      pig_list=pnums, ds=ds, ws=ws, nfeats=nfeats, view_subset=view_subset,
      valid_labels=valid_labels)
  data = aggregate_multipig_data(pig_data, n=npts)
  data = rescale(data)
  v_sizes = [data[vi].shape[1] for vi in data]
  config = default_RMAE_config(v_sizes)

  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}
  tr_frac = 0.8
  split_inds = [0, int(tr_frac * npts), npts]
  (tr_data, te_data), _ = split_data(data, split_inds=split_inds)

  config.drop_scale = drop_scale
  config.zero_at_input = zero_at_input
  config.max_iters = 10000

  # IPython.embed()
  model = robust_multi_ae.RobustMultiAutoEncoder(config)
  model.fit(tr_data)
  # vlens = [data[vi].shape[1] for vi in range(len(data))]
  # msplit_inds = np.cumsum(vlens)[:-1]
  IPython.embed()

  # plot_heatmap(model.nullspace_matrix(), msplit_inds


def test_video_vitals_rmae(
    num_pigs=-1, npts=-1, phase="EndBaseline", vs_ftype="vs_orig",
    drop_scale=True, zero_at_input=True):
  pnums = pig_videos.COMMON_PNUMS
  if num_pigs > 0:
    pnums = pnums[:num_pigs]

  if vs_ftype == "tdPCA":
    ds = 5
    ws = 30
    nfeats = 6
    view_subset = [1, 2, 3, 4, 5, 6, 10] #None
    valid_labels = None
    f_kwargs = {
        "ds": ds, "ws": ws, "nfeats": nfeats, "view_subset": view_subset,
        "valid_labels": valid_labels}
  elif vs_ftype == "vs_orig":
    raise NotImplementedError("Fix this.")
    # ds=5, ds_factor=10,
    # feature_columns=ALL_FEATURE_COLUMNS, view_feature_sets=None,
    # save_new=False, valid_labels=None
  phases = [phase]
  mv_dset = pig_videos.load_synced_vidfeat_data(num_pigs, phases, f_kwargs)
  all_data = {}
  for key, dset in mv_dset.items():
    for i, f in dset.items():
      if i not in all_data: all_data[i] = []
      all_data[i].extend(f)
  all_data = {i: np.array(dset) for i, dset in all_data.items()}
  IPython.embed()
  data = rescale(all_data)
  v_sizes = [data[vi].shape[1] for vi in data]
  config = default_RMAE_config(v_sizes)

  if npts > 0:
    data = {vi: d[:npts] for vi, d in data.items()}
  else: npts = data[0].shape[0]

  tr_frac = 0.8
  split_inds = [0, int(tr_frac * npts), npts]
  (tr_data, te_data), _ = split_data(data, split_inds=split_inds)

  config.drop_scale = drop_scale
  config.zero_at_input = zero_at_input
  config.max_iters = 10000

  # IPython.embed()
  model = robust_multi_ae.RobustMultiAutoEncoder(config)
  model.fit(tr_data)
  # vlens = [data[vi].shape[1] for vi in range(len(data))]
  # msplit_inds = np.cumsum(vlens)[:-1]
  IPython.embed()

  # plot_heatmap(model.nullspace_matrix(), msplit_inds


if __name__ == "__main__":
  import sys
  enum = int(sys.argv[1]) if len(sys.argv) > 1 else 2
  lnum = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  # dirname = "/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/waveform/slow/numpy_arrays"
  # cols = "[0, 3, 4, 5, 6, 7, 11]"
  # ds_factor = 25
  # fls = [os.path.join(dirname, fl) for fl in os.listdir(dirname) if cols in fl]
  # sigma = compute_cov(fls, ds_factor=ds_factor)
  # plot_heatmap(sigma)

  # IPython.embed()
  ph_name = pig_videos.PHASE_MAP[lnum]
  num_pigs = -1
  npts = 5000
  if enum == 0:
    test_vitals_only_opt(num_pigs=num_pigs, npts=npts, lnum=lnum)
  elif enum == 1:
    test_vitals_only_rmae(num_pigs=num_pigs, npts=npts, lnum=lnum)
  else:
    test_video_vitals_rmae(num_pigs=num_pigs, npts=npts, phase=ph_name)

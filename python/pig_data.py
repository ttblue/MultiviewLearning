from __future__ import print_function, division

import glob
import sys
import os
import time

import multiprocessing

import numpy as np

import dataset
import lstm
import multi_task_learning as mtl
# import time_series_ml as tsml
import time_series_utils as tsu
import utils

import IPython

FREQUENCY = 250
VERBOSE = tsu.VERBOSE
DATA_DIR = os.getenv("PIG_DATA_DIR")
SAVE_DIR = os.getenv("PIG_FEATURES_DIR")

np.set_printoptions(suppress=True, precision=3)

mm_rff = None

def save_pigdata_window_rff(args):
  data_file = args["data_file"]
  features_file = args["features_file"]
  time_channel = args["time_channel"]
  ts_channels = args["ts_channels"]
  channel_taus = args["channel_taus"]
  downsample = args["downsample"]
  window_length_s = args["window_length_s"]
  num_windows = args["num_windows"]
  d_lag = args["d_lag"]
  d_features = args["d_features"]

  if VERBOSE:
    t_start = time.time()
    print("Pig %s."%(os.path.basename(data_file).split('.')[0]))
    print("Loading data.")
  _, data = utils.load_csv(data_file)
    
  mc_ts = data[::downsample, ts_channels]
  tstamps = data[::downsample, time_channel]

  # Parameters for features
  window_length = int(FREQUENCY*window_length_s/downsample)

  mcts_rff, window_tstamps = tsu.compute_multichannel_timeseries_window_only(
      mc_ts, tstamps, channel_taus=channel_taus, mm_rff=mm_rff,
      window_length=window_length, num_windows=num_windows, d_lag=d_lag,
      d_features=d_features)

  if VERBOSE:
    print("Saving features.")
  save_data = {"features": mcts_rff, "tstamps": window_tstamps}
  np.save(features_file, save_data)

  if VERBOSE:
    print("Time taken for pig: %.2f"%(time.time() - t_start))


def save_window_rff_slow_pigs(num_pigs=-1, ws=30, parallel=False, num_workers=5):
  global mm_rff

  time_channel = 0
  ts_channels = range(2, 13)
  downsample = 5
  window_length_s = ws
  num_windows = None
  d_lag = 3
  d_features = 1000
  bandwidth = 0.5

  data_dir = os.path.join(DATA_DIR, "extracted/waveform/slow")
  save_dir = os.path.join(SAVE_DIR, "waveform/slow/window_rff/")
  params_dir = os.path.join(SAVE_DIR, "waveform/slow/")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  mmrff_file = os.path.join(params_dir, "mmrff_di_%i_do_%i_bw_%.3f"%(d_lag, d_features, bandwidth))
  mm_rff = tsu.mm_rbf_fourierfeatures(d_lag, d_features, bandwidth, mmrff_file)

  suffix = "_window_rff_ds_%i_ws_%i"%(downsample, window_length_s)
  data_files, features_files = utils.create_data_feature_filenames(
      data_dir, save_dir, suffix)

  # Not re-computing the stuff already computed.
  channel_taus = None
  # taus_file = os.path.join(params_dir, "taus_ds_%i_ws_%i.npy"%(downsample, window_length_s))
  taus_file = os.path.join(params_dir, "taus_ds_%i.npy"%(downsample))
  channel_taus = np.load(taus_file).tolist()["taus"]

  already_finished = [os.path.exists(ffile + ".npy") for ffile in features_files]
  restart = any(already_finished)

  if restart:
    if VERBOSE:
      print("Already finished pigs: %s"%(
                [int(os.path.basename(data_files[i]).split('.')[0])
                 for i in range(len(already_finished))
                 if already_finished[i]]))

    not_finished = [not finished for finished in already_finished]
    data_files = [data_files[i] for i in xrange(len(data_files)) if not_finished[i]]
    features_files = [features_files[i] for i in xrange(len(features_files)) if not_finished[i]]

  if num_pigs > 0:
    data_files = data_files[:num_pigs]
    features_files = features_files[:num_pigs]

  IPython.embed()

  if parallel:
    all_args = [{
        "data_file": data_file,
        "features_file": features_file,
        "time_channel": time_channel,
        "ts_channels": ts_channels,
        "channel_taus": channel_taus,
        "downsample": downsample,
        "window_length_s": window_length_s,
        "num_windows": num_windows,
        "d_lag": d_lag,
        "d_features": d_features,
      } for data_file, features_file in zip(data_files, features_files)]

    pl = multiprocessing.Pool(num_workers)
    pl.map(save_pigdata_window_rff, all_args)

  else:
    for data_file, features_file in zip(data_files, features_files):
      args = {
          "data_file": data_file,
          "features_file": features_file,
          "time_channel": time_channel,
          "ts_channels": ts_channels,
          "channel_taus": channel_taus,
          "downsample": downsample,
          "window_length_s": window_length_s,
          "num_windows": num_windows,
          "d_lag": d_lag,
          "d_features": d_features,
        } 
      channel_taus = save_pigdata_window_rff(args)

  print("DONE")

################################################################################

def save_window_basis_slow_pigs(num_training=30, ds=5, ws=30):
  num_from_each_pig = 300
  d_reduced = 10

  features_dir = os.path.join(SAVE_DIR, "waveform/slow/window_rff/")
  fdict = utils.create_number_dict_from_files(
      features_dir, wild_card_str="*_window_rff_ds_%i_ws_%i.npy"%(ds, ws))

  if VERBOSE:
    print("Loading random fourier features from the pigs.")

  training_ids = fdict.keys()
  if num_training > 0:
    np.random.shuffle(training_ids)

  num_channels = None
  d_features = None
  channel_features = None
  num_selected = 0
  basis_pigs = []

  for key in training_ids:
    fdata = np.load(fdict[key]).tolist()
    features = fdata["features"]
    num_windows = features[0].shape[0]

    if num_windows < num_from_each_pig:
      if VERBOSE:
        print("\tPig %i does not have enough data."%key)
      continue

    if VERBOSE:
      print("\tAdding random windows from pig %i."%key, end='\r')
      sys.stdout.flush()

    if channel_features is None:
      num_channels = len(features)
      d_features = features[0].shape[1]
      channel_features = {i:np.empty((0, d_features)) for i in xrange(num_channels)}

    rand_inds = np.random.permutation(num_windows)[:num_from_each_pig]

    for i in xrange(num_channels):
      channel_features[i] = np.r_[channel_features[i], features[i][rand_inds]]

    basis_pigs.append(key)
    if num_training > 0:
      num_selected += 1
      if num_selected >= num_training:
        break

  if VERBOSE:
    print("\tAdding random windows from pig %i."%key)
    print("Pigs used for basis computation: %s"%(basis_pigs))
    print("Computing basis:")

  IPython.embed()
  basis = {}
  for channel in channel_features:
    if VERBOSE:
      print("\tChannel %i."%channel, end='\r')
      sys.stdout.flush()
    basis[channel] = tsu.compute_window_PCA(channel_features[i], d_reduced)

  if VERBOSE:
    print("\tChannel %i."%channel)
    print("Saving basis.")

  IPython.embed()

  basis_file = os.path.join(features_dir, "window_basis_ds_%i_ws_%i.npy"%(ds, ws))
  np.save(basis_file, [basis[i] for i in xrange(num_channels)])
  training_ids_file = os.path.join(features_dir, "training_ids_ds_%i_ws_%i.npy"%(ds, ws))
  np.save(training_ids_file, sorted(basis_pigs))


# Global variable for simple access
basis = None


def save_pigdata_features_given_basis(args):
  rff_file = args["rff_file"]
  features_file = args["features_file"]
  d_reduced = args["d_reduced"]

  fdata = np.load(rff_file).tolist()
  features = fdata["features"]
  tstamps = fdata["tstamps"]

  num_channels = len(features)
  d_features = features[0].shape[0]

  mcts_f = []
  for channel in xrange(num_channels):
    if VERBOSE:
      print("Channel:", channel + 1)

    c_f = features[channel]
    c_b = basis[channel][:d_reduced]
    mcts_f.append(c_f.dot(c_b.T))

  save_data = {"features": mcts_f, "tstamps": tstamps}
  np.save(features_file, save_data)


def save_features_slow_pigs_given_basis(num_pigs=-1, ws=30, parallel=False, num_workers=5):
  global basis

  window_length_s = ws
  downsample = 5
  d_reduced = 6
  num_training = 30

  rffeatures_dir = os.path.join(SAVE_DIR, "waveform/slow/window_rff/")
  rffdict = utils.create_number_dict_from_files(
      rffeatures_dir,
      wild_card_str="*_window_rff_ds_%i_ws_%i.npy"%(downsample, window_length_s))

  basis_file = os.path.join(
      rffeatures_dir, "window_basis_ds_%i_ws_%i.npy"%(downsample, window_length_s))
  if not os.path.exists(basis_file):
    save_window_basis_slow_pigs(num_training, downsample, window_length_s)
  basis = np.load(basis_file)

  features_dir = os.path.join(SAVE_DIR, "waveform/slow/")
  suffix = "_features_ds_%i_ws_%i"%(downsample, window_length_s)
  fdict = {key: os.path.join(features_dir, '%i'%key + suffix)
           for key in rffdict}
  already_finished ={key:os.path.exists(fdict[key] + ".npy")
                     for key in fdict}
  restart = any(already_finished.values())

  if restart:
    if VERBOSE:
      print(
          "Already finished pigs: %s"%(
              [key for key in already_finished if already_finished[key]]))
    rffdict = {key:rffdict[key] for key in rffdict if not already_finished[key]}
    fdict = {key:fdict[key] for key in fdict if not already_finished[key]}

  # import IPython
  IPython.embed()

  if num_pigs > 0:
    keys = rffdict.keys()
    rffdict = {key:rffdict[key] for key in keys[:num_pigs]}
    fdict = {key:fdict[key] for key in keys[:num_pigs]}

  if parallel:

    all_args = [{
        "rff_file": rffdict[key],
        "features_file": fdict[key],
        "d_reduced": d_reduced,
      } for key in rffdict]

    pl = multiprocessing.Pool(num_workers)
    pl.map(save_pigdata_features_given_basis, all_args)

  else:
    for key in rffdict:
      args = {
          "rff_file": rffdict[key],
          "features_file": fdict[key],
          "d_reduced": d_reduced,
      }
      save_pigdata_features_given_basis(args)

  print("DONE")

################################################################################

def save_pigdata_features(args):
  data_file = args["data_file"]
  features_file = args["features_file"]
  time_channel = args["time_channel"]
  ts_channels = args["ts_channels"]
  channel_taus = args["channel_taus"]
  downsample = args["downsample"]
  window_length_s = args["window_length_s"]
  tau_range = args["tau_range"]
  num_samples = args["num_samples"]
  num_windows = args["num_windows"]
  d_lag = args["d_lag"]
  d_reduced = args["d_reduced"]
  d_features = args["d_features"]
  bandwidth = args["bandwidth"]

  if VERBOSE:
    t_start = time.time()
    print("Pig %s."%(os.path.basename(data_file).split('.')[0]))
    print("Loading data.")
  _, data = utils.load_csv(data_file)
  
  mc_ts = data[::downsample, ts_channels]
  tstamps = data[::downsample, time_channel]

  # Parameters for features
  tau_range = int(tau_range/downsample)
  window_length = int(FREQUENCY*window_length_s/downsample)

  mcts_f, window_tstamps, channel_taus = tsu.feature_multi_channel_timeseries(
    mc_ts, tstamps, channel_taus=channel_taus, tau_range=tau_range,
    window_length=window_length, num_samples=num_samples,
    num_windows=num_windows, d_lag=d_lag, d_reduced=d_reduced,
    d_features=d_features, bandwidth=bandwidth)

  nan_inds = [np.isnan(c_f).any(1).nonzero()[0].tolist() for c_f in mcts_f]
  invalid_inds = np.unique([i for inds in nan_inds for i in inds])
  if len(invalid_inds) > 0:
    valid_locs = np.ones(window_tstamps.shape[0]).astype(bool)
    valid_locs[invalid_inds] = False
    mcts_f = [c_f[valid_locs] for c_f in mcts_f]
    window_tstamps = window_tstamps[valid_locs]

  if VERBOSE:
    print("Saving features.")
  save_data = {"features": mcts_f, "tstamps": window_tstamps, "taus": channel_taus}
  np.save(features_file, save_data)

  if VERBOSE:
    print("Time taken for pig: %.2f"%(time.time() - t_start))

  return channel_taus


def save_features_slow_pigs(num_pigs=-1, parallel=False, num_workers=5):
  time_channel = 0
  ts_channels = range(2, 13)
  downsample = 5
  window_length_s = 30
  tau_range = 200
  num_samples = 500
  num_windows = None
  d_lag = 3
  d_reduced = 6
  d_features = 1000
  bandwidth = 0.5

  data_dir = os.path.join(DATA_DIR, "extracted/waveform/slow")
  save_dir = os.path.join(SAVE_DIR, "waveform/slow")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  suffix = "_features_ds_%i_ws_%i"%(downsample, window_length_s)
  data_files, features_files = utils.create_data_feature_filenames(
      data_dir, save_dir, suffix)
  # Hack to put pig 33 up in front:
  idx33 = map(os.path.basename, data_files).index("33.csv")
  data_files[0], data_files[idx33] = data_files[idx33], data_files[0]
  features_files[0], features_files[idx33] = features_files[idx33], features_files[0]

  # Not re-computing the stuff already computed.
  channel_taus = None
  taus_file = os.path.join(save_dir, "taus_ds_%i_ws_%i.npy"%(downsample, window_length_s))
  if os.path.exists(taus_file):
    channel_taus = np.load(taus_file).tolist()["taus"]

  already_finished = [os.path.exists(ffile + ".npy") for ffile in features_files]
  restart = any(already_finished)

  if restart:
    if channel_taus is None: 
      first_finished_idx = already_finished.index(True)
      ffile = features_files[first_finished_idx] + ".npy"
      channel_taus = np.load(ffile).tolist()["taus"]

    not_finished = [not finished for finished in already_finished]
    data_files = [data_files[i] for i in xrange(len(data_files)) if not_finished[i]]
    features_files = [features_files[i] for i in xrange(len(features_files)) if not_finished[i]]

  # import IPython
  IPython.embed()

  if num_pigs > 0:
    data_files = data_files[:num_pigs]
    features_files = features_files[:num_pigs]

  if parallel:
    # First one is done separately to calculate taus
    if channel_taus is None:
      data_file, features_file = data_files[0], features_files[0]
      args = {
          "data_file": data_file,
          "features_file": features_file,
          "time_channel": time_channel,
          "ts_channels": ts_channels,
          "channel_taus": channel_taus,
          "downsample": downsample,
          "window_length_s": window_length_s,
          "tau_range": tau_range,
          "num_samples": num_samples,
          "num_windows": num_windows,
          "d_lag": d_lag,
          "d_reduced": d_reduced,
          "d_features": d_features,
          "bandwidth": bandwidth,
      }
      channel_taus = save_pigdata_features(args)
      data_files = data_files[1:]
      features_files = features_files[1:]

    all_args = [{
        "data_file": data_file,
        "features_file": features_file,
        "time_channel": time_channel,
        "ts_channels": ts_channels,
        "channel_taus": channel_taus,
        "downsample": downsample,
        "window_length_s": window_length_s,
        "tau_range": tau_range,
        "num_samples": num_samples,
        "num_windows": num_windows,
        "d_lag": d_lag,
        "d_reduced": d_reduced,
        "d_features": d_features,
        "bandwidth": bandwidth,
      } for data_file, features_file in zip(data_files, features_files)]

    pl = multiprocessing.Pool(num_workers)
    pl.map(save_pigdata_features, all_args)

    print("DONE")

  else:
    for data_file, features_file in zip(data_files, features_files):
      args = {
          "data_file": data_file,
          "features_file": features_file,
          "time_channel": time_channel,
          "ts_channels": ts_channels,
          "channel_taus": channel_taus,
          "downsample": downsample,
          "window_length_s": window_length_s,
          "tau_range": tau_range,
          "num_samples": num_samples,
          "num_windows": num_windows,
          "d_lag": d_lag,
          "d_reduced": d_reduced,
          "d_features": d_features,
          "bandwidth": bandwidth,
      }
      channel_taus = save_pigdata_features(args)

################################################################################

def compute_tau_means(features_dir, ds=5, ws=30):
  ffiles = glob.glob(os.path.join(features_dir, "*_ds_%i_ws_%i.npy"%(ds, ws)))
  all_taus = []
  for ffile in ffiles:
    channel_taus = np.load(ffile).tolist()["taus"]
    all_taus.append(channel_taus)
  mean_taus = np.round(np.mean(all_taus, axis=0)).astype(int)
  taus_file = os.path.join(features_dir, "taus_ds_%i_ws_%i.npy"%(ds, ws))
  np.save(taus_file, {"taus":mean_taus})

################################################################################

def numpy_convert_func(args):
  utils.convert_csv_to_np(
      args["data_file"], args["out_file"], downsample=args["downsample"],
      columns=args["columns"])


def save_pigs_as_numpy_arrays(num_pigs=-1, ds=1, parallel=False, num_workers=5):
  data_dir = os.path.join(DATA_DIR, "extracted/waveform/slow")
  save_dir = os.path.join(SAVE_DIR, "waveform/slow/numpy_arrays/")

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  columns = [0, 3, 4, 5, 6, 7, 11]
  # columns = [0, 7]
  columns = sorted(columns)
  suffix = "_numpy_ds_%i_cols_%s"%(ds, columns)
  data_files, out_files = utils.create_data_feature_filenames(
      data_dir, save_dir, suffix, extension=".csv")

  already_finished = [os.path.exists(ofile + ".npy") for ofile in out_files]
  restart = any(already_finished)

  if restart:
    if VERBOSE:
      print("Already finished pigs: %s"%(
                [int(os.path.basename(data_files[i]).split('.')[0])
                 for i in range(len(already_finished))
                 if already_finished[i]]))

    not_finished = [not finished for finished in already_finished]
    data_files = [data_files[i] for i in xrange(len(data_files)) if not_finished[i]]
    out_files = [out_files[i] for i in xrange(len(out_files)) if not_finished[i]]

  if num_pigs > 0:
    data_files = data_files[:num_pigs]
    out_files = out_files[:num_pigs]

  import IPython
  IPython.embed()

  if parallel:
    all_args = [{
        "data_file": data_file,
        "out_file": out_file,
        "downsample": downsample,
        "columns": columns}
        for data_file, out_file in zip(data_files, out_files)]

    pl = multiprocessing.Pool(num_workers)
    pl.map(numpy_convert_func, all_args)

  else:
    for data_file, out_file in zip(data_files, out_files):
      utils.convert_csv_to_np(
          data_file, out_file, downsample=ds, columns=columns)

################################################################################

def create_label_timeline(labels):  
  label_dict = {i:lbl for i,lbl in enumerate(labels)}
  label_dict[len(labels)] = labels[-1]

  return label_dict


def convert_tstamps_to_labels(tstamps, critical_times, label_dict, window_length_s):
  mean_tstamps = [t + window_length_s / 2.0 for t in tstamps]
  segment_inds = np.searchsorted(critical_times, mean_tstamps)
  labels = np.array([label_dict[idx] for idx in segment_inds])

  return labels


def load_slow_pig_features_and_labels(num_pigs=-1, ds=5, ws=30, category="both"):
  features_dir = os.path.join(SAVE_DIR, "waveform/slow")
  rfeatures_dir = os.path.join(SAVE_DIR, "waveform/slow/window_rff")
  ann_dir = os.path.join(DATA_DIR, "raw/annotation/slow")
  
  fdict = utils.create_number_dict_from_files(
      features_dir, wild_card_str="*_ds_%i_ws_%i.npy"%(ds, ws))
  adict = utils.create_number_dict_from_files(ann_dir, wild_card_str="*.xlsx")

  common_keys = np.intersect1d(fdict.keys(), adict.keys()).tolist()

  unused_pigs = []
  unused_pigs.extend([p for p in fdict if p not in common_keys])
  unused_pigs.extend([p for p in adict if p not in common_keys])
  if VERBOSE:
    print("Not using pigs %s. Either annotations or data missing."%(unused_pigs))

  pig_ids = common_keys
  if category in ["train", "test"]:
    training_ids_file = os.path.join(rfeatures_dir, "training_ids_ds_%i_ws_%i.npy"%(ds, ws))
    training_ids = np.load(training_ids_file)
    if category == "train":
      pig_ids = [idx for idx in pig_ids if idx in training_ids]
    else:
      pig_ids = [idx for idx in pig_ids if idx not in training_ids]

  if num_pigs > 0:
    np.random.shuffle(pig_ids)

  all_data = {}
  curr_unused_pigs = len(unused_pigs)
  num_selected = 0
  for key in pig_ids:
    pig_data = np.load(fdict[key]).tolist()
    tstamps = pig_data["tstamps"]
    features = pig_data["features"]
    
    ann_time, ann_text = utils.load_xlsx_annotation_file(adict[key])
    # if key == 37:
    #   critical_anns, ann_labels = utils.create_annotation_labels(ann_text, True)
    # else:
    critical_anns, ann_labels = utils.create_annotation_labels(ann_text, False)
    critical_times = [ann_time[idx] for idx in critical_anns]
    critical_text = {idx:ann_text[idx] for idx in critical_anns}
    label_dict = create_label_timeline(ann_labels)
    labels = convert_tstamps_to_labels(tstamps, critical_times, label_dict, ws)

    valid_inds = (labels != -1)
    features = [c_f[valid_inds, :] for c_f in features]
    labels = labels[valid_inds]

    lvals, counts = np.unique(labels, False, False, True)

    # Debugging and data-quality checks:
    # print(counts, counts.astype(float)/counts.sum())
    # if (counts.astype(float)/counts.sum() < 0.05).any(): IPython.embed()
    # if (labels == 0).sum() > 70: IPython.embed()
    # if len(counts) < 6: IPython.embed()
    # if (labels == 0).sum() < 50: IPython.embed()
    # if key == 18: IPython.embed()

    # Something weird happened with the data:
    # 1. Too label types are missing
    # 2. The stabilization period is too small
    if len(lvals) < 5:
      if VERBOSE:
        print("Not using pig %i. Missing data from some phases."%key)
      unused_pigs.append(key)
      continue
    if counts[0] < 10:
      if VERBOSE:
        print("Not using pig %i. Stabilization period is too small."%key)
      unused_pigs.append(key)
      continue

    all_data[key] = {"features": features, "labels": labels, "ann_text":critical_text}
    if num_pigs > 0:
      num_selected += 1
      if num_selected >= num_pigs:
        break
  new_unused_pigs = len(unused_pigs) - curr_unused_pigs
  if num_pigs > 0:
    print("Not using pigs %s. Already have enough."%(pig_ids[num_selected+new_unused_pigs:]))
  unused_pigs.extend(pig_ids[num_selected+new_unused_pigs:])

  return all_data, unused_pigs

################################################################################

def rbf_fourierfeatures(d_in, d_out, a):
  # Returns a function handle to compute random fourier features.
  W = np.random.normal(0., 1., (d_in, d_out))
  h = np.random.uniform(0., 2 * np.pi, (1, d_out))
  def rbf_ff(x):
    ff = np.cos((1 / a) * x.dot(W) + h) / np.sqrt(d_out) * np.sqrt(2)
    return ff
  return rbf_ff


def mt_krc_pigs_slow():
  # For now:
  # np.random.seed(1)

  all_data, _ = load_slow_pig_features_and_labels(num_pigs=-1, ds=5, ws=30)

  pig_ids = all_data.keys()
  num_train, num_test = 1, 1
  # rand_inds = np.random.permutation(len(pig_ids))[:(num_train + num_test)]
  # train_inds = [pig_ids[i] for i in rand_inds[:num_train]]
  # test_inds = [pig_ids[i] for i in rand_inds[num_train:]]
  train_inds = [33]
  test_inds = [34]
  print("Train inds: %s\nTest inds: %s"%(train_inds, test_inds))

  binary_class = 1
  training_tasks = []
  channels = len(all_data[train_inds[0]]["features"])
  labels = np.array([y for idx in train_inds for y in all_data[idx]["labels"]])
  labels = (labels == binary_class).astype(int)
  for channel in xrange(channels):
    features = np.vstack([all_data[idx]["features"][channel] for idx in train_inds])
    training_tasks.append((features, labels))

  test_tasks = []
  for channel in xrange(channels):
    features = np.vstack([all_data[idx]["features"][channel] for idx in test_inds])
    test_tasks.append((features, ))

  IPython.embed()
  d_in = training_tasks[0][0].shape[1]
  d_out = 100
  a = 1.
  feature_gen = rbf_fourierfeatures(d_in, d_out, a)
  omega = mtl.create_independent_omega(T=channels, lambda_s=1.)

  pred_y = mtl.mt_krc(training_tasks, test_tasks, omega, feature_gen)
  test_labels = np.array([y for idx in test_inds for y in all_data[idx]["labels"]])

  IPython.embed()

################################################################################

def cluster_slow_pigs(num_pigs=4, ws=30):
  all_data, _ = load_slow_pig_features_and_labels(num_pigs=num_pigs, ds=5, ws=ws, category="both")
  class_names = [
      "Ground_Truth", "EKG", "Art_pressure_MILLAR", "Art_pressure_Fluid_Filled",
      "Pulmonary_pressure", "CVP", "Plethysmograph", "CCO", "SVO2", "SPO2",
      "Airway_pressure", "Vigeleo_SVV"]

  pig_ids = all_data.keys()
  # num_pigs = 4
  # rand_inds = np.random.permutation(len(pig_ids))[:num_pigs]
  # pig_inds = [pig_ids[idx] for idx in rand_inds]
  print("Pig inds: %s"%pig_ids)

  channels = len(all_data[pig_ids[0]]["features"])
  labels = np.array([y for idx in pig_ids for y in all_data[idx]["labels"]])
  all_features = []
  for channel in xrange(channels):
    features = np.vstack([all_data[idx]["features"][channel] for idx in pig_ids])
    all_features.append(features)

  mi_matrix = tsml.cluster_windows(all_features, labels, class_names)

################################################################################

def select_random_windows(data, num_per_window=10, num_per_pig=10, channel=None, pos_label=None):
  windows = {}
  labels = {}

  half_window_len = int(num_per_window/2)
  for idx in data:
    pig_features = data[idx]["features"]
    pig_labels = data[idx]["labels"]

    window_labels = [
        pig_labels[i+half_window_len]
        for i in range(0, pig_labels.shape[0]-num_per_window, num_per_window)
    ]
    if pos_label is None:
      choice_inds = np.random.permutation(len(window_labels))[:num_per_pig]
    else:
      window_labels = (np.array(window_labels) == pos_label).astype(int)
      # choice_inds = np.random.permutation(len(window_labels))[:num_per_pig]
      half_num_per_pig = int(num_per_pig/2)
      pos_inds = np.nonzero(window_labels)[0]
      neg_inds = np.nonzero(window_labels == 0)[0]
      np.random.shuffle(pos_inds)
      np.random.shuffle(neg_inds)
      choice_inds = pos_inds[:half_num_per_pig].tolist()
      choice_inds.extend(neg_inds[:num_per_pig-len(choice_inds)].tolist())

    if channel is None:
      window_split = [
          [f[i:i+num_per_window] for i in range(0, f.shape[0]-num_per_window, num_per_window)]
          for f in pig_features
      ]
      pig_windows = [
          [channel_windows[i] for i in choice_inds]
          for channel_windows in window_split
      ]
    elif isinstance(channel, list):
      window_split = [
          [pig_features[c][i:i+num_per_window] 
           for i in range(0, pig_features[c].shape[0]-num_per_window, num_per_window)]
          for c in channel
      ]
      pig_windows = [
          [channel_windows[i] for i in choice_inds]
          for channel_windows in window_split
      ]
    else:
      window_split = [
          pig_features[channel][i:i+num_per_window] 
          for i in range(0, pig_features[channel].shape[0]-num_per_window, num_per_window)
      ]
      pig_windows = [window_split[i] for i in choice_inds]

    windows[idx] = pig_windows
    labels[idx] = [window_labels[i] for i in choice_inds]

  return windows, labels


def pred_nn_slow_pigs(ws=30):
  # Only using a single channel for now.
  channel = 6
  pos_label = None
  num_per_window = 30
  num_per_pig = 50

  num_train_pigs = -1
  num_test_pigs = -1

  train_data, _ = load_slow_pig_features_and_labels(num_pigs=num_train_pigs, ds=5, ws=ws, category="train")
  test_data, _ = load_slow_pig_features_and_labels(num_pigs=num_test_pigs, ds=5, ws=ws, category="test")

  train_ids = train_data.keys()
  train_ts = [train_data[idx]["features"][channel] for idx in train_ids]
  train_labels = [np.array(train_data[idx]["labels"]) for idx in train_ids]
  if pos_label is not None:
    train_labels = [(lbls == pos_label).astype(int) for lbls in train_labels]

  train_windows, train_window_labels = select_random_windows(
      train_data, num_per_window=num_per_window, num_per_pig=num_per_pig,
      channel=channel, pos_label=pos_label)
  test_windows, test_window_labels = select_random_windows(
      test_data, num_per_window=num_per_window, num_per_pig=num_per_pig,
      channel=channel, pos_label=pos_label)

  # train_pred_labels = {}
  # train_pred_acc = {
  # pig_idx = 1
  # for idx in train_windows:
  #   t1 = time.time()
  #   train_pred_labels[idx] = tsml.predict_nn_dtw(train_ts, train_labels, train_windows[idx])
  #   train_pred_acc[idx] = np.sum(np.array(train_pred_labels[idx]) == np.array(train_window_labels[idx]))/num_per_pig
  #   ttaken = time.time() - t1
  #   print ("Train pig number: %i, ID: %i. Accuracy: %.2f. Time taken: %.2fs"%
  #           (pig_idx, idx, train_pred_acc[idx], ttaken))
  #   pig_idx += 1
  #   if pig_idx > 2:
  #     break
  # final_train_acc = np.mean(train_pred_acc.values())
  # print("Final training accuracy over %i pigs: %.2f"%
  #       (len(train_windows), final_train_acc))

  # print()

  test_pred_labels = {}
  test_pred_acc = {}
  pig_idx = 1
  for idx in test_windows:
    t1 = time.time()
    test_pred_labels[idx] = tsml.predict_nn_dtw(train_ts, train_labels, test_windows[idx])
    test_pred_acc[idx] = np.sum(np.array(test_pred_labels[idx]) == np.array(test_window_labels[idx]))/num_per_pig
    ttaken = time.time() - t1
    print ("Test pig number: %i, ID: %i. Accuracy: %.2f. Time taken: %.2fs"%
           (pig_idx, idx, test_pred_acc[idx], ttaken))
    pig_idx += 1
  final_test_acc = np.mean(test_pred_acc.values())
  print("Final test accuracy over %i pigs: %.2f"%
        (len(test_windows), final_test_acc))
  IPython.embed()
################################################################################

def pred_lstm_slow_pigs(ws=5):
  # Only using a single channel for now.
  channel = 6
  allowed_labels = [0, 1, 2]
  pos_label = None
  if pos_label not in allowed_labels:
    pos_label is None

  num_train_pigs = -1
  num_test_pigs = -1

  train_data, _ = load_slow_pig_features_and_labels(num_pigs=num_train_pigs, ds=5, ws=ws, category="train")
  test_data, _ = load_slow_pig_features_and_labels(num_pigs=num_test_pigs, ds=5, ws=ws, category="test")

  train_ids = train_data.keys()
  train_ts = [train_data[idx]["features"][channel] for idx in train_ids]
  train_labels = [np.array(train_data[idx]["labels"]) for idx in train_ids]
  if allowed_labels is not None:
    valid_inds = [[l in allowed_labels for l in lbls] for lbls in train_labels]
    train_ts = [ts[vi] for ts, vi in zip(train_ts, valid_inds)]
    train_labels = [lbls[vi] for lbls, vi in zip(train_labels, valid_inds)]
  if pos_label is not None:
    train_labels = [(lbls == pos_label).astype(int) for lbls in train_labels]

  test_ids = test_data.keys()
  test_ts = [test_data[idx]["features"][channel] for idx in test_ids]
  test_labels = [np.array(test_data[idx]["labels"]) for idx in test_ids]
  if allowed_labels is not None:
    valid_inds = [[l in allowed_labels for l in lbls] for lbls in test_labels]
    test_ts = [ts[vi] for ts, vi in zip(test_ts, valid_inds)]
    test_labels = [lbls[vi] for lbls, vi in zip(test_labels, valid_inds)]
  if pos_label is not None:
    test_labels = [(lbls == pos_label).astype(int) for lbls in test_labels]

  dset_train = dataset.TimeseriesDataset(train_ts, train_labels)
  dset_test = dataset.TimeseriesDataset(test_ts, test_labels)
  dset_test, dset_validate = dset_test.split([0.8, 0.2])
  # IPython.embed()
  # LSTM Config:
  num_classes = 6 if pos_label is None else 2
  num_features = train_ts[0].shape[1]

  hidden_size = 600
  forget_bias = 0.5
  use_sru = False
  keep_prob = 1.0
  num_layers = 2
  init_scale = 0.1
  max_grad_norm = 5
  max_epochs = 5 ##
  max_max_epochs = 50 ##
  init_lr = 1.0
  lr_decay = 0.9
  batch_size = 20
  num_steps = 10
  verbose = True

  config = lstm.LSTMConfig(
      num_classes=num_classes, num_features=num_features, use_sru=use_sru,
      hidden_size=hidden_size, forget_bias=forget_bias, keep_prob=keep_prob,
      num_layers=num_layers, init_scale=init_scale, max_grad_norm=max_grad_norm,
      max_epochs=max_epochs, max_max_epochs=max_max_epochs, init_lr=init_lr,
      lr_decay=lr_decay, batch_size=batch_size, num_steps=num_steps,
      verbose=verbose)
  lstm_classifier = lstm.LSTM(config)
  lstm_classifier.fit(dset=dset_train, dset_v=dset_validate)
  IPython.embed()


if __name__ == "__main__":
  save_pigs_as_numpy_arrays(num_pigs=1, ds=1, parallel=False, num_workers=5)
  # save_window_rff_slow_pigs(-1, True, 7)
  # save_window_basis_slow_pigs()
  # save_features_slow_pigs_given_basis(-1, True, 7)
  # class_names = [
  #     "Ground_Truth", "EKG", "Art_pressure_MILLAR", "Art_pressure_Fluid_Filled",
  #     "Pulmonary_pressure", "CVP", "Plethysmograph", "CCO", "SVO2", "SPO2",
  #     "Airway_pressure", "Vigeleo_SVV"]
  # tsml.cluster_windows(feature_file)
  # all_data, unused_pigs = load_slow_pig_features_and_labels()
  # IPython.embed()
  # mt_krc_pigs_slow()
  # cluster_slow_pigs(10)
  # pred_nn_slow_pigs(ws=5)
  # pred_lstm_slow_pigs(ws=5)
  # for j in range(1, 11):
  #    cluster_slow_pigs(j)


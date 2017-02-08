from __future__ import print_function, division

import glob
import os
import time

import multiprocessing

import numpy as np

import multi_task_learning as mtl
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


def save_window_rff_slow_pigs(num_pigs=-1, parallel=False, num_workers=5):
  global mm_rff

  time_channel = 0
  ts_channels = range(2, 13)
  downsample = 5
  window_length_s = 30
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
  taus_file = os.path.join(params_dir, "taus_ds_%i_ws_%i.npy"%(downsample, window_length_s))
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
    rand_inds = np.random.permutation(len(training_ids))[:num_training]
    training_ids = [training_ids[idx] for idx in rand_inds]

  if VERBOSE:
    print("Pigs used for basis computation: %s"%(training_ids))

  num_channels = None
  d_features = None
  channel_features = None
  for key in training_ids:
    if VERBOSE:
      print("\tAdding random windows from pig %i."%key, end='\r')

    fdata = np.load(fdict[key]).tolist()
    features = fdata["features"]

    if channel_features is None:
      num_channels = len(features)
      d_features = features[0].shape[1]
      channel_features = {i:np.empty((0, d_features)) for i in xrange(num_channels)}

    num_windows = features[0].shape[0]
    rand_inds = np.random.permutation(num_windows)[:num_from_each_pig]

    for i in xrange(num_channels):
      channel_features[i] = np.r_[channel_features[i], features[i][rand_inds]]

  IPython.embed()
  if VERBOSE:
    print("\tAdding random windows from pig %i."%key)
    print("Computing basis:")

  basis = {}
  for channel in channel_features:
    if VERBOSE:
      print("\tChannel %i."%channel, end='\r')
    basis[channel] = tsu.compute_window_PCA(channel_features[i], d_reduced)

  if VERBOSE:
    print("\tChannel %i."%channel)
    print("Saving basis.")

  basis_file = os.path.join(features_dir, "window_basis_ds_%i_ws_%i.npy"%(ds, ws))
  np.save(basis_file, [basis[i] for i in xrange(num_channels)])
  training_ids_file = os.path.join(features_dir, "training_ids_ds_%i_ws_%i.npy"%(ds, ws))
  np.save(training_ids_file, sorted(training_ids))


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


def save_features_slow_pigs_given_basis(num_pigs=-1, parallel=False, num_workers=5):
  global basis

  window_length_s = 30
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

def create_label_timeline(labels):  
  label_dict = {i:lbl for i,lbl in enumerate(labels)}
  label_dict[len(labels)] = labels[-1]

  return label_dict


def convert_tstamps_to_labels(tstamps, critical_times, label_dict, window_length_s):
  mean_tstamps = [t + window_length_s / 2.0 for t in tstamps]
  segment_inds = np.searchsorted(critical_times, mean_tstamps)
  labels = np.array([label_dict[idx] for idx in segment_inds])

  return labels


def load_slow_pig_features_and_labels(num_pigs=-1, ds=5, ws=30):
  features_dir = os.path.join(SAVE_DIR, "waveform/slow")
  ann_dir = os.path.join(DATA_DIR, "raw/annotation/slow")
  
  fdict = utils.create_number_dict_from_files(
      features_dir, wild_card_str="*_ds_%i_ws_%i.npy"%(ds, ws))
  adict = utils.create_number_dict_from_files(ann_dir, wild_card_str="*.xlsx")

  common_keys = np.intersect1d(fdict.keys(), adict.keys()).tolist()

  unused_pigs = []
  unused_pigs.extend([p for p in fdict if p not in common_keys])
  unused_pigs.extend([p for p in adict if p not in common_keys])
  print("Not using pigs %s. Either annotations or data missing."%(unused_pigs))

  if num_pigs > 0:
    np.random.shuffle(common_keys)    

  all_data = {}
  for key in common_keys[:num_pigs]:
    pig_data = np.load(fdict[key]).tolist()
    tstamps = pig_data["tstamps"]
    features = pig_data["features"]
    
    ann_time, ann_text = utils.load_xlsx_annotation_file(adict[key])
    critical_anns, ann_labels = utils.create_annotation_labels(ann_text)
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
      print("Not using pig %i. Missing data from some phases."%key)
      unused_pigs.append(key)
      continue
    if counts[0] < 10:
      print("Not using pig %i. Stabilization period is too small."%key)
      unused_pigs.append(key)
      continue

    all_data[key] = {"features": features, "labels": labels, "ann_text":critical_text}

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

def cluster_slow_pigs(num_pigs=4):
  all_data, _ = load_slow_pig_features_and_labels(num_pigs=num_pigs, ds=5, ws=30)
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

  mi_matrix = tsu.cluster_windows(all_features, labels, class_names)

################################################################################

if __name__ == "__main__":
  # save_window_rff_slow_pigs(-1, True, 7)
  save_features_slow_pigs_given_basis()
  # class_names = [
  #     "Ground_Truth", "EKG", "Art_pressure_MILLAR", "Art_pressure_Fluid_Filled",
  #     "Pulmonary_pressure", "CVP", "Plethysmograph", "CCO", "SVO2", "SPO2",
  #     "Airway_pressure", "Vigeleo_SVV"]
  # tsu.cluster_windows(feature_file)
  # all_data, unused_pigs = load_slow_pig_features_and_labels()
  # IPython.embed()
  # mt_krc_pigs_slow()
  # for j in range(1, 11):
  #    cluster_slow_pigs(j)

from __future__ import print_function, division

import os
import time
import IPython

import multiprocessing

import numpy as np

import time_series_utils as tsu
import utils

VERBOSE = tsu.VERBOSE
DATA_DIR = os.getenv('PIG_DATA_DIR')
SAVE_DIR = os.getenv('PIG_FEATURES_DIR')


def create_label_timeline(critical_inds, labels):
  
  label_dict = {i:lbl for i,lbl in enumerate(labels)}
  label_dict[len(labels)] = labels[-1]

  return label_dict


def save_pigdata_features(args):
  try:
    data_file = args['data_file']
    features_file = args['features_file']
    time_channel = args['time_channel']
    ts_channels = args['ts_channels']
    channel_taus = args['channel_taus']
    downsample = args['downsample']
    window_length_s = args['window_length_s']
    tau_range = args['tau_range']
    num_samples = args['num_samples']
    num_windows = args['num_windows']
    d_lag = args['d_lag']
    d_reduced = args['d_reduced']
    d_features = args['d_features']
    bandwidth = args['bandwidth']

    if VERBOSE:
      t_start = time.time()
      print('Pig %s.'%(os.path.basename(data_file).split('.')[0]))
      print('Loading data.')
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
      print('Saving features.')
    save_data = {'features': mcts_f, 'tstamps': window_tstamps, 'taus': channel_taus}
    np.save(features_file, save_data)

    if VERBOSE:
      print('Time taken for pig: %.2f'%(time.time() - t_start))

  except Exception as e:
    print(e)

  return channel_taus
  # HACK:
  # Loading a downsampled version, so all the indices need to be adjusted.
  # downsample = 5
  # IPython.embed()
  # Labels are:
  # 0: Stabilization
  # 1: Bleeding
  # 2: Between bleeds
  # 3: Resuscitation
  # 4: Between resuscitations
  # 5: Recovery
  # -1: None
  # ann_file = os.path.join(DATA_DIR, '33_annotation.txt')
  # ann_idx, ann_text = utils.load_annotation_file(ann_file)
  # critical_anns = [2, 7, 8, 13, 18, 19, 26, 27, 30, 35, 38, 39, 43, 46, 47, 48, 51, 52, 60]
  # ann_labels = [-1, 0, -1, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 5]
  # critical_inds = [ann_idx[anni]/downsample for anni in critical_anns]
  # label_dict = create_label_timeline(critical_inds, ann_labels)

  # mean_inds = [t + window_length/2.0 for t in tvals]
  # segment_inds = np.searchsorted(critical_inds, mean_inds)
  # labels = np.array([label_dict[idx] for idx in segment_inds])

  # valid_inds = (labels != -1)
  # mcts_f = [c_f[valid_inds, :] for c_f in mcts_f]
  # labels = labels[valid_inds]
  
  # return mcts_f, labels


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

  data_dir = os.path.join(DATA_DIR, 'waveform/slow')
  save_dir = os.path.join(SAVE_DIR, 'waveform/slow')
  for dr in (data_dir, save_dir):
    if not os.path.exists(dr):
      os.makedirs(dr)
  suffix = '_features_ds_%i_ws_%i'%(downsample, window_length_s)
  data_files, features_files = utils.create_data_feature_filenames(
      data_dir, save_dir, suffix)
  # Hack to put pig 33 up in front:
  idx33 = map(os.path.basename, data_files).index('33.csv')
  data_files[0], data_files[idx33] = data_files[idx33], data_files[0]
  features_files[0], features_files[idx33] = features_files[idx33], features_files[0]

  # Not re-computing the stuff already computed.
  already_finished = [os.path.exists(ffile + '.npy') for ffile in features_files]
  restart = any(already_finished)
  if restart:
    taus_file = os.path.join(save_dir, 'taus_ds_%i_ws_%i'%(downsample, window_length_s))
    if os.path.exists(taus_file):
      channel_taus = np.load(taus_file).tolist()['taus']
    else:
      first_finished_idx = already_finished.index(True)
      ffile = features_files[first_finished_idx] + '.npy'
      channel_taus = np.load(ffile).tolist()['taus']

    not_finished = [not finished for finished in already_finished]
    data_files = [data_files[i] for i in xrange(len(data_files)) if not_finished[i]]
    features_files = [features_files[i] for i in xrange(len(features_files)) if not_finished[i]]

  else:
    channel_taus = None

  # import IPython
  # IPython.embed()

  if num_pigs > 0:
    data_files = data_files[:num_pigs]
    features_files = features_files[:num_pigs]

  if parallel:
    # First one is done separately to calculate taus
    if channel_taus is None:
      data_file, features_file = data_files[0], features_files[0]
      args = {
          'data_file': data_file,
          'features_file': features_file,
          'time_channel': time_channel,
          'ts_channels': ts_channels,
          'channel_taus': channel_taus,
          'downsample': downsample,
          'window_length_s': window_length_s,
          'tau_range': tau_range,
          'num_samples': num_samples,
          'num_windows': num_windows,
          'd_lag': d_lag,
          'd_reduced': d_reduced,
          'd_features': d_features,
          'bandwidth': bandwidth,
      }
      channel_taus = save_pigdata_features(args)
      data_files = data_files[1:]
      features_files = features_files[1:]

    all_args = [{
        'data_file': data_file,
        'features_file': features_file,
        'time_channel': time_channel,
        'ts_channels': ts_channels,
        'channel_taus': channel_taus,
        'downsample': downsample,
        'window_length_s': window_length_s,
        'tau_range': tau_range,
        'num_samples': num_samples,
        'num_windows': num_windows,
        'd_lag': d_lag,
        'd_reduced': d_reduced,
        'd_features': d_features,
        'bandwidth': bandwidth,
      } for data_file, features_file in zip(data_file, features_files)]

    pl = multiprocessing.Pool(num_workers)
    pl.map(save_pigdata_features, all_args)

    print('DONE')

  else:
    for data_file, features_file in zip(data_files, features_files):
      args = {
          'data_file': data_file,
          'features_file': features_file,
          'time_channel': time_channel,
          'ts_channels': ts_channels,
          'channel_taus': channel_taus,
          'downsample': downsample,
          'window_length_s': window_length_s,
          'tau_range': tau_range,
          'num_samples': num_samples,
          'num_windows': num_windows,
          'd_lag': d_lag,
          'd_reduced': d_reduced,
          'd_features': d_features,
          'bandwidth': bandwidth,
      }
      channel_taus = save_pigdata_features(args)


def compute_tau_means(features_dir, ds=5, ws=30):
  ffiles = glob.glob(os.path.join(features_dir, '*_ds_%i_ws_%i.npy'%(ds, ws)))
  all_taus = []
  for ffile in ffiles:
    channel_taus = np.load(ffile).tolist()['taus']
    all_taus.append(channel_taus)
  mean_taus = np.round(np.mean(all_taus, axis=0)).astype(int)
  taus_file = os.path.join(features_dir, 'taus_ds_%i_ws_%i.npy'%(ds, ws))
  np.save(taus_file, {'taus':mean_taus})


def load_slow_pig_features_and_labels(num_pigs=-1):
  features_dir = os.path.join(SAVE_DIR, 'waveform/slow')


if __name__ == '__main__':
  save_features_slow_pigs(-1, True, 5)
  # pass
  # data_file = os.path.join(DATA_DIR, '33.csv')
  # col_names, data = utils.load_csv(data_file)
  # data = np.load('tmp2.npy')
  # mcts_f, labels = create_pigdata33_features_labels(data)
  # # IPython.embed()
  # np.save(save_file, {'features': mcts_f, 'labels': labels})
  # class_names = [
  #     'Ground_Truth', 'EKG', 'Art_pressure_MILLAR', 'Art_pressure_Fluid_Filled',
  #     'Pulmonary_pressure', 'CVP', 'Plethysmograph', 'CCO', 'SVO2', 'SPO2',
  #     'Airway_pressure', 'Vigeleo_SVV']
  # feature_file = os.path.join(DATA_DIR, '33_features.npy')
  # cluster_windows(feature_file)
  save_dir = os.path.join(SAVE_DIR, 'waveform/slow')
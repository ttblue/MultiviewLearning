from __future__ import print_function, division

import glob
import os
import sys
import IPython

import matplotlib.pyplot as plt, matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np, numpy.random as nr, numpy.linalg as nlg

import pandas as pd
from sklearn.cluster import KMeans 

import mutual_info as mi

import plot_utils
import utils

VERBOSE = True
FREQUENCY = 250
DATA_DIR = os.getenv('PIG_DATA_DIR')#'/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/slow'

# ==============================================================================
# 1 - time
# 2 - X Value
# 3 - EKG
# 4 - Art pressure MILLAR
# 5 - Art pressure Fluid Filled
# 6 - Pulmonary pressure
# 7 - CVP
# 8 - Plethysmograph
# 9 - CCO
# 10 - SVO2
# 11 - SPO2
# 12 - Airway pressure
# 13 - Vigeleo_SVV
# ==============================================================================


# ==============================================================================
# Utility functions
# ==============================================================================

def mm_rbf_fourierfeatures(d_in, d_out, a):
  # Returns a function handle to compute random fourier features.
  W = nr.normal(0., 1., (d_in, d_out))
  h = nr.uniform(0., 2*np.pi, (1, d_out))

  def mm_rbf(x):
    xhat = np.mean(np.cos((1/a)*x.dot(W)+h)/np.sqrt(d_out)*np.sqrt(2), axis=0)
    return xhat

  return mm_rbf


def compute_tau(y, M=200, show=True):
  # Computes time lag for ts-features using Mutual Information
  # This time lag is where the shifted function is most dissimilar.
  # y -- time series
  # M -- search iterations

  ysmooth = y
  # ysmooth = pd.rolling_window(y, 5, "triang")
  # ysmooth = ysmooth[~np.isnan(ysmooth)]

  N = ysmooth.shape[0]
  minfo = np.zeros(M)
  for m in xrange(M):
    minfo[m] = mi.mutual_information_2d(ysmooth[:N-m], ysmooth[m:])
    if VERBOSE:
      print('\t\tSearched over %i out of %i values.'%(m+1, M), end='\r')
      sys.stdout.flush()
  if VERBOSE:
    print('\t\tSearched over %i out of %i values.'%(m+1, M))

  tau = np.argmin(minfo)

  if show:
    plt.plot(minfo)
    if VERBOSE:
      print(tau)
    plt.show()

  return tau


def mean_scale_transform(x):
  x = np.array(x)
  x -= x.mean(0)
  x /= np.abs(x).max()
  return x


def compute_window_mean_embdedding(y, tau, mm_rff, f_transform=mean_scale_transform, d=3):
  # Computes the explicit mean embedding in some RKHS space of the TS y
  # tau -- time lag
  # mm_rff -- mean map of random fourier features
  # f_transform -- transformation of time-lagged features before mm_rff
  # D -- dimensionality of time-lagged features

  Ny = y.shape[0]
  if f_transform is None:
    f_transform = lambda v: v

  # TODO: maybe not do this here because you lose edge-of-window information
  # of about (d-1)*tau samples
  x = np.zeros((Ny-(d-1)*tau, 0))
  for i in xrange(d):
    x = np.c_[x, y[i*tau:Ny-(d-i-1)*tau]]

  mm_xhat = mm_rff(f_transform(x))
  return mm_xhat


def compute_window_PCA(Xhat, wdim, evecs=False):
  valid_inds = ~np.isnan(Xhat).any(axis=1)
  Xhat = Xhat[valid_inds]  # remove NaN rows.

  E, V, _ = nlg.svd(Xhat, full_matrices=0)
  v = V[:wdim]**2
  e = E[:, :wdim]

  if evecs:
    return e, v, valid_inds
  else:
    return np.diag(1./v).dot(e.T).dot(Xhat)


def compute_window_features(mm_windows, basis):
  # mm_windows -- n x df mean map features for windows
  # basis: fdim x df basis window embeddings

  mm_windows = np.atleast_2d(mm_windows)
  return mm_windows.dot(basis.T)


def featurize_single_timeseries(
    ts, tau_range=200, window_length=5000, num_samples=100, num_windows=1000,
    d_lag=3, tau=None, d_reduced=6, d_features=1000, bandwidth=0.5):

  if tau is None:
    if VERBOSE:
      print('\tComputing time lag tau.')
    tau = compute_tau(ts, M=tau_range, show=False)
    if VERBOSE:
      print('\tValue of tau: %i'%tau)

  mm_rff = mm_rbf_fourierfeatures(d_lag, d_features, bandwidth)

  if VERBOSE:
    print('\tCreating random windows for finding bases.')
  random_inds = nr.randint(1, ts.shape[0] - window_length, size=(num_samples,))
  Z = np.zeros((num_samples, d_features))
  for i in range(num_samples):
    if VERBOSE:
      print('\t\tRandom window %i out of %i.'%(i+1, num_samples), end='\r')
      sys.stdout.flush()
    window = ts[random_inds[i]:random_inds[i]+window_length+(d_lag-1)*tau]
    Z[i, :] = compute_window_mean_embdedding(window, tau, mm_rff, d=d_lag)
  if VERBOSE:
    print('\t\tRandom window %i out of %i.'%(i+1, num_samples))

  valid_inds = ~np.isnan(Z).any(axis=1)
  valid_locs = np.nonzero(valid_inds)[0]
  Z = Z[valid_inds]  # remove NaN rows.
  num_samples = Z.shape[0]
  
  if VERBOSE:
    print('\tComputing basis.')
  Zhat = compute_window_PCA(Z, d_reduced)

  if VERBOSE:
    print('\tComputing window features.')
  ts_features = np.zeros((num_windows, d_reduced))
  tvals = np.floor(np.linspace(0, ts.shape[0]-window_length, num_windows))
  tvals = tvals.astype(int)
  for i, t in enumerate(tvals):
    if VERBOSE:
      print('\t\tWindow %i out of %i.'%(i+1, num_windows), end='\r')
      sys.stdout.flush()
    window = ts[t:t+window_length+(d_lag-1)*tau]
    window_mm = compute_window_mean_embdedding(window, tau, mm_rff, d=d_lag)
    ts_features[i,:] = window_mm.dot(Zhat.T)
  if VERBOSE:
    print('\t\tWindow %i out of %i.'%(i+1, num_windows))

  return ts_features, tvals, tau


def compute_average_tau(mc_ts, M=200):
  taus = []
  for channel in range(mc_ts.shape[1]):
    if VERBOSE:
      print(channel)
    taus.append(compute_tau(mc_ts[:,channel], M, show=True))

  return int(np.mean(taus))


def feature_multi_channel_timeseries(mc_ts, tstamps, channel_taus=None,
    tau_range=50, window_length=3000, num_samples=100, num_windows=None,
    d_lag=3, d_reduced=6, d_features=1000, bandwidth=0.5):
  
  # tstamps = mc_ts[:, time_channel]
  # mc_ts = mc_ts[:, ts_channels]

  if num_windows is None:
    num_windows = int(mc_ts.shape[0]/window_length)

  # if VERBOSE:
  #   print('Computing average tau.')
  # tau = compute_average_tau(mc_ts, M=tau_range)
  all_taus = [] if channel_taus is None else channel_taus
  if channel_taus is None:
    channel_taus = [None for _ in xrange(mc_ts.shape[1])]

  tvals = None
  mcts_features = []
  for channel in xrange(mc_ts.shape[1]):
    tau = channel_taus[channel]
    if VERBOSE:
      print('Channel:', channel + 1)
    channel_features, channel_tvals, tau_channel = featurize_single_timeseries(
        mc_ts[:, channel], tau_range=tau_range, window_length=window_length,
        num_samples=num_samples, num_windows=num_windows, d_lag=d_lag, tau=tau,
        d_reduced=d_reduced, d_features=d_features, bandwidth=bandwidth)
    if VERBOSE:
      print()

    mcts_features.append(channel_features)
    if tau is not None:
      all_taus.append(tau_channel)
    if tvals is None:
      tvals = channel_tvals

  window_tstamps = tstamps[tvals]
  return mcts_features, window_tstamps, all_taus


def create_label_timeline(critical_inds, labels):
  
  label_dict = {i:lbl for i,lbl in enumerate(labels)}
  label_dict[len(labels)] = labels[-1]

  return label_dict


# # ==============================================================================
# # Putting things together
# # ==============================================================================
def save_pigdata_features(
    data_file, features_file, time_channel=0, ts_channels=range(2, 13),
    channel_taus=None, downsample=1, window_length_s=30, tau_range=200,
    num_samples=500, num_windows=None, d_lag=3, d_reduced=6, d_features=1000,
    bandwidth=0.5):

  _, data = utils.load_csv(data_file)
  
  mc_ts = data[::downsample, ts_channels]
  tstamps = data[::downsample, time_channel]

  # Parameters for features
  tau_range = int(tau_range/downsample)
  window_length = int(FREQUENCY*window_length_s/downsample)

  mcts_f, window_tstamps, channel_taus = feature_multi_channel_timeseries(mc_ts,
    tstamps, tau_range=tau_range, window_length=window_length,
    num_samples=num_samples, num_windows=num_windows, d_lag=d_lag,
    d_reduced=d_reduced, d_features=d_features, bandwidth=bandwidth)

  nan_inds = [np.isnan(c_f).any(1).nonzero()[0].tolist() for c_f in mcts_f]
  invalid_inds = np.unique([i for inds in nan_inds for i in inds])
  valid_locs = np.ones(window_tstamps.shape[0]).astype(bool)
  valid_locs[invalid_inds] = False
  
  mcts_f = [c_f[valid_locs] for c_f in mcts_f]
  window_tstamps = window_tstamps[valid_locs]

  save_data = {'features': mcts_f, 'tstamps': window_tstamps, 'taus': channel_taus}
  np.save(features_file, save_data)

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


def save_features_slow_pigs():
  time_channel = 0
  ts_channels = range(2, 13)
  downsample = 1
  window_length_s = 30
  tau_range = 200
  num_samples = 500
  num_windows = None
  d_lag = 3
  d_reduced = 6
  d_features = 1000
  bandwidth = 0.5

  channel_taus = None

  data_files, features_files = utils.create_data_feature_filenames(
      os.path.join(DATA_DIR, 'waveform/slow'))

  for data_file, features_file in zip(data_files, features_files):
    channel_taus = save_pigdata_features(
        data_file=data_file, features_file=features_file,
        time_channel=time_channel, ts_channels=ts_channels,
        channel_taus=channel_taus, downsample=downsample,
        window_length_s=window_length_s, tau_range=tau_range,
        num_samples=num_samples, num_windows=num_windows, d_lag=d_lag,
        d_reduced=d_reduced, d_features=d_features, bandwidth=bandwidth)


def cluster_windows(feature_file):
  data_dict = np.load(feature_file).tolist()
  mcts_f = data_dict['features']
  labels = data_dict['labels']
  
  num_clusters = np.unique(labels).shape[0]
  num_channels = len(mcts_f)

  nan_inds = [np.isnan(c_f).any(1).nonzero()[0].tolist() for c_f in mcts_f]
  invalid_inds = np.unique([i for inds in nan_inds for i in inds])
  valid_locs = np.ones(labels.shape[0]).astype(bool)
  valid_locs[invalid_inds] = False
  
  mcts_f = [c_f[valid_locs] for c_f in mcts_f]
  labels = labels[valid_locs]

  kmeans = [KMeans(num_clusters) for _ in xrange(num_channels)]
  for km, c_f in zip(kmeans, mcts_f):
    km.fit(c_f)
  all_labels = [labels] + [km.labels_ for km in kmeans]

  mi_matrix = np.zeros((num_channels+1, num_channels+1))
  for i in range(num_channels + 1):
    for j in range(i, num_channels + 1):
      lbl_mi = mi.mutual_information_2d(all_labels[i], all_labels[j])
      mi_matrix[i, j] = mi_matrix[j, i] = lbl_mi

  # max_mi = mi_matrix.max()
  plot_utils.plot_matrix(mi_matrix, class_names)
  plt.show()

  IPython.embed()


if __name__ == '__main__':

  
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
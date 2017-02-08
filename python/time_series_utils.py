from __future__ import print_function, division

import os
import time
import sys
import IPython

import multiprocessing

try:
  import matplotlib.pyplot as plt, matplotlib.cm as cm
  from mpl_toolkits.mplot3d import Axes3D
  import plot_utils
  PLOTTING = True
except ImportError:
  PLOTTING = False

import numpy as np, numpy.random as nr, numpy.linalg as nlg

import pandas as pd
from sklearn.cluster import KMeans 

import mutual_info as mi

import utils

VERBOSE = True
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

def mm_rbf_fourierfeatures(d_in, d_out, a, file_name=None):
  # Returns a function handle to compute random fourier features.
  # Can load from/save to file if one exists.
  if file_name is not None:
    # Some simple checks for filename.
    basename = os.path.basename(file_name)
    if len(basename.split('.')) == 1:
      file_name = file_name + ".npy"

    if os.path.exists(file_name):
      W, h, a = np.load(file_name)
    else:
      W = nr.normal(0., 1., (d_in, d_out))
      h = nr.uniform(0., 2*np.pi, (1, d_out))
      np.save(file_name, [W, h, a])
  else:
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
      print("\t\tSearched over %i out of %i values."%(m+1, M), end='\r')
      sys.stdout.flush()
  if VERBOSE:
    print("\t\tSearched over %i out of %i values."%(m+1, M))

  tau = np.argmin(minfo)

  if show and PLOTTING:
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
  v = V[:wdim] ** 2
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
      print("\tComputing time lag tau.")
    tau = compute_tau(ts, M=tau_range, show=False)
    if VERBOSE:
      print("\tValue of tau: %i"%tau)

  mm_rff = mm_rbf_fourierfeatures(d_lag, d_features, bandwidth)

  if VERBOSE:
    print("\tCreating random windows for finding bases.")
  random_inds = nr.randint(1, ts.shape[0] - window_length, size=(num_samples,))
  Z = np.zeros((num_samples, d_features))
  for i in range(num_samples):
    if VERBOSE:
      print("\t\tRandom window %i out of %i."%(i+1, num_samples), end='\r')
      sys.stdout.flush()
    window = ts[random_inds[i]:random_inds[i]+window_length+(d_lag-1)*tau]
    Z[i, :] = compute_window_mean_embdedding(window, tau, mm_rff, d=d_lag)
  if VERBOSE:
    print("\t\tRandom window %i out of %i."%(i+1, num_samples))

  valid_inds = ~np.isnan(Z).any(axis=1)
  valid_locs = np.nonzero(valid_inds)[0]
  Z = Z[valid_inds]  # remove NaN rows.
  num_samples = Z.shape[0]
  
  if VERBOSE:
    print("\tComputing basis.")
  Zhat = compute_window_PCA(Z, d_reduced)

  if VERBOSE:
    print("\tComputing window features.")
  ts_features = np.zeros((num_windows, d_reduced))
  tvals = np.floor(np.linspace(0, ts.shape[0]-window_length, num_windows))
  tvals = tvals.astype(int)
  for i, t in enumerate(tvals):
    if VERBOSE:
      print("\t\tWindow %i out of %i."%(i+1, num_windows), end='\r')
      sys.stdout.flush()
    window = ts[t:t+window_length+(d_lag-1)*tau]
    window_mm = compute_window_mean_embdedding(window, tau, mm_rff, d=d_lag)
    ts_features[i,:] = window_mm.dot(Zhat.T)
  if VERBOSE:
    print("\t\tWindow %i out of %i."%(i+1, num_windows))

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
  #   print("Computing average tau.")
  # tau = compute_average_tau(mc_ts, M=tau_range)
  all_taus = [] if channel_taus is None else channel_taus
  if channel_taus is None:
    channel_taus = [None for _ in xrange(mc_ts.shape[1])]

  tvals = None
  mcts_features = []
  for channel in xrange(mc_ts.shape[1]):
    tau = channel_taus[channel]
    if VERBOSE:
      print("Channel:", channel + 1)
    channel_features, channel_tvals, tau_channel = featurize_single_timeseries(
        mc_ts[:, channel], tau_range=tau_range, window_length=window_length,
        num_samples=num_samples, num_windows=num_windows, d_lag=d_lag, tau=tau,
        d_reduced=d_reduced, d_features=d_features, bandwidth=bandwidth)
    if VERBOSE:
      print()

    mcts_features.append(channel_features)
    if tau is None:
      all_taus.append(tau_channel)
    if tvals is None:
      tvals = channel_tvals

  window_tstamps = tstamps[tvals]
  return mcts_features, window_tstamps, all_taus


def compute_timeseries_windows_only(
    ts, tvals, tau, mm_rff, window_length, d_lag, d_features):

  if VERBOSE:
    print("\tComputing window RF features.")

  num_windows = tvals.shape[0]
  ts_features = np.zeros((num_windows, d_features))

  for i, t in enumerate(tvals):
    if VERBOSE:
      print("\t\tWindow %i out of %i."%(i+1, num_windows), end='\r')
      sys.stdout.flush()
    window = ts[t:t+window_length+(d_lag-1)*tau]
    ts_features[i,:] = compute_window_mean_embdedding(window, tau, mm_rff, d=d_lag)
  if VERBOSE:
    print("\t\tWindow %i out of %i."%(i+1, num_windows))

  # Remove NaN rows outside.
  valid_inds = (~np.isnan(ts_features).any(axis=1)).nonzero()[0]

  return ts_features, valid_inds


def compute_multichannel_timeseries_window_only(
    mc_ts, tstamps, channel_taus, mm_rff, window_length=5000, num_windows=None,
    d_lag=3, d_features=1000):
  
  if num_windows is None:
    num_windows = int(mc_ts.shape[0] / window_length)

  tvals = np.floor(np.linspace(0, mc_ts.shape[0]-window_length, num_windows))
  tvals = tvals.astype(int)

  all_features = []
  all_valid_inds = []
  for channel in xrange(mc_ts.shape[1]):
    if VERBOSE:
      print("\tChannel: %i"%(channel+1))
    ts_features, valid_inds = compute_timeseries_windows_only(
        mc_ts[:, channel], tvals, channel_taus[channel], mm_rff, window_length,
        d_lag, d_features)
    all_features.append(ts_features)
    all_valid_inds.append(valid_inds)

  final_valid_inds = all_valid_inds[0]
  for valid_inds in all_valid_inds[1:]:
    final_valid_inds = np.intersect1d(final_valid_inds, valid_inds)

  all_features = [f[final_valid_inds] for f in all_features]
  tvals = tvals[final_valid_inds]

  return all_features, tstamps[tvals]


def cluster_windows(features, labels, class_names):
  num_clusters = np.unique(labels).shape[0]
  num_channels = len(features)

  nan_inds = [np.isnan(c_f).any(1).nonzero()[0].tolist() for c_f in features]
  invalid_inds = np.unique([i for inds in nan_inds for i in inds])
  if len(invalid_inds) > 0:
    valid_locs = np.ones(labels.shape[0]).astype(bool)
    valid_locs[invalid_inds] = False
    
    features = [c_f[valid_locs] for c_f in features]
    labels = labels[valid_locs]

  kmeans = [KMeans(num_clusters) for _ in xrange(num_channels)]
  for km, c_f in zip(kmeans, features):
    km.fit(c_f)
  all_labels = [labels] + [km.labels_ for km in kmeans]

  mi_matrix = np.zeros((num_channels + 1, num_channels + 1))
  for i in range(num_channels + 1):
    for j in range(i, num_channels + 1):
      lbl_mi = mi.mutual_information_2d(all_labels[i], all_labels[j])
      mi_matrix[i, j] = mi_matrix[j, i] = lbl_mi

  # max_mi = mi_matrix.max()
  if PLOTTING:
    plot_utils.plot_matrix(mi_matrix, class_names)
    plt.show()

  # IPython.embed()
  return mi_matrix

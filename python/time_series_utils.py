from __future__ import print_function, division

import multiprocessing
import os
import sys
import time

import numpy as np, numpy.random as nr, numpy.linalg as nlg
import pandas as pd
from sklearn.cluster import KMeans
import mutual_info as mi

try:
  import matplotlib.pyplot as plt, matplotlib.cm as cm
  from mpl_toolkits.mplot3d import Axes3D
  import plot_utils
  PLOTTING = True
except ImportError:
  PLOTTING = False

import IPython

from tvregdiff import TVRegDiff
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

################################################################################
# Derivatives stuff:

### UTILITY FUNCTIONS:
def compute_finite_differences(T, X):
  # Computes derivatives at midpoints of T.
  T = np.array(T)
  X = np.array(X)

  DX = X[1:] - X[:-1]
  DT = T[1:] - T[:-1]

  return DX / DT


def compute_total_variation_derivatives(
    X, dt, max_len=20000, overlap=100, max_iter=100, alpha=1e-1, ep=1e-2,
    scale="large", plotting=False, verbose=False):

  X = np.squeeze(X)
  if len(X.shape) > 1:
    raise ValueError("Data must be one dimensional.")

  if X.shape[0] <= max_len:
    return TVRegDiff(
        X, max_iter, alpha, dx=dt, ep=ep, scale=scale,
        plotflag=int(plotting), diagflag=verbose)

  stride = max_len - overlap
  # stride = max_len
  start_inds = np.arange(0, X.shape[0], stride)
  start_inds[-1] = X.shape[0] - stride

  end_inds = start_inds + max_len
  end_inds[-1] = X.shape[0]

  num_inds = len(start_inds)
  DX = None
  # Stitch together derivatives.
  for idx in xrange(num_inds):
    print("\t\tWindow %i."%(idx+1))
    Xsub = X[start_inds[idx]:end_inds[idx]]
    DXsub = TVRegDiff(
        Xsub, max_iter, alpha, dx=dt, ep=ep, scale=scale,
        plotflag=int(plotting), diagflag=verbose)

    if idx == 0:
      DX = DXsub
    else:
      num_overlap = end_inds[idx-1] - start_inds[idx]
      half_overlap = num_overlap // 2
      DX = np.r_[DX[:-half_overlap], DXsub[half_overlap:]]

  return DX


def _single_channel_tvd_helper(args):
  print("Channel: %i"%(args["channel"] + 1))
  return compute_total_variation_derivatives(
    args["X"], args["dt"], args["max_len"], args["overlap"], args["max_iter"],
    args["alpha"], args["ep"], args["scale"], args["plotting"], args["verbose"])  


def compute_multi_channel_tv_derivs(Xs, dt, max_len=20000, overlap=100,
    max_iter=100, alpha=1e-1, ep=1e-2, scale="large", n_jobs=None,
    verbose=False):

  DX = []
  Xs = np.array(Xs)
  if len(Xs.shape) == 1:
    Xs = np.atleast_2d(Xs).T
    n_jobs = None
  if n_jobs is None:
    args = {
        "dt": dt,
        "max_len": max_len,
        "overlap": overlap,
        "max_iter": max_iter,
        "alpha": alpha,
        "ep": ep,
        "scale": scale,
        "plotting": False,
        "verbose": verbose,
    }
    for channel in xrange(Xs.shape[1]):
      args["channel"] = channel
      args["X"] = Xs[:, channel]
      DX.append(_single_channel_tvd_helper(args))

    return np.atleast_2d(DX).T

  n_jobs = Xs.shape[1] if n_jobs == -1 else n_jobs
  all_args = [{
      "channel": channel,
      "X" : Xs[:, channel],
      "dt": dt,
      "max_len": max_len,
      "overlap": overlap,
      "max_iter": max_iter,
      "alpha": alpha,
      "ep": ep,
      "scale": scale,
      "plotting": False,
      "verbose": verbose,
  } for channel in xrange(Xs.shape[1])]

  pl = multiprocessing.Pool(n_jobs)
  DX = pl.map(_single_channel_tvd_helper, all_args)

  return np.atleast_2d(DX).T


if __name__ == "__main__":
  data = np.load('/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/waveform/slow/numpy_arrays/10_numpy_ds_1_cols_[0, 3, 4, 5, 6, 7, 11].npy')
  ds_factor = 10
  data = data[:3000:ds_factor]

  T = data[:, 0]
  X = data[:, 1:]
  dt = np.mean(T[1:] - T[:-1])

  t1 = time.time()
  # DXs1 = compute_multi_channel_tv_derivs(
  #     X, dt, max_len=200, overlap=10, alpha=1e-3, scale="large", max_iter=100, n_jobs=-1)
  DXs2 = compute_multi_channel_tv_derivs(
      X, dt, max_len=200, overlap=10, alpha=1e-2, scale="large", max_iter=100, n_jobs=-1)
  DXs3 = compute_multi_channel_tv_derivs(
      X, dt, max_len=200, overlap=10, alpha=5e-3, scale="large", max_iter=100, n_jobs=-1)
  # DXs4 = compute_multi_channel_tv_derivs(
  #     X, dt, max_len=200, overlap=10, alpha=1, scale="large", max_iter=100, n_jobs=-1)
  # DX = compute_total_variation_derivatives(
  #     X, dt, max_len=2000, overlap=50, scale="large", max_iter=100,
  #     plotting=False)
  print("TIME:", time.time() - t1)

  # import IPython
  # IPython.embed()
  titles = {0: "CVP", 1: "Pleth", 2: "Airway Pressure"}
  import matplotlib.pyplot as plt
  for channel in xrange(X.shape[1]):
    # dxc1 = DXs1[:, channel].cumsum()
    dxc2 = DXs2[:, channel].cumsum()
    dxc3 = DXs3[:, channel].cumsum()
    # dxc4 = DXs4[:, channel].cumsum()

    x0 = X[0, channel]
    # x1 = x0 + dxc1*dt
    x2 = x0 + dxc2*dt
    x3 = x0 + dxc3*dt
    # x4 = x0 + dxc4*dt

    plt.figure()
    plt.plot(X[:, channel], color='b')
    # plt.plot(x1, color='r', label="1e-3")
    # plt.plot(x2, color='g', label="1e-2")
    plt.plot(DXs3[:, channel], color='r')#, label="5e-3")
    # plt.plot(x4, color='k', label="1")
    # plt.plot(dxc)
    plt.legend()
    plt.title(titles[channel])

  # dxc = DX2.squeeze().cumsum()
  # x3 = x0+dxc*dt

  # plt.plot(X, color='b')
  # plt.plot(x2, color='r')
  # plt.plot(x3, color='g')
  plt.show()
  import IPython
  IPython.embed()
  # t1 = time.time()
  # DX = compute_total_variation_derivatives(x[:5000, 0], x[:5000, 0], scale="large", max_iter=5, plotting=False)
  # print("TIME:", time.time() - t1)
  # import cProfile
#   # cProfile.run("DX = compute_total_variation_derivatives(x[:5000, 0], x[:5000, 0], max_iter=2, plotting=False)")

# import matplotlib.pyplot as plt, numpy as np  
# idx = 11
# data = np.load('/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/waveform/slow/numpy_arrays/derivs/%i_derivs_numpy_10_columns_[0, 3, 4, 5, 6, 7, 11].npy'%idx).tolist()
# Xs = data['X']
# T = data['tstamps']
# DXs = data['DX']
# dt = np.mean(T[1:] - T[:-1])

# channel = 2
# x0 = Xs[0, channel]
# dxc = DXs[:, channel].cumsum()
# x2 = x0 + dxc*dt
# plt.plot(Xs[:, channel], color='b')
# plt.plot(x2, color='r')
# plt.show()
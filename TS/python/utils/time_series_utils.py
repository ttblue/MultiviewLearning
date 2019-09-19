import multiprocessing
import numpy as np
import os
import sys
import time

from sklearn.cluster import KMeans

try:
  import mutual_info as mi
  _MI_IMPORTED = True
except ImportError:
  print("Mutual information import failed.")
  _MI_IMPORTED = False


try:
  import matplotlib.pyplot as plt, matplotlib.cm as cm
  from mpl_toolkits.mplot3d import Axes3D
  import plot_utils
  PLOTTING = True
except ImportError:
  PLOTTING = False

import math_utils as mu
from tvregdiff import TVRegDiff
import utils


import IPython


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


def compute_tau(y, M=200, show=True):
  # Computes time lag for ts-features using Mutual Information
  # This time lag is where the shifted function is most dissimilar.
  # y -- time series
  # M -- search iterations
  if not _MI_IMPORTED:
    raise ImportError("Mutual information import failed.")
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


def compute_window_mean_embedding(y, tau, mm_rff, f_transform=mean_scale_transform, d=3):
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

  E, V, _ = np.linalg.svd(Xhat, full_matrices=0)
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

  mm_rff = mu.mm_rbf_fourierfeatures(d_lag, d_features, bandwidth)

  if VERBOSE:
    print("\tCreating random windows for finding bases.")
  random_inds = np.random.randint(
      1, ts.shape[0] - window_length, size=(num_samples,))
  Z = np.zeros((num_samples, d_features))
  for i in range(num_samples):
    if VERBOSE:
      print("\t\tRandom window %i out of %i."%(i+1, num_samples), end='\r')
      sys.stdout.flush()
    window = ts[random_inds[i]:random_inds[i]+window_length+(d_lag-1)*tau]
    Z[i, :] = compute_window_mean_embedding(window, tau, mm_rff, d=d_lag)
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
    window_mm = compute_window_mean_embedding(window, tau, mm_rff, d=d_lag)
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
    ts_features[i,:] = compute_window_mean_embedding(window, tau, mm_rff, d=d_lag)
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
    t1 = time.time()
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
    print("\t\tWindow %i out of %i took %.2fs."%
              (idx + 1, num_inds, time.time() - t1))
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


def compute_time_delay_embedding(X, dt, tau=None, d=3, tau_s_to_search=2.):
  # tau_s_to_search: number of seconds to search over for tau
  # Note -- samples on the edge of the window will be lost
  X = np.squeeze(X)
  if len(X.shape) > 1:
    raise ValueError("Input must be 1-D.")

  if tau is None:
    M = tau_s_to_search // dt
    tau = compute_tau(X, M=M)

  n = X.shape[0]
  X_td = np.empty((n - (d - 1) * tau, 0))
  for i in xrange(d):
    X_td = np.c_[X_td, X[i * tau: n - (d - i - 1) * tau]]

  return X_td


################################################################################
# Data processing stuff


def split_ts_into_windows(ts, window_size, ignore_rest=False, shuffle=True):
  # round_func = np.floor if ignore_rest else np.ceil
  n_win = int(np.ceil(ts.shape[0] / window_size))
  split_inds = np.arange(1, n_win).astype(int) * window_size
  split_data = np.split(ts, split_inds, axis=0)
  last_ts = split_data[-1]
  n_overlap = window_size - last_ts.shape[0]
  if n_overlap > 0:
    if ignore_rest:
      split_data = split_data[:-1]
    else:
      last_ts = np.r_[split_data[-2][-n_overlap:], last_ts]
      split_data[-1] = last_ts
  windows = np.array(split_data)

  if shuffle:
    r_inds = np.random.permutation(windows.shape[0])
    windows = windows[r_inds]

  return windows


def split_discnt_ts_into_windows(
    ts, tstamps, window_size, ignore_rest=False, shuffle=True):
  if len(ts.shape) < 2:
    ts = ts.reshape(-1, 1)
  tstamps_and_ts = np.c_[tstamps.reshape(-1, 1), ts]

  tdiffs = tstamps[1:] - tstamps[:-1]
  gap_inds = (tdiffs > _WINDOW_SPLIT_THRESH_S).nonzero()[0] + 1

  if len(gap_inds) == 0:
    windows = split_ts_into_windows(
        tstamps_and_ts, window_size, ignore_rest, shuffle=shuffle)
    w_tstamps = windows[:, :, 0]
    windows = windows[:, :, 1:]
    return w_tstamps, windows

  windows = []
  w_tstamps = []
  cnt_tstamps_and_ts = np.split(tstamps_and_ts, gap_inds, axis=0)
  for cts in cnt_ts:
    wcts = split_ts_into_windows(cts, window_size, ignore_rest, shuffle=False)
    w_tstamps.append(wcts[:, :, 0])
    windows.append(wcts[:, :, 1:])

  windows = np.concatenate(windows, axis=0)
  w_tstamps = np.concatenate(w_tstamps, axis=0)
  if shuffle:
    r_inds = np.random.permutation(windows.shape[0])
    windows = windows[r_inds]
    w_tstamps = w_tstamps[r_inds]

  return w_tstamps, windows


def wt_avg_smooth(ts, n_neighbors=3):
  if len(ts.shape) > 1:
    individual_smooth = [
        wt_avg_smooth(ts[:, i], n_neighbors).reshape(-1, 1)
        for i in range(ts.shape[1])]
    return np.concatenate(individual_smooth, axis=1)
  box = np.ones(n_neighbors) / n_neighbors
  ts_smooth = np.convolve(ts, box, mode="same")

  return ts_smooth


def smooth_data(ts, tstamps):  #, coeff=0.8):
  tdiffs = tstamps[1:] - tstamps[:-1]
  gap_inds = (tdiffs > _WINDOW_SPLIT_THRESH_S).nonzero()[0] + 1
  if len(gap_inds) == 0:
    cnts_ts = [ts]
  else:
    cnts_ts = np.split(ts, gap_inds, axis=0)

  smooth_ts = []
  n_neighbors = 3
  for cts in cnts_ts:
    smooth_ts.append(wt_avg_smooth(cts, n_neighbors))
  # Put the ts back into the original shape
  smooth_ts = np.concatenate(smooth_ts, axis=0)
  return smooth_ts


_STD_OUTLIERS = 10
def rescale_single_ts(ts, noise_std):
  unwrapped_ts = ts.reshape(-1, ts.shape[-1]) if len(ts.shape) > 2 else ts
  valid_inds = (
      unwrapped_ts - np.mean(unwrapped_ts, axis=0) <
      _STD_OUTLIERS * np.std(unwrapped_ts, axis=0))
  mins = []
  maxs = []
  for (ch_ts, vidx) in zip(unwrapped_ts.T, valid_inds.T):
    mins.append(ch_ts[vidx].min())
    maxs.append(ch_ts[vidx].max())

  mins = np.array(mins)
  maxs = np.array(maxs)
  diffs = maxs - mins
  diffs = np.where(diffs, diffs, 1)

  noise = np.random.randn(*ts.shape) * noise_std
  scaled_ts = (ts - mins) / diffs + noise

  # IPython.embed()
  return scaled_ts


def rescale(data, noise_std=1e-3):
  unwrapped_data = {i: d.reshape(-1, d.shape[2]) for i, d in data.items()}
  mins = {i: d.min(axis=0) for i, d in unwrapped_data.items()}
  maxs = {i: d.max(axis=0) for i, d in unwrapped_data.items()}
  diffs = {i: (maxs[i] - mins[i]) for i in mins}
  diffs = {i: np.where(diff, diff, 1) for i, diff in diffs.items()}

  noise = {
      i: np.random.randn(*dat.shape) * noise_std for i, dat in data.items()
  }
  data = {i: ((data[i] - mins[i]) / diffs[i] + noise[i]) for i in data}
  # IPython.embed()
  return data


def convert_data_into_windows(
    key_data, window_size=100, n=1000, smooth=True, scale=True, noise_std=1e-3,
    shuffle=True):
  ignore_rest = False

  nviews = len(key_data[utils.get_any_key(key_data)]["features"])
  tstamps = {i: [] for i in range(nviews)}
  labels = {i: [] for i in range(nviews)}
  data = {i:[] for i in range(nviews)}

  for pnum in key_data:
    vfeats = key_data[pnum]["features"]
    vtstamps = key_data[pnum]["tstamps"]
    vlabels = key_data[pnum]["labels"]
    for i, vf in enumerate(vfeats):
      # Shuffle at the end
      vl = vlabels[i]
      if scale:
        vf = rescale_single_ts(vf, noise_std)
      if smooth:
        vf = smooth_data(vf, vtstamps)
      # Appending labels:
      vf_with_labels = np.c_[vlabels.reshape(-1, 1), vf]
      vf_tstamps, vf_windows = split_discnt_ts_into_windows(
          vf, vtstamps, window_size, ignore_rest, shuffle=False)
      labels[i].append(vf_windows[:, :, 0])
      data[i].append(vf_windows[:, :, 1:])
      tstamps[i].append(vf_tstamps)

  for i in data:
    data[i] = np.concatenate(data[i], axis=0)
    tstamps[i] = np.concatenate(tstamps[i], axis=0)
    labels[i] = np.concatenate(labels[i], axis=0)

  if shuffle:
    npts = data[0].shape[0]
    shuffle_inds = np.random.permutation(npts)
    data = {i: data[i][shuffle_inds] for i in data}
    tstamps = {i: tstamps[i][shuffle_inds] for i in tstamps}
    labels = {i: labels[i][shuffle_inds] for i in labels}

  if n > 0:
    data = {i: data[i][:n] for i in data}
    tstamps = {i: tstamps[i][:n] for i in tstamps}
    labels = {i: labels[i][:n] for i in labels}

  return tstamps, data, labels


if __name__ == "__main__":
  data = np.load('/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/waveform/slow/numpy_arrays/10_numpy_ds_1_cols_[0, 3, 4, 5, 6, 7, 11].npy')
  ds_factor = 10
  data = data[:500000:ds_factor]

  T = data[:, 0]
  X = data[:, 3]
  dt = np.mean(T[1:] - T[:-1])

  t1 = time.time()
  DXs1 = compute_multi_channel_tv_derivs(
      X, dt, max_len=2000, overlap=10, alpha=5e-3, scale="large", max_iter=100, n_jobs=-1)
  t2 = time.time()
  print("TIME:", t2 - t1)
  DXs2 = compute_multi_channel_tv_derivs(
      X, dt, max_len=5000, overlap=10, alpha=5e-3, scale="large", max_iter=100, n_jobs=1)
  t3 = time.time()
  print("TIME:", t3 - t2)
  DXs3 = compute_multi_channel_tv_derivs(
      X, dt, max_len=20000, overlap=10, alpha=5e-3, scale="large", max_iter=100, n_jobs=1)
  t4 = time.time()
  print("TIME:", t4 - t3)
  # DXs4 = compute_multi_channel_tv_derivs(
  #     X, dt, max_len=200, overlap=10, alpha=1, scale="large", max_iter=100, n_jobs=-1)
  # DX = compute_total_variation_derivatives(
  #     X, dt, max_len=2000, overlap=50, scale="large", max_iter=100,
  #     plotting=False)

  import IPython
  IPython.embed()
  titles = {0: "CVP", 1: "Pleth", 2: "Airway Pressure"}
  import matplotlib.pyplot as plt
  for channel in xrange(X.shape[1]):
    dxc1 = DXs1[:, channel].cumsum()
    dxc2 = DXs2[:, channel].cumsum()
    dxc3 = DXs3[:, channel].cumsum()
    # dxc4 = DXs4[:, channel].cumsum()

    x0 = X[0, channel]
    # x1 = x0 + dxc1*dt
    # x2 = x0 + dxc2*dt
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

  plt.show()

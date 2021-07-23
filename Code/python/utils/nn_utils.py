from __future__ import print_function, division

import gc
import multiprocessing
import sys
import time

import numpy as np
import scipy.spatial as ss
# import _ucrdtw as ucrdtw

try:
  import rpy2.robjects.numpy2ri
  from rpy2.robjects.packages import importr

  # Set up our R namespaces
  rpy2.robjects.numpy2ri.activate()
  R = rpy2.robjects.r
  dtw = importr("dtw")
  _R_UTILS = True
except ImportError:
  _R_UTILS = False

import math_utils as mu

import IPython


VERBOSE = 1


class NNUtilsException(Exception):
  pass


def choose_nn_dtw(target_ts, all_ts, warp_width=0.05):
  if not _R_UTILS:
    raise NNUtilsException("No R utils available. Cannot do DTW.")

  target_ts = np.atleast_2d(target_ts)
  all_ts = [np.array(ts) for ts in all_ts]
  locs = []
  dists = []


  # num_ts = len(all_ts)
  # tidx = 0
  # tstart = time.time()
  for ts in all_ts:
    alignment = R.dtw(target_ts, ts, open_begin=True, open_end=True,
                        step_pattern=dtw.asymmetric)
    # loc, dist = ucrdtw.ucrdtw(ts, target_ts, warp_width)
    index2 = np.squeeze(alignment.rx("index2"))
    loc = int(index2[int(index2.shape[0]/2)]) # Take the middle index.
    dist = float(np.squeeze(alignment.rx("distance")))

    locs.append(loc)
    dists.append(dist)

    # tidx += 1
    # print("Checked %i out of %i ts. Time taken: %.2f"%
    #       (tidx, num_ts, time.time() - tstart), end='\r')
    # sys.stdout.flush()

  best_idx = np.argmin(dists)

  return best_idx, locs[best_idx], dists[best_idx]


def find_loc_dist_euclidean(source_ts, target_ts):
  target_size = target_ts.shape[0]

  dists = []
  for idx in range(source_ts.shape[0] - target_size + 1):
    source_window = source_ts[idx:idx+target_size, :]
    dists.append(np.linalg.norm(source_window - target_ts))
  min_idx = np.argmin(dists) + int(target_size / 2)

  return min_idx, dists[min_idx]


def choose_nn_euclidean(target_ts, all_ts):
  target_ts = np.atleast_2d(target_ts)
  all_ts = [np.array(ts) for ts in all_ts]

  locs = []
  dists = []

  for ts in all_ts:
    loc, dist = find_loc_dist_euclidean(ts, target_ts)
    locs.append(loc)
    dists.append(dist)

  best_idx = np.argmin(dists)

  return best_idx, locs[best_idx], dists[best_idx]


# Currently only uses dot-product similarity.
def one_step_wn_forecast(test_pts, train_window, feature_gen, dr="forward"):
  test_pts = np.atleast_2d(test_pts)
  train_f = feature_gen(train_window)
  test_f = feature_gen(test_pts)

  if dr not in ["forward", "backward", "both"]:
    raise ValueError("Invalid direction %s."%dr)

  preds = []
  W = test_f.dot(train_f.T.dot(np.ones(train_f.shape[0])))
  if dr in ["forward", "both"]:
    wts = (W - test_f.dot(train_f[-1])).reshape(-1, 1)
    preds.append(test_f.dot(train_f[:-1].T.dot(train_window[1:])) / wts)

  if dr in ["backward", "both"]:
    wts = (W - test_f.dot(train_f[0])).reshape(-1, 1)
    preds.append(test_f.dot(train_f[1:].T.dot(train_window[:-1])) / wts)

  # IPython.embed()

  return preds if dr == "both" else preds[0]


# RBF kernel:
def one_step_wn_forecast_RBF(test_pts, train_window, gammak, dr="forward"):
  test_pts = np.atleast_2d(test_pts)
  dist = ss.distance.cdist(test_pts, train_window)
  W = np.exp(-gammak * np.square(dist))
  Ws = np.sum(W, axis=1)

  if dr not in ["forward", "backward", "both"]:
    raise ValueError("Invalid direction %s."%dr)

  preds = []
  if dr in ["forward", "both"]:
    preds.append(W[:, :-1].dot(train_window[1:]) / (Ws - W[:, -1])[:, None])

  if dr in ["backward", "both"]:
    preds.append(W[:, 1:].dot(train_window[1:]) / (Ws - W[:, 0])[:, None])

  # if not np.any(np.isnan(preds[0])):
  #   IPython.embed()
  return preds if dr == "both" else preds[0]


class ForecasterKDTree:
  def __init__(self, train_pts, leaf_size=10, dr="forward"):
    import scipy.spatial as ss
    self.data = train_pts
    if dr in ["forward", "both"]:
      self._kdtree_f = ss.cKDTree(train_pts[:-1], leafsize=leaf_size)
    if dr in ["backward", "both"]:
      self._kdtree_b = ss.cKDTree(train_pts[1:], leafsize=leaf_size)

    # Not being checked now.
    self.allowed_drs = ["forward", "backward"] if dr == "both" else [dr]

  def query(self, test_pts, k, dr="forward"):
    if dr == "forward":
      return self._kdtree_f.query(test_pts, k)
    elif dr == "backward":
      dists, nnidx = self._kdtree_b.query(test_pts, k)
      return dists, nnidx + 1


def one_step_forecast_knn(test_pts, train_kdtree, nn=10, dr="forward"):
  if dr not in ["forward", "backward", "both"]:
    raise ValueError("Invalid direction %s."%dr)

  preds = []
  if dr in ["forward", "both"]:
    nn_dists, nn_idxs = train_kdtree.query(test_pts, k=nn, dr="forward")

    pred_idxs = nn_idxs + 1
    if nn > 1:
      # wts = np.exp(-nn_dists)
      # wts = wts / wts.sum(axis=1)[:, None]
      f_pred = train_kdtree.data[pred_idxs].mean(axis=1)
      # f_pred = (train_kdtree.data[pred_idxs] * wts[:,:, None]).sum(axis=1)
    else:
      f_pred = train_kdtree.data[pred_idxs]
    preds.append(f_pred)

  if dr in ["backward", "both"]:
    nn_dists, nn_idxs = train_kdtree.query(test_pts, k=nn, dr="backward")

    pred_idxs = nn_idxs - 1
    if nn > 1:
      # wts = np.exp(-nn_dists)
      # wts = wts / wts.sum(axis=1)[:, None]
      f_pred = train_kdtree.data[pred_idxs].mean(axis=1)
      # f_pred = (train_kdtree.data[pred_idxs]   * wts[:,:, None]).sum(axis=1)
    else:
      f_pred = train_kdtree.data[pred_idxs]

    preds.append(f_pred)

  # IPython.embed()
  return preds if dr == "both" else preds[0]


def k_step_wn_forecast(
    test_pts, train_window, param, k=1, forecast_type="knn", dr="forward"):

  preds = {"forward": [], "backward": []} if dr == "both" else {dr: []}

  prev_pts = {dr: test_pts for dr in preds}

  if forecast_type == "knn":
    one_step_func = one_step_forecast_knn
  elif forecast_type == "rbf":
    one_step_func = one_step_wn_forecast_RBF
  else:
    one_step_func = one_step_wn_forecast

  for step in xrange(k):
    if VERBOSE > 50:
      print("Step %i out of %i"%(step + 1, k))
    for dr in preds:
      next_pred = one_step_func(
          prev_pts[dr], train_window, param, dr)
      preds[dr].append(next_pred)
      prev_pts[dr] = next_pred

  # IPython.embed()
  return preds


def wn_forecast_distance(
    test_window, train_data, num_steps, sample_inds, forecast_type="knn",
    feature_gen=None, gammak=1., nn=10, dr="forward", step_wts=None):

  if step_wts is not None:
    step_wts = np.squeeze(step_wts)
    if len(step_wts.shape) > 1:
      raise ValueError("Invalid step weights. Expecting 1-D array.")
    if step_wts.shape[0] != num_steps:
      raise ValueError("Invalid step weights. Expecting <num_steps> weights.")

  if step_wts is None:
    step_wts = np.ones(num_steps)
    # print("Warning: step_wts is currently not being used.")

  test_pts = test_window[sample_inds]

  if forecast_type == "knn":
    param = nn
  elif forecast_type == "rbf":
    param = gammak
  else:
    param = feature_gen

  preds = k_step_wn_forecast(
      test_pts, train_data, param, num_steps, forecast_type, dr)

  true_vals = {}
  for dr in preds:
    sign = 1 if dr == "forward" else -1
    true_vals[dr] = [
        test_window[sample_inds + sign*i] for i in xrange(1, num_steps + 1)]

  dists = {dr: np.array([
          np.linalg.norm(tv - pd, axis=1)
          for tv, pd in zip(true_vals[dr], preds[dr])
      ]).T for dr in preds}
  # Should probably do something fancy to ignore points when many are nans
  dists = {dr: dists[dr][-np.any(np.isnan(dists[dr]), axis=1)] for dr in dists}
  if np.any([dist.shape[0] == 0 for dist in dists.values()]):
    # IPython.embed()
    return np.inf
  
  weighted_dists = {dr: dists[dr].dot(step_wts).mean()}
  # Can weight forward and backward dynamics differently but not doing so now.
  return np.mean(weighted_dists.values())


def _compute_kernel_bandwidth(windows, c=1.):
  # Compute the bandwidth based onthe difference of first elements of all windows

  diffs = np.concatenate([w[1:, 0] - w[:-1, 0] for w in windows], axis=0)
  return 0.5 * c / (np.std(diffs)**2)


g_all_train_windows = None
g_num_train = None
g_test_window = None
g_num_steps = None
g_sample_inds = None
g_dr = None
g_forecast_type = None
g_feature_gen = None
g_gammak = None
g_nn = None


def _find_nearest_window_forcast_dist_single_ts(widx):
  # Helper for parallelizing
  if VERBOSE > 10:
    
    print("\t\tTrain ts %i out of %i."%(widx + 1, g_num_train))
  dists = []
  num_windows = len(g_all_train_windows[widx])
  for i, window in enumerate(g_all_train_windows[widx]):
    if VERBOSE > 50:
      print("\t\t\tWindow %i out of %i."%(i + 1, num_windows), end='\r')
    sys.stdout.flush()
    dists.append(wn_forecast_distance(
        g_test_window, window, g_num_steps, g_sample_inds, g_forecast_type,
        feature_gen=g_feature_gen, gammak=g_gammak, nn=g_nn, dr=g_dr))

  if VERBOSE > 50:
    print("\t\t\tWindow %i out of %i."%(i + 1, num_windows))
  # IPython.embed()
  return dists


def _find_min_window_inds(dists, nw=10):
  wlens = [len(wdist) for wdist in dists]
  w_ends = np.cumsum(wlens)
  w_starts = np.array([0] + np.cumsum(wlens)[:-1].tolist())

  dists = [dist for wdist in dists for dist in wdist]
  best_inds = np.argsort(dists)[:nw]

  ts_inds = np.searchsorted(w_ends, best_inds, side="right").tolist()
  w_inds = [bi - w_starts[ti] for bi, ti in zip(best_inds, ts_inds)]

  best_dists = [dists[bi] for bi in best_inds]

  return ts_inds, w_inds, best_dists


# TODO: Maybe parallelize it within each ts as opposed to over all ts.
def find_nearest_windows_forecast_dist(
    test_window, all_train_windows, num_steps, nw=10, num_samples=50, dr="forward",
    forecast_type="knn", feature_gen=None, gammak=1., nn=10, n_jobs=None):
  # all_train_windows is a list of training time-series,
  # each of which is a list of windows/kdtrees.
  global g_all_train_windows, g_num_train, g_test_window, g_num_steps,\
      g_sample_inds, g_dr, g_forecast_type, g_feature_gen, g_gammak, g_nn

  if g_all_train_windows is None or all_train_windows is not None:
    g_all_train_windows = all_train_windows
    g_num_train = len(all_train_windows)

  test_window = np.array(test_window)
  g_test_window = test_window
  g_num_steps = num_steps
  g_dr = dr
  g_forecast_type = forecast_type
  g_feature_gen = feature_gen
  g_gammak = gammak
  g_nn = nn

  ntest = test_window.shape[0]
  start_ind = num_steps if dr in ["backward", "both"] else 0
  end_ind = ntest - num_steps - 1 if dr in ["forward", "both"] else ntest - 1
  ind_range = np.arange(start_ind, end_ind + 1)
  g_sample_inds = np.array([
      ind_range[idx] for idx in
      np.random.permutation(ind_range.shape[0])[:num_samples]
  ])

  all_args = xrange(g_num_train)
  if n_jobs is not None and n_jobs != 1:
    pl = multiprocessing.Pool(n_jobs)
    window_dists =  pl.map(
        _find_nearest_window_forcast_dist_single_ts, all_args)
    pl.close()
    pl.join()
  else:
    window_dists = map(
        _find_nearest_window_forcast_dist_single_ts, all_args)

  gc.collect()
  return _find_min_window_inds(window_dists, nw)
  # min_window_idx = np.argmin([dist for dist, _ in window_dists_idxs])
  # best_tw = window_dists_idxs[min_window_idx]

  # IPython.embed()
  # if VERBOSE > 0:
  #   print (window_dists_idxs)
  # return min_window_idx, best_tw[1], best_tw[0]

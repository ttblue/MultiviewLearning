from __future__ import print_function, division

import sys
import time

import numpy as np
# import _ucrdtw as ucrdtw

import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

# Set up our R namespaces
rpy2.robjects.numpy2ri.activate()
R = rpy2.robjects.r
dtw = importr("dtw")

def choose_nn_dtw(target_ts, all_ts, warp_width=0.05):
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


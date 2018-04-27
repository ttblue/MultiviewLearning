import numpy as np
import sklearn.neighbors as sn

import matplotlib.pyplot as plt

import synthetic.lorenz as lorenz

import IPython


def compute_td_embedding(X, tau, d):
  X = np.squeeze(X)
  if len(X.shape) > 1:
    raise ValueError("Input must be 1-D.")

  n = X.shape[0]
  X_td = np.empty((n - (d - 1) * tau, 0))
  for i in xrange(d):
    X_td = np.c_[X_td, X[i * tau: n - (d - i - 1) * tau]]

  return X_td


def synchronization_test (ts1, ts2, n=20):
  # Assuming the TS are already in phase-space or embedded form
  # Check if they are synchronized -- closer the outputs are to 1, the 
  # more synchronized they are

  ts_len = np.minimum(ts1.shape[0], ts2.shape[0])
  ts1 = ts1[:ts_len]
  ts2 = ts2[:ts_len]

  # Generate random sample to perform test
  rand_inds = np.random.choice(ts_len, n, replace=False)
  #rand_inds2 = np.random.choice(ts_len, n, replace=False)
  sample1 = ts1[rand_inds]
  sample2 = ts2[rand_inds]

  # Find NN - can probably do this outside instead of computing every time
  nn1 = sn.NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(ts1)
  nn2 = sn.NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(ts2)
  dists1, inds1 = nn1.kneighbors(sample1)
  dists2, inds2 = nn2.kneighbors(sample2)
  # Extract second closest neighbor (as 1st one is point itself)
  inds1 = inds1[:, 1].squeeze()
  inds2 = inds2[:, 1].squeeze()

  pts_nn1_ts1 = ts1[inds1]
  pts_nn1_ts2 = ts2[inds1]
  pts_nn2_ts1 = ts1[inds2]
  pts_nn2_ts2 = ts2[inds2]

  d1_nn1 = np.linalg.norm(sample1 - pts_nn1_ts1, axis=1)
  d1_nn2 = np.linalg.norm(sample1 - pts_nn2_ts1, axis=1)
  d2_nn1 = np.linalg.norm(sample2 - pts_nn1_ts2, axis=1)
  d2_nn2 = np.linalg.norm(sample2 - pts_nn2_ts2, axis=1)

  # Hack
  if (d1_nn2 < 1e-3).any():
    d1_nn1 = d1_nn1[(d1_nn2 > 1e-3)]
    d1_nn2 = d1_nn2[(d1_nn2 > 1e-3)]

  if (d2_nn1 < 1e-3).any():
    d2_nn2 = d2_nn2[(d2_nn1 > 1e-3)]
    d2_nn1 = d2_nn1[(d2_nn1 > 1e-3)]

  r1 = np.mean(d1_nn1 / d1_nn2)
  r2 = np.mean(d2_nn2 / d2_nn1)

  return r1, r2


def find_sync_distance(ts1, ts2, tau, ntau, n_check=20, max_ndt=30):
  # Probably should be different taus/ntaus?
  # Maybe implement the second metric robust to ntau
  ts1_td = compute_td_embedding(ts1, tau, ntau)
  ts2_td = compute_td_embedding(ts2, tau, ntau)

  dt_range = range(-max_ndt, max_ndt + 1)

  rs = []
  for dt in dt_range:
    ts1_check = ts1_td[dt:] if dt > 0 else ts1_td
    ts2_check = ts2_td[-dt:] if dt < 0 else ts2_td

    r1, r2 = synchronization_test(ts1_check, ts2_check, n_check)
    r = (r1 + r2) / 2.
    r = r if r >= 1 else 1. / r
    rs.append(r)

  rs = np.array(rs)
  min_ind = np.argmin(rs)

  dtf, rf = dt_range[min_ind], rs[min_ind]

  return dtf, rf


if __name__ == "__main__":

  istart = 0
  tmax = 100
  nt = 10000 + istart
  x, y, z = lorenz.generate_lorenz_attractor(tmax, nt)

  tau = 10
  ntau = 3
  n_check = 50
  max_ndt = 300
  dt = 150

  ts1 = x[istart - dt:] if dt < 0 else x[istart:]
  ts2 = y[istart + dt:] if dt > 0 else y[istart:]

  dtf, rf = find_sync_distance(ts1, ts2, tau, ntau, n_check, max_ndt)


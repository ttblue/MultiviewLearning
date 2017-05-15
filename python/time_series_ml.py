from __future__ import print_function, division

import time
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm as sksvm
import mutual_info as mi

try:
  import matplotlib.pyplot as plt, matplotlib.cm as cm
  from mpl_toolkits.mplot3d import Axes3D
  import plot_utils
  PLOTTING = True
except ImportError:
  PLOTTING = False

import nn_utils


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


def predict_nn_dtw(train_ts, train_labels, test_windows):
  train_ts = [np.array(ts) for ts in train_ts]
  test_windows = [np.array(window) for window in test_windows]
  
  # Use only the first feature for each ts.
  # train_ts = [np.squeeze(ts[:, 0]) if len(ts.shape) > 1 else ts for ts in train_ts]

  pred_labels = []
  num_windows = len(test_windows)
  widx = 0
  tstart = time.time()
  for window in test_windows:

    idx, loc, dist = nn_utils.choose_nn_dtw(window, train_ts)
    pred_labels.append(train_labels[idx][loc])

    widx += 1
    print("Window %i out of %i done. Time taken: %.2fs"%
          (widx, num_windows, time.time() - tstart), end='\r')
    sys.stdout.flush()

  print("Window %i out of %i done. Time taken: %.2fs"%
        (widx, num_windows, time.time() - tstart))

  return np.array(pred_labels)


def predict_nn_euclidean(train_ts, train_labels, test_windows):
  train_ts = [np.array(ts) for ts in train_ts]
  test_windows = [np.array(window) for window in test_windows]
  
  pred_labels = []
  num_windows = len(test_windows)
  widx = 0
  tstart = time.time()
  for window in test_windows:
    idx, loc, dist = nn_utils.choose_nn_euclidean(window, train_ts)
    pred_labels.append(train_labels[idx][loc])

    widx += 1
    print("Window %i out of %i done. Time taken: %.2fs"%
          (widx, num_windows, time.time() - tstart), end='\r')
    sys.stdout.flush()

  print("Window %i out of %i done. Time taken: %.2fs"%
        (widx, num_windows, time.time() - tstart))

  return np.array(pred_labels)


def compute_dynamics_coefficients_simple(Xs, DXs):
  classifier = sksvm.SVC(kernel='linear')
  coeffs = []
  for X, DX in zip(Xs, DXs):
    classifier.fit(X, DX)
    coeffs.append(classifier.coef_.squeeze())

  return coeffs
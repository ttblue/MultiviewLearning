from __future__ import print_function, division

import os
import sys
import IPython

import matplotlib.pyplot as plt, matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np, numpy.random as nr, numpy.linalg as nlg

import pandas as pd
from sklearn.cluster import KMeans 

import mutual_info as mi

import utils

VERBOSE = True
FREQUENCY = 250
DATA_DIR = '/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/slow'

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

  return ts_features, tvals


def compute_average_tau(mc_ts, M=200):
  taus = []
  for channel in range(mc_ts.shape[1]):
    if VERBOSE:
      print(channel)
    taus.append(compute_tau(mc_ts[:,channel], M, show=True))

  return int(np.mean(taus))


def feature_multi_channel_timeseries(mc_ts, time_channel, ts_channels,
    tau_range=50, downsample=5, window_length=3000, num_samples=100,
    num_windows=None, d_lag=3, d_reduced=6, d_features=1000, bandwidth=0.5):
  
  tstamps = mc_ts[::downsample, time_channel]
  mc_ts = mc_ts[::downsample, ts_channels]

  if num_windows is None:
    num_windows = int(mc_ts.shape[0]/window_length)

  # if VERBOSE:
  #   print('Computing average tau.')
  # tau = compute_average_tau(mc_ts, M=tau_range)
  tau = None

  tvals = None
  mcts_features = []
  for channel in xrange(len(ts_channels)):
    if VERBOSE:
      print('Channel:', ts_channels[channel])
    channel_features, channel_tvals = featurize_single_timeseries(
        mc_ts[:, channel], tau_range=tau_range, window_length=window_length,
        num_samples=num_samples, num_windows=num_windows, d_lag=d_lag, tau=tau,
        d_reduced=d_reduced, d_features=d_features, bandwidth=bandwidth)
    if VERBOSE:
      print()

    mcts_features.append(channel_features)
    if tvals is None:
      tvals = channel_tvals

  return mcts_features, tvals


def create_label_timeline(critical_inds, labels):
  
  label_dict = {i:lbl for i,lbl in enumerate(labels)}
  label_dict[len(labels)] = labels[-1]

  return label_dict

# # ==============================================================================
# # Putting things together
# # ==============================================================================
def create_pigdata33_features_labels(data):

  ann_file = os.path.join(DATA_DIR, '33_annotation.txt')
  ann_idx, ann_text = utils.load_annotation_file(ann_file)

  time_channel = 0
  ts_channels = range(2, 13)

  # Parameters for features
  # One minute long windows
  downsample = 1
  window_length_s = 30  # Size of window in seconds.
  tau_range = int(200/downsample)
  window_length = int(FREQUENCY*window_length_s/downsample)
  
  num_samples = 500
  num_windows = None
  
  d_lag = 3
  d_reduced = 6
  d_features = 1000
  bandwidth = 0.5

  mcts_f, tvals = feature_multi_channel_timeseries(data,
    time_channel=time_channel, ts_channels=ts_channels, tau_range=tau_range,
    downsample=downsample, window_length=window_length, num_samples=num_samples,
    num_windows=num_windows, d_lag=d_lag, d_reduced=d_reduced,
    d_features=d_features, bandwidth=bandwidth)


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
  critical_anns = [2, 7, 8, 13, 18, 19, 26, 27, 30, 35, 38, 39, 43, 46, 47, 48, 51, 52, 60]
  ann_labels = [-1, 0, -1, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 5]
  critical_inds = [ann_idx[anni]/downsample for anni in critical_anns]
  label_dict = create_label_timeline(critical_inds, ann_labels)

  mean_inds = [t + window_length/2.0 for t in tvals]
  segment_inds = np.searchsorted(critical_inds, mean_inds)
  labels = np.array([label_dict[idx] for idx in segment_inds])

  valid_inds = (labels != -1)
  mcts_f = [c_f[valid_inds, :] for c_f in mcts_f]
  labels = labels[valid_inds]

  return mcts_f, labels


if __name__ == '__main__':
  # data_file = os.path.join(DATA_DIR, '33.csv')
  # col_names, data = utils.load_csv(data_file)
  data = np.load('tmp2.npy')
  mcts_f, labels = create_pigdata33_features_labels(data)
  save_file = os.path.join(DATA_DIR, '33_features')
  # IPython.embed()
  np.save(save_file, {'features': mcts_f, 'labels': labels})
  # IPython.embed()


# ann_file = os.path.join(os.getenv('HOME'), 'Research/TransferLearning/data/33/33_annotation.txt')
# with open(ann_file, 'r') as fh:
#   ann = fh.readlines()

# k = 1
# ann_idx = {}
# ann_text = {}
# for s in ann:
#   s = s.strip()
#   s_split = s.split('\t')
#   if len(s_split) == 1: continue
#   ann_idx[k] = float(s_split[0])
#   ann_text[k] = ' '.join(s_split[1:])
#   k += 1

# ann_use = [[3, 8], [9, 14], [19, 20], [27, 28], [31, 36], [41, 41], [44, 47],
#            [48, 49], [52, 53], [59, 59], [60, 60], [61, 61]]

# labels, data = utils.load_csv(os.path.join(DATA_DIR, '33.csv'))
# y = data[:,4]

# ysmooth = pd.rolling_window(y, 5, "triang")*5./3  # Scaling factor for normalization.
# ysmooth = ysmooth[~np.isnan(ysmooth)]
# Ny = ysmooth.shape[0]
# plt.plot(ysmooth)

# # # ==============================================================================
# # Average Mutual Information & embedding
# M = 100
# N = 5000

# y0 = ysmooth[:N]
# ami = np.zeros(M)
# alpha = 1.0
# tau = 0  # Point where the shifted function is most dissimilar.
# fmv = np.inf
# for m in xrange(M):
#   print m
#   ami[m] = mi.mutual_information_2d(y0[:N-m], y0[m:])
#   if ami[m] <= alpha*fmv:
#     tau = m
#     fmv = ami[m]
#   else:
#     fmv = -np.inf

# plt.plot(ami)
# print tau
# plt.show()

# D = 3
# x = np.zeros((Ny-(D-1)*tau, 0))
# for d in xrange(D):
#   x = np.c_[x, ysmooth[d*tau:Ny-(D-d-1)*tau]]

# L = 5000
# stepsize = 1
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[:L:stepsize, 0], x[:L:stepsize, 1], x[:L:stepsize, 2])

# # ==============================================================================
# # Compute the features: Basis coefficients

# Dred = 6  # number of dimensions to reduce to
# Ns = 100  # number of samples
# a = 0.5  # kernel bandwidth
# df = 1000  # random features
# mm_rbf = mm_rbf_fourierfeatures(D, df, a)

# s = nr.randint(1, x.shape[0]-L, size=(Ns,))
# Z = np.zeros((Ns, df))
# # parfor_progress(Ns)
# for i in range(Ns):
#   print i
#   xhat = x[s[i]:s[i]+L:stepsize, :]
#   xhat -=  xhat.mean(axis=0)
#   xhat /= np.abs(xhat).max()
#   Z[i] = mm_rbf(xhat)
# parfor_progress;

# parfor_progress(0);
# valid_inds = ~np.isnan(Z).any(axis=1)
# valid_locs = np.nonzero(valid_inds)[0]
# Z = Z[valid_inds]  # remove NaN rows.
# Ns = Z.shape[0]

# E, V, _ = nlg.svd(Z, full_matrices=0)
# v = V[:Dred]**2
# e = E[:, :Dred]
# nl = int(L/stepsize)

# # ==============================================================================
# # Computing the basis windows -- not required in the end.
# phis = []
# for i in xrange(Dred):
#   print i
#   phi = np.zeros((nl, D))
#   for j in xrange(Ns):
#     xhat = x[s[valid_locs[j]]:s[valid_locs[j]]+L:stepsize, :]
#     xhat -= xhat.mean(axis=0)
#     xhat /= np.abs(xhat).max()
#     phi += e[j, i]*xhat
  
#   phis.append(phi)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(phis[0][:, 0], phis[0][:, 1], phis[0][:, 2], color='r')
# ax.scatter(phis[1][:, 0], phis[1][:, 1], phis[1][:, 2], color='g')
# ax.scatter(phis[2][:, 0], phis[2][:, 1], phis[2][:, 2], color='b')
# plt.show()

# xhat = x[s[0]:s[0]+L:stepsize, :]
# xhat -= xhat.mean(axis=0)
# xhat /= np.abs(xhat).max()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xhat[:, 0], xhat[:, 1], xhat[:, 2])
# plt.show()
# # ==============================================================================

# nsteps = 1000
# comp = np.zeros((nsteps, Dred))
# shat = np.floor(np.linspace(0, x.shape[0]-L, nsteps)).astype(int)
# Zhat = Z.T.dot(e).dot(np.diag(1./v))

# for i in range(nsteps):
#   print i
#   xhat = x[shat[i]:shat[i]+L:stepsize, :]
#   xhat -= xhat.mean(axis=0)
#   xhat /= np.abs(xhat).max()
#   comp[i, :] = mm_rbf(xhat).dot(Zhat)

# valid_windows = ~np.isnan(comp).any(axis=1)
# comp = comp[valid_windows]
# shat = shat[valid_windows]
# # ==============================================================================

# comp2 = pd.rolling_window(comp, 10, 'triang')
# shat2 = shat[~np.isnan(comp2).any(axis=1)]
# shat2 -= shat2.min()
# shat2 /= np.abs(shat2).max()
# comp2 = comp2[~np.isnan(comp2).any(axis=1)]
# j1, j2 = 1, 4
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(shat2, comp2[:, j1], comp2[:, j2])#, color=[0.9, 0.9, 0.9])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for j in range(comp2.shape[0]):
#   print j
#   fig.clf()
#   # ax.cla()
#   # ax = fig.add_subplot(111, projection='3d')
#   ax.scatter(comp2[:j+1, 0], comp2[:j+1, 1], comp2[:j+1, 2], color=colors[:j+1])
#   plt.show(block=False)
#   time.sleep(0.1)

# py = comp[:, 4]
# py10 = pd.rolling_window(py, 10, 'triang')
# py10[np.isnan(py10)] = py[np.isnan(py10)]
# plt.plot(shat, py10)
# ymin = py.min()
# ymax = py.max()
# yvals = [ymin, ymax, ymin, ymax]
# for i in xrange(len(ann_use)):
#   xvals = [ann_idx[ann_use[i][0]], ann_idx[ann_use[i][0]], ann_idx[ann_use[i][1]], ann_idx[ann_use[i][1]]]
#   print xvals
#   plt.plot(xvals, yvals)

# plt.show()

# for i=1:length(ann_use);
# idx = shat>=ann_idx(ann_use(i,1)) & shat<=ann_idx(ann_use(i,2));
# plot3(shat(idx),comp2(idx,j1),comp2(idx,j2))


# d=20;
# i=randi(length(s));
# j=randi(length(s));
# xhat = x(s(i):d:s(i)+L-1,:);
# xhat=xhat-repmat(mean(xhat),size(xhat,1),1); xhat=xhat/max(max(abs(xhat)));
# xhat2 = x(s(j):d:s(j)+L-1,:);
# xhat2=xhat2-repmat(mean(xhat2),size(xhat2,1),1); xhat2=xhat2/max(max(abs(xhat2)));
# rbf(xhat,xhat2,a)
# sum(mm_rbf(xhat,a).*mm_rbf(xhat2,a))

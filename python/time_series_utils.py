import os

import matplotlib.pyplot as plt, matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import pandas as pd

import utils
import mutual_info as mi

DATA_DIR = '/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/slow'

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


# def compute_tau(y, M=200, show=True):
#   # Computes time lag for ts-features using Mutual Information
#   # This time lag is where the shifted function is most dissimilar.
#   # y -- time series
#   # M -- search iterations
#   N = y.shape[0]
#   minfo = np.zeros(M)
#   for m in xrange(M):
#     minfo[m] = mi.mutual_information_2d(y[:N-m], y[m:])

#   tau = np.argmin(minfo)

#   if show:
#     plt.plot(minfo)
#     print tau
#     plt.show()

#   return tau


# def mean_scale_transform(x):
#   x = np.array(x)
#   x -= x.mean(0)
#   x /= np.abs(x).max()
#   return x


# def compute_window_mean_embdedding(y, tau, mm_rff, f_transform=None, D=3):
#   # Computes the explicit mean embedding in some RKHS space of the TS y
#   # tau -- time lag
#   # mm_rff -- mean map of random fourier features
#   # f_transform -- transformation of time-lagged features before mm_rff
#   # D -- dimensionality of time-lagged features

#   Ny = y.shape[0]
#   if f_transform is None:
#     f_transform = lambda v: v

#   # TODO: maybe not do this here because you lose edge-of-window information
#   # of about (D-1)*tau samples
#   x = np.zeros((Ny-(D-1)*tau, 0))
#   for d in xrange(D):
#     x = np.c_[x, y[d*tau:Ny-(D-d-1)*tau]]

#   mm_xhat = mm_rff(f_transform(x))
#   return mm_xhat


# def compute_window_PCA(Xhat, wdim, evecs=False):
#   valid_inds = ~np.isnan(Xhat).any(axis=1)
#   Xhat = Xhat[valid_inds]  # remove NaN rows.

#   E, V, _ = nlg.svd(Xhat, full_matrices=0)
#   v = V[:wdim]**2
#   e = E[:, :wdim]

#   if evecs:
#     return e, v, valid_inds
#   else:
#     return np.diag(1./v).dot(e.T).dot(Xhat)


# def compute_window_features(mm_windows, basis):
#   # mm_windows -- n x df mean map features for windows
#   # basis: fdim x df basis window embeddings

#   mm_windows = np.atleast_2d(mm_windows)
#   return mm_windows.dot(basis.T)


# def featurize_timeseries(ts):

# # ==============================================================================
# # Putting things together
# # ==============================================================================
# def compute_TS_basis(signal, D=3, wdim=6):

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

# TODO:
# ann = strsplit(fileread('33_annotation.txt'),'\n');
# ann_idx=[]; ann_text={};k=1;
# for i=1:length(ann)
#   s = ann{i};
#   idx = find(isspace(s));
#   if length(idx)==0, continue; end;
#   ann_idx(k) = str2num(s(1:idx(1)));
#   ann_text{k} = s(idx(1)+1:end);
#   k=k+1;
# end
# ann_use = [3,8; 9,14; 19,20; 27,28; 31,36; 41,41; 44,47; 48,49; 52,53; 59,59; 60,60; 61,61];

labels, data = utils.load_csv(os.path.join(DATA_DIR, '33.csv'))
y = data[:,4]

ysmooth = pd.rolling_window(y, 5, "triang")*5./3  # Scaling factor for normalization.
ysmooth = ysmooth[~np.isnan(ysmooth)]
Ny = ysmooth.shape[0]
plt.plot(ysmooth)

# # ==============================================================================
# Average Mutual Information & embedding
M = 100
N = 5000

y0 = ysmooth[:N]
ami = np.zeros(M)
alpha = 1.0
tau = 0  # Point where the shifted function is most dissimilar.
fmv = np.inf
for m in xrange(M):
  print m
  ami[m] = mi.mutual_information_2d(y0[:N-m], y0[m:])
  if ami[m] <= alpha*fmv:
    tau = m
    fmv = ami[m]
  else:
    fmv = -np.inf

plt.plot(ami)
print tau
plt.show()

D = 3
x = np.zeros((Ny-(D-1)*tau, 0))
for d in xrange(D):
  x = np.c_[x, ysmooth[d*tau:Ny-(D-d-1)*tau]]

L = 5000
stepsize = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:L:stepsize, 0], x[:L:stepsize, 1], x[:L:stepsize, 2])

# ==============================================================================
# Compute the features: Basis coefficients

Dred = 6  # number of dimensions to reduce to
Ns = 100  # number of samples
a = 0.5  # kernel bandwidth
df = 1000  # random features
mm_rbf = mm_rbf_fourierfeatures(D, df, a)

s = nr.randint(1, x.shape[0]-L, size=(Ns,))
Z = np.zeros((Ns, df))
# parfor_progress(Ns)
for i in range(Ns):
  print i
  xhat = x[s[i]:s[i]+L:stepsize, :]
  xhat -=  xhat.mean(axis=0)
  xhat /= np.abs(xhat).max()
  Z[i] = mm_rbf(xhat)
# parfor_progress;

# parfor_progress(0);
valid_inds = ~np.isnan(Z).any(axis=1)
valid_locs = np.nonzero(valid_inds)[0]
Z = Z[valid_inds]  # remove NaN rows.
Ns = Z.shape[0]

E, V, _ = nlg.svd(Z, full_matrices=0)
v = V[:Dred]**2
e = E[:, :Dred]
nl = int(L/stepsize)

# ==============================================================================
# Computing the basis windows -- not required in the end.
phis = []
for i in xrange(Dred):
  print i
  phi = np.zeros((nl, D))
  for j in xrange(Ns):
    xhat = x[s[valid_locs[j]]:s[valid_locs[j]]+L:stepsize, :]
    xhat -= xhat.mean(axis=0)
    xhat /= np.abs(xhat).max()
    phi += e[j, i]*xhat
  
  phis.append(phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(phis[0][:, 0], phis[0][:, 1], phis[0][:, 2], color='r')
ax.scatter(phis[1][:, 0], phis[1][:, 1], phis[1][:, 2], color='g')
ax.scatter(phis[2][:, 0], phis[2][:, 1], phis[2][:, 2], color='b')
plt.show()

xhat = x[s[0]:s[0]+L:stepsize, :]
xhat -= xhat.mean(axis=0)
xhat /= np.abs(xhat).max()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xhat[:, 0], xhat[:, 1], xhat[:, 2])
plt.show()
# ==============================================================================

nsteps = 100
comp = np.zeros((nsteps, Dred))
shat = np.floor(np.linspace(0, x.shape[0]-L, nsteps)).astype(int)
Zhat = Z.T.dot(e).dot(np.diag(1./v))

for i in range(nsteps):
  print i
  xhat = x[shat[i]:shat[i]+L:stepsize, :]
  xhat -= xhat.mean(axis=0)
  xhat /= np.abs(xhat).max()
  comp[i, :] = mm_rbf(xhat, a).dot(Zhat)
# ==============================================================================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for j in range(comp2.shape[0]):
  print j
  fig.clf()
  # ax.cla()
  # ax = fig.add_subplot(111, projection='3d')
  ax.scatter(comp2[:j+1, 0], comp2[:j+1, 1], comp2[:j+1, 2], color=colors[:j+1])
  plt.show(block=False)
  time.sleep(0.1)

py = comp[:, 4]
plt.plot(shat, pd.rolling_window(py, 10, 'triang'))
plt.show()
# ymin = py.min()
# ymax = py.max()
# for i xrange(length(ann_use)  
#    plot([ann_idx(ann_use(i,1));ann_idx(ann_use(i,1));ann_idx(ann_use(i,2));ann_idx(ann_use(i,2))],[y1;y2;y1;y2]);
# end
# hold off;

comp2 = pd.rolling_window(comp, 10, 'triang')
j1, j2 = 1, 4
ax = fig.add_subplot(111, projection='3d')
plt.scatter(shat, comp2[:, j1], comp2[:, j2], color=[0.9, 0.9, 0.9])
plt.show()
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

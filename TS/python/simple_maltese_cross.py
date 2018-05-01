## Multiple approaches for the Simple Maltese Cross (without DNs)
## Overall idea:
## 1. Generate two coupled systems which are asynchronous
## 2. Learn:
##   a. Transformation from each system to latent space
##   b. Dynamical system in the latent space
## 3. Predict: One stream given the other

import cvxpy as cvx
import numpy as np
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import dynamical_systems_learning as dsl
import gaussianRandomFeatures as grf
import synthetic.lorenz as lorenz
import time_sync as tsync

import IPython

###############################################################################
## Approach 1: EM-style alternating projections
## Treat one TS (say A) as the canonical state-space. Then, alternate:
## (1) Learn dynamical system from A + transformed B
## (2) Learn transform from B to A given the dynamical system

# Rewrite this later.

###############################################################################
## Approach 2:
## Learn the dynamical system in one TS alone (say A), and then find a transform
## from B to A such that the dynamical system learned on the transformed B is
## the same.
class TransformLearner(nn.Module):

  def __init__(self, dim, rff, target_alpha, thresh=1e-3, affine=True):
    super(TransformLearner, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    self.dim = dim
    self.affine = affine  # For the RFFs
    self.thresh = thresh
    self.update_torch_rff(rff)
    self.update_target_alpha(target_alpha)
    self.reset_transform()

    self.criterion = nn.MSELoss()

  def update_torch_rff(self, rff):
    assert self.dim == rff.dim
    self.sine = rff.sine
    self.rn = rff.rn
    # IPython.embed()
    self.ws = torch.from_numpy(rff.ws.astype(np.float32))
    self.ws.requires_grad_(False)
    if not self.sine:
      self.bs = torch.from_numpy(rff.bs.astype(np.float32))
      self.bs.requires_grad_(False)  

  def _compute_random_features(self, xt):
    # xt is an n x r tensor of transformed points, with the transform being
    # parameterized by trainable variables
    # IPython.embed()
    if self.sine:
      c = np.sqrt(1 / self.rn)
      rf_cos = torch.cos(torch.mm(self.ws, xt.t())) * c
      rf_sin = torch.sin(torch.mm(self.ws, xt.t())) * c
      rf = torch.cat((rf_cos, rf_sin), dim=0)
    else:
      c = np.sqrt(2 / self.rn)
      rf = torch.cos(torch.mm(self.ws, xt.t()) + self.bs[:, None]) * c

    rf_final = (
        torch.cat((rf.t(), torch.ones((rf.shape[1], 1))), dim=1)
        if self.affine else rf.t())
    return rf_final

  def update_target_alpha(self, target_alpha):
    self.target_alpha = torch.from_numpy(target_alpha.astype(np.float32))
    self.target_alpha.requires_grad_(False)

  def reset_transform(self):
    # Currently linear: rot + trans
    # self.R = autograd.Variable(torch.eye(self.dim), requires_grad=True)
    # self.t = autograd.Variable(torch.zeros(self.dim, 1), requires_grad=True)
    self.R = nn.Parameter(torch.eye(self.dim), requires_grad=True)
    self.t = nn.Parameter(torch.zeros(self.dim, 1), requires_grad=True)

  def forward(self, x):
    # Solve for alpha
    if not isinstance(x, torch.Tensor):
      # Assuming it is numpy arry
      x = torch.from_numpy(x)
    x.requires_grad_(False)

    # IPython.embed()
    x_transformed = (torch.mm(self.R, x.t()) + self.t).t()
    # IPython.embed()
    x_rff = self._compute_random_features(x_transformed)
    self.x_ff = x_rff

    # Now solve ||x_rff(T) * alpha - x_transformed(T+1)||
    x_next = x_transformed[1:]
    x_rff_prev = x_rff[:-1]
    A = torch.mm(x_rff_prev.t(), x_rff_prev)

    # u, s, v = torch.svd(A)
    # si = torch.where(s > self.thresh, 1. / s, torch.zeros_like(s))
    # Ainv = torch.mm(torch.mm(u, torch.diag(si)), v)
    Ainv = torch.inverse(A)

    alpha = torch.mm(torch.mm(Ainv, x_rff_prev.t()), x_next)
    # alpha_cols = []
    # for col in range(x_next.shape[1]):
    #   alpha_col = torch.mm(torch.mm(Ainv, x_rff_prev.t()), x_next[:, col])
    #   alpha_cols.append(alpha_col)
    # alpha = torch.cat(alpha_cols, dim=1)

    return alpha

  def loss(self, alpha):
    return self.criterion(alpha, self.target_alpha)


class TrainerConfig(object):

  def __init__(self, lr=1e-3, max_iters=100, verbose=True):
    self.lr = lr
    self.max_iters = max_iters
    self.verbose = verbose


class TransformTrainer(object):

  def __init__(self, tlearner, config):
    self.config = config
    self._setup_optimizer()

  def _setup_optimizer(self):
    self.opt = optim.Adam(tlearner.parameters(), self.config.lr)

  def train_loop(self, x):
    self.opt.zero_grad()
    alpha = tlearner(x)
    self.loss = tlearner.loss(alpha)
    self.loss.backward()
    self.opt.step()

  def fit(self, x):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    for itr in range(self.config.max_iters):
      if self.config.verbose:
        itr_start_time = time.time()
        print("\nIteration %i out of %i." % (itr + 1, self.config.max_iters))

      self.train_loop(x)

      if self.config.verbose:
        itr_duration = time.time() - itr_start_time
        print("Loss: %.2f" % float(self.loss.detach()))
        print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))

    print("Training finished in %0.2f s." % (time.time() - all_start_time))


if __name__ == "__main__":

  visualize = True
  tmax = 100
  nt = 10000
  x, y, z = lorenz.generate_lorenz_attractor(tmax, nt)

  tau = 10
  ntau = 3
  xts = tsync.compute_td_embedding(x, tau, ntau)
  yts = tsync.compute_td_embedding(y, tau, ntau)
  xts = xts.astype(np.float32)
  yts = yts.astype(np.float32)

  affine = True
  rn = 500
  gammak = 1.0
  sine = False
  rff = grf.GaussianRandomFeatures(dim=ntau, rn=rn, gammak=gammak, sine=sine)
  feature_func = rff.computeRandomFeatures
  A, b, err = dsl.learn_linear_dynamics(xts, feature_func, affine)
  target_alpha = np.c_[A, b].T

  # Visualize
  if visualize:
    ts0 = xts[0]
    ts_pred = [ts0]
    for _ in range(len(xts) - 1):
      ts_pred.append(A.dot(feature_func(ts_pred[-1]).squeeze()) + b)
    ts_pred = np.array(ts_pred)
    IPython.embed()

  thresh = 1e-2
  tlearner = TransformLearner(
      dim=ntau, rff=rff, target_alpha=target_alpha, thresh=thresh,
      affine=affine)
  # ap2 = tlearner.forward(yts)
  # IPython.embed()
  print("Set up learner.")

  lr = 1e-3
  max_iters = 1
  verbose = True
  config = TrainerConfig(lr=lr, max_iters=max_iters, verbose=verbose)
  ttrainer = TransformTrainer(tlearner, config)
  print("Set up trainer.")
  print("Training.")
  ttrainer.fit(yts)


  # n = 1000
  # d = 3
  # rn = 100
  # gammak = 1.0
  # thresh = 1e-3
  # sine = True
  # rn_true = rn * (2 if sine else 1)
  # rff = grf.GaussianRandomFeatures(dim=d, rn=rn, gammak=gammak, sine=True)

  # x = np.random.rand(n, d)
  # xr = rff.computeRandomFeatures(x)
  # xr = torch.from_numpy(xr)
  # xr.requires_grad_()
  # A = torch.mm(xr.t(), xr)

  # u, s, v = torch.svd(A)
  # si = torch.where(s > thresh, 1. / s, torch.zeros_like(s))
  # Ainv = torch.mm(torch.mm(u, torch.diag(si)), v)

  # alpha = torch.mm(torch.mm(Ainv, xr.t()), y)
  # alpha_base = torch.zeros(rn_true, 1, dtype=torch.float64)
  # loss = torch.norm(alpha-alpha_base)

  # loss.backward()

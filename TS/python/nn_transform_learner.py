## Multiple approaches for the Simple Maltese Cross (without DNs)
## Overall idea:
## 1. Generate two coupled systems which are asynchronous
## 2. Learn:
##   a. Transformation from each system to latent space
##   b. Dynamical system in the latent space
## 3. Predict: One stream given the other

import cvxpy as cvx
import numpy as np
import os
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import gaussianRandomFeatures as grf
import synthetic.simple_systems as ss
import tfm_utils as tu
import time_sync as tsync

import IPython

torch.set_default_dtype(torch.float64)


_RCOND = 1e-10


raise Exception("Fix this file with new stuff.")


def pairwise_sqeuclidean(x, y=None):
    """
    Input: 
      x -- Nxd torch tensor
      y -- optional Mxd torch tensor
    Output:
      dist -- NxM torch tensor where dist[i,j] is the square norm between
          x[i,:] and y[j,:]
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    
    Note:Iif y is not given then use y=x.

    From - 
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        if len(y.shape) < 2:
          y = y.unsqueeze(0)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y for numerical stability issues
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    # IPython.embed()
    return torch.clamp(dist, 0.0, np.inf)


class NNTransformLearner(nn.Module):

  def __init__(self, X, dim, gammak=1.0):
    super(NNTransformLearner, self).__init__()
    if X is not None and not isinstance(X, torch.Tensor):
      # Assuming it is numpy arry
      X = torch.from_numpy(X)
      X.requires_grad_(False)
    self.X = X  # base dataset
    self.dim = dim
    self.gammak = gammak

    self.reset_transform()
    self.criterion = nn.MSELoss()

  def _update_base_data(self, X):
      # Assuming it is numpy arry
    self.X = torch.from_numpy(X)
    self.X.requires_grad_(False)

  def reset_transform(self, init_R=None, init_t=None):
    # Currently linear: rot + trans
    # self.R = autograd.Variable(torch.eye(self.dim), requires_grad=True)
    # self.t = autograd.Variable(torch.zeros(self.dim, 1), requires_grad=True)
    init_R = torch.eye(self.dim) if init_R is None else torch.from_numpy(init_R)
    init_t = (
        torch.zeros(self.dim, 1) if init_t is None
        else torch.from_numpy(init_t))
    self.R = nn.Parameter(init_R, requires_grad=True)
    self.t = nn.Parameter(init_t, requires_grad=True)

  def _transform_pts(self, x):
    # Assuming it's a linear transform.
    return (torch.mm(self.R, x.t()) + self.t).t()

  def _pinvert_mat(self, A):
    # Computes pseudo-inverse of given condition number
    u, s, v = torch.svd(A)
    si = torch.where(s >= _RCOND * s[0], 1. / s, torch.zeros_like(s))
    Ainv = torch.mm(torch.mm(v, torch.diag(si)), u.t())
    # Ainv = torch.inverse(A)
    return Ainv

  def _gaussian_kernel(self, x, y):
    pdist = pairwise_sqeuclidean(x, y)
    return torch.exp(-self.gammak * pdist)

  def forward_simulate(self, y0, nsteps=100, use_tf=False):
    # Solve for alpha without transform
    if len(y0.shape) >= 2:  # We forward simulate from one point only
      y0 = y0[:1]

    if not isinstance(y0, torch.Tensor):
      # Assuming it is numpy array
      y0 = torch.from_numpy(y0)
    y0.requires_grad_(False)
    if use_tf:
      y0 = self._transform_pts(y0)

    y_pred = [y0.unsqueeze(0)]
    for i in range(nsteps):
      print("Simulating step %i out of %i" % (i + 1, nsteps), end='\r')
      G = self._gaussian_kernel(self.X[:-1], y_pred[-1]).view(-1, 1)
      Dinv = 1. / (G.squeeze().sum())
      y_pred.append(torch.mm(self.X[1:].t(), Dinv * G).t())

    print("Simulating step %i out of %i" % (i + 1, nsteps))
    # IPython.embed()

    return torch.cat(y_pred, 0)

  def forward(self, y, y_next):
    if not isinstance(y, torch.Tensor):
      y = torch.from_numpy(y)
    if not isinstance(y_next, torch.Tensor):
      y_next = torch.from_numpy(y_next)

    y.requires_grad_(False)
    y_next.requires_grad_(False)

    y_transformed = self._transform_pts(y)
    yn_transformed = self._transform_pts(y_next)

    G = self._gaussian_kernel(self.X[:-1], y_transformed)
    Dinv = torch.diag(1. / G.sum(1))

    y_pred = torch.mm(torch.mm(Dinv, G), self.X[1:])
    return y_pred, y_next

  def loss(self, y_pred, y_next):
    return self.criterion(y_pred, y_next)


class NNTrainerConfig(object):

  def __init__(self, lr=1e-3, max_iters=100, verbose=True):
    self.lr = lr
    self.max_iters = max_iters
    self.verbose = verbose


class NNTransformTrainer(object):

  def __init__(self, tlearner, config):
    self.config = config
    self.tlearner = tlearner
    self._setup_optimizer()

  def _setup_optimizer(self):
    self.opt = optim.Adam(self.tlearner.parameters(), self.config.lr)

  def train_loop(self, y, y_next):
    self.opt.zero_grad()
    y_pred, yn_transformed = self.tlearner(y, y_next)
    self.loss = self.tlearner.loss(y_pred, yn_transformed)
    self.loss.backward()
    self.opt.step()

  def fit(self, y):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    y_t1 = y[:-1]
    y_t2 = y[1:]
    for itr in range(self.config.max_iters):
      if self.config.verbose:
        itr_start_time = time.time()
        print("\nIteration %i out of %i." % (itr + 1, self.config.max_iters))

      self.train_loop(y_t1, y_t2)

      if self.config.verbose:
        itr_duration = time.time() - itr_start_time
        print("Loss: %.2f" % float(self.loss.detach()))
        print("Current transform estimate: \nR: %s\nt:%s" % 
            (self.tlearner.R.detach().numpy(),
             self.tlearner.t.detach().numpy()))
        print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))

    print("Training finished in %0.2f s." % (time.time() - all_start_time))


if __name__ == "__main__":

  visualize = True
  tmax = 100
  nt = 1000
  x, _, _ = ss.generate_lorenz_system(tmax, nt)

  tau = 1
  ntau = 3
  xts = tsync.compute_td_embedding(x, tau, ntau)
  # yts = tsync.compute_td_embedding(y, tau, ntau)

  # Simple rotation + translation
  # The correct transform will be the inverse of the true one.
  theta = 0 # np.pi / 4
  new_xdir = np.atleast_2d([np.cos(theta), np.sin(theta), 0.]).T
  new_ydir = np.atleast_2d([- np.sin(theta), np.cos(theta), 0.]).T
  new_zdir = np.atleast_2d([0., 0., 1.]).T
  scaling = np.diag([1.5, 0.75, 1.0])
  R_true_base = np.c_[new_xdir, new_ydir, new_zdir].astype(np.float64)
  R_true = R_true_base.dot(scaling)
  t_true = 0 * np.array([3, -4, 0], dtype=np.float64).reshape(-1, 1)
  yts = (R_true.dot(xts.T) + t_true).T

  # R_guess, t_guess = tu.guess_best_transform(yts, xts)
  # t_guess = t_guess.reshape(-1, 1)
  R_guess = np.linalg.inv(R_true_base)
  t_guess = np.zeros([ntau, 1])
  xR = (R_guess.dot(yts.T) + t_guess).T

  # plt.plot(xR[:, 0], xR[:, 1], color="r")
  # plt.plot(xts[:, 0], xts[:, 1], color="b")
  # plt.plot(yts[:, 0], yts[:, 1], color="g")
  # plt.show()
  # IPython.embed()

  gammak = 1e1
  tlearner = NNTransformLearner(xts, dim=ntau, gammak=gammak)
  # IPython.embed()
  print("Finished setting up learner.")

  tlearner.reset_transform(init_R=R_guess.copy(), init_t=t_guess.copy())
  # Visualize
  show_x = True
  show_y = True
  show_both = True
  if visualize:
    xts_pred = tlearner.forward_simulate(xts[0], nt).detach().numpy()
    yts_pred = tlearner.forward_simulate(yts[0], nt).detach().numpy()
    # IPython.embed()
    if show_x:
      plt.plot(xts[:, 0], xts[:, 1], color="b")
      plt.plot(xts_pred[:, 0], xts_pred[:, 1], color="r")
      plt.show()
    if show_y:
      plt.plot(yts[:, 0], yts[:, 1], color="b")
      plt.plot(yts_pred[:, 0], yts_pred[:, 1], color="r")
      plt.show()
    if show_both:
      pass
    IPython.embed()

  lr = 1e-3
  max_iters = 500
  verbose = True
  config = NNTrainerConfig(lr=lr, max_iters=max_iters, verbose=verbose)
  ttrainer = NNTransformTrainer(tlearner, config)
  print("Set up trainer.")
  print("Training.")
  ttrainer.fit(yts)

  IPython.embed()
  R, t = tlearner.R.detach().numpy(), tlearner.t.detach().numpy()
  yR = (R.dot(yts.T) + t).T

  ts_pred = tlearner.forward_simulate(yR[0], nt).detach().numpy()
  # plt.plot(xR[:, 0], xR[:, 1], color="g")
  plt.plot(yts[:, 0], yts[:, 1], color="g", label="y")
  plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
  plt.plot(yR[:, 0], yR[:, 1], color="r", label="f(y)")
  plt.plot(ts_pred[:, 0], ts_pred[:, 1], color="k", label="sim")
  plt.legend()
  plt.show()
  # n = 1000
  # d = 3
  # rn = 100
  # gammak = 1.0
  # rcond = 1e-3
  # sine = True
  # rn_true = rn * (2 if sine else 1)
  # rff = grf.GaussianRandomFeatures(dim=d, rn=rn, gammak=gammak, sine=True)

  # x = np.random.rand(n, d)
  # xr = rff.computeRandomFeatures(x)
  # xr = torch.from_numpy(xr)
  # xr.requires_grad_()
  # A = torch.mm(xr.t(), xr)

  # u, s, v = torch.svd(A)
  # si = torch.where(s > rcond, 1. / s, torch.zeros_like(s))
  # Ainv = torch.mm(torch.mm(u, torch.diag(si)), v)

  # alpha = torch.mm(torch.mm(Ainv, xr.t()), y)
  # alpha_base = torch.zeros(rn_true, 1)
  # loss = torch.norm(alpha-alpha_base)

  # loss.backward()

## Multiple approaches for the Simple Maltese Cross (without DNs)
## Overall idea:
## 1. Generate two coupled systems which are asynchronous
## 2. Learn:
##   a. Transformation from each system to latent space
##   b. Dynamical system in the latent space
## 3. Predict: One stream given the other

import cvxpy as cvx
from itertools import combinations_with_replacement as combinations_w_r
from itertools import chain
import numpy as np
import os
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import dynamical_systems_learning as dsl
import gaussianRandomFeatures as grf
import synthetic.simple_systems as ss
import tfm_utils as tu
import time_sync as tsync

import IPython


torch.set_default_dtype(torch.float64)
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
class RFFTransformLearner(nn.Module):

  def __init__(
      self, dim, feature_func, target_alpha, rcond=1e-3, affine=True,
      ff_type="RFF"):
    super(RFFTransformLearner, self).__init__()
    self.dim = dim
    self.affine = affine  # For the RFFs
    self.rcond = rcond

    self.feature_func = feature_func
    self.ff_type = ff_type
    if ff_type == "RFF":
      self.update_torch_rff(feature_func)
      self._feature_func = self._compute_random_features
    elif ff_type == "poly":
      self.update_torch_poly(feature_func)
      self._feature_func = self._compute_poly_features
    else:
      raise NotImplementedError(
          "Feature function type %s not implemented." % feature_func)

    if target_alpha is not None:
      self.update_target_alpha(target_alpha)
    self.reset_transform()

    self.criterion = nn.MSELoss(size_average=False)

  def update_torch_rff(self, rff):
    if self.ff_type != "RFF":
      raise Exception(
          "Feature function type must be RFF for this. Currently: %s" %
          self.ff_type)

    assert self.dim == rff.dim
    self.sine = rff.sine
    self.rn = rff.rn
    # IPython.embed()
    self.ws = torch.from_numpy(rff.ws)
    self.ws.requires_grad_(False)
    if not self.sine:
      self.bs = torch.from_numpy(rff.bs)
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
        torch.cat((rf.t(), torch.ones(rf.shape[1], 1)), dim=1)
        if self.affine else rf.t())
    return rf_final

  def update_torch_poly(self, pf):
    if self.ff_type != "poly":
      raise Exception(
          "Feature function type must be poly for this. Currently: %s" %
          self.ff_type)

    self.degree = pf.degree
    self.include_bias = pf.include_bias

  def _combinations(self, n_features):
    #From sk-learn
    start = int(not self.include_bias)
    return chain.from_iterable(combinations_w_r(range(n_features), i)
                               for i in range(start, self.degree + 1))

  def _compute_poly_features(self, xt):
    # xt is an n x r tensor of transformed points, with the transform being
    # parameterized by trainable variables
    # IPython.embed()

    pf = []
    combs = self._combinations(xt.shape[1])
    for i, c in enumerate(combs):
      if len(c) > 0:
        pf.append(xt[:, c].prod(1).unsqueeze(1))
      else:
        pf.append(torch.ones((xt.shape[0], 1)))
    pf_final = torch.cat(pf, 1)
    return pf_final

  def update_target_alpha(self, target_alpha):
    self.target_alpha = torch.from_numpy(target_alpha)
    self.target_alpha.requires_grad_(False)

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
    si = torch.where(s >= self.rcond * s[0], 1. / s, torch.zeros_like(s))
    Ainv = torch.mm(torch.mm(v, torch.diag(si)), u.t())
    # Ainv = torch.inverse(A)
    return Ainv

  def forward_no_tf(self, x):
    # Solve for alpha without transform
    if not isinstance(x, torch.Tensor):
      # Assuming it is numpy arry
      x = torch.from_numpy(x)
    x.requires_grad_(False)


    x_ff = self._feature_func(x)
    x_next = x[1:]
    x_ff_prev = x_ff[:-1]

    A = torch.mm(x_ff_prev.t(), x_ff_prev)
    Ainv = self._pinvert_mat(A)

    alpha = torch.mm(torch.mm(Ainv, x_ff_prev.t()), x_next)
    return alpha

  def forward(self, x):
    # Solve for alpha
    if not isinstance(x, torch.Tensor):
      # Assuming it is numpy arry
      x = torch.from_numpy(x)
    x.requires_grad_(False)

    # IPython.embed()
    x_transformed = self._transform_pts(x)
    # IPython.embed()
    x_ff = self._feature_func(x_transformed)

    # Now solve ||x_rff(T) * alpha - x_transformed(T+1)||
    x_next = x_transformed[1:]
    x_ff_prev = x_ff[:-1]

    A = torch.mm(x_ff_prev.t(), x_ff_prev)
    Ainv = self._pinvert_mat(A)
    
    alpha = torch.mm(torch.mm(Ainv, x_ff_prev.t()), x_next)
    return alpha

  def loss(self, alpha):
    return self.criterion(alpha, self.target_alpha)


class RFFTrainerConfig(object):

  def __init__(self, lr=1e-3, max_iters=100, verbose=True):
    self.lr = lr
    self.max_iters = max_iters
    self.verbose = verbose


class RFFTransformTrainer(object):

  def __init__(self, tlearner, config):
    self.config = config
    self.tlearner = tlearner
    self._setup_optimizer()

  def _setup_optimizer(self):
    self.opt = optim.Adam(self.tlearner.parameters(), self.config.lr)

  def train_loop(self, x):
    self.opt.zero_grad()
    alpha = self.tlearner(x)
    self.loss = self.tlearner.loss(alpha)
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
        print("Loss: %.5f" % float(self.loss.detach()))
        print("Current transform estimate: \nR: %s\nt:%s" % 
            (self.tlearner.R.detach().numpy(), self.tlearner.t.detach().numpy()))
        print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))

    print("Training finished in %0.2f s." % (time.time() - all_start_time))


if __name__ == "__main__":

  visualize = True
  tmax = 100
  nt = 10000
  x, _, _ = ss.generate_lorenz_system(tmax, nt)

  tau = 10
  ntau = 2
  xts = tsync.compute_td_embedding(x, tau, ntau)
  # yts = tsync.compute_td_embedding(y, tau, ntau)

  # Simple rotation + translation
  # The correct transform will be the inverse of the true one.
  theta = 0 # np.pi / 4
  new_x = np.atleast_2d([np.cos(theta), np.sin(theta)]).T
  new_y = np.atleast_2d([- np.sin(theta), np.cos(theta)]).T
  scaling = np.diag([1.5, 0.75])
  R_true_base = np.c_[new_x, new_y].astype(np.float64)
  R_true = R_true_base.dot(scaling)
  t_true = 0 * np.array([3, -4], dtype=np.float64).reshape(-1, 1)
  yts = (R_true.dot(xts.T) + t_true).T

  # R_guess, t_guess = tu.guess_best_transform(yts, xts)
  # t_guess = t_guess.reshape(-1, 1)
  R_guess = np.linalg.inv(R_true_base)
  t_guess = np.zeros([ntau, 1])
  xR = (R_guess.dot(yts.T) + t_guess).T

  plt.plot(xR[:, 0], xR[:, 1], color="r")
  plt.plot(xts[:, 0], xts[:, 1], color="b")
  plt.plot(yts[:, 0], yts[:, 1], color="g")
  plt.show()
  # IPython.embed()

  load_rff = True
  rn = 500
  gammak = 1.0
  sine = True
  fl = "rff_data/rff_dim_%i_rn_%i_gammak_%.3f_sine_%i.pkl" % (
      ntau, rn, gammak, sine)
  if load_rff and os.path.exists(fl):
    rff = grf.GaussianRandomFeatures(fl=fl)
  else:  
    rff = grf.GaussianRandomFeatures(dim=ntau, rn=rn, gammak=gammak, sine=sine)
    rff.SaveToFile(fl)
  
  feature_func = rff.computeRandomFeatures

  affine = True
  rcond = 1e-10
  Ax, bx, errx = dsl.learn_linear_dynamics(xts, feature_func, affine, rcond)
  Ay, by, erry = dsl.learn_linear_dynamics(yts, feature_func, affine, rcond)

  tlearner = RFFTransformLearner(
      dim=ntau, rff=rff, target_alpha=None, rcond=rcond,
      affine=affine)
  # ap2 = tlearner.forward_no_tf(yts)
  # IPython.embed()
  print("Set up learner.")

  target_alpha = tlearner.forward_no_tf(xts).detach().numpy()  # np.c_[A, b].T
  Axf, bxf = target_alpha.T[:, :-1], target_alpha.T[:, -1]
  alpha_y = tlearner.forward_no_tf(yts).detach().numpy()  # np.c_[A, b].T
  Ayf, byf = alpha_y.T[:, :-1], alpha_y.T[:, -1]

  tlearner.update_target_alpha(target_alpha)
  tlearner.reset_transform(init_R=R_guess.copy(), init_t=t_guess.copy())
  # Visualize
  show_x = True
  show_y = True
  if visualize:
    xts_pred = [xts[0]]
    xts_predf = [xts[0]]
    for _ in range(len(xts) - 1):
      xts_pred.append(Ax.dot(feature_func(xts_pred[-1]).squeeze()) + bx)
      xts_predf.append(Axf.dot(feature_func(xts_predf[-1]).squeeze()) + bxf)
    xts_pred = np.array(xts_pred)
    xts_predf = np.array(xts_predf)

    yts_pred = [yts[0]]
    yts_predf = [yts[0]]
    for _ in range(len(yts) - 1):
      yts_pred.append(Ay.dot(feature_func(yts_pred[-1]).squeeze()) + by)
      yts_predf.append(Ayf.dot(feature_func(yts_predf[-1]).squeeze()) + byf)
    yts_pred = np.array(yts_pred)
    yts_predf = np.array(yts_predf)

    if show_x:
      plt.plot(xts[:, 0], xts[:, 1], color="b")
      # plt.plot(xts_pred[:, 0], xts_pred[:, 1], color="r")
      plt.plot(xts_predf[:, 0], xts_predf[:, 1], color="g")
      plt.show()
    if show_y:
      plt.plot(yts[:, 0], yts[:, 1], color="b")
      # plt.plot(yts_pred[:, 0], yts_pred[:, 1], color="r")
      plt.plot(yts_predf[:, 0], yts_predf[:, 1], color="g")
      plt.show()
    IPython.embed()

  lr = 1e-3
  max_iters = 500
  verbose = True
  config = RFFTrainerConfig(lr=lr, max_iters=max_iters, verbose=verbose)
  ttrainer = RFFTransformTrainer(tlearner, config)
  print("Set up trainer.")
  print("Training.")
  ttrainer.fit(yts)

  IPython.embed()
  R, t = tlearner.R.detach().numpy(), tlearner.t.detach().numpy()
  yR = (R.dot(yts.T) + t).T
  alpha2 = tlearner.forward(yts).detach().numpy().T
  A2, b2 = alpha2[:, :-1], alpha2[:, -1]

  ts0 = yR[0]
  ts_pred = [ts0]
  for _ in range(len(yR) - 1):
    ts_pred.append(A2.dot(feature_func(ts_pred[-1]).squeeze()) + b2)
  ts_pred = np.array(ts_pred)
  plt.plot(yR[:, 0], yR[:, 1], color="r")
  plt.plot(xR[:, 0], xR[:, 1], color="g")
  plt.plot(xts[:, 0], xts[:, 1], color="b")
  plt.plot(ts_pred[:, 0], ts_pred[:, 1], color="r")
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

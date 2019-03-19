## Multiple approaches for the Simple Maltese Cross (without DNs)
## Overall idea:
## 1. Generate two coupled systems which are asynchronous
## 2. Learn:
##   a. Transformation from each system to latent space
##   b. Dynamical system in the latent space
## 3. Predict: One stream given the other

import cvxpy as cvx
import itertools
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import dynamical_systems_learning as dsl
import gaussian_random_features as grf
import synthetic.simple_systems as ss
from utils import tfm_utils as tu
from utils import time_sync as tsync

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
class FFTransformLearner(nn.Module):
  def __init__(
      self, transform, feature_func, target_alpha, regularizers=[], rcond=1e-3):
      # self, dim, feature_func,  affine=True, ff_type="RFF"):
    super(FFTransformLearner, self).__init__()

    self.feature_func = feature_func
    self.transform = transform
    self.regularizers = (
        regularizers if isinstance(regularizers, list) else [regularizers])
    self.rcond = rcond

    if target_alpha is not None:
      self.update_target_alpha(target_alpha)
    self.reset_transform()

    self.criterion = nn.MSELoss(size_average=True)

  def update_target_alpha(self, target_alpha):
    self.target_alpha = torch.from_numpy(target_alpha)
    self.target_alpha.requires_grad_(False)

  def reset_transform(self, **values):
    self.transform.reset_transform(**values)
    for pname, pval in self.transform.get_parameters(as_numpy=False).items():
      self.__setattr__(pname, pval)

  def update_ff_params(self, **params):
    self.feature_func.update_params(**params)

  def transform_pts(self, x):
    return self.transform(x)

  def featurize(self, x):
    return self.feature_func(x)

  def _pinvert_mat(self, A):
    # Computes pseudo-inverse of given condition number
    u, s, v = torch.svd(A)
    si = torch.where(s >= self.rcond * s[0], 1. / s, torch.zeros_like(s))
    Ainv = torch.mm(torch.mm(v, torch.diag(si)), u.t())
    # Ainv = torch.inverse(A)
    return Ainv

  def forward_simulate(self, x, use_tf=False):
    # Solve for system parameters without transform

    if not isinstance(x, torch.Tensor):
      # Assuming it is numpy arry
      x = torch.from_numpy(x)
    x.requires_grad_(False)

    if use_tf:
      x = self.transform_pts(x)

    x_ff = self.featurize(x)
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
    x_transformed = self.transform_pts(x)
    # IPython.embed()
    x_ff = self.featurize(x_transformed)

    # Now solve ||x_ff(T) * target_alpha - x_transformed(T+1)||
    # for R, t
    x_next = x_transformed[1:]
    x_ff_prev = x_ff[:-1]
    x_next_pred = torch.mm(x_ff_prev, self.target_alpha)

    return x_next_pred, x_next

  def loss(self, xn, xn_pred):
    zeros = torch.zeros_like(xn)
    zeros.requires_grad_(False)
    obj = self.criterion(xn - xn_pred, zeros)
    obj -= 1e-1 * self.criterion(self.transform.A, torch.zeros_like(self.transform.A))
    obj -= 1e-1 * self.criterion(self.transform.B, torch.zeros_like(self.transform.B))
    
    reg = torch.tensor(0.)
    for r in self.regularizers:
      reg += r()

    return obj + reg

  # def parameters(self):
  #   params = super(FFTransformLearner, self).parameters()
  #   return itertools.chain([params, self.transform.parameter_generator()])


class FFTrainerConfig(object):

  def __init__(self, lr=1e-3, max_iters=100, verbose=True):
    self.lr = lr
    self.max_iters = max_iters
    self.verbose = verbose


class FFTransformTrainer(object):

  def __init__(self, tlearner, config):
    self.config = config
    self.tlearner = tlearner
    self._setup_optimizer()

  def _setup_optimizer(self):
    self.opt = optim.Adam(self.tlearner.parameters(), self.config.lr)

  def train_loop(self, x):
    self.opt.zero_grad()
    xn, xn_pred = self.tlearner(x)
    self.loss = self.tlearner.loss(xn, xn_pred)
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
        print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))

    print("Training finished in %0.2f s." % (time.time() - all_start_time))
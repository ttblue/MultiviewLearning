# Different feature functions with torch support
from itertools import combinations_with_replacement as cwr, chain
import numpy as np
import torch

import gaussianRandomFeatures


torch.set_default_dtype(torch.float64)


# Abstract class for feature functions
class FeatureFunction(object):

  def __call__(self, X):
    raise NotImplementedError

  def numpy_features(self, X):
    raise NotImplementedError

  def update_params(self, *params):
    raise NotImplementedError


# Random Fourier Features
class RandomFourierFeatures(FeatureFunction):

  def __init__(
      self, dim=None, rn=None, gammak=1.0, sine=True, affine=False, rff=None):

    if rff is not None:
      self.update_params(rff)
    else:
      self.dim = dim
      self.rn = rn
      self.gammak = gammak
      self.sine = sine
      self.affine = affine
      self._generateCoefficients()

  def _generateCoefficients (self):
    # From gaussianRandomFeatures.py
    ws = []
    mean = np.zeros(self.dim)
    cov = np.eye(self.dim)*(2*self.gammak)

    if self.sine:
      for _ in range(self.rn):
        ws.append(nr.multivariate_normal(mean, cov))
        self.bs = None
    else:
      bs = []
      for _ in range(self.rn):
        ws.append(nr.multivariate_normal(mean, cov))
        bs.append(nr.uniform(0.0, 2*np.pi))
      bs = np.array(bs)
      self.bs = torch.from_numpy(bs)
      self.bs.requires_grad_(False)
   
    ws = np.array(ws)
    self.ws = torch.from_numpy(ws)
    self.ws.requires_grad_(False)

  def update_params(self, rff):
    assert self.dim == rff.dim
    self.sine = rff.sine
    self.rn = rff.rn

    self.ws = torch.from_numpy(rff.ws)
    self.ws.requires_grad_(False)
    if not self.sine:
      self.bs = torch.from_numpy(rff.bs)
      self.bs.requires_grad_(False)

  def __call__(self, X):
    if not isinstance(X, torch.Tensor):
      # Assuming it is numpy arry
      X = torch.from_numpy(X)
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

  def numpy_features(self, X):
    X = np.atleast_2d(X)
    if self.sine:
      c = np.sqrt(1 / self.rn)
      rf_cos = np.cos(self.ws.dot(X.T)) * c
      rf_sin = np.sin(self.ws.dot(X.T)) * c
      return np.r_[rf_cos, rf_sin].T
    else:
      c = np.sqrt(2 / self.rn)
      rf = np.cos(self.ws.dot(X.T) + self.bs[:, None]) * c
      return rf.T


# Polynomial features
class PolynomialFeatures(FeatureFunction):

  def __init__(self, degree, include_bias=True):
    self.update_params(degree, include_bias)

  def _combinations(self, n_features):
    #From sk-learn
    start = int(not self.include_bias)
    return chain.from_iterable(cwr(range(n_features), i)
                               for i in range(start, self.degree + 1))

  def update_params(self, degree, include_bias, pfunc=None):
    # If pfunc is given, degree and include bias are ignored
    if pfunc is not None:
      degree = pfunc.degree
      include_bias = pfunc.include_bias

    self.degree = degree
    self.include_bias = include_bias

  def __call__(self, X):
    # X is an n x r tensor of transformed points, with the transform being
    # parameterized by trainable variables
    if not isinstance(X, torch.Tensor):
      # Assuming it is numpy arry
      X = torch.from_numpy(X)
    pf = []
    combs = self._combinations(X.shape[1])
    for c in combs:
      if len(c) > 0:
        pf.append(X[:, c].prod(1).unsqueeze(1))
      else:
        pf.append(torch.ones((X.shape[0], 1)))
    pf_final = torch.cat(pf, 1)
    return pf_final

  def numpy_features(self, X):
    # From scipy's PolynomialFeatures
    combs = self._combinations(X.shape[1])
    pf = []
    for c in combs:
        pf.append(X[:, c].prod(1).reshape(-1, 1))
    pf = np.concatenate(pf, axis=1)
    return pf
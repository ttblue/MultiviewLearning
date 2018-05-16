import cvxpy as cvx
import numpy as np
import scipy.spatial.distance as ssd

import matplotlib.pyplot as plt

import gaussianRandomFeatures as grf
import synthetic.simple_systems as ss
import time_sync as tsync

import IPython


def learn_linear_dynamics(
    ts, feature_func=None, affine=True, rcond=1e-3, verbose=True):
  # We assume time-delay/phase-space embedding has been done
  ts = np.atleast_2d(ts)
  tsf = ts if feature_func is None else feature_func(ts)
  tlen = ts.shape[0]

  X = np.c_[tsf[:-1], np.ones((tlen - 1, 1))] if affine else tsf[:-1]
  Y = ts[1:]

  nr = Y.shape[1]
  nc = X.shape[1]

  A = X.T.dot(X)
  Asol = np.linalg.pinv(A, rcond=rcond).dot(X.T).dot(Y).T
  obj_val = np.linalg.norm((X.dot(Asol.T) - Y))
  # A = cvx.Variable(nr, nc)
  # error = cvx.sum_squares(X*A.T - Y)
  # cnts = []
  # prob = cvx.Problem(cvx.Minimize(error), cnts)
  # obj_val = prob.solve(solver="CVXOPT", verbose=verbose)
  # Asol = np.array(A.value)
  
  bsol = np.zeros(ts.shape[1])
  if affine:
    Asol, bsol = Asol[:, :-1], Asol[:, -1]
  return Asol, bsol, obj_val


class GaussianKernel(object):
  def __init__ (self, gamma=1.0):
    self.gamma = gamma
    self.parameters = {"bandwidth": gamma}

  def __call__(self, x, y):
    x = np.squeeze(x)
    y = np.squeeze(y)

    if len(x.shape) > 1 or len(y.shape) > 1:
      x = np.atleast_2d(x)
      y = np.atleast_2d(y)
      cdist = np.squeeze(ssd.cdist(x, y, "sqeuclidean"))
      return np.exp(-self.gamma * cdist)
    else:
      return np.exp(-self.gamma * (np.linalg.norm(x - y) ** 2))


class NNDynamicsIterator(object):
  def __init__(self, X, kernel=GaussianKernel()):
    self.X = X
    self.kernel = kernel

    # Don't need to redo computation
    self.prev_output = None

  def __call__(self, Y, same=True):
    if same is True and self.prev_output is not None:
      return self.prev_output

    G = self.kernel(self.X, self.Y[:-1])
    Dinv = np.diag(1. / G.sum(1))
    output = Dinv.dot(G).dot(self.Y[1:])
    
    self.prev_output = output
    return output


if __name__ == "__main__":

  tmax = 100
  nt = 10000
  x, y, z = ss.generate_lorenz_system(tmax, nt)

  tau = 10
  ntau = 3
  ts = tsync.compute_td_embedding(x, tau, ntau)

  affine = True
  rn = 500
  gammak = 1.0
  rcond = 1e-3
  rff = grf.GaussianRandomFeatures(dim=ntau, rn=rn, gammak=gammak)
  feature_func = rff.computeRandomFeatures
  A, b, err = learn_linear_dynamics(ts, feature_func, affine, rcond)

  ts0 = ts[0]
  ts_pred = [ts0]
  for _ in range(len(ts) - 1):
    ts_pred.append(A.dot(feature_func(ts_pred[-1]).squeeze()) + b)
  ts_pred = np.array(ts_pred)
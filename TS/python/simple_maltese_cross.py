## Multiple approaches for the Simple Maltese Cross (without DNs)
## Overall idea:
## 1. Generate two coupled systems which are asynchronous
## 2. Learn:
##   a. Transformation from each system to latent space
##   b. Dynamical system in the latent space
## 3. Predict: One stream given the other

import cvxpy as cvx
import numpy as np

import matplotlib.pyplot as plt

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

def learn_linear_dynamics(ts, feature_func=None, affine=True, verbose=True):
  # We assume time-delay/phase-space embedding has been done
  ts = np.atleast_2d(ts)
  tsf = ts if feature_func is None else feature_func(ts)
  tlen = ts.shape[0]

  X = np.c_[tsf[:-1], np.ones((tlen - 1, 1))] if affine else tsf[:-1]
  Y = ts[1:]

  nr = Y.shape[1]
  nc = X.shape[1]
  A = cvx.Variable(nr, nc)

  error = cvx.sum_squares(X*A.T - Y)
  cnts = []
  prob = cvx.Problem(cvx.Minimize(error), cnts)

  obj_val = prob.solve(solver="CVXOPT", verbose=verbose)
  Asol = np.array(A.value)
  bsol = None

  if affine:
    Asol, bsol = Asol[:, :-1], Asol[:, -1]

  return Asol, bsol, obj_val


if __name__ == "__main__":

  tmax = 100
  nt = 10000
  x, y, z = lorenz.generate_lorenz_attractor(tmax, nt)

  tau = 10
  ntau = 2
  ts = tsync.compute_td_embedding(x, tau, ntau)

  affine = True
  rn = 500
  gammak = 5.0
  rff = grf.GaussianRandomFeatures(dim=ntau, rn=rn)
  feature_func = rff.computeRandomFeatures
  A, b, err = learn_linear_dynamics(ts, feature_func, affine)

  ts0 = ts[0]
  ts_pred = [ts0]
  for _ in range(len(ts) - 1):
    ts_pred.append(A.dot(feature_func(ts_pred[-1]).squeeze()) + b)
  ts_pred = np.array(ts_pred)
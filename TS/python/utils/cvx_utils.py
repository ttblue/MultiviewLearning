# Some convex-opt utils like prox operators and such.
# Most of the prox operator stuff is taken from:
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

import cvxpy as cvx
import numpy as np

# from utils import utils
import utils

import IPython


_SOLVER = cvx.GUROBI


class OptException(Exception):
  pass


_ORDER_MAP = {
    "inf": np.inf,
    "1": 1,
    "2": 2,
}
# Group norm:
def group_norm(x, G, order="inf"):
  if not isinstance(order, int) and order[0] == "L":
    order = order[1:]
  order = _ORDER_MAP[order]
  return np.sum([np.linalg.norm(x[g], order) for g in G])


# Prox operators:
def prox_generic(f, v, lmbda=1.):
  # prox_{\lambda f} (v) = argmin_x f(x) + 1/(2 \lambda) ||x-v||_2^2
  if len(f.variables()) != 1:
    raise OptException("Loss function must have exactly one variable.")

  if lmbda <= 0:
    raise OptException("lambda must be positive.")

  # Make sure at least 2d
  v = np.array(v)
  x = f.variables()[0]

  # Super hacky
  if len(v.shape) == len(x.shape) - 1 and x.shape[-1] == 1:
    if v.shape != x.shape[:-1]:
      raise OptException("Variable and v must be of same shape.")
  elif v.shape != x.shape:
    raise OptException("Variable and v must be of same shape.")

  dist_loss = 1. / (2. * lmbda) * cvx.norm2(x - v) ** 2
  prox_loss = f + dist_loss

  prob = cvx.Problem(cvx.Minimize(prox_loss))
  prob.solve(solver=_SOLVER)

  return x.value


# Some simple prox operators and operations over prox operators
# Helper functions:
def bisection_solver(f, lb, ub, tol=1e-6):
  # Solve for 0 given monotonic f and range.
  func = (lambda x: -f(x)) if f(lb) > f(ub) else f
  # lb, ub = ub, lb
  lbr, ubr = lb, ub


  x = (lb + ub) / 2.
  idx = 0
  while np.abs(func(x)) > tol and (ub - lb) > tol:
    if func(x) < 0:
      lb = x
    else:
      ub = x
    x = (lb + ub) / 2.
    idx += 1
    # if idx > 100:
    #   IPython.embed()

  return x


def soft_thresholding(v, lmbda=1.):
  return (
      np.where(v >= lmbda, v - lmbda, 0) + np.where(v <= -lmbda, v + lmbda, 0))


# Prox operators for common norms:
def prox_L2(v, lmbda=1.):
  v_norm = np.linalg.norm(v, 2)
  p = v * (0 if v_norm < lmbda else (1 - lmbda / v_norm))
  return p


def prox_L1(v, lmbda=1.):
  return soft_thresholding(v, lmbda)


# Helper function for prox operator for Linf norm
def _lambda_proj(v, lmbda=1.):
  v_abs_minus_lambda = np.abs(v) - lmbda
  return np.where(v_abs_minus_lambda > 0, v_abs_minus_lambda, 0).sum() - 1


def prox_Linf(v, lmbda=1.):
  if lmbda == 0:
    return v
  f = lambda l: _lambda_proj(v / lmbda, l)
  lb, ub = 0, np.abs(v).max()
  # IPython.embed()
  # Using Moreau decomposition
  l = bisection_solver(f, lb, ub)
  p = v - lmbda * soft_thresholding(v / lmbda, l)
  return p


_PROX_NORMS = {
    "L2": prox_L2,
    "L1": prox_L1,
    "Linf": prox_Linf,
}


# Group norm prox operator
def prox_group_norm(v, G, lmbda, norm="Linf"):
  if lmbda == 0:
    return v
  # This is just the prox operator over individual groups
  prox_func = _PROX_NORMS.get(norm, None)
  if prox_func is None:
    raise OptException("No prox operator for norm: %s." % norm)

  v = np.array(v)
  # Assume this is ok for now.
  # if not utils.is_valid_partitioning(G, v.shape[1]):
  #   raise OptException("G not a valid partition of features.")
  p = np.empty(v.shape[0])
  for g in G:
    p[g] = prox_func(v[g], lmbda)
  return p


# Prox for projections onto norm balls
def prox_proj_L2_norm_ball(v, lmbda=1., radius=1.):
  v_norm = np.linalg.norm(v, 2)
  return v if v_norm < radius else (v / v_norm) * radius


# Basic Prox operator operations
def prox_affine_addition(v, prox_f, c, lmbda):
  # g(x) = f(x) + c^T x
  # prox_{\lambda g} (v) = prox_{\lambda f}(v - \lambda c)
  return prox_f(v - lmbda * c, lmbda)


# Some optimization algorithms:
def basic_admm(
    prox_f, prox_g, dim, lmbda=0., init_x=None, init_z=None, init_u=None,
    max_iter=1000, tol=1e-6):
    
  x = np.zeros(dim) if init_x is None else init_x
  z = np.zeros(dim) if init_z is None else init_z
  u = np.zeros(dim) if init_u is None else init_u

  for itr in range(max_iter):
    x = prox_f(z - u, lmbda)
    z = prox_g(x + u, lmbda)
    u += x - z

    if np.linalg.norm(x - z) < tol:
      break

  return x

if __name__ == "__main__":
  v = np.random.randn(10)
# Some convex-opt utils like prox operators and such.

import cvxpy as cvx
import numpy as np


from utils import utils


_SOLVER = cvx.GUROBI


class OptException(Exception):
  pass


def prox(f, v, lmbda=1.):
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
# First norms:
def prox_Linf(v, lmbda):
  pass


def prox_L1(v, lmbda):
  pass


def prox_L2(v, lmbda):
  pass


# Group norm prox operator
def prox_group_Ln(v, G, lmbda):
  # This is just the prox operator over individual groups
  v = np.array(v)

  # Assume this is ok.
  # if not utils.is_valid_partitioning(G, v.shape[1]):
  #   raise OptException("G not a valid partition of features.")

# From Junier Oliva's funcLearn package:
# https://github.com/junieroliva/funcLearn

import numpy as np

import IPython


class FunctionBasis(object):
  # Abstract class for basis functions
  def __init__(self, k):
    self._default_inds = None if k is None else list(range(0, k + 1))

  def eval_basis(self, xvals):
    raise NotImplementedError()

  def project(self, fvals):
    raise NotImplementedError()


class SinusoidalBasis(FunctionBasis):
  # Sinusoidal basis
  def __init__(self, k=None, cosine_basis=True):
    self.cosine_basis = cosine_basis
    super(SinusoidalBasis, self).__init__(k)

    self._bfunc = self._cos_bfunc if cosine_basis else self._sin_cos_bfunc

  def _sin_cos_bfunc(self, x, kidx):
    feval = ((kidx == 0) +
             (kidx < 0) * np.sqrt(2.) * np.cos(2 * np.pi * kidx * x) +
             (kidx > 0) * np.sqrt(2.) * np.sin(2 * np.pi * kidx * x))
    return feval

  def _cos_bfunc(self, x, kidx):
    feval = ((kidx == 0) +
             (kidx > 0) * np.sqrt(2.) * np.cos(np.pi * kidx * x))
    return feval

  def _eval_single_basis_func(self, xvals, kidx):
    kidx = np.squeeze(kidx).reshape(1, -1)
    xvals = np.atleast_2d(xvals)
#     if kidx.shape[1] != xvals.shape[1]:
#       IPython.embed()
#       raise TypeError("Something is wrong with the input.")

    kidx = np.tile(kidx, (xvals.shape[0], 1))
    feval = self._bfunc(xvals, kidx)
    return np.prod(feval, 1)

  def eval_basis(self, xvals, inds, eval_all=True):
    # xvals: n x d data matrix of points to evaluate bases.
    # inds: n x d indices of basis to evaluate each xval point
    # OR inds: m x d array (or list) of indices for basis functions 
    # to evaluate all points at.
    # eval_all: Flag to evaluate every basis at every point 

    xvals = np.atleast_2d(xvals)
    inds = np.asarray(inds)
    n, d = xvals.shape
    m = inds.shape[0]

    if not eval_all and m != n:
      raise TypeError("Basis function indices and data don't match.")

    phix = np.ones((n, m)) if eval_all else np.ones(n)

    if eval_all:
      for midx in range(m):
        phix[:, midx] = self._eval_single_basis_func(xvals, inds[midx])
    else:
      for didx in range(d):
        # % evaluate one-d basis functions
        mav = np.max(inds[:, di])
        miv = np.min(inds[:, di])
        inds_di = list(np.range(miv, mav))

        phidi = self._bfunc(x[:, di], inds_di)
        phix *= phidi

    return phix

  def eval_func(self, xvals, inds, pcoeffs):
    phix = self.eval_basis(xvals, inds, eval_all=True)
    return phix.dot(pcoeffs)

  def _get_single_coeff(self, xvals, fvals, kidx):
    # TODO: Replace with linear regression
    n, d = xvals.shape
    kidx = np.atleast_2d(kidx)
    phix = self.eval_basis(xvals, kidx, eval_all=True)
    return phix.dot(fvals) / n

  def project(self, xvals, fvals, inds):
    xvals = np.squeeze(xvals)
    if len(xvals.shape) == 1:
      xvals = xvals.reshape(-1, 1)
    fvals = fvals.reshape(-1, 1)
    bevals = self.eval_basis(xvals, inds, eval_all=True)
    # Replaced w/ linear regression
    # IPython.embed()
    return np.linalg.pinv(bevals).dot(fvals).squeeze()
    # return bevals.T.dot(fvals) / fvals.shape[0]

  def func_approximation(self, xvals, fvals, inds):
    pcoeffs = self.project(xvals, fvals, inds)
    return self.eval_func(xvals, inds, pcoeffs)
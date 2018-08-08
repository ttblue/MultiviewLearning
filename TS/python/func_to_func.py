# Function to function regression stuff
# From Junier Oliva's funcLearn package:
# https://github.com/junieroliva/funcLearn

import numpy as np
import sys

import func_utils as fu  # :)
import gaussian_random_features as grf

import IPython


_DEFAULT_REG_COEFF = 1.0


class TripleBasisEstimator(object):

  def __init__(
    self, input_basis, output_basis, basis, fgen_args, reg=None, reg_options={},
    rcond=1e-6, verbose=True):
    # input and output basis variables are indices for the basis_func
    # basis needs to have a project function which takes x, y and inds
    # fgen_args is the list of arguments to the random feature generator
    self.input_inds = input_basis
    self.output_inds = output_basis
    self.basis = basis
    self.feature_gen = grf.GaussianRandomFeatures(
        dim=len(input_basis), **fgen_args)

    self.reg = reg
    if reg is not None:
      if reg != "ridge":
        raise NotImplementedError("%s regularization not available." % reg)
      
      self.reg_options = reg_options
      self.lm = self.reg_options.get("lm", _DEFAULT_REG_COEFF)

    self.rcond = rcond
    self.verbose = verbose

  def _project_and_featurize(self, P_dset):
    if self.verbose:
      n_pts = len(P_dset)
      aP_Us = []
      for idx, (xvals, yvals) in enumerate(P_dset):
        aP_Us.append(self.basis.project(xvals, yvals, self.input_inds))
        print("Done with %i/%i of P." % (idx + 1, n_pts), end='\r')
      print("Done with %i/%i of P." % (idx + 1, n_pts))
      aP_Us = np.atleast_2d(aP_Us)
    else:
      aP_Us = np.atleast_2d(
          [self.basis.project(xvals, yvals, self.input_inds)
           for xvals, yvals in P_dset])

    Z = self.feature_gen.computeRandomFeatures(aP_Us)
    return Z

  def _fit_no_reg(self, P_dset, Q_dset):
    Z = self._project_and_featurize(P_dset)
    if self.verbose:
      n_pts = len(Q_dset)
      Av = []
      for idx, (xvals, yvals) in enumerate(Q_dset):
        Av.append(self.basis.project(xvals, yvals, self.output_inds))
        print("Done with %i/%i of Q." % (idx + 1, n_pts), end='\r')
      print("Done with %i/%i of Q." % (idx + 1, n_pts))
      Av = np.atleast_2d(Av)
    else:
      Av = np.atleast_2d(
          [self.basis.project(xvals, yvals, self.output_inds)
           for xvals, yvals in Q_dset])

    self._Psi = np.linalg.pinv(Z, rcond=self.rcond).dot(Av)

  def _fit_ridge(self, P_dset, Q_dset):
    Z = self._project_and_featurize(P_dset)
    if self.verbose:
      n_pts = len(Q_dset)
      Av = []
      for idx, (xvals, yvals) in enumerate(Q_dset):
        Av.append(self.basis.project(xvals, yvals, self.output_inds))
        print("Done with %i/%i of Q." % (idx + 1, n_pts), end='\r')
      print("Done with %i/%i of Q." % (idx + 1, n_pts))
      Av = np.atleast_2d(Av)
    else:
      Av = np.atleast_2d(
          [self.basis.project(xvals, yvals, self.output_inds)
           for xvals, yvals in Q_dset])

    I = np.eye(Z.shape[1])
    self._Psi = np.linalg.inv(Z.T.dot(Z) + self.lm * I).dot(Z.T).dot(Av)

  def fit(self, P_dset, Q_dset):
    # P_dset, Q_dset are two datasets, each being a list of sets of samples:
    # P_dset_i -- (X_i, p(X_i)), Q_dset_i -- (X_l, q(X_l))
    # Index i in both are corresponding.
    if self.reg == None:
      self._fit_no_reg(P_dset, Q_dset)
    else:
      self._fit_ridge(P_dset, Q_dset)

  def get_output_coeffs(self, P):
    zP = self._project_and_featurize(P).squeeze()
    return self._Psi.T.dot(zP.T)

  def predict(self, P, Q_xvals):
    if not isinstance(P, list): P = [P]
    if not isinstance(Q_xvals, list): Q_xvals = [Q_xvals]
    assert len(P) == len(Q_xvals)

    fQ = self.get_output_coeffs(P)
    Qx_eval = [
        self.basis.eval_basis(qx, inds=self.output_inds, eval_all=True)
        for qx in Q_xvals]

    Q_yvals = [qx.dot(f) for qx, f in zip(Qx_eval, fQ.T)]
    return Q_yvals
import cvxpy as cvx
import numpy as np
import scipy as sp

from utils import math_utils as mu
from utils import utils


_SOLVER = cvx.GUROBI
_REGULARIZERS = {
  "L1": None,
  "Linf": None,
  "L1_inf": (lambda mat: cvx.mixed_norm(mat, "inf", 1)),
  "Linf_1": None,
}


_OPT_ALGORITHMS = ["alt_proj", "admm", "tfocs"]


class CCAConfig(object):
  # General config for all CCA classes.
  def __init__(
      self, ndim=None, info_frac=0.8, scale=True, use_diag_cov=True,
      regularizer="L_inf", lmbda=1.0, gamma=1.0, opt_algorithm="alt_proj",
      init="auto", max_iter=100, tol=1e-3, verbose=True):
    self.ndim = ndim
    self.info_frac = info_frac
    self.scale = scale

    # Not for basic CCA
    self.use_diag_cov = use_diag_cov

    self.regularizer = regularizer
    self.lmbda = lmbda
    self.gamma = gamma

    self.opt_algorithm = opt_algorithm
    self.init = init

    self.tol = tol
    self.max_iter = max_iter

    self.verbose = verbose


class CCA(object):
  def __init__(self, config):
    self.config = config
    # self.ndim = ndim
    # self.info_frac = info_frac
    # self.scale = scale

  def fit(self, Xs, Ys):
    Xs = np.array(Xs)
    Ys = np.array(Ys)

    if Xs.shape[0] != Ys.shape[0]:
      raise ModelException(
          "Xs and Ys don't have the same number of data points.")

    # if self.config.ndim is not None:
    #   cca_model = scd.CCA(self.config.ndim, scale=self.config.scale)
    #   cca_model.fit(Xs, Ys)
    #   X_proj, Y_proj = cca_model.transform(Xs, Ys)
    #   Xw, Yw = cca_model.x_weights_, cca_model.y_weights_
    # else:
    Xs_data = mu.shift_and_scale(Xs, scale=self.config.scale)
    Ys_data = mu.shift_and_scale(Ys, scale=self.config.scale)

    X_centered = Xs_data[0]
    Y_centered = Ys_data[0]
    n = X_centered.shape[0]
    X_cov = X_centered.T.dot(X_centered) / n
    Y_cov = Y_centered.T.dot(Y_centered) / n

    # IPython.embed()

    X_cov_sqrt_inv = mu.sym_matrix_power(X_cov, -0.5)
    Y_cov_sqrt_inv = mu.sym_matrix_power(Y_cov, -0.5)

    cov = X_centered.T.dot(Y_centered) / n
    normalized_cov = X_cov_sqrt_inv.dot(cov.dot(Y_cov_sqrt_inv))
    U, S, VT = np.linalg.svd(normalized_cov)

    ndim = self.config.ndim
    if ndim is None:
      ndim = mu.get_index_p_in_ordered_pdf(S, self.config.info_frac)
    self.Wx = X_cov_sqrt_inv.dot(U[:, :ndim])
    self.Wy = Y_cov_sqrt_inv.dot(VT[:ndim].T)
    self.X_proj = X_centered.dot(Wx)
    self.Y_proj = Y_centered.dot(Wy)
    self.S = S[:ndim]

    self._X_cov = X_cov
    self._Y_cov = Y_cov
    self._XY_cov = cov

    return self


def _is_valid_partitioning(G, dim):
  if np.sum([len(g) for g in G]) != dim:
    return False

  flat_G = utils.flatten(G)
  if len(flat_G) != dim:
    return False

  d_range = list(range(dim))
  for g in flat_G:
    if g not in d_range:
      return False
  return True

# Loss function, given a partition of the features Gx for X, Gy for Y:
# \min_{u, v} -\tr{u^T X^T Y v} + \lambda \sum_{g \in G_x} R(u_g) +
#             \gamma \sum_{h \in G_y} R(v_h)
# where R is some regularizer (like L_inf) to encourage group sparsity.
# Index-set subscript indicates sub-matrix with the corresponding rows.
class GroupSparseCCA(CCA):

  def __init__(self, config):
    super(CCA, self).__init__(config)

  def _load_funcs(self):
    self._r_func = _REGULARIZERS.get(self.config.regularizer, None)
    if self._r_func is None:
      raise classifier.ClassiferException(
          "Regularizer %s not available." % self.config.regularizer)

    _opt_func_map = {
        "alt_proj": self._alternating_projections,
    }
    self._opt_func  = _opt_func_map.get(self.config.opt_algorithm)
    if self._opt_func is None:
      raise classifier.ClassiferException(
          "Optimization algorithm %s not available." %
          self.config.opt_algorithm)

  def _initialize_uv (self, X, Y):
    _, S, VT = np.linalg.svd(X, full_matrices=False)
    ux_init = VT[:self.config.ndim].T / S[:self.config.ndim]
    _, S, VT = np.linalg.svd(Y, full_matrices=False)
    vy_init = VT[:self.config.ndim].T / S[:self.config.ndim]
    return ux_init, vy_init

  def _admm(self):
    # Solve using linearized ADMM
    pass

  def _alternating_projections(self):
    # Similar to ADMM but simply alternating between solving for each variable
    # keeping the other one fixed.

    X_centered, Y_centered = self._X_centered, self._Y_centered
    dx, dy = X_centered.shape[0], Y_centerd.shape[0]

    if self.config.init == "auto":
      ux_init, vy_init = self._initialize_uv(X_centered, Y_centerd)
    else:
      raise classifier.ClassificationException(
          "Initialization method %s unavailable." % self.config.init)

    # We can initialize the problem outside
    ux = cvx.Variable((dx, self.config.ndim))
    vy = cvx.Variable((dy, self.config.ndim))
    ux_fixed = cvx.Parameter((dx, self.config.ndim))
    vy_fixed = cvx.Parameter((dy, self.config.ndim))

    ux_cov_loss = -cvx.trace((Xs_centered * ux).T * (Ys_centered * vy_fixed))
    ux_reg = np.sum([self._r_func(ux[g].T) for g in self._Gx])
    ux_obj = ux_cov_loss + self.config.lmbda * ux_reg
    ux_cnstrs = [
        ux.T * (X_centered.T).dot(X_centered) * ux == np.eye(self.config.ndim)]
    ux_prob = cvx.Problem(cvx.Minimize(ux_obj), ux_cnstrs)

    vy_cov_loss = -cvx.trace((Xs_centered * ux_fixed).T * (Ys_centered * vy))
    vy_reg = np.sum([self._r_func(vy[g].T) for g in self._Gy])
    vy_obj = vy_cov_loss + self.config.gamma * vy_reg
    vy_cnstrs = [
        vy.T * (Y_centered.T).dot(Y_centered) * vy == np.eye(self.config.ndim)]
    vy_prob = cvx.Problem(cvx.Minimize(vy_obj), vy_cnstrs)

    ux_opt, vy_opt = ux_init, vy_init
    for itr in range(self.config.max_iter):

      # Solve for u.
      if self.config.verbose:
        print("Iter %i: Solving for ux..." % (itr + 1))
      ux.value = ux_opt
      vy_fixed.value = vy_opt
      ux_prob.solve(solve=_SOLVER, warm_start=True)
      ux_opt = ux.value
      if self.config.verbose:
        print("Current objective value: %.3f\n" % ux_prob.value)

      # Solve for u.
      if self.config.verbose:
        print("Iter %i: Solving for vy..." % (itr + 1))
      vy.value = vy_opt
      ux_fixed.value = ux_opt
      vy_prob.solve(solve=_SOLVER, warm_start=True)
      vy_opt = vy.value
      if self.config.verbose:
        print("Current objective value: %.3f" % vy_prob.value)

      if np.abs(ux_prob.value - vy_prob) < self.config.tol:
        if self.config.verbose:
          print("Change in objective below tolerance. Done.")
        break
    return ux_opt, vy_opt

  def fit(self, Xs, Ys, Gx, Gy):

    self._load_funcs()

    Xs = np.array(Xs)
    Ys = np.array(Ys)
    n = Xs.shape[0]
    dx, dy = Xs.shape[1], Ys.shape[1]

    if Ys.shape[0] != n:
      raise ModelException(
          "Xs and Ys don't have the same number of data points.")

    if not _is_valid_partitioning(Gx, dx) or not _is_valid_partitioning(Gy, dy):
      raise ModelException(
          "Index groupings not valid partition of feature dimension.")

    Xs_data = mu.shift_and_scale(Xs, scale=self.config.scale)
    Ys_data = mu.shift_and_scale(Ys, scale=self.config.scale)
    self._X_centered, self._Y_centered = Xs_data[0], Ys_data[0]
    self._Gx, self._Gy = Gx, Gy

    # Optimization function
    self._ux, self._vy = self._opt_func()
    # ux = cvx.Variable((dx, self.config.ndim))
    # vy = cvx.Variable((dy, self.config.ndim))

    # # Trace objective for covariance maximization
    # cov_loss = -cvx.trace((Xs_centered * ux).T * (Ys_centered * uy))
    # # Group regularization for ux, vy
    # xreg, yreg = 0, 0
    # for g in gx:
    #   xreg += self._r_func(ux[g].T)
    # for g in gy:
    #   yreg += self._r_func(vy[g].T)
    # # Constraints for orthogonality of projections
    # ux_constraint = 


    # obj = cvx.Minimize()
    # prob = cvx.Problem()
    return self


def correlation_info(v_Xs):
  # This is just for assuming that the views are not0 indexed from 0 to K-1. 
  n_views = len(v_Xs)
  views = list(range(n_views))
  v_map = {i:vi for i, vi in zip(views, np.sort(list(v_Xs.keys())))}

  # concat_X = np.concatenate([v_Xs[v_map[i]] for i in views], axis=1)
  cca_info = {}
  for vi in views:
    X = v_Xs[v_map[vi]]
    X_all_other = np.concatenate(
        [v_Xs[v_map[i]] for i in views if i != vi], axis=1)
    cca_info[vi] = CCA(X, X_all_other, info_frac=1.0)

  IPython.embed()
  return cca_info




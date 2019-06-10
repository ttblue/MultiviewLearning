# Some approaches to learn CCA/PCA/CAA embeddings
# Some of the resources used for this:
# Structured sparse CCA: https://arxiv.org/pdf/1705.10865.pdf
# Prox algorithms: https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sys
import time

from models import classifier
from utils import cvx_utils
from utils import math_utils
from utils import utils

import IPython

_EPS = 1e-6

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
      regularizer="Linf", tau_u=1.0, tau_v=1.0, lmbda=1., mu=1.,
      opt_algorithm="alt_proj", init="auto", max_inner_iter=1000, max_iter=100,
      tol=1e-6, sp_tol=1e-5, plot=True, verbose=True):
    self.ndim = ndim
    self.info_frac = info_frac
    self.scale = scale

    # Not for basic CCA
    self.use_diag_cov = use_diag_cov

    self.regularizer = regularizer
    self.tau_u = tau_u
    self.tau_v = tau_v

    # Misc. optimization hyperparameters
    # Right now required for linearized ADMM
    self.lmbda = lmbda
    self.mu = mu

    self.opt_algorithm = opt_algorithm
    self.init = init

    self.tol = tol
    self.sp_tol = sp_tol
    self.max_inner_iter = max_inner_iter
    self.max_iter = max_iter

    self.plot = plot
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
    Xs_data = math_utils.shift_and_scale(Xs, scale=self.config.scale)
    Ys_data = math_utils.shift_and_scale(Ys, scale=self.config.scale)

    X_centered = Xs_data[0]
    Y_centered = Ys_data[0]
    n = X_centered.shape[0]
    X_cov = X_centered.T.dot(X_centered) / n
    Y_cov = Y_centered.T.dot(Y_centered) / n

    # IPython.embed()

    X_cov_sqrt_inv = math_utils.sym_matrix_power(X_cov, -0.5)
    Y_cov_sqrt_inv = math_utils.sym_matrix_power(Y_cov, -0.5)

    cov = X_centered.T.dot(Y_centered) / n
    normalized_cov = X_cov_sqrt_inv.dot(cov.dot(Y_cov_sqrt_inv))
    U, S, VT = np.linalg.svd(normalized_cov)

    ndim = self.config.ndim
    if ndim is None:
      ndim = math_utils.get_index_p_in_ordered_pdf(S, self.config.info_frac)
    self.Wx = X_cov_sqrt_inv.dot(U[:, :ndim])
    self.Wy = Y_cov_sqrt_inv.dot(VT[:ndim].T)
    self.X_proj = X_centered.dot(Wx)
    self.Y_proj = Y_centered.dot(Wy)
    self.S = S[:ndim]

    self._X_cov = X_cov
    self._Y_cov = Y_cov
    self._XY_cov = cov

    return self


def is_valid_partitioning(G, dim):
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
class GroupRegularizedCCA(CCA):

  def __init__(self, config):
    super(GroupRegularizedCCA, self).__init__(config)
    if self.config.use_diag_cov:
      raise NotImplementedError(
          "Using diagonal covariance not implemented.")
    self._training = True

  # def _load_funcs(self):
    # self._r_func = _REGULARIZERS.get(self.config.regularizer, None)
    # if self._r_func is None:
    #   raise classifier.ClassifierException(
    #       "Regularizer %s not available." % self.config.regularizer)

    # self._opt_func  = _opt_func_map.get(self.config.opt_algorithm)
    # if self._opt_func is None:
    #   raise classifier.ClassifierException(
    #       "Optimization algorithm %s not available." %
    #       self.config.opt_algorithm)

  def _initialize_uv (self, X, Y):
    _, Sx, VxT = np.linalg.svd(X, full_matrices=False)
    ux_init = VxT[:self.config.ndim].T
    _, Sy, VyT = np.linalg.svd(Y, full_matrices=False)
    vy_init = VyT[:self.config.ndim].T

    if not self.config.use_diag_cov:
      ux_init /= Sx[:self.config.ndim]
      vy_init /= Sy[:self.config.ndim]

    return ux_init, vy_init

  def _linearized_admm_single_var (
      self, prox_f, prox_g, X, U, lmbda=1., mu=1., init_u=None, init_z=None,
      init_p=None):
    # Solve using linearized ADMM
    # Details: https://arxiv.org/pdf/1705.10865.pdf
    # Slight change: Solving for --
    # min_{x, z} f(x) + g(z)
    # s.t. [A1; A2] x - [I; 0]z = 0
    # -- The paper is precise about solving for the canonical vectors 2 onwards.

    n, dim = X.shape
    nvec = self._cvec_id
    u = np.zeros(dim) if init_u is None else init_u
    z = np.zeros(n) if init_z is None else init_z
    p = np.zeros(n + nvec) if init_p is None else init_p

    I = np.eye(dim)
    zpad = np.zeros(nvec)

    A = X if U.shape[1] == 0 else np.r_[X, U.T.dot(X.T.dot(X))]
    B = lambda z: -np.r_[z, zpad]

    # lmbda = lmbda * n #np.sqrt(n)
    primal_residual = []
    dual_residual = []
    obj_vals = []
    # IPython.embed()
    # uns, zns, pns = [], [], []
    def temp_prox(v, mu):
      return cvx_utils.soft_thresholding(v + mu * self._c, mu * self._tau)

    # if self._cvec_id > 0:
    #   IPython.embed()
    u_var = cvx.Variable(dim)
    z_fixed = cvx.Parameter(n + nvec)
    p_fixed = cvx.Parameter(n + nvec)
    f_obj = (
        self._c * u_var +
        self._tau * cvx_utils.group_norm(
            u_var, self._G, order=self.config.regularizer, use_cvx=True) +
        0.5 / lmbda * cvx.norm(A * u_var + z_fixed + p_fixed) ** 2)
    prob = cvx.Problem(cvx.Minimize(f_obj))
    def temp_min_u(zf, pf):
      z_fixed.value = zf
      p_fixed.value = pf
      try:
        prob.solve(solver=_SOLVER, warm_start=True)
      except KeyError as e:
        print("        Error using solver %s: %s.\n"
              "        Using default solver." %
              (_SOLVER, e))
        try:
          prob.solve()#warm_sart=True)
        except Exception as e2:
          print("Issue with solvers.")
          IPython.embed()
      return u_var.value

      # return (
      #     np.where(x + mu * c > tau * mu, v + mu * c - tau * mu, 0) + 
      #     np.where(x + mu * c < -tau * mu, v + mu * c + tau * mu, 0))
    def obj(u, z, p, lin=False, u0=None):
      fval1 = u.dot(self._c)
      fval2 = (
          self._tau * cvx_utils.group_norm(u, self._G, self.config.regularizer))
      pval = (1 / lmbda * A.T.dot(A.dot(u0) + B(z)).dot(u) if lin else 
              0.5 / lmbda * np.linalg.norm(A.dot(u) + B(z) + p))
      cval = mu / 2 * np.linalg.norm(u - u0) ** 2 if lin else 0
      # pval = (1 / lmbda * A.T.dot(A.dot(u0) + B(z)).dot(u) if lin else
      #         p.T.dot(A.dot(u) + B(z)))
      # cval = (1 / (2 * mu) * np.linalg.norm(u - u0)**2 if lin else
      #         1 / (2 * lmbda) * np.linalg.norm(A.dot(u) + B(z))**2)
      return fval1, fval2, pval, cval

    # IPython.embed()
    for itr in range(self.config.max_inner_iter):
      u0, z0, p0 = u, z, p
      # u = prox_f(u - mu / lmbda * A.T.dot(A.dot(u) + B(z) + p), mu)
      # u = temp_prox(u - mu / lmbda * A.T.dot(A.dot(u) + B(z) + p), mu)
      # Solving original problem for u
      u = temp_min_u(B(z), p)
      z = prox_g(X.dot(u) + p[:n], lmbda)
      p += A.dot(u) + B(z)
      # Old updates. They work for only first canonical vector.
      # u = prox_f(u - mu / lmbda * X.T.dot(X.dot(u) - z + p), mu)
      # z = prox_g(X.dot(u) + p, lmbda)
      # p += X.dot(u) - z
      # uns.append(np.linalg.norm(u-u0))
      # zns.append(np.linalg.norm(z-z0))
      # pns.append(np.linalg.norm(p-p0))
      primal_residual.append(np.linalg.norm(A.dot(u) + B(z)))
      dual_residual.append(np.linalg.norm(A.T.dot(B(z - z0) / lmbda)))#(A.dot(u) + B(z)).dot(p))
      obj_vals.append(obj(u, z, p, False, u0))
      # if np.linalg.norm(X.dot(u) - z) < self.config.tol:
      #   break
      dvecs = [np.linalg.norm(dvec) for dvec in [u - u0, z - z0, p - p0]]
      if np.all(np.array(dvecs) < self.config.tol):
        break
      if self.config.verbose:
        fval1, fval2, pval, cval = obj_vals[-1]
        print("      Inner iter %i out of %i."
              "      Obj. val: %.6f, %.6f, %.6f, %.6f"
              "      Cnt. val: %.6f" %
              (itr + 1, self.config.max_inner_iter,
               fval1, fval2 / self._tau, pval, cval,
               primal_residual[-1]), end='\r')
        sys.stdout.flush()
    # IPython.embed()
    # plt.plot(primal_residual, color='r', label='primal residual')
    # plt.plot(dual_residual, color='b', label='dual residual')
    if self.config.plot:
      obj_plot = [np.sum(qd) for qd in obj_vals]
      plt.plot(obj_plot, color='g', label='obj')
      plt.legend()
      plt.show(block=False)
      plt.pause(1.5)
      plt.close()
    if np.any(np.array(obj_vals[-2:]) > 1e5):
      IPython.embed()
    if self.config.verbose:
      fval1, fval2, pval, cval = obj_vals[-1]
      print("      Inner iter %i out of %i."
            "      Obj. val: %.6f, %.6f, %.6f, %.6f"
            "      Cnt. val: %.6f" %
            (itr + 1, self.config.max_inner_iter,
             fval1, fval2 / self._tau, pval, cval,
             primal_residual[-1]))
      sys.stdout.flush()

    return u, z, p

  def _hand_off_to_solver(self, var_name):
    self._var_name = var_name
    if var_name == "u":
      X = self._X
      U = self._ux

      c = -X.T.dot(self._Y.dot(self._v))
      G = self._Gx
      tau = self.config.tau_u

      init_u = self._u
      init_z = X.dot(init_u)
      init_p = self._pu
    else:
      X = self._Y
      U = self._vy

      c = -X.T.dot(self._X.dot(self._u))
      G = self._Gy
      tau = self.config.tau_v

      init_u = self._v
      init_z = X.dot(init_u)
      init_p = self._pv

    # Temp stuff
    # Fixing tau so that the problem doesn't blow up:
    # tau = max(tau, np.abs(c).mean()) + _EPS
    lmbda = self.config.lmbda #* X.shape[0]
    self._c = c
    self._tau = tau
    self._G = G
    self._lmbda = lmbda
    ###

    prox_group_norm = (
        lambda u, lmbda: cvx_utils.prox_group_norm(
            u, G, lmbda * tau, norm=self.config.regularizer))
    prox_f = (
        lambda u, lmbda: cvx_utils.prox_affine_addition(
            u, prox_group_norm, c, lmbda))
    prox_g = cvx_utils.prox_proj_L2_norm_ball

    u, z, p = self._linearized_admm_single_var(
        prox_f=prox_f, prox_g=prox_g, X=X, U=U, lmbda=lmbda, mu=self.config.mu,
        init_u=init_u, init_z=init_z, init_p=init_p)
    if var_name == "u":
      self._u, self._zu, self._pu = u, z, p
    else:
      self._v, self._zv, self._pv = u, z, p

  def _objective_value(self):
    obj = -(self._u.T.dot(self._X.T)).dot(self._Y.dot(self._v))
    order = self.config.regularizer
    obj += self.config.tau_u * cvx_utils.group_norm(self._u, self._Gx, order)
    obj += self.config.tau_v * cvx_utils.group_norm(self._v, self._Gy, order)

    return obj

  def _constraint_value(self, var_name):
    X, U, u, z = (
        (self._X, self._ux, self._u, self._zu) if var_name == "u" else
        (self._Y, self._vy, self._v, self._zv))
    A = X if U.shape[1] == 0 else np.r_[X, U.T.dot(X.T.dot(X))]
    z = np.r_[z, np.zeros(U.shape[1])]

    return np.linalg.norm(A.dot(u) - z)

  def _alternating_projections_single_canonical_vector_pair(self):
    # Alternate between u and v to solve for next pair of canonical vectors.
    self._u = self._ux_init[:, self._cvec_id]
    self._v = self._vy_init[:, self._cvec_id]
    self._pu, self._pv = None, None

    for itr in range(self.config.max_iter):
      if self.config.verbose:
        u_start_time = time.time()
        print("  Canonical vector %i: Iter %i of %i." % (
            self._cvec_id + 1, itr + 1, self.config.max_iter))
        print("    Solving for u...")

      self._hand_off_to_solver("u")
      if np.isnan(self._objective_value()):
        # if self.config.verbose:
        print("NaN value for objective. Cannot continue.")
        self._training = False
        break

      if self.config.verbose:
        v_start_time = time.time()
        print("    Solving for u... Done in %.2fs" %
              (v_start_time - u_start_time))
        print("      Objective value: %.6f" % self._objective_value())
        print("      Constraint satisfaction: %.6f" %
              self._constraint_value("u"))
        print("    Solving for v...")

      self._hand_off_to_solver("v")
      if np.isnan(self._objective_value()):
        # if self.config.verbose:
        print("NaN value for objective. Cannot continue.")
        self._training = False
        break

      if self.config.verbose:
        print("    Solving for v... Done in %.2fs" %
              (time.time() - v_start_time))
        print("      Objective value: %.6f" % self._objective_value())
        print("      Constraint satisfaction: %.6f" %
              self._constraint_value("v"))
    # Add learned projection vectors to bases
    u = np.where(np.abs(self._u) >= self.config.sp_tol, self._u, 0)
    v = np.where(np.abs(self._v) >= self.config.sp_tol, self._v, 0)
    self._ux = np.c_[self._ux, u.reshape(-1, 1)]
    self._vy = np.c_[self._vy, v.reshape(-1, 1)]

  def _optimize(self):
    # Similar to ADMM but simply alternating between solving for each variable
    # keeping the other one fixed.
    # X_centered, Y_centered = self._X_centered, self._Y_centered
    # dx, dy = X_centered.shape[1], Y_centered.shape[1]

    if self.config.init == "auto":
      self._ux_init, self._vy_init = self._initialize_uv(self._X, self._Y)
    else:
      raise classifier.ClassificationException(
          "Initialization method %s unavailable." % self.config.init)

    if self._ux is None or self._vy is None:
      init_dim = 0  
      self._ux = np.empty((self._X.shape[1], 0))
      self._vy = np.empty((self._Y.shape[1], 0))
    else:
      init_dim = self._ux.shape[1]
      if self._vy.shape[1] != init_dim:
        raise classifier.ClassificationException(
            "Initial U/V projection values do not have matching dimensions.")

    for self._cvec_id in range(init_dim, self.config.ndim):
      if self.config.verbose:
        cvec_start_time = time.time()
        print("\nSolving for canonical vector %i out of %i." % 
              (self._cvec_id + 1, self.config.ndim))

      self._alternating_projections_single_canonical_vector_pair()
      if not self._training:
        print("Breaking out of training.")
        break

      if self.config.verbose:
        print("Solved for canonical vector %i in %.2fs " % 
              (self._cvec_id + 1, time.time() - cvec_start_time))

  def fit(self, Xs, Ys, Gx, Gy, ux_init=None, vy_init=None):
    self._training = True
    # self._load_funcs()
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    n = Xs.shape[0]
    dx, dy = Xs.shape[1], Ys.shape[1]

    if Ys.shape[0] != n:
      raise ModelException(
          "Xs and Ys don't have the same number of data points.")

    if (not utils.is_valid_partitioning(Gx, dx) or
        not utils.is_valid_partitioning(Gy, dy)):
      raise ModelException(
          "Index groupings not valid partition of feature dimension.")

    self._ux, self._vy = ux_init, vy_init

    Xs_data = math_utils.shift_and_scale(Xs, scale=self.config.scale)
    Ys_data = math_utils.shift_and_scale(Ys, scale=self.config.scale)
    self._X, self._Y = Xs_data[0], Ys_data[0]
    self._Gx, self._Gy = Gx, Gy
    # Optimization function
    if self.config.verbose:
      train_start_time = time.time()
    try:
      self._optimize()
    except KeyboardInterrupt:
      print("Training interrupted. Exiting...")
    self._training = False
    if self.config.verbose:
      print("Finished training in %.2fs." % (time.time() - train_start_time))

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

  return cca_info


### Old, incorrect code for alternating projections:
    # # We can initialize the problem outside
    # ux = cvx.Variable((dx, self.config.ndim))
    # vy = cvx.Variable((dy, self.config.ndim))
    # ux_fixed = cvx.Parameter((dx, self.config.ndim))
    # vy_fixed = cvx.Parameter((dy, self.config.ndim))
    # # Using Identity for diagonal covariance
    # x_cov = 1 if self.config.use_diag_cov else (X_centered.T).dot(X_centered)
    # y_cov = 1 if self.config.use_diag_cov else (Y_centered.T).dot(Y_centered)

    # ux_cov_loss = -cvx.trace((X_centered * ux).T * (Y_centered * vy_fixed))
    # ux_reg = np.sum([self._r_func(ux[g].T) for g in self._Gx])
    # ux_obj = ux_cov_loss + self.config.lmbda * ux_reg
    # ux_constraints = [] # [ux.T * x_cov * ux <= np.eye(self.config.ndim)]
    # ux_prob = cvx.Problem(cvx.Minimize(ux_obj), ux_constraints)

    # vy_cov_loss = -cvx.trace((X_centered * ux_fixed).T * (Y_centered * vy))
    # vy_reg = np.sum([self._r_func(vy[g].T) for g in self._Gy])
    # vy_obj = vy_cov_loss + self.config.gamma * vy_reg
    # vy_constraints = [] # [vy.T * y_cov * vy <= np.eye(self.config.ndim)]
    # vy_prob = cvx.Problem(cvx.Minimize(vy_obj), vy_constraints)

    # ux_opt, vy_opt = ux_init, vy_init
    # for itr in range(self.config.max_iter):

    #   # Solve for u.
    #   if self.config.verbose:
    #     print("\nIter %i: Solving for ux..." % (itr + 1))
    #   ux.value = ux_opt
    #   vy_fixed.value = vy_opt
    #   ux_prob.solve(solver=_SOLVER, warm_start=True)
    #   ux_opt = ux.value
    #   if self.config.verbose:
    #     print("Current objective value: %.6f" % ux_prob.value)
    #     print("Covariance maximization loss: %.6f" % ux_cov_loss.value)
    #     print("Regularization loss: %.6f" % ux_reg.value)

    #   # Solve for u.
    #   if self.config.verbose:
    #     print("\nIter %i: Solving for vy..." % (itr + 1))
    #   vy.value = vy_opt
    #   ux_fixed.value = ux_opt
    #   vy_prob.solve(solver=_SOLVER, warm_start=True)
    #   vy_opt = vy.value
    #   if self.config.verbose:
    #     print("Current objective value: %.6f" % vy_prob.value)
    #     print("Covariance maximization loss: %.6f" % vy_cov_loss.value)
    #     print("Regularization loss: %.6f" % vy_reg.value)

    #   if np.abs(ux_prob.value - vy_prob) < self.config.tol:
    #     if self.config.verbose:
    #       print("Change in objective below tolerance. Done.")
    #     break
    # return ux_opt, vy_opt

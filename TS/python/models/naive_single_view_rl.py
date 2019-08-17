# Naive single-view solvers defined for naive MV-RL.
import cvxpy as cvx
import numpy as np
import time
import torch

from models.model_base import ModelException
from utils import cvx_utils


import IPython


_SOLVER = cvx.GUROBI
_OBJ_ORDER = ["error", "gs", "reg", "total"]


class AbstractSVSConfig(object):
  def __init__(self, verbose, *args, **kwargs):
    self.verbose = verbose


# Simple abstract base class 
class AbstractSingleViewSolver(object):
  def __init__(self, view_id, config):
    self.view_id = view_id
    self.config = config

    self._has_data = False
    self.projections = None

    self.has_history = False

  def initialize(self):
    raise NotImplemented("Abstract class method.")

  def set_data(self, data):
    # raise NotImplemented("Abstract class method.")
    self._nviews = len(data)
    self._view_data = data[self.view_id]
    self._npts, self._dim = self._view_data.shape
    self._rest_data = {vi:data[vi] for vi in data if vi != self.view_id}
    self._rest_dims = {vi:vd.shape[1] for vi, vd in self._rest_data.items()}

    self._has_data = True

  def fit(self):
    raise NotImplemented("Abstract class method.")

  def compute_projections(self):
    raise NotImplemented("Abstract class method.")

  def get_objective(self, obj_type):
    raise NotImplemented("Abstract class method.")


################################################################################
# CVX based optimization
################################################################################


class SVOSConfig(AbstractSVSConfig):
  def __init__(
      self, group_regularizer, global_regularizer, lambda_group, lambda_global,
      n_solves, lambda_group_init, lambda_group_beta, resolve_change_thresh,
      n_resolve_attempts, sp_eps, *args, **kwargs):
    super(SVOSConfig, self).__init__(*args, **kwargs)

    self.group_regularizer = group_regularizer
    self.global_regularizer = global_regularizer
    self.lambda_group = lambda_group
    self.lambda_global = lambda_global

    self.n_solves = n_solves
    self.lambda_group_init = lambda_group_init
    self.lambda_group_beta = lambda_group_beta
    # self.global_reg_beta = global_reg_beta

    self.resolve_change_thresh = resolve_change_thresh
    self.n_resolve_attempts = n_resolve_attempts

    self.sp_eps = sp_eps


class OptimizationSolver(AbstractSingleViewSolver):
  def __init__(self, view_id, config):
    super(OptimizationSolver, self).__init__(view_id, config)

    self._p_vars = {}
    # self._G = None
    self._error = None
    self._gs_reg = None
    self._global_reg = None
    self._prob = None

  def initialize(self):
    if not self._has_data:
      raise ModelException("Data has not been set. Use set_data function.")
    # group_inds = [0] + np.cumsum(self._rest_dims).tolist()
    # self._G = [np.arange(sidx, eidx) for sidx, eidx in 
    #            zip(group_inds[:-1], group_inds[1:])]
    self._lambda_group = cvx.Parameter((), value=0.)
    self._lambda_global = cvx.Parameter((), value=self.config.lambda_global)

    self._p_vars = {
        vi:cvx.Variable((self._rest_dims[vi], self._dim))
        for vi in self._rest_dims
    }
    # IPython.embed()
    self._reconstruction = np.sum(
        [(self._rest_data[vi] * self._p_vars[vi]) for vi in self._p_vars])
    self._error = cvx.norm(self._reconstruction - self._view_data) / self._npts
    self._group_reg = np.sum(
        [cvx_utils.cvx_norm_wrapper(p_var, self.config.group_regularizer)
         for p_var in self._p_vars.values()])
    self._global_reg = cvx_utils.cvx_norm_wrapper(
        cvx.vstack(self._p_vars.values()), self.config.global_regularizer)
    self._final_obj = (
        self._error + cvx.abs(self._lambda_group) * self._group_reg +
        cvx.abs(self._lambda_global) * self._global_reg)

    self._obj_map = {
        "error": self._error,
        "gs": self._group_reg,
        "reg": self._global_reg,
        "total": self._final_obj
    }

    self._prob = cvx.Problem(cvx.Minimize(self._final_obj))

  def compute_projections(self):
    projs = {vi: p_var.value for vi, p_var in self._p_vars.items()}
    self.projections = {
        vi: np.where(np.abs(proj) < self.config.sp_eps, 0., proj)
        for vi, proj in projs.items()}

  def get_objective(self, obj_type="all"):
    if obj_type == "all":
      return [self._obj_map[otype].value for otype in _OBJ_ORDER]
    return self._obj_map.get(obj_type, "total").value

  def _solve(self, verbose=None):
    verbose = self.config.verbose if verbose is None else verbose
    try:
      self._prob.solve(
          solver=_SOLVER, warm_start=True, verbose=verbose)
    except (KeyError, cvx.SolverError) as e:
      try:
        self._prob.solve(verbose=verbose)
      except Exception as e2:
        raise ModelException("Issues with solvers: \n(1) %s\n(2) %s" % (e, e2))
    # return self

  def _check_if_resolve(self, curr_objs):
    # TODO: Maybe a better way to check if resolve needed.
    if self._prev_objs is None:
      return False
    # Only using reconstruction error
    prev_error = self._prev_objs[0]
    curr_error = curr_objs[0]
    return (curr_error - prev_error > self.config.resolve_change_thresh)

  # def iterated_solve(self):
  def fit(self):
    self._proj_history = []
    self._objs_history = []
    self._lambda_group_history = []
    self._main_iters = []
    self._prev_objs = None

    _n_solves = self.config.n_solves
    _n_resolve_attempts = self.config.n_resolve_attempts
    _lambda_group = self.config.lambda_group
    _lambda_group_init = self.config.lambda_group_init
    _lambda_group_beta = self.config.lambda_group_beta

    group_lambda_iter = 0.

    solve_idx = 0
    iter_idx = 0
    reset_objs = None
    curr_n_resolves = 0
    last_solve_iter = False
    while solve_idx < _n_solves:
      if group_lambda_iter > _lambda_group and curr_n_resolves == 0:
        if last_solve_iter:
          if self.config.verbose:
            print("  Current group-regularizer coefficient > threshold. Done.")
          break
        else:
          group_lambda_iter = _lambda_group
          last_solve_iter = True

      self._lambda_group.value = group_lambda_iter
      if self.config.verbose:
        if curr_n_resolves > 0:
          print("    Resolve iteration (%i/%i)). Current lambda_group: %.5f" %
                    (curr_n_resolves, _n_resolve_attempts, group_lambda_iter))
        else:
          print("  Solver iteration %i/%i. Current lambda_group: %.5f" % (
              solve_idx + 1, _n_solves, group_lambda_iter))

        itr_start_time = time.time()

      self._solve()
      self.compute_projections()
      self._proj_history.append(self.projections)

      curr_objs = self.get_objective("all")
      self._objs_history.append(curr_objs)
      needs_resolve = self._check_if_resolve(curr_objs)
      self._lambda_group_history.append(group_lambda_iter)

      if self.config.verbose:
        print(
            "    Solver iteration done in %.2fs. " %
            (time.time() - itr_start_time))
        print("    Objective value: %.4f" % self._final_obj.value)
        print("    Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
                self._error.value, self._group_reg.value,
                self._global_reg.value))

      # Keep track of all iters, including resolves
      iter_idx += 1
      if needs_resolve:
        if curr_n_resolves >= _n_resolve_attempts:
          if self.config.verbose:
            print("    Max resolve attempts reached. Moving on.")
          curr_objs = reset_objs
          group_lambda_iter = reset_group_lambda
        else:
          # Some bookkeeping for when resetting:
          if curr_n_resolves == 0:
            reset_objs = curr_objs
            reset_group_lambda = group_lambda_iter
          group_lambda_iter = (prev_lambda_iter + group_lambda_iter) / 2.
          curr_n_resolves += 1
          if self.config.verbose:
            print("    Recon. error change too large. Re-solving.")
          continue
      # Reset when, after some resolve iters, we no longer need to resolve.
      elif curr_n_resolves > 0:
        curr_objs = reset_objs
        group_lambda_iter = reset_group_lambda

      self._prev_objs = curr_objs
      # Some bookkeeping for resolving:
      curr_n_resolves = 0
      prev_lambda_iter = group_lambda_iter
      # needs_resolve = False  # Don't need this.
      # Keeping track of redundancy matrix for main iters
      self._main_iters.append(iter_idx - 1)
      solve_idx += 1
      if group_lambda_iter == 0:
        group_lambda_iter = _lambda_group_init
      else:
        group_lambda_iter *= _lambda_group_beta

    self.has_history = True
    return self


_SINGLE_VIEW_SOLVERS = {
    "opt": OptimizationSolver,
    # "nn": NNSolver,
}
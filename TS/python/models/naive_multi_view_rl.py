import cvxpy as cvx
import multiprocessing as mp
import numpy as np
import torch
import time

from models import naive_single_view_rl
from models.naive_single_view_rl import _SOLVERS
from models.model_base import ModelException, BaseConfig
from utils import cvx_utils


import IPython

# Hack to get rid of some issues with parallel running
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


_SINGLE_VIEW_SOLVERS = _SOLVERS

# class NBSMVRLConfig(BaseConfig):
#   def __init__(
      # self, group_regularizer="inf", global_regularizer="L1", lambda_group=1.0,
      # lambda_global=1.0, sp_eps=1e-5, n_solves=1, lambda_group_init=1e-5,
      # lambda_group_beta=10, resolve_change_thresh=0.05, n_resolve_attempts=3,
      # solve_joint=False, parallel=True, n_jobs=None, verbose_interval=10.,
      # verbose=True):
    # self.group_regularizer = group_regularizer
    # self.global_regularizer = global_regularizer
    # self.lambda_group = lambda_group
    # self.lambda_global = lambda_global

    # self.sp_eps = sp_eps

    # self.n_solves = n_solves
    # self.lambda_group_init = lambda_group_init
    # self.lambda_group_beta = lambda_group_beta
    # # self.global_reg_beta = global_reg_beta

    # self.resolve_change_thresh = resolve_change_thresh
    # self.n_resolve_attempts = n_resolve_attempts

    # self.solve_joint = solve_joint
    # self.parallel = parallel
    # self.n_jobs = n_jobs

    # self.verbose = verbose

class NBSMVRLConfig(BaseConfig):
  def __init__(
      self, single_view_solver_type, single_view_config, parallel, n_jobs,
      *args, **kwargs):

    self.single_view_solver_type = single_view_solver_type
    self.single_view_config = single_view_config

    self.parallel = parallel
    self.n_jobs = n_jobs

    super(NBSMVRLConfig, self).__init__(*args, **kwargs)


# TODO: Fix this for OPT
def view_solve_func(view_solver):
  view_solver.fit()
  fname = "./temp_models/tmp%i" % view_solver.view_id
  torch.save(view_solver.state_dict(), fname)
  return fname


class NaiveBlockSparseMVRL(object):
  def __init__(self, config):
    self.config = config
    # self._training = True
    self._initialized = False
    self._trained = False

  def _initialize_views(self):
    if self.config.single_view_solver_type not in _SINGLE_VIEW_SOLVERS:
      raise ModelException(
          "Invalid single-view solver type: %s" %
          self.config.single_view_solver_type)

    _solver_type = _SINGLE_VIEW_SOLVERS[self.config.single_view_solver_type]

    self._view_solvers = {}
    for vi in range(self._nviews):
      vi_solver = _solver_type(vi, self.config.single_view_config)
      vi_solver.set_data(self._view_data)
      vi_solver.initialize()
      self._view_solvers[vi] = vi_solver
    #   self._final_obj += vi_solver._final_obj

    # self._joint_problem = cvx.Problem(cvx.Minimize(self._final_obj))
    self._initialized = True

  def get_objective(self, vi=None, obj_type=None):
    if not self._initialized:
      raise ModelException("Model has not been initialized.")
    if vi is not None:
      return self._view_solvers[vi].get_objective(obj_type)

    if obj_type == "all":
      objs = np.array(
          [self._view_solvers[vi].get_objective("all")
          for vi in range(self._nviews)])
      return objs.sum(axis=0).tolist()
    return np.sum([
        self._view_solvers[vi].get_objective(obj_type)
        for vi in range(self._nviews)])

  def _solve_parallel(self):
    if self.config.verbose:
      print("  Solving problems in parallel...")
      start_time = time.time()
    pool = mp.Pool(processes=self._n_jobs)

    solvers = [self._view_solvers[vi] for vi in range(self._nviews)]
    result = pool.map_async(view_solve_func, solvers)
    pool.close()
    pool.join()
    for vi, fname in enumerate(result.get()):
      self._view_solvers[vi].load_state_dict(torch.load(fname))
      self._view_solvers[vi].eval()

    if self.config.verbose:
      objs = self.get_objective(vi=None, obj_type="all")
      diff_time = time.time() - start_time
      print("    Finished solving for all views. Overall objective values:")
      print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
            tuple(objs + [diff_time]))

  def _solve_sequential(self):
    if self.config.verbose:
      print("  Solving problems in sequence...")
      start_time = time.time()

    for vi in range(self._nviews):
      if self.config.verbose:
        print("    Solving for view %i..." % (vi + 1))
        vi_start_time = time.time()

      self._view_solvers[vi].fit()

      if self.config.verbose:
        objs = self.get_objective(vi=vi, obj_type="all")
        diff_time = time.time() - vi_start_time
        print("    Finished solving for view %i. Objective values:" % (vi + 1))
        print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
              tuple(objs + [diff_time]))

    if self.config.verbose:
      objs = self.get_objective(vi=None, obj_type="all")
      diff_time = time.time() - start_time
      print("    Finished solving for all views. Overall objective values:")
      print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
            tuple(objs + [diff_time]))

  # def _solve_joint(self):
  #   if self.config.verbose:
  #     # print("  Solving joint problem...")
  #     start_time = time.time()
  #   safe_solve(
  #       self._joint_problem, parallel=self.config.parallel, verbose=False)

  #   if self.config.verbose:
  #     objs = self._get_current_obj(vi=None, obj_type="all")
  #     diff_time = time.time() - start_time
  #     print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
  #           tuple(objs + [diff_time]))

  def compute_projections(self):
    self.view_projections = {}
    for vi in range(self._nviews):
      self._view_solvers[vi].compute_projections()
      self.view_projections[vi] = self._view_solvers[vi].projections

  def _optimize(self):
    if self.config.verbose:
      print("Optimizing for group sparse transforms.")
      start_time = time.time()

    self._n_jobs = (
        self._nviews if (self.config.parallel and self.config.n_jobs is None)
        else self.config.n_jobs
    )

    # if self.config.solve_joint:
    #   # solve_func = self._solve_joint
    # else:
    solve_func = (
        self._solve_parallel if self.config.parallel and self._n_jobs != 1 else
        self._solve_sequential
    )
    solve_func()

    if self.config.verbose:
      print("Finished training in %.2fs" % (time.time() - start_time))

  def fit(self, view_data):
    self._nviews = len(view_data)
    self._view_data = view_data
    self._npts = view_data[0].shape[0]
    self._tot_dim = np.sum([view_data[i].shape[1] for i in view_data])

    self._initialize_views()
    self._optimize()

    self._trained = True
    return self

  def redundancy_matrix(self, projections=None):
    if projections is None: self.compute_projections()
    # if self._training is True:
    #   raise ModelException("Model has not been trained yet.")
    view_dims = [self._view_data[vi].shape[1] for vi in range(self._nviews)]
    view_dims_cs = [0] + np.cumsum(view_dims).tolist()
    start_inds, end_inds = view_dims_cs[:-1], view_dims_cs[1:]

    projections = self.view_projections if projections is None else projections
    rmat = np.zeros((self._tot_dim, self._tot_dim))
    for vi in range(self._nviews):
      si_ind, ei_ind = start_inds[vi], end_inds[vi]
      for vo in range(self._nviews):
        if vo == vi: continue
        so_ind, eo_ind = start_inds[vo], end_inds[vo]
        rmat[so_ind:eo_ind, si_ind:ei_ind] = projections[vi][vo]
    return rmat

  def predict(self, view_data, vi_out=None):
    if vi_out is None:
      vi_out = list(range(self._nviews))
    preds = {vo: self._view_solvers[vo].predict(view_data) for vo in vi_out}
    return preds

  def has_history(self):
    return all([vs.has_history for vs in self._view_solvers.values()])

  def redundancy_matrix_history(self):
    if not self.has_history():
      raise ModelException("History not present for all view solvers.")
    # if self._training is True:
    #   raise ModelException("Model has not been trained yet.")
    view_dims = [self._view_data[vi].shape[1] for vi in range(self._nviews)]
    view_dims_cs = [0] + np.cumsum(view_dims).tolist()
    start_inds, end_inds = view_dims_cs[:-1], view_dims_cs[1:]

    proj_histories = {}
    max_len = 0
    for vi, solver in self._view_solvers.items():
        proj_histories[vi] = [
            solver._proj_history[itr] for itr in solver._main_iters]
        max_len = max(max_len, len(solver._main_iters))
    for vi in proj_histories:
      vi_len = len(proj_histories[vi])
      if vi_len < max_len:
        proj_histories[vi].extend([proj_histories[vi][-1]] * (max_len - vi_len))

    rmats = []
    for i in range(max_len):
      projections = {vi: proj_histories[vi][i] for vi in proj_histories}
      rmat = np.zeros((self._tot_dim, self._tot_dim))
      for vi in range(self._nviews):
        si_ind, ei_ind = start_inds[vi], end_inds[vi]
        for vo in range(self._nviews):
          if vo == vi: continue
          so_ind, eo_ind = start_inds[vo], end_inds[vo]
          rmat[so_ind:eo_ind, si_ind:ei_ind] = projections[vi][vo]
      rmats.append(rmat)
    return rmats

  def nullspace_matrix(self, projections=None):
    return -np.eye(self._tot_dim) + self.redundancy_matrix(projections).T

  def nullspace_matrix_history(self):
    rmats = self.redundancy_matrix_history()
    return [(-np.eye(self._tot_dim) + rmat.T) for rmat in rmats]

  def obj_history(self):
    raise NotImplementedError("obj_history not yet implemented.")

  def save_to_file(self, fname):
    raise NotImplementedError("Saving to file not yet implemented.")    
  #   nmat = self.nullspace_matrix()
  #   vproj = self.view_projections
  #   cfg = self.config.__dict__
  #   np.save(fname, [nmat, vproj, cfg])
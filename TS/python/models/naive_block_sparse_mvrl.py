import cvxpy as cvx
import multiprocessing as mp
import numpy as np
import time

from models.model_base import ModelException
from utils import cvx_utils


import IPython  


_SOLVER = cvx.GUROBI
_OBJ_ORDER = ["error", "gs", "reg", "total"]


class NBSMVRLConfig(object):
  def __init__(
      self, group_regularizer="inf", global_regularizer="L1", lambda_group=1.0,
      lambda_global=1.0, sp_eps=1e-5, n_solves=1, lambda_group_init=1e-5,
      lambda_group_beta=10, resolve_change_thresh=0.05, n_resolve_attempts=3,
      solve_joint=False, parallel=True, n_jobs=None, verbose_interval=10.,
      verbose=True):
    self.group_regularizer = group_regularizer
    self.global_regularizer = global_regularizer
    self.lambda_group = lambda_group
    self.lambda_global = lambda_global

    self.sp_eps = sp_eps

    self.n_solves = n_solves
    self.lambda_group_init = lambda_group_init
    self.lambda_group_beta = lambda_group_beta
    # self.global_reg_beta = global_reg_beta

    self.resolve_change_thresh = resolve_change_thresh
    self.n_resolve_attempts = n_resolve_attempts

    self.solve_joint = solve_joint
    self.parallel = parallel
    self.n_jobs = n_jobs
    # If parallel, how often to print current objective values
    self.verbose_interval = verbose_interval
    self.verbose = verbose


def safe_solve(prob, parallel=False, verbose=False):
  try:
    prob.solve(
        solver=_SOLVER, warm_start=True, parallel=parallel, verbose=verbose)
  except (KeyError, cvx.SolverError) as e:
    # pass
    # print("        Error using solver %s: %s.\n"
    #       "        Using default solver." %
    #       (_SOLVER, e))
    try:
      # prob.solve(warm_sart=True, parallel=parallel, verbose=verbose)
      prob.solve(parallel=parallel, verbose=verbose)
    except Exception as e2:
      raise ModelException("Issue with solvers: %s" % e2)
  return prob


class NaiveSingleViewBSRL(object):
  def __init__(self, view_id, parent_config):
    self._view_id = view_id
    self._parent_config = parent_config

    self._group_regularizer = parent_config.group_regularizer
    self._global_regularizer = parent_config.global_regularizer

    self.sp_eps = self._parent_config.sp_eps
    self.resolve_change_thresh = self._parent_config.resolve_change_thresh

    self.parallel = parent_config.parallel
    self.verbose = parent_config.verbose

    self._p_var = None
    self._G = None

    self._error = None
    self._gs_reg = None
    self._global_reg = None
    self._prob = None

    self.projections = None

  def initialize_cvx(self):
    group_inds = [0] + np.cumsum(self._rest_dims).tolist()
    self._G = [np.arange(sidx, eidx) for sidx, eidx in 
               zip(group_inds[:-1], group_inds[1:])]

    self._lambda_group = cvx.Parameter((), value=0.)
    self._lambda_global = cvx.Parameter(
        (), value=self._parent_config.lambda_global)

    self._p_var = cvx.Variable((group_inds[-1], self._dim))
    # IPython.embed()
    self._error = cvx.norm(
        self._rest_data * self._p_var - self._view_data) / self._npts
    self._gs_reg = cvx_utils.group_norm(
        self._p_var, self._G, order=self._group_regularizer, use_cvx=True)
    self._global_reg = cvx_utils.cvx_norm_wrapper(
        self._p_var, self._global_regularizer)
    self._final_obj = (
        self._error + cvx.abs(self._lambda_group) * self._gs_reg +
        cvx.abs(self._lambda_global) * self._global_reg)

    self._obj_map = {
        "error": self._error,
        "gs": self._gs_reg,
        "reg": self._global_reg,
        "total": self._final_obj
    }

    self._prob = cvx.Problem(cvx.Minimize(self._final_obj))

  def set_data(self, data):
    self._nviews = len(data)
    self._view_data = data[self._view_id]
    self._npts, self._dim = self._view_data.shape
    rest_data_list = [data[i] for  i in range(len(data)) if i != self._view_id]
    self._rest_data = np.concatenate(rest_data_list, axis=1)
    self._rest_dims = [vd.shape[1] for vd in rest_data_list]

  # def set_lambda_global(self, val):
  #   self._lambda_global.value = val

  # def set_lambda_group(self, val):
  #   self._lambda_group.value = val

  def solve(self, parallel=None, verbose=None):
    verbose = self.verbose if verbose is None else verbose
    parallel = self.parallel if parallel is None else parallel
    try:
      self._prob.solve(
          solver=_SOLVER, warm_start=True, parallel=parallel, verbose=verbose)
    except (KeyError, cvx.SolverError) as e:
      # pass
      # print("        Error using solver %s: %s.\n"
      #       "        Using default solver." %
      #       (_SOLVER, e))
      try:
        # prob.solve(warm_sart=True, parallel=parallel, verbose=verbose)
        self._prob.solve(parallel=parallel, verbose=verbose)
      except Exception as e2:
        raise ModelException("Issue with solvers: %s" % e2)
    return self

  def _recompute_projections(self):
    # Just extracting the indices to split the concatenated projection at.
    split_inds = [g[0] for g in self._G[1:]]
    other_views = [vo for vo in range(self._nviews) if vo != self._view_id]
    projs = self._p_var.value
    projs_sp = np.where(np.abs(projs) < self.sp_eps, 0., projs)
    projs_split = np.split(projs_sp, split_inds, axis=0)
    self.projections = {
        vo: pr for vo, pr in zip(other_views, projs_split)
    }

  def get_obj(self, obj_type="all"):
    if obj_type == "all":
      return [self._obj_map[otype].value for otype in _OBJ_ORDER]
    return self._obj_map.get(obj_type, "total").value

  def _check_if_resolve(self, curr_objs):
    # Naive way to do this. TODO: Do something smarter.
    # TODO
    if self._prev_objs is None:
      return False
    prev_error = self._prev_objs[0]
    curr_error = curr_objs[0]
    return (curr_error - prev_error > self.resolve_change_thresh)

  def iterated_solve(self):
    self._proj_history = []
    self._objs_history = []
    self._lambda_group_history = []
    self._main_iters = []
    self._prev_objs = None
    # self._training = True

    _n_solves = self._parent_config.n_solves
    _n_resolve_attempts = self._parent_config.n_resolve_attempts
    _lambda_group = self._parent_config.lambda_group
    _lambda_group_init = self._parent_config.lambda_group_init
    _lambda_group_beta = self._parent_config.lambda_group_beta

    group_lambda_iter = 0.

    solve_idx = 0
    iter_idx = 0
    reset_objs = None
    curr_n_resolves = 0
    last_solve_iter = False
    while solve_idx < _n_solves:
      if group_lambda_iter > _lambda_group and curr_n_resolves == 0:
        if last_solve_iter:
          if self.verbose:
            print("  Current group-regularizer coefficient > threshold. Done.")
          break
        else:
          group_lambda_iter = _lambda_group
          last_solve_iter = True

      self._lambda_group.value = group_lambda_iter
      if self.verbose:
        if curr_n_resolves > 0:
          print("    Resolve iteration (%i/%i)). Current lambda_group: %.5f" %
                    (curr_n_resolves, _n_resolve_attempts, group_lambda_iter))
        else:
          print("  Solver iteration %i/%i. Current lambda_group: %.5f" % (
              solve_idx + 1, _n_solves, group_lambda_iter))

        itr_start_time = time.time()

      # solve_func()
      self.solve()
      self._recompute_projections()
      self._proj_history.append(self.projections)
      # self._nmat_history.append(self.nullspace_matrix())

      curr_objs = self.get_obj("all")
      self._objs_history.append(curr_objs)
      needs_resolve = self._check_if_resolve(curr_objs)
      self._lambda_group_history.append(group_lambda_iter)

      if self.verbose:
        print(
            "    Solver iteration done in %.2fs. " %
            (time.time() - itr_start_time))
        print("    Objective value: %.4f" % self._final_obj.value)
        # print("  Individual objective values:")
        # for vi in range(self._nviews):
        #   eobj, gobj, robj = [o.value for o in vi_objs[vi]]
        #   print("\n  View %i:" % (vi + 1))
        #   print("    Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
        #           eobj, gobj, robj))
        #   all_error += eobj
        #   all_gs += gobj
        #   all_reg += robj
        # print("\n    Overall objective values:")
        print("    Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
                self._error.value, self._gs_reg.value, self._global_reg.value))

      # Keep track of all iters, including resolves
      iter_idx += 1
      if needs_resolve:
        if curr_n_resolves >= _n_resolve_attempts:
          if self.verbose:
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
          if self.verbose:
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

    return self

# def safe_solve(single_view_solver, parallel=False, verbose=False):
#   try:
#     single_view_solver.solve(
#         solver=_SOLVER, warm_start=True, parallel=parallel, verbose=verbose)
#   except (KeyError, cvx.SolverError) as e:
#     # pass
#     # print("        Error using solver %s: %s.\n"
#     #       "        Using default solver." %
#     #       (_SOLVER, e))
#     try:
#       # prob.solve(warm_sart=True, parallel=parallel, verbose=verbose)
#       single_view_solver.solve(parallel=parallel, verbose=verbose)
#     except Exception as e2:
#       raise ModelException("Issue with solvers: %s" % e2)
#   return single_view_solver

def view_solve_func(view_solver):
  return view_solver.iterated_solve()


class NaiveBlockSparseMVRL(object):
  def __init__(self, config):
    self.config = config
    # self._training = True
    self._initialized = False

  # def _setup_cvx_for_view(self, vi):
  #   vi_data = self._view_data[vi]
  #   vi_dim = vi_data.shape[1]
  #   rest_data_list = [
  #       self._view_data[i] for  i in range(self._nviews) if i != vi]
  #   rest_data = np.concatenate(rest_data_list, axis=1)
  #   rest_dims = [vd.shape[1] for vd in rest_data_list]
  #   group_inds = [0] + np.cumsum(rest_dims).tolist()
  #   G = [np.arange(sidx, eidx) for sidx, eidx in 
  #        zip(group_inds[:-1], group_inds[1:])]

  #   p_var = cvx.Variable((group_inds[-1], vi_dim))
  #   # IPython.embed()
  #   error_obj = cvx.norm(rest_data * p_var - vi_data) / self._npts
  #   group_sparse_obj = cvx_utils.group_norm(
  #       p_var, G, order=self.config.group_regularizer, use_cvx=True)
  #   reg_obj = cvx_utils.cvx_norm_wrapper(p_var, self.config.global_regularizer)
  #   total_obj = (error_obj + cvx.abs(self._lambda_group) * group_sparse_obj +
  #         cvx.abs(self._lambda_global) * reg_obj)

  #   prob = cvx.Problem(cvx.Minimize(total_obj))
  #   objs = (error_obj, group_sparse_obj, reg_obj, total_obj)

  #   return p_var, prob, objs, G

  # def _initialize_cvx(self):
  #   self._view_p_vars = {}
  #   self._view_objs = {} # vi: None for vi in range(self._nviews)}
  #   self._view_groups = {} # vi: None for vi in range(self._nviews)}
  #   self._view_probs = {}  # vi: Can solve for each view independently
  #   self._final_obj = 0
  #   self._total_error = 0.
  #   self._total_gs_obj = 0.
  #   self._total_reg_obj = 0.

  #   self._lambda_group = cvx.Parameter((), value=0.)
  #   self._lambda_global = cvx.Parameter((), value=self.config.lambda_global)

  #   for vi in range(self._nviews):
  #     p_var, prob, objs, G = self._setup_cvx_for_view(vi)

  #     self._view_p_vars[vi] = p_var
  #     self._view_probs[vi] = prob
  #     self._view_objs[vi] = objs
  #     self._view_groups[vi] = G

  #     error_obj, group_sparse_obj, reg_obj, total_obj = objs
  #     self._total_error += error_obj
  #     self._total_gs_obj += group_sparse_obj
  #     self._total_reg_obj += reg_obj
  #     self._final_obj += total_obj

  #   self._joint_problem = cvx.Problem(cvx.Minimize(self._final_obj))
  #   self._initialized = True
  def _initialize_views(self):
    self._view_solvers = {}
    self._final_obj = 0
    # self._total_error = 0.
    # self._total_gs_obj = 0.
    # self._total_reg_obj = 0.

    self._lambda_group = cvx.Parameter((), value=0.)
    self._lambda_global = cvx.Parameter((), value=self.config.lambda_global)

    for vi in range(self._nviews):
      vi_solver = NaiveSingleViewBSRL(view_id=vi, parent_config=self.config)
          # view_id=vi, group_regularizer=self.config.group_regularizer,
          # global_regularizer=self.config.global_regularizer,
          # parallel=self.config.parallel, verbose=self.config.verbose)
 
      vi_solver.set_data(self._view_data)
      vi_solver.initialize_cvx()
      self._view_solvers[vi] = vi_solver
      # self._total_error += vi_solver._error
      # self._total_gs_obj += vi_solver._gs_reg
      # self._total_reg_obj += vi_solver._global_reg
      self._final_obj += vi_solver._final_obj

    self._joint_problem = cvx.Problem(cvx.Minimize(self._final_obj))
    self._initialized = True

  def _get_current_obj(self, vi=None, obj_type=None):
    if not self._initialized:
      raise ModelException("Model has not been initialized.")
    if vi is not None:
      return self._view_solvers[vi].get_obj(obj_type)
      # if obj_type == "all":
      #   return [obj.value for obj in self._view_objs[vi]]
      # obj_map = {"error": 0, "gs": 1, "reg": 2, "total": 3}.get(obj_type, 3)
      # return self._view_objs[vi][obj_index].value

    # objs = {
    #     "error": self._total_error,
    #     "gs": self._total_gs_obj,
    #     "reg": self._total_reg_obj,
    #     "total": self._final_obj
    # }
    if obj_type == "all":
      objs = np.array(
          [self._view_solvers[vi].get_obj("all") for vi in range(self._nviews)])
      return objs.sum(axis=0).tolist()
      # return [objs[otype].value for otype in ["error", "gs", "reg", "total"]]
    return np.sum([
        self._view_solvers[vi].get_obj(obj_type) for vi in range(self._nviews)])

  def _solve_parallel(self):
    # raise NotImplementedError("Non-joint parallel solver not working.")
    if self.config.verbose:
      print("  Solving problems in parallel...")
      start_time = time.time()
    pool = mp.Pool(processes=self._n_jobs)

    solvers = [self._view_solvers[vi] for vi in range(self._nviews)]
    # result = pool.map_async(safe_solve, probs)
    result = pool.map_async(view_solve_func, solvers)
    pool.close()
    pool.join()
    solvers = result.get()
    self._view_solvers = {vi: solver for vi, solver in enumerate(solvers)}
    # if self.config.verbose:
    #   while not result.ready():
    #     objs = self._get_current_obj(vi=None, obj_type="all")
    #     if not any([obj is None for obj in objs]):
    #       diff_time = time.time() - start_time
    #       print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
    #             tuple(objs + [diff_time]), end='\r')
    #     IPython.embed()
    #     time.sleep(self.config.verbose_interval)

    #   # IPython.embed()
    #   objs = self._get_current_obj(vi=None, obj_type="all")
    #   diff_time = time.time() - start_time
    #   print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
    #         tuple(objs + [diff_time]))
    #   print("Parallel optimizers have finished.")
    # else:
    #   pool.join()
    # return result.get()

  def _solve_sequential(self):
    if self.config.verbose:
      print("  Solving problems in sequence...")
      start_time = time.time()

    for vi in range(self._nviews):
      if self.config.verbose:
        print("    Solving for view %i..." % (vi + 1))
        vi_start_time = time.time()

      self._view_solvers[vi].iterated_solve()
      # safe_solve(
      #     self._view_probs[vi], parallel=False, verbose=self.config.verbose)

      if self.config.verbose:
        objs = self._get_current_obj(vi=vi, obj_type="all")
        diff_time = time.time() - vi_start_time
        print("    Finished solving for view %i." % (vi + 1))
        print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
              tuple(objs + [diff_time]))

  def _solve_joint(self):
    if self.config.verbose:
      # print("  Solving joint problem...")
      start_time = time.time()
    # IPython.embed()
    safe_solve(
        self._joint_problem, parallel=self.config.parallel, verbose=False)
        # verbose=self.config.verbose)

    if self.config.verbose:
      objs = self._get_current_obj(vi=None, obj_type="all")
      diff_time = time.time() - start_time
      print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
            tuple(objs + [diff_time]))

  def _recompute_view_projections(self):
    self.view_projections = {}
    for vi in range(self._nviews):
      self._view_solvers[vi]._recompute_projections()
      self.view_projections[vi] = self._view_solvers[vi].projections
      # Just extracting the indices to split the concatenated projection at.
      # split_inds = [g[0] for g in self._view_groups[vi][1:]]
      # other_views = [vo for vo in range(self._nviews) if vo != vi]
      # projs = self._view_p_vars[vi].value
      # projs_sp = np.where(np.abs(projs) < self.config.sp_eps, 0., projs)
      # projs_split = np.split(projs_sp, split_inds, axis=0)
      # self.view_projections[vi] = {
      #     vo: pr for vo, pr in zip(other_views, projs_split)
      # }

  # def _compute_all_objs(self):
  #   obj_struct = {}
  #   for vi in range(self._nviews):
  #     obj_struct[vi] = self._get_current_obj(vi=vi, obj_type="all")
  #   obj_struct["total"] = self._get_current_obj(vi=None, obj_type="all")
  #   return obj_struct

  # def _check_if_resolve(self, curr_objs):
  #   # Naive way to do this. TODO: Do something smarter.
  #   # TODO
  #   if self._prev_objs is None:
  #     return False
  #   prev_error = self._prev_objs["total"][0]
  #   curr_error = curr_objs["total"][0]
  #   return (curr_error - prev_error > self.config.resolve_change_thresh)

  def _optimize(self):
    # Add individual view objectives together
    # vi_objs = {}
    # vi_vars = {}
    # vi_groups = {}
    # combined_objs = []
    # for vi in range(self._nviews):
    #   vi_vars[vi], vi_objs[vi], vi_groups[vi] = (
    #       self._block_sparse_obj_for_view(vi))
    #   error_obj, group_sparse_obj, reg_obj = vi_objs[vi]
    #   combined_objs.append(
    #       error_obj + self.config.lambda_group * group_sparse_obj +
    #           self.config.lambda_global * reg_obj
    #   )
    # obj = np.sum(combined_objs)
    # prob = cvx.Problem(cvx.Minimize(obj))
    # self._initialize_cvx()
    self._initialize_views()
    if self.config.verbose:
      print("Optimizing for group sparse transforms.")
      start_time = time.time()

    self._n_jobs = (
        self._nviews if (self.config.parallel and self.config.n_jobs is None)
        else self.config.n_jobs
    )

    if self.config.solve_joint:
      solve_func = self._solve_joint
    else:
      solve_func = (
          self._solve_parallel if self.config.parallel else
          self._solve_sequential
      )
    solve_func()
    # group_lambda_iter = 0.

    # # Just to keep track of past solves.
    # self._nmat_history = []
    # self._objs_history = []
    # self._lambda_group_history = []
    # self._prev_objs = None
    # # self._training = True
    
    # solve_idx = 0
    # reset_objs = None
    # curr_n_resolves = 0
    # last_solve_iter = False
    # while solve_idx < self.config.n_solves:
    #   if group_lambda_iter > self.config.lambda_group and curr_n_resolves == 0:
    #     if last_solve_iter:
    #       if self.config.verbose:
    #         print("  Current group-regularizer coefficient > threshold. Done.")
    #       break
    #     else:
    #       group_lambda_iter = self.config.lambda_group
    #       last_solve_iter = True

    #   self._lambda_group.value = group_lambda_iter
    #   if self.config.verbose:
    #     if curr_n_resolves > 0:
    #       print("    Resolve iteration (%i/%i)). Current lambda_group: %.5f" %
    #                 (curr_n_resolves, self.config.n_resolve_attempts,
    #                  group_lambda_iter))
    #     else:
    #       print("  Solver iteration %i/%i. Current lambda_group: %.5f" % (
    #           solve_idx + 1, self.config.n_solves, group_lambda_iter))

    #     itr_start_time = time.time()

    #   solve_func()
    #   self._recompute_view_projections()
    #   self._nmat_history.append(self.nullspace_matrix())

    #   curr_objs = self._compute_all_objs()
    #   self._objs_history.append(curr_objs)
    #   needs_resolve = self._check_if_resolve(curr_objs)
    #   self._lambda_group_history.append(group_lambda_iter)

    #   if self.config.verbose:
    #     print(
    #         "    Solver iteration done in %.2fs. " %
    #         (time.time() - itr_start_time))
    #     print("    Objective value: %.4f" % self._final_obj.value)
    #     # print("  Individual objective values:")
    #     # for vi in range(self._nviews):
    #     #   eobj, gobj, robj = [o.value for o in vi_objs[vi]]
    #     #   print("\n  View %i:" % (vi + 1))
    #     #   print("    Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
    #     #           eobj, gobj, robj))
    #     #   all_error += eobj
    #     #   all_gs += gobj
    #     #   all_reg += robj
    #     # print("\n    Overall objective values:")
    #     print("    Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
    #             self._total_error.value, self._total_gs_obj.value,
    #             self._total_reg_obj.value))

    #   if needs_resolve:
    #     if curr_n_resolves >= self.config.n_resolve_attempts:
    #       if self.config.verbose:
    #         print("    Max resolve attempts reached. Moving on.")
    #       curr_objs = reset_objs
    #       group_lambda_iter = reset_group_lambda
    #     else:
    #       # Some bookkeeping for when resetting:
    #       if curr_n_resolves == 0:
    #         reset_objs = curr_objs
    #         reset_group_lambda = group_lambda_iter
    #       group_lambda_iter = (prev_lambda_iter + group_lambda_iter) / 2.
    #       curr_n_resolves += 1
    #       if self.config.verbose:
    #         print("    Recon. error change too large. Re-solving.")
    #       continue
    #   # Reset when, after some resolve iters, we no longer need to resolve.
    #   elif curr_n_resolves > 0:
    #     curr_objs = reset_objs
    #     group_lambda_iter = reset_group_lambda

    #   self._prev_objs = curr_objs
    #   # Some bookkeeping for resolving:
    #   curr_n_resolves = 0
    #   prev_lambda_iter = group_lambda_iter
    #   # needs_resolve = False  # Don't need this.
    #   solve_idx += 1
    #   if group_lambda_iter == 0:
    #     group_lambda_iter = self.config.lambda_group_init
    #   else:
    #     group_lambda_iter *= self.config.lambda_group_beta

    if self.config.verbose:
      print("Finished training in %.2fs" % (time.time() - start_time))

    # if self.config.verbose:
    #   print("Optimization done in %.2fs. " % (time.time() - start_time))
    #   print("Final objective value: %.4f" % obj.value)
    #   print("Individual objective values:")
    #   all_error = 0.
    #   all_gs = 0.
    #   all_reg = 0.
    #   for vi in range(self._nviews):
    #     eobj, gobj, robj = [o.value for o in vi_objs[vi]]
    #     print("\n  View %i:" % (vi + 1))
    #     print("    Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
    #             eobj, gobj, robj))
    #     all_error += eobj
    #     all_gs += gobj
    #     all_reg += robj
    #   print("\nOverall objective values:")
    #   print("  Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
    #           all_error, all_gs, all_reg))

  def fit(self, view_data):
    self._nviews = len(view_data)
    self._view_data = view_data
    self._npts = view_data[0].shape[0]
    self._tot_dim = np.sum([view_data[i].shape[1] for i in view_data])

    self._optimize()
    return self

  def redundancy_matrix(self, projections=None):
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

  def redundancy_matrix_history(self):
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
          print(vo, vi)
          so_ind, eo_ind = start_inds[vo], end_inds[vo]
          rmat[so_ind:eo_ind, si_ind:ei_ind] = projections[vi][vo]
      rmats.append(rmat)
    return rmats

  def nullspace_matrix(self, projections=None):
    return -np.eye(self._tot_dim) + self.redundancy_matrix(projections).T

  def nullspace_matrix_history(self):
    rmats = self.redundancy_matrix_history()
    return [(-np.eye(self._tot_dim) + rmat.T) for rmat in rmats]
    # return -np.eye(self._tot_dim) + self.redundancy_matrix(projections).T

  def obj_history(self):

  def save_to_file(self, fname):
    nmat = self.nullspace_matrix()
    vproj = self.view_projections
    cfg = self.config.__dict__
    np.save(fname, [nmat, vproj, cfg])
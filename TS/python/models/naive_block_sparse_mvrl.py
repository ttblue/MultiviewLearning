import cvxpy as cvx
import multiprocessing as mp
import numpy as np
import time

from models.model_base import ModelException
from utils import cvx_utils


import IPython  


_SOLVER = cvx.GUROBI


class NBSMVRLConfig(object):
  def __init__(
      self, group_regularizer="inf", global_regularizer="L1", lambda_group=1.0,
      lambda_global=1.0, sp_eps=1e-5, n_solves=1, lambda_group_init=1e-5,
      lambda_group_beta=10, resolve_change_thresh=0.05, solve_joint=False,
      parallel=True, n_jobs=None, verbose_interval=10., verbose=True):
    self.group_regularizer = group_regularizer
    self.global_regularizer = global_regularizer
    self.lambda_group = lambda_group
    self.lambda_global = lambda_global

    self.sp_eps = sp_eps

    self.n_solves = n_solves
    self.lambda_group_init = lambda_group_init
    self.lambda_group_beta = lambda_group_beta
    # self.global_reg_beta = global_reg_beta

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
    print("        Error using solver %s: %s.\n"
          "        Using default solver." %
          (_SOLVER, e))
    try:
      # prob.solve(warm_sart=True, parallel=parallel, verbose=verbose)
      prob.solve(parallel=parallel, verbose=verbose)
    except Exception as e2:
      raise ModelException("Issue with solvers: %s" % e2)
  return prob


class NaiveBlockSparseMVRL(object):
  def __init__(self, config):
    self.config = config
    # self._training = True
    self._initialized = False

  def _setup_cvx_for_view(self, vi):
    vi_data = self._view_data[vi]
    vi_dim = vi_data.shape[1]
    rest_data_list = [
        self._view_data[i] for i in range(self._nviews) if i != vi]
    rest_data = np.concatenate(rest_data_list, axis=1)
    rest_dims = [vd.shape[1] for vd in rest_data_list]
    group_inds = [0] + np.cumsum(rest_dims).tolist()
    G = [np.arange(sidx, eidx) for sidx, eidx in 
         zip(group_inds[:-1], group_inds[1:])]

    p_var = cvx.Variable((group_inds[-1], vi_dim))
    # IPython.embed()
    error_obj = cvx.norm(rest_data * p_var - vi_data) / self._npts
    group_sparse_obj = cvx_utils.group_norm(
        p_var, G, order=self.config.group_regularizer, use_cvx=True)
    reg_obj = cvx_utils.cvx_norm_wrapper(p_var, self.config.global_regularizer)
    total_obj = (error_obj + cvx.abs(self._lambda_group) * group_sparse_obj +
          cvx.abs(self._lambda_global) * reg_obj)

    prob = cvx.Problem(cvx.Minimize(total_obj))
    objs = (error_obj, group_sparse_obj, reg_obj, total_obj)

    return p_var, prob, objs, G

  def _initialize_cvx(self):
    self._view_p_vars = {}
    self._view_objs = {} # vi: None for vi in range(self._nviews)}
    self._view_groups = {} # vi: None for vi in range(self._nviews)}
    self._view_probs = {}  # vi: Can solve for each view independently
    self._final_obj = 0
    self._total_error = 0.
    self._total_gs_obj = 0.
    self._total_reg_obj = 0.

    self._lambda_group = cvx.Parameter((), value=0.)
    self._lambda_global = cvx.Parameter((), value=self.config.lambda_global)

    for vi in range(self._nviews):
      p_var, prob, objs, G = self._setup_cvx_for_view(vi)

      self._view_p_vars[vi] = p_var
      self._view_probs[vi] = prob
      self._view_objs[vi] = objs
      self._view_groups[vi] = G

      error_obj, group_sparse_obj, reg_obj, total_obj = objs
      self._total_error += error_obj
      self._total_gs_obj += group_sparse_obj
      self._total_reg_obj += reg_obj
      self._final_obj += total_obj

    self._joint_problem = cvx.Problem(cvx.Minimize(self._final_obj))
    self._initialized = True

  def _get_current_obj(self, vi=None, obj_type=None):
    if not self._initialized:
      raise ModelException("Model has not been initialized.")
    if vi is not None:
      if obj_type == "all":
        return [obj.value for obj in self._view_objs[vi]]
      obj_index = {"error": 0, "gs": 1, "reg": 2, "total": 3}.get(obj_type, 3)
      return self._view_objs[vi][obj_index].value

    objs = {
        "error": self._total_error,
        "gs": self._total_gs_obj,
        "reg": self._total_reg_obj,
        "total": self._final_obj
    }
    if obj_type == "all":
      return [objs[otype].value for otype in ["error", "gs", "reg", "total"]]
    return objs.get(obj_type, self._final_obj).value

  def _solve_parallel(self):
    raise NotImplementedError("Non-joint parallel solver not working.")
  #   if self.config.verbose:
  #     print("  Solving problems in parallel...")
  #     start_time = time.time()
  #   pool = mp.Pool(processes=self._n_jobs)

  #   probs = [self._view_probs[vi] for vi in range(self._nviews)]
  #   result = pool.map_async(safe_solve, probs)
  #   if self.config.verbose:
  #     while not result.ready():
  #       objs = self._get_current_obj(vi=None, obj_type="all")
  #       if not any([obj is None for obj in objs]):
  #         diff_time = time.time() - start_time
  #         print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
  #               tuple(objs + [diff_time]), end='\r')
  #       # IPython.embed()
  #       time.sleep(self.config.verbose_interval)

  #     # IPython.embed()
  #     objs = self._get_current_obj(vi=None, obj_type="all")
  #     diff_time = time.time() - start_time
  #     print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
  #           tuple(objs + [diff_time]))
  #     print("Parallel optimizers have finished.")
  #   else:
  #     pool.join()
  #   # return result.get()

  def _solve_sequential(self):
    if self.config.verbose:
      print("  Solving problems in sequence...")
      start_time = time.time()

    for vi in range(self._nviews):
      if self.config.verbose:
        print("    Solving for view %i..." % (vi + 1))
        vi_start_time = time.time()

      safe_solve(
          self._view_probs[vi], parallel=False, verbose=self.config.verbose)

      if self.config.verbose:
        objs = self._get_current_obj(vi=vi, obj_type="all")
        diff_time = time.time() - vi_start_time
        print("    Finished solving for view %i." % (vi + 1))
        print("    Error: %.3f, GS: %.3f, Reg: %.3f, Tot: %.3f (in %.2fs)" %
              tuple(objs + [diff_time]))

  def _solve_joint(self):
    if self.config.verbose:
      print("  Solving joint problem...")
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
      # Just extracting the indices to split the concatenated projection at.
      split_inds = [g[0] for g in self._view_groups[vi][1:]]
      other_views = [vo for vo in range(self._nviews) if vo != vi]
      projs = self._view_p_vars[vi].value
      projs_sp = np.where(np.abs(projs) < self.config.sp_eps, 0., projs)
      projs_split = np.split(projs_sp, split_inds, axis=0)
      self.view_projections[vi] = {
          vo: pr for vo, pr in zip(other_views, projs_split)
      }

  def _compute_all_objs(self):
    obj_struct = {}
    for vi in range(self._views):
      obj_struct[vi] = self._get_current_obj(vi=vi, obj_type="all")
    obj_struct["total"] = self._get_current_obj(vi=None, obj_type="all")
    return obj_struct

  def _check_if_resolve(self, curr_objs):
    # Naive way to do this. TODO: Do something smarter.
    # TODO
    if self._prev_objs is None:
      return False
    prev_error = self._prev_objs["total"][0]
    curr_error = curr_objs["total"][0]
    return (prev_error < self.config.resolve_frac * curr_error)

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
    self._initialize_cvx()
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
    group_lambda_iter = 0.

    # Just to keep track of past solves.
    self._nmat_history = []
    self._objs_history = []
    self._prev_objs = None
    # self._training = True

    for solve_idx in range(self.config.n_solves):
      if group_lambda_iter > self.config.lambda_group:
        if self.config.verbose:
          print("  Current group-regularizer coefficient > threshold. Done.")
        break

      self._lambda_group.value = group_lambda_iter
      if self.config.verbose:
        print("  Solver iteration %i/%i. Current lambda_group: %.5f" % (
            solve_idx + 1, self.config.n_solves, group_lambda_iter))
        itr_start_time = time.time()

      solve_func()
      self._recompute_view_projections()
      self._nmat_history.append(self.nullspace_matrix())

      curr_objs = self._compute_all_objs()
      self._objs_history.append(curr_objs)
      needs_resolve = self._check_if_resolve(self, curr_objs)

      if self.config.verbose:
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
                self._total_error.value, self._total_gs_obj.value,
                self._total_reg_obj.value))

      if needs_resolve:
        if self.config.verbose:
          print("    Recon. error change too large."
                " Resolving with smaller reg. penalty.")
        solve_idx -= 1
        prev_lambda = group_lambda_iter / self.config.lambda_group_beta
        group_lambda_iter = (prev_lambda + group_lambda_iter) / 2.
        continue

      if group_lambda_iter == 0:
        group_lambda_iter = self.config.lambda_group_init
      else:
        group_lambda_iter *= self.config.lambda_group_beta

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

  def redundancy_matrix(self):
    # if self._training is True:
    #   raise ModelException("Model has not been trained yet.")

    view_dims = [self._view_data[vi].shape[1] for vi in range(self._nviews)]
    view_dims_cs = [0] + np.cumsum(view_dims).tolist()
    start_inds, end_inds = view_dims_cs[:-1], view_dims_cs[1:]

    rmat = np.zeros((self._tot_dim, self._tot_dim))
    for vi in range(self._nviews):
      si_ind, ei_ind = start_inds[vi], end_inds[vi]
      for vo in range(self._nviews):
        if vo == vi: continue
        so_ind, eo_ind = start_inds[vo], end_inds[vo]
        rmat[so_ind:eo_ind, si_ind:ei_ind] = self.view_projections[vi][vo]

    return rmat

  def nullspace_matrix(self):
    return -np.eye(self._tot_dim) + self.redundancy_matrix().T

  def save_to_file(self, fname):
    nmat = self.nullspace_matrix()
    vproj = self.view_projections
    cfg = self.config.__dict__
    np.save(fname, [nmat, vproj, cfg])
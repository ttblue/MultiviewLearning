import cvxpy as cvx
import numpy as np
import time

from models.model_base import ModelException
from utils import cvx_utils


import IPython  


_SOLVER = cvx.GUROBI


class NBSMVRLConfig(object):
  def __init__(
      self, group_regularizer="inf", global_regularizer="L1", lambda_group=1.0,
      lambda_global=1.0, sp_eps=1e-5, verbose=True):
    self.group_regularizer = group_regularizer
    self.global_regularizer = global_regularizer
    self.lambda_group = lambda_group
    self.lambda_global = lambda_global

    self.sp_eps = sp_eps

    self.verbose = verbose


class NaiveBlockSparseMVRL(object):
  def __init__(self, config):
    self.config = config
    self._training = True

  def _block_sparse_obj_for_view(self, vi):
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
    objs = (error_obj, group_sparse_obj, reg_obj)
        
    return p_var, objs, G

  def _optimize(self):
    # Add individual view objectives together
    vi_objs = {}
    vi_vars = {}
    vi_groups = {}
    combined_objs = []
    for vi in range(self._nviews):
      vi_vars[vi], vi_objs[vi], vi_groups[vi] = (
          self._block_sparse_obj_for_view(vi))
      error_obj, group_sparse_obj, reg_obj = vi_objs[vi]
      combined_objs.append(
          error_obj + self.config.lambda_group * group_sparse_obj +
              self.config.lambda_global * reg_obj
      )
    obj = np.sum(combined_objs)
    prob = cvx.Problem(cvx.Minimize(obj))

    if self.config.verbose:
      print("Optimizing for group sparse transforms.")
      start_time = time.time()
    try:
      prob.solve(solver=_SOLVER, verbose=self.config.verbose)
    except (KeyError, cvx.SolverError) as e:
      print("        Error using solver %s: %s.\n"
            "        Using default solver." %
            (_SOLVER, e))
      try:
        prob.solve(verbose=self.config.verbose)#warm_sart=True)
      except Exception as e2:
        print("Issue with solvers.")
        IPython.embed()

    self.view_projections = {}
    for vi in range(self._nviews):
      # Just extracting the indices to split the concatenated projection at.
      split_inds = [g[0] for g in vi_groups[vi][1:]]
      other_views = [vo for vo in range(self._nviews) if vo != vi]
      projs = vi_vars[vi].value
      projs_sp = np.where(np.abs(projs) < self.config.sp_eps, 0., projs)
      projs_split = np.split(projs_sp, split_inds, axis=0)
      self.view_projections[vi] = {
          vo: pr for vo, pr in zip(other_views, projs_split)
      }

    if self.config.verbose:
      print("Optimization done in %.2fs. " % (time.time() - start_time))
      print("Final objective value: %.4f" % obj.value)
      print("Individual objective values:")
      all_error = 0.
      all_gs = 0.
      all_reg = 0.
      for vi in range(self._nviews):
        eobj, gobj, robj = [o.value for o in vi_objs[vi]]
        print("\n  View %i:" % (vi + 1))
        print("    Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
                eobj, gobj, robj))
        all_error += eobj
        all_gs += gobj
        all_reg += robj
      print("\nOverall objective values:")
      print("  Recon. error: %.3f, GS val: %.3f, Reg val: %.3f" % (
              all_error, all_gs, all_reg))

    self._training = False

  def fit(self, view_data):
    self._nviews = len(view_data)
    self._view_data = view_data
    self._npts = view_data[0].shape[0]
    self._tot_dim = np.sum([view_data[i].shape[1] for i in view_data])
    self._optimize()

  def redundancy_matrix(self):
    if self._training is True:
      raise ModelException("Model has not been trained yet.")

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
    np.save(fname, [nmat, vproj])
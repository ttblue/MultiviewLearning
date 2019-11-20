# Naive single-view solvers defined for naive MV-RL.
import cvxpy as cvx
import numpy as np
import time
import torch
from torch import nn, optim

from models.abstract_single_view_rl import\
    AbstractSVSConfig, AbstractSingleViewSolver
from models.model_base import ModelException
from models import torch_models
from utils import cvx_utils, torch_utils, utils


import IPython


_SOLVER = cvx.GUROBI
_OBJ_ORDER = ["error", "gs", "reg", "total"]


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

  def set_data(self, data):
    # raise NotImplemented("Abstract class method.")
    self._nviews = len(data)
    self._view_data = data[self.view_id]
    self._npts, self._dim = self._view_data.shape
    self._rest_data = {vi:data[vi] for vi in data if vi != self.view_id}
    self._rest_dims = {vi:vd.shape[1] for vi, vd in self._rest_data.items()}

    self._has_data = True

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

  def get_objective(self, obj_type=None):
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


################################################################################
# Torch based NN model
################################################################################

class SVNNSConfig(AbstractSVSConfig):
  def __init__(
      self, nn_config, group_regularizer, global_regularizer, lambda_group,
      lambda_global, batch_size, lr, max_iters, *args, **kwargs): 
      # n_solves, lambda_group_init, lambda_group_beta, resolve_change_thresh,
      # n_resolve_attempts, sp_eps, *args, **kwargs):
    super(SVNNSConfig, self).__init__(*args, **kwargs)

    self.nn_config = nn_config
    self.group_regularizer = group_regularizer
    self.global_regularizer = global_regularizer
    self.lambda_group = lambda_group
    self.lambda_global = lambda_global

    # self.n_solves = n_solves
    # self.lambda_group_init = lambda_group_init
    # self.lambda_group_beta = lambda_group_beta
    # # self.global_reg_beta = global_reg_beta

    self.batch_size = batch_size
    self.lr = lr
    self.max_iters = max_iters
    # self.resolve_change_thresh = resolve_change_thresh
    # self.n_resolve_attempts = n_resolve_attempts

    # self.sp_eps = sp_eps


# Some default options for torch norms
_TORCH_NORM_MAP = {
    "inf": np.inf,
    "Linf": np.inf,
    "fro": "fro",
    "L1": 1,
    "L2": 2,
}


class NNSolver(AbstractSingleViewSolver, nn.Module):
  def __init__(self, view_id, config):
    nn.Module.__init__(self)
    AbstractSingleViewSolver.__init__(self, view_id, config)
    # super(NNSolver, self).__init__(view_id, config)

    self.recon_criterion = nn.MSELoss(reduction="mean")

    self._obj_map = None

  def set_data(self, data):
    # raise NotImplemented("Abstract class method.")
    self._nviews = len(data)
    self._data_torch = torch_utils.dict_numpy_to_torch(data)
    self._view_data = self._data_torch[self.view_id]
    self._npts, self._dim = self._view_data.shape
    self._rest_data = {
        vi: vd for vi, vd in self._data_torch.items() if vi != self.view_id}

    self._initialized = False
    self._has_data = True

  def initialize(self):
    if not self._has_data:
      raise ModelException("Data has not been set. Use set_data function.")
    if self._initialized:
      print("Model already initialized.")
      return

    # self._lambda_global = value=self.config.lambda_global
    gl_reg = self.config.global_regularizer
    self._global_norm_p = _TORCH_NORM_MAP.get(gl_reg, gl_reg)
    self._lambda_global = self.config.lambda_global

    gs_reg = self.config.group_regularizer
    self._group_norm_p = _TORCH_NORM_MAP.get(gs_reg, gs_reg)
    self._lambda_group = self.config.lambda_group

    self.config.nn_config.set_sizes(output_size=self._dim)
    self._p_tfms = nn.ModuleDict()
    self._p_last_layers = {}
    for vi in self._rest_data:
      vi_config = self.config.nn_config.copy()
      vi_config.set_sizes(input_size=self._rest_data[vi].shape[1])
      vi_transform = torch_models.MultiLayerNN(vi_config)
      self._p_tfms[vi] = vi_transform

      vi_layer = vi_transform.get_layer_params(-1)
      # IPython.embed()
      if vi_layer.bias is None:
        vi_param = vi_layer.weight
      else:
        vi_param = torch.cat(
            [vi_layer.weight, torch.reshape(vi_layer.bias, (1, -1))], dim=0)
      self._p_last_layers[vi] = vi_param
    # IPython.embed()
      # vi_transform.get_layer_params(-1)  #(vi_layer.weight, vi_layer.bias)

    # self._view_data_torch = torch.from_numpy(self._view_data.astype(torch_utils._DTYPE))
    # self._rest_data_torch = torch_utils.dict_numpy_to_torch(self._rest_data)

    self.opt = optim.Adam(self.parameters(), self.config.lr)
    self._initialized = True

  def _reconstruction(self, data=None):
    data = self._rest_data if data is None else data
    p_recons = torch.stack(
        [self._p_tfms[vi](data[vi]) for vi in self._p_tfms], dim=0)
    return torch.sum(p_recons, dim=0)

  def _error(self):
    error = self.recon_criterion(self._reconstruction(), self._view_data)
    return error.detach().numpy()

  def _group_reg(self):
    group_reg = torch.sum(torch.Tensor(
        [torch.norm(l, p=self._group_norm_p)
        for l in self._p_last_layers.values()]))
    return group_reg.detach().numpy()

  def _global_reg(self):
    all_params = torch.cat([l for l in self._p_last_layers.values()])
    global_reg = torch.norm(all_params, p=self._global_norm_p)
    return global_reg

  def _total_obj(self):
    return self._error() + self._lambda_group * self._group_reg()

  def forward(self, data):
    # reconstruction = np.sum(
    #     [self._p_tfms[vi](data[vi])
    #      for vi in self._p_tfms if vi in data])
    return self._reconstruction(data)

  def loss(self, vi_data, recons):
    # xv = self._split_views(x, rtn_torch=True)
    obj = self.recon_criterion(recons, vi_data)
    # Group sparsity penalty:
    group_norms = torch.Tensor([
        torch.norm(l, p=self._group_norm_p)
        for l in self._p_last_layers.values()])
    obj += self._lambda_group * torch.sum(group_norms)
    # Global penalty
    global_norm = torch.norm(torch.cat(
        [l for l in self._p_last_layers.values()]), p=self._global_norm_p)
    obj += self._lambda_global * global_norm
    return obj

  def _shuffle(self, xvs):
    npts = xvs[utils.get_any_key(xvs)].shape[0]
    r_inds = np.random.permutation(npts)
    return {vi:xv[r_inds] for vi, xv in xvs.items()}

  def _train_loop(self):
    xvs = self._shuffle(self._data_torch)
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      xvs_batch = {vi:xv[b_start:b_end] for vi, xv in xvs.items()}
      xvs_batch_view = xvs_batch[self.view_id]
      xvs_batch_rest = {
          vi: xv for vi, xv in xvs_batch.items() if vi != self.view_id}
      # keep_subsets = next(self._view_subset_shuffler)
      # xvs_dropped_batch = {vi:xvs_batch[vi] for vi in keep_subsets}

      self.opt.zero_grad()
      recons = self.forward(xvs_batch_rest)
      loss_val = self.loss(xvs_batch_view, recons)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self):
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    self._n_batches = int(np.ceil(self._npts / self.config.batch_size))

    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("\nIteration %i out of %i." % (itr + 1, self.config.max_iters))
        self._train_loop()

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          print("Loss: %.5f" % float(self.itr_loss.detach()))
          print("Iteration %i took %0.2fs." % (itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
    return self

  def get_objective(self, obj_type="all"):
    if self._obj_map is None:
      self._obj_map = {
        "error": self._error,
        "gs": self._group_reg,
        "reg": self._global_reg,
        "total": self._total_obj
      }

    if obj_type == "all":
      return [self._obj_map[otype]() for otype in _OBJ_ORDER]
    return self._obj_map.get(obj_type, "total")()

  def compute_projections(self):
    projections = {}
    for i, pl in self._p_last_layers.items():
      projections[i] = pl.detach().numpy().T
    self.projections = projections

  def predict(self, xvs, rtn_torch=False):
    if self.training:
      raise ModelException("Model not yet trained!")

    xvs = torch_utils.dict_numpy_to_torch(xvs)
    preds = self.forward(xvs)

    return preds if rtn_torch else preds.detach().numpy()
    # return preds


################################################################################
################################################################################

_SOLVERS = {
    "naive_opt": OptimizationSolver,
    "naive_nn": NNSolver,
}
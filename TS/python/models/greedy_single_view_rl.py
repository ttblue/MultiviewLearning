# Naive single-view solvers defined for naive MV-RL.
import cvxpy as cvx
import multiprocessing as mp
import numpy as np
import time
import torch
from torch import nn, optim

from models.model_base import ModelException, BaseConfig
from models.naive_single_view_rl import\
    AbstractSVSConfig, AbstractSingleViewSolver
from models import torch_models
from utils import cvx_utils, torch_utils, utils


import IPython


_SOLVER = cvx.GUROBI
_OBJ_ORDER = ["error", "gs", "reg", "total"]
# Some default options for torch norms
_TORCH_NORM_MAP = {
    "inf": np.inf,
    "Linf": np.inf,
    "fro": "fro",
    "L1": 1,
    "L2": 2,
}


################################################################################
# CVX based optimization
################################################################################

class GOSConfig(AbstractSVSConfig):
  def __init__(
      self, regularizer, lambda_reg, *args, **kwargs):
    super(GOSConfig, self).__init__(*args, **kwargs)
    self.regularizer = regularizer
    self.lambda_reg = lambda_reg


class GreedyOptimizationSolver(AbstractSingleViewSolver):
  def __init__(self, config):
    self.config = config


################################################################################
# Torch based NN model
################################################################################

class GNNSConfig(AbstractSVSConfig):
  def __init__(
        self, nn_config, regularizer, lambda_reg, batch_size, lr, max_iters,
        parallel, n_jobs, *args, **kwargs):
    self.nn_config = nn_config
    self.regularizer = regularizer
    self.lambda_reg = lambda_reg

    self.batch_size = batch_size
    self.lr = lr
    self.max_iters = max_iters

    self.parallel = parallel
    self.n_jobs = n_jobs

    super(GNNSConfig, self).__init__(*args, **kwargs)


class SingleIterationNNSolver(nn.Module):
  def __init__(self, parent_config, base_view_id):
    self.config = parent_config
    self.base_view_id = base_view_id
    self._initialized = False

    super(SingleIterationNNSolver, self).__init__()

  def initialize(self, new_view_id, fixed_view_ids, data):
    self.new_view_id = new_view_id
    self.fixed_view_ids = fixed_view_ids
    self.trainable_views = fixed_view_ids + [new_view_id]

    # Assume all the data is already in torch.Tensor format.
    self._npts = data[self.base_view_id].shape[0]
    self._view_data = {vi: data[vi] for vi in self.trainable_views}
    self._view_data[self.base_view_id] = data[self.base_view_id]

    reg = self.config.regularizer
    self._norm_p = _TORCH_NORM_MAP.get(reg, reg)
    self._lambda_reg = self.config.lambda_reg

    self._p_tfms = nn.ModuleDict()
    self._p_last_layers = {}
    # Setup fixed-view projections
    for vi in self.trainable_views:
      vi_config = self.config.nn_config.copy()
      vi_config.set_sizes(input_size=self._view_data[vi].shape[1])
      vi_transform = torch_models.MultiLayerNN(vi_config)
      self._p_tfms["T%i"%vi] = vi_transform

      vi_layer = vi_transform.get_layer_params(-1)
      if vi_layer.bias is None:
        vi_param = vi_layer.weight
      else:
        vi_param = torch.cat(
            [vi_layer.weight, torch.reshape(vi_layer.bias, (1, -1))], dim=0)
      self._p_last_layers[vi] = vi_param

    self.recon_criterion = nn.MSELoss(reduction="mean")
    self.opt = optim.Adam(self.parameters(), self.config.lr)
    self._initialized = True

  def _reconstruction(self, data=None):
    data = self._view_data if data is None else data
    p_recons = torch.stack(
        [self._p_tfms["T%i"%vi](data[vi]) for vi in self.trainable_views],
        dim=0)
    return torch.sum(p_recons, dim=0)

  def _error(self, data=None):
    data = self._view_data if data is None else data
    error = self.recon_criterion(
        self._reconstruction(data), data[self.base_view_id])
    return error.detach().numpy()

  def _regularization(self):
    reg = torch.sum(torch.Tensor(
        [torch.norm(l, p=self._norm_p)
        for l in self._p_last_layers.values()]))
    return reg.detach().numpy()

  def _total_obj(self):
    return self._error() + self._lambda_reg * self._regularization()

  def forward(self, data):
    # reconstruction = np.sum(
    #     [self._p_tfms["T%i"%i](data[vi])
    #      for vi in self.trainable_views])
    return self._reconstruction(data)

  def loss(self, base_view_data, recons):
    # xv = self._split_views(x, rtn_torch=True)
    obj = self.recon_criterion(recons, base_view_data)
    # Regularization penalty:
    reg_norms = torch.Tensor([
        torch.norm(l, p=self._norm_p)
        for l in self._p_last_layers.values()])
    obj += self._lambda_reg * torch.sum(reg_norms)
    return obj

  def _shuffle(self, xvs):
    npts = xvs[utils.get_any_key(xvs)].shape[0]
    r_inds = np.random.permutation(npts)
    return {vi:xv[r_inds] for vi, xv in xvs.items()}

  def _train_loop(self):
    xvs = self._shuffle(self._view_data)
    self.itr_loss = 0.
    for bidx in range(self._n_batches):
      b_start = bidx * self.config.batch_size
      b_end = b_start + self.config.batch_size
      xvs_batch = {vi:xv[b_start:b_end] for vi, xv in xvs.items()}
      xvs_batch_view = xvs_batch[self.base_view_id]
      xvs_batch_rest = {
          vi: xv for vi, xv in xvs_batch.items() if vi in self.trainable_views}
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
        self._train_loop()

        if self.config.verbose:
          itr_diff_time = time.time() - itr_start_time
          loss_val = float(self.itr_loss.detach())
          print("\nIteration %i out of %i (in %.2fs). Loss: %.5f" %
                (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
                end='\r')
      if self.config.verbose:
        print("\nIteration %i out of %i (in %.2fs). Loss: %.5f" %
              (itr + 1, self.config.max_iters, itr_diff_time, loss_val),
              end='\r')
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    self.eval()
    print("Training finished in %0.2f s." % (time.time() - all_start_time))
    return self


def view_solve_func(view_solver):
  view_solver.fit()
  obj_val = view_solver._total_obj()
  fname = "./temp_models/tmp_greedy_%i" % view_solver.new_view_id
  torch.save(view_solver.state_dict(), fname)
  return obj_val, fname


class GreedyNNSolver(AbstractSingleViewSolver):
  def __init__(self, view_id, config):
    super(GreedyNNSolver, self).__init__(view_id, config)
    self._initialized = False
    self.training = True

  def set_data(self, data):
    # raise NotImplemented("Abstract class method.")
    self._nviews = len(data)
    self._data_torch = torch_utils.dict_numpy_to_torch(data)
    self._npts, self._dim = self._data_torch[self.view_id].shape
    # self._view_data = self._data_torch[self.view_id]
    # self._rest_data = {
    #     vi: vd for vi, vd in self._data_torch.items() if vi != self.view_id}
    self._has_data = True

  def initialize(self):
    if not self._has_data:
      raise ModelException("Data has not been set. Use set_data function.")
    if self._initialized:
      print("Model already initialized.")
      return

    self.config.nn_config.set_sizes(output_size=self._dim)

    self._greedy_view_order = []
    self._greedy_view_solver = {}

  def _initialize_solve_iteration(self):
    fixed_views = self._greedy_view_order + [self.view_id]
    self._next_views = [i for i in range(self._nviews) if i not in fixed_views]
    iter_solvers = {
        vi: SingleIterationNNSolver(self.config, self.view_id)
        for vi in self._next_views}
    for vi, solver in iter_solvers.items():
      solver.initialize(vi, self._greedy_view_order, self._data_torch)
    self._initialized = True
    return iter_solvers

  def _solve_sequential(self, iter_solvers):
    obj_vals = []
    for vi in self._next_views:
      solver = iter_solvers[vi]
      if self.config.verbose:
        vi_start_time = time.time()
        print("  Solving for view %i from %s." % (vi, self._next_views))

      solver.fit()
      vi_obj = solver._total_obj()
      obj_vals.append(vi_obj)

      if self.config.verbose:
        diff_time = time.time() - vi_start_time
        print("  Solved for view %i in %.2fs. Obj val: %.3f" %
              (vi, diff_time, vi_obj))

    self._iter_best_view = self._next_views[np.argmin(obj_vals)]
    best_solver = iter_solvers[self._iter_best_view]
    self._greedy_view_solver[self._n_greedy_views] = best_solver

  def _solve_parallel(self, iter_solvers):
    if self.config.verbose:
      print("  Solving problems in parallel...")
      start_time = time.time()
    pool = mp.Pool(processes=self._n_jobs)

    solvers = [iter_solvers[vi] for vi in self._next_views]
    result = pool.map_async(view_solve_func, solvers)
    pool.close()
    pool.join()

    result = result.get()

    obj_vals = [oval for oval, _ in result]
    best_ind = np.argmin(obj_vals)
    best_fname = result[best_ind][1]

    self._iter_best_view = self._next_views[best_ind]    
    best_solver = iter_solvers[self._iter_best_view]
    best_solver.load_state_dict(torch.load(best_fname))
    best_solver.eval()
    self._greedy_view_solver[self._n_greedy_views] = best_solver

  def _greedy_solve_iteration(self):
    if self.config.verbose:
      iter_start_time = time.time()
      if self._n_greedy_views > 1:
        print("  Greedy solve iteration for %i views." % self._n_greedy_views)
        print("  Fixed views so far: %s" % self._greedy_view_order)
      else:
        print("  Greedy solve iteration for 1 view.")

    iter_solvers = self._initialize_solve_iteration()
    self._solve_func(iter_solvers)
    self._greedy_view_order.append(self._iter_best_view)

    if self.config.verbose:
      print("  Best next view for greedy selection: %i" % self._iter_best_view)
      diff_time = time.time() - iter_start_time

  def fit(self):
    if self.config.verbose:
      all_start_time = time.time()

    self._n_jobs = (
        self._nviews - 1
        if (self.config.parallel and self.config.n_jobs is None)
        else self.config.n_jobs
    )

    self._solve_func = (
        self._solve_parallel if self.config.parallel else
        self._solve_sequential)
    for self._n_greedy_views in range(1, self._nviews):
      self._greedy_solve_iteration()
    self._all_view_solver = self._greedy_view_solver[self._n_greedy_views]

    self.training = False
    if self.config.verbose:
      diff_time = time.time() - all_start_time
      print("Overall greedy solving procedure took %.3fs" % diff_time)

    return self

  def compute_projections(self, num_greedy_views=-1):
    solver = (
        self._greedy_view_solver[num_greedy_views] if num_greedy_views > 0 else
        self._all_view_solver)
    projections = {}
    for i, pl in solver._p_last_layers.items():
      projections[i] = pl.detach().numpy().T
    self.projections = projections

  def get_objective(self, obj_type=None):
    return self._all_view_solver._total_obj()

  def _get_appropriate_solver(self, available_views):
    first_view = self._greedy_view_order[0]
    if first_view not in available_views:
      raise("Prediction requires %i at the very least." % first_view)

    for idx, view in enumerate(self._greedy_view_order):
      if view not in available_views:
        return idx, self._greedy_view_solver[idx]
    return idx + 1, self._all_view_solver

  def predict(self, xvs, rtn_torch=False):
    if self.training:
      raise ModelException("Model not yet trained!")    

    available_views = list(xvs.keys())
    n_greedy_views, solver = self._get_appropriate_solver(available_views)

    xvs = torch_utils.dict_numpy_to_torch(xvs)
    preds = solver.forward(xvs)

    return preds if rtn_torch else preds.detach().numpy()


################################################################################
################################################################################

_SOLVERS = {
    "greedy_nn": GreedyNNSolver,
}
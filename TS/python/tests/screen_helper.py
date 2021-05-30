import itertools
import numpy as np

def all_subset_accuracy_cat(model, data):
  data = {vi:np.array(vdat) for vi, vdat in data.items()}
  view_range = list(range(len(data)))
  npts = data[0].shape[0]
  v_sizes = {vi: vdat.shape[1] for vi, vdat in data.items()}
  zero_pads = {vi: np.zeros((npts, vs)) for vi, vs in v_sizes.items()}

  v_splits = np.cumsum([v_sizes[vi] for vi in view_range[:-1]])

  all_errors = {}
  subset_errors = {}
  
  for nv in view_range:
    s_error = []
    for subset in itertools.combinations(view_range, nv + 1):
      input_data = np.concatenate([
          (data[vi] if vi in subset else zero_pads[vi])
          for vi in view_range
          ], axis=1)
      # IPython.embed()
      pred = model.predict({0:input_data})

      pred_split = np.array_split(pred[0], v_splits, axis=1)
      pred_dict = {vi: pred_split[vi] for vi in view_range}

      err = error_func(data, pred_dict)
      s_error.append(err)
      all_errors[subset] = err
    subset_errors[(nv + 1)] = np.mean(s_error)
  return subset_errors, all_errors


def spaghetti_evals_single_view(
    n_views, error_evals, start_view=0, max_curves=100, rtn_orders=False):
  other_views = [vi for vi in range(n_views) if vi != start_view]

  def get_subset_error(subset):
    sorted_subset = tuple(sorted(subset))
    return error_evals[sorted_subset]

  # Need to be careful if there are too many views:
  view_orders = list(itertools.permutations(other_views, n_views - 2))
  n_view_orders = len(view_orders)
  shuffled_orders = [
      view_orders[i] for i in np.random.permutation(n_view_orders)[:max_curves]]

  error_curves = []
  all_error = get_subset_error(np.arange(n_views))
  start_error = get_subset_error([start_view])
  for vorder in shuffled_orders:
    v_list = [start_view]
    err_curve = [start_error]
    for vi in vorder:
      v_list.append(vi)
      err_curve.append(get_subset_error(v_list))
    err_curve.append(all_error)
    error_curves.append(err_curve)

  if rtn_orders:
    return error_curves, shuffled_orders
  return error_curves


def error_mat(model, data, error_func):
  nv = len(data)
  view_range = list(range(nv))
  errors = np.empty((nv, nv))
  for vi in view_range:
    v_error = {}
    input_data = {vi:data[vi]}
    pred = model.predict(input_data)
    for vj in view_range:
      errors[vi, vj] = error_func(data, {vj:pred[vj]})
  return errors

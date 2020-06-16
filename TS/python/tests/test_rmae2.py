# Testing multi-view autoencoder
import itertools
import numpy as np
import torch
from torch import nn

from dataprocessing import multiview_datasets
from models import robust_multi_ae, torch_models, multi_ae
from synthetic import multimodal_systems
from utils import torch_utils, utils

try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


# Models for classification 
from sklearn.linear_model import \
    LogisticRegression, PassiveAggressiveClassifier,\
    SGDClassifier, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier


import IPython


torch.set_default_dtype(torch.float32)


def load_3news(ndims_red=None, rtn_proj_mean=True):
  view_data, view_sizes, labels, multi_labels, news_sources, terms_dat = (
      multiview_datasets.load_3sources_dataset())
  if ndims_red:
    proj_base_data, projs, means = multiview_datasets.dim_reduce(
        view_data, ndims=ndims_red, fill=False, rtn_proj_mean=True)
    if rtn_proj_mean:
      return proj_base_data, view_sizes, labels, projs, means
    return proj_base_data, view_sizes, labels
  return view_data, view_sizes, labels


def default_RMAE_config(v_sizes, hidden_size=16, joint_code_size=32):
  n_views = len(v_sizes)

  # Default Encoder config:
  output_size = hidden_size
  layer_units = [32] # [32, 64]
  use_vae = False
  activation = nn.ReLU  # nn.functional.relu
  last_activation = nn.Sigmoid  # functional.sigmoid
  dropout_p = 0
  encoder_params = {}
  for i in range(n_views):
    input_size = v_sizes[i]
    layer_types, layer_args = torch_utils.generate_linear_types_args(
        input_size, layer_units, output_size)
    encoder_params[i] = torch_models.MNNConfig(
        input_size=input_size, output_size=output_size, layer_types=layer_types,
        layer_args=layer_args, activation=activation,
        last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)

  input_size = joint_code_size
  layer_units = [32]  #[64, 32]
  use_vae = False
  last_activation = torch_models.Identity
  dropout_p = 0.
  decoder_params = {}
  for i in range(n_views):
    output_size = v_sizes[i]
    layer_types, layer_args = torch_utils.generate_linear_types_args(
        input_size, layer_units, output_size)
    decoder_params[i] = torch_models.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)

  input_size = hidden_size * len(v_sizes)
  output_size = joint_code_size
  layer_units = [64]  #[64, 64]
  layer_types, layer_args = torch_utils.generate_linear_types_args(
      input_size, layer_units, output_size)
  use_vae = False
  joint_coder_params = torch_models.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)

  drop_scale = True
  zero_at_input = True

  code_sample_noise_var = 0.
  max_iters = 1000
  batch_size = 50
  lr = 1e-3
  verbose = True
  config = robust_multi_ae.RMAEConfig(
      joint_coder_params=joint_coder_params, drop_scale=drop_scale,
      zero_at_input=zero_at_input, v_sizes=v_sizes, code_size=joint_code_size,
      encoder_params=encoder_params, decoder_params=decoder_params,
      code_sample_noise_var=code_sample_noise_var, max_iters=max_iters,
      batch_size=batch_size, lr=lr, verbose=verbose)

  return config


# Assuming < 5 views for now
_COLORS = ['b', 'r', 'g', 'y']
def plot_recon(true_vals, pred_vals, labels, title=None):
  if not MPL_AVAILABLE:
    print("Matplotlib not available.")
    return

  for tr, pr, l, c in zip(true_vals, pred_vals, labels, _COLORS):
    plt.plot(tr[:, 0], color=c, label=l + " True")
    plt.plot(pr[:, 0], color=c, ls='--', label=l + " Pred")
  plt.legend()
  if title:
    plt.title(title)
  plt.show()


def plot_simple(tr, pr, l, title=None):
  plt.plot(tr[:, 0], color='b', label=l + " True")
  plt.plot(pr[:, 0], color='r', ls='--', label=l + " Pred")
  plt.legend()
  if title:
    plt.title(title)
  plt.show()


def plot_heatmap(mat, msplit_inds, misc_title=""):
  fig = plt.figure()
  hm = plt.imshow(mat)
  plt.title("Redundancy Matrix: %s" % misc_title)
  cbar = plt.colorbar(hm)
  for mind in msplit_inds:
    mind -= 0.5
    plt.axvline(x=mind, ls="--")
    plt.axhline(y=mind, ls="--")
  plt.show(block=True)


def make_subset_list(nviews):
    view_subsets = []
    view_range = list(range(nviews))
    for nv in view_range:
      view_subsets.extend(list(itertools.combinations(view_range, nv + 1)))


def error_func(true_data, pred):
  return np.sum([np.linalg.norm(true_data[vi] - pred[vi]) for vi in pred])


def all_subset_accuracy(model, data):
  view_range = list(range(len(data)))
  all_errors = {}
  subset_errors = {}
  for nv in view_range:
    s_error = []
    for subset in itertools.combinations(view_range, nv + 1):
      input_data = {vi:data[vi] for vi in subset}
      pred = model.predict(input_data)
      err = error_func(data, pred)
      s_error.append(err)
      all_errors[subset] = err
    subset_errors[(nv + 1)] = np.mean(s_error)

  return subset_errors, all_errors


def evaluate_downstream_task(tr_x, tr_y, te_x, te_y):
  # Logreg
  # Keep same model?
  kwargs = {'C': 1e2, "max_iter": 500}#, "solver": "newton-cg"}
  model = LogisticRegression(**kwargs)
  mtype = "Logreg"

  ntrain, ntest = tr_x.shape[0], te_x.shape[0]

  print(mtype)
  print("Original dsets --")
  model.fit(tr_x, tr_y)
  tr_pred = model.predict(tr_x)
  te_pred = model.predict(te_x)
  train_acc = (tr_pred == tr_y).sum() / ntrain
  test_acc = (te_pred == te_y).sum() / ntest
  print("  [%s] All views:\nTrain accuracy: %.3f\nTest accuracy: %.3f" % (mtype, train_acc, test_acc))


def setup_RMAE(v_sizes, drop_scale=True, zero_at_input=False, max_iters=1000):
  hidden_size = 32
  joint_code_size = 64
  config = default_RMAE_config(
      v_sizes, hidden_size=hidden_size, joint_code_size=joint_code_size)

  config.drop_scale = drop_scale
  config.zero_at_input = zero_at_input
  config.max_iters = max_iters

  model = robust_multi_ae.RobustMultiAutoEncoder(config)
  return model


def setup_intersection_mae(v_sizes, max_iters=1000):
  code_size = 64
  config = multi_ae.default_MAE_config(v_sizes, code_size=code_size)
  config.max_iters = max_iters

  model = multi_ae.MultiAutoEncoder(config)
  return model


def setup_cat_ae(v_sizes, max_iters=1000):
  cat_dim = np.sum([sz for sz in v_sizes.values()])
  code_size = 64

  cat_vsizes = {0: cat_dim}
  config = multi_ae.default_MAE_config(cat_vsizes, code_size=code_size)
  config.max_iters = max_iters  

  model = multi_ae.MultiAutoEncoder(config)
  return model


def make_proper_dset(xvs, ys):
  npts = len(xvs[utils.get_any_key(xvs)])
  valid_pts = np.zeros(npts)
  for vi, xv in xvs.items():
    vi_valid = np.array([i for i in range(npts) if xv[i] is not None])
    valid_pts[vi_valid] += 1
  valid_inds = valid_pts.nonzero()[0]

  xvs_valid = {vi: [xv[i] for i in valid_inds] for vi, xv in xvs.items()}
  ys_valid = np.array([ys[i] for i in valid_inds])

  return xvs_valid, ys_valid


def fill_missing(xvs, v_sizes):
  npts = len(xvs[utils.get_any_key(xvs)])
  xvs_filled = {}
  for vi in v_sizes:
    vdim = v_sizes[vi]
    if vi in xvs:
      xv = xvs[vi]
      vzeros = np.zeros(vdim)
      xv_filled = np.array([
          (x if x is not None else vzeros) for x in xv])
    else:
      xv_filled = np.zeros((npts, vdim))
    xvs_filled[vi] = xv_filled
  return xvs_filled



def test_3news(args):
  ndims_red = args.ndims_red
  drop_scale = args.drop_scale
  zero_at_input = args.zero_at_input

  if ndims_red is not None:
    data, v_sizes, labels, projs, means = load_3news(
        ndims_red, rtn_proj_mean=True)
  else:
    data, v_sizes, labels = load_3news()

  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}
  tr_frac = 0.8
  split_frac = [tr_frac, 1. - tr_frac]
  (tr_data, te_data), split_inds = multiview_datasets.split_data(
      data, split_frac, get_inds=True)
  tr_labels, te_labels = labels[split_inds[0]], labels[split_inds[1]]

  tr_cat = multiview_datasets.fill_missing(tr_data, cat_dims=True)
  te_cat = multiview_datasets.fill_missing(te_data, cat_dims=True)

  # IPython.embed()
  max_iters = 3000
  rmae_model = setup_RMAE(
      v_sizes, drop_scale, zero_at_input, max_iters=max_iters)
  # IPython.embed()
  imae_model = setup_intersection_mae(v_sizes, max_iters=max_iters)
  cmae_model = setup_cat_ae(v_sizes, max_iters=max_iters)
  # IPython.embed()2
  rmae_model.fit(tr_data)
  imae_model.fit(tr_data)
  cmae_tr, cmae_te = {0: tr_cat}, {0: te_cat}
  cmae_model.fit(cmae_tr)

  model = imae_model
  IPython.embed()

  # Everything together
  tr_y, te_y = labels[split_inds[0]], labels[split_inds[1]]

  rtr_x, _ = rmae_model.encode(tr_data, aggregate="mean")
  rte_x, _ = rmae_model.encode(te_data, aggregate="mean")
  rtr_x, rte_x = torch_utils.torch_to_numpy(rtr_x), torch_utils.torch_to_numpy(rte_x)

  itr_x, _ = imae_model.encode(tr_data, aggregate="mean")
  ite_x, _ = imae_model.encode(te_data, aggregate="mean")
  itr_x, ite_x = torch_utils.torch_to_numpy(itr_x), torch_utils.torch_to_numpy(ite_x)

  ctr_x, _ = cmae_model.encode(cmae_tr, aggregate="mean")
  cte_x, _ = cmae_model.encode(cmae_te, aggregate="mean")
  ctr_x, cte_x = torch_utils.torch_to_numpy(ctr_x), torch_utils.torch_to_numpy(cte_x)

  tr_cat = multiview_datasets.fill_missing(tr_data, cat_dims=True)
  te_cat = multiview_datasets.fill_missing(te_data, cat_dims=True)

  print("\n\nRMAE: Robust Multi-view AE")
  evaluate_downstream_task(rtr_x, tr_y, rte_x, te_y)

  print("\n\nIMAE: Intersection Multi-view AE")
  evaluate_downstream_task(itr_x, tr_y, ite_x, te_y)

  print("\n\nCAE: Concatenated AE")
  evaluate_downstream_task(ctr_x, tr_y, cte_x, te_y)

  print("\n\nCAT: Simple concatenation")
  evaluate_downstream_task(tr_cat, tr_y, te_cat, te_y)

  # Only one view
  # model = rmae_model
  # nviews = len(v_sizes)
  for vi in range(nviews):
    tr_vi = {vi:tr_data[vi]}
    te_vi = {vi:te_data[vi]}
    tr_vi_x, tr_vi_y = make_proper_dset(tr_vi, tr_y)
    te_vi_x, te_vi_y = make_proper_dset(te_vi, te_y)

    ntr = len(tr_vi_x[vi])
    nte = len(te_vi_x[vi])
    tr_vi_with_None = {
        vj: tr_vi_x[vi] if vj == vi else [None] * ntr
        for vj in range(nviews)
    }
    te_vi_with_None = {
        vj: te_vi_x[vi] if vj == vi else [None] * nte
        for vj in range(nviews)
    }
    tr_vi_cat = {0: fill_missing(tr_vi_with_None, cat_dims=True)}
    # te_vi_cat = {0: multiview_datasets.fill_missing(te_vi_with_None, cat_dims=True)}
    # tr_vi = {vj:tr_data[vj] for vj in range(nviews) if vj != vi}
    # te_vi = {vj:te_data[vj] for vj in range(nviews) if vj != vi}

    rtr_x, _ = rmae_model.encode(tr_vi_x, aggregate="mean")
    rte_x, _ = rmae_model.encode(te_vi_x, aggregate="mean")
    rtr_x, rte_x = torch_utils.torch_to_numpy(rtr_x), torch_utils.torch_to_numpy(rte_x)

    itr_x, _ = imae_model.encode(tr_vi_x, aggregate="mean")
    ite_x, _ = imae_model.encode(te_vi_x, aggregate="mean")
    itr_x, ite_x = torch_utils.torch_to_numpy(itr_x), torch_utils.torch_to_numpy(ite_x)

    # ctr_x, _ = cmae_model.encode(tr_vi_cat, aggregate="mean")
    # cte_x, _ = cmae_model.encode(te_vi_cat, aggregate="mean")
    # ctr_x, cte_x = torch_utils.torch_to_numpy(ctr_x), torch_utils.torch_to_numpy(cte_x)


    print("\n\n\nView %i" % vi)
    print("\n\nRMAE")
    evaluate_downstream_task(rtr_x, tr_y, rte_x, te_y)

    print("\n\nIMAE")
    evaluate_downstream_task(itr_x, tr_y, ite_x, te_y)

    # print("\n\nCMAE")
    # evaluate_downstream_task(ctr_x, tr_y, cte_x, te_y)

    # print("\n\nsimple CAT")
    # evaluate_downstream_task(tr_cat, tr_y, te_cat, te_y)

  # Leave one out


    # for ovi in range(nviews):
    #   if ovi != vi:
    #     tr_vi[ovi] = [None] * ntr
    #     te_vi[ovi] = [None] * nte

    # trv_cat = multiview_datasets.fill_missing(tr_vi, cat_dims=True)
    # tev_cat = multiview_datasets.fill_missing(te_vi, cat_dims=True)

    # print("cat")
    # evaluate_downstream_task(trv_cat, trv_y, tev_cat, tev_y)

  # plot_heatmap(model.nullspace_matrix(), msplit_inds
  # plt.plot(x, y)
  # plt.title("Reconstruction error vs. number of views", fontsize=20)
  # plt.xticks([1,2,3,4,5], fontsize=15)
  # plt.yticks(fontsize=15)
  # plt.xlabel("Available views", fontsize=18)
  # plt.ylabel("Error", fontsize=18)

def make_synthetic_data(args):
  npts = args.npts
  n_views = args.nviews  
  s_dim = args.s_dim
  scale = args.scale
  noise_eps = args.noise_eps
  peps = args.peps

  # n_views = 3
  # subsets = [(0, 1), (1, 2), (2, 0)]
  subsets = [(i, (i + 1) % n_views) for i in range(n_views)]
  # subsets = [
  #     [j for j in range(n_views) if j != i] for i in range(n_views)]

  tfm_final = False
  rtn_correspondences = True
  view_data, ptfms, corrs = multimodal_systems.subset_redundancy_data(
      npts=npts, n_views=n_views, subsets=subsets, s_dim=s_dim,
      noise_eps=noise_eps, tfm_final=tfm_final, peps=peps,
      rtn_correspondences=rtn_correspondences)

  return view_data, ptfms, corrs


def error_func(true_data, pred, rtn_mean=True):
  denom = len(true_data[utils.get_any_key(true_data)]) if rtn_mean else 1.
  return np.sum(
      [np.linalg.norm(true_data[vi] - pred[vi]) for vi in pred]) / denom


def all_subset_accuracy(model, data):
  view_range = list(range(len(data)))
  all_errors = {}
  subset_errors = {}
  for nv in view_range:
    s_error = []
    for subset in itertools.combinations(view_range, nv + 1):
      input_data = {vi:data[vi] for vi in subset}
      pred = model.predict(input_data)
      err = error_func(data, pred)
      s_error.append(err)
      all_errors[subset] = err
    subset_errors[(nv + 1)] = np.mean(s_error)
  return subset_errors, all_errors


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

  error_curves = np.array(error_curves)
  if rtn_orders:
    return error_curves, shuffled_orders
  return error_curves


def get_all_start_view_spaghets(n_views, error_evals, max_curves=100):
  v_error_curves = {}
  for vi in range(n_views):
    v_error_curves[vi] = spaghetti_evals_single_view(
        n_views, error_evals, start_view=vi, max_curves=max_curves,
        rtn_orders=False)
  return v_error_curves


def error_mat(model, data):
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


def error_mat_cat(model, data):
  nv = len(data)
  view_range = list(range(nv))

  data = {vi:np.array(vdat) for vi, vdat in data.items()}
  npts = data[0].shape[0]
  v_sizes = {vi: vdat.shape[1] for vi, vdat in data.items()}
  zero_pads = {vi: np.zeros((npts, vs)) for vi, vs in v_sizes.items()}
  v_splits = np.cumsum([v_sizes[vi] for vi in view_range[:-1]])

  errors = np.empty((nv, nv))
  for vi in view_range:
    input_data = np.concatenate(
        [(data[vi] if vj == vi else zero_pads[vj]) for vj in view_range],
        axis=1)
    pred = model.predict({0:input_data})
    pred_split = np.array_split(pred[0], v_splits, axis=1)
    pred_dict = {vi: pred_split[vi] for vi in view_range}
    for vj in view_range:
      errors[vi, vj] = error_func(data, {vj:pred_dict[vj]})
  return errors


# Synthetic dataset:
# 3 views, 7-subspaces in all subsets of intersections of venn diagrams
def test_RMAE_synthetic_subset_redundancy(args):
  drop_scale = args.drop_scale
  zero_at_input = args.zero_at_input
  n_views = args.nviews

  data, ptfms, corrs = make_synthetic_data(args)
  v_sizes = {vi: vdat.shape[1] for vi, vdat in data.items()}

  #   data = {vi: d[:npts] for vi, d in data.items()}
  tr_frac = 0.8
  split_frac = [tr_frac, 1. - tr_frac]
  (tr_data, te_data), split_inds = multiview_datasets. split_data(
      data, split_frac, get_inds=True)

  tr_cat = multiview_datasets.fill_missing(tr_data, cat_dims=True)
  te_cat = multiview_datasets.fill_missing(te_data, cat_dims=True)

  # IPython.embed()
  max_iters = args.max_iters
  rmae_model = setup_RMAE(
      v_sizes, drop_scale, zero_at_input, max_iters=max_iters)
  # IPython.embed()
  imae_model = setup_intersection_mae(v_sizes, max_iters=max_iters)
  cmae_model = setup_cat_ae(v_sizes, max_iters=max_iters)

  rmae_model.fit(tr_data)
  imae_model.fit(tr_data)
  cmae_tr, cmae_te = {0: tr_cat}, {0: te_cat}
  cmae_model.fit(cmae_tr)
  IPython.embed()

  # savemodels = True
  # rnum = np.random.randn()
  # if savemodels:
  #   torch.save(rmae_model.state_dict(), "rmae_model_synth_nv%i.tmdl" % (n_views, rnum))
  #   torch.save(imae_model.state_dict(), "imae_model_synth_nv%i_%.4f" % (n_views, rnum))
  #   torch.save(cmae_model.state_dict(), "cmae_model_synth_nv%i_%.4f" % (n_views, rnum))
    # np.save("synth_data%.2f" % rnum, [tr_data, te_data])

  # loadmodels = False
  # if loadmodels:
  #   rmae_model
  # rnum = np.random.randn()
  def save_data():
    r_sub_tr, r_all_tr = all_subset_accuracy(rmae_model, tr_data)
    r_sub_te, r_all_te = all_subset_accuracy(rmae_model, te_data)
    i_sub_tr, i_all_tr = all_subset_accuracy(imae_model, tr_data)
    i_sub_te, i_all_te = all_subset_accuracy(imae_model, te_data)
    c_sub_tr, c_all_tr = all_subset_accuracy_cat(cmae_model, tr_data)
    c_sub_te, c_all_te = all_subset_accuracy_cat(cmae_model, te_data)

    err_matr_tr, err_matr_te = error_mat(rmae_model, tr_data), error_mat(rmae_model, te_data)
    err_mati_tr, err_mati_te = error_mat(imae_model, tr_data), error_mat(imae_model, te_data)
    err_matc_tr, err_matc_te = error_mat_cat(cmae_model, tr_data), error_mat_cat(cmae_model, te_data)

    torch.save(rmae_model.state_dict(), "rmae_model_synth_nv%i.tmdl" % (n_views))
    torch.save(imae_model.state_dict(), "imae_model_synth_nv%i.tmdl" % (n_views))
    torch.save(cmae_model.state_dict(), "cmae_model_synth_nv%i.tmdl" % (n_views))
    # np.save("synth_data%.2f" % rnum, [tr_data, te_data])
    np.save("synth_nv%i_all_dat" % (n_views),
        [tr_data, te_data,
         r_sub_tr, r_all_tr, r_sub_te, r_all_te,
         c_sub_tr, c_all_tr, c_sub_te, c_all_te,
         i_sub_tr, i_all_tr, i_sub_te, i_all_te,
         err_matr_tr, err_matr_te,
         err_mati_tr, err_mati_te,
         err_matc_tr, err_matc_te,
         ]
    )

  IPython.embed()
  # [tr_data, te_data, r_sub_tr, r_all_tr, r_sub_te, r_all_te, c_sub_tr, c_all_tr, c_sub_te, c_all_te, i_sub_tr, i_all_tr, i_sub_te, i_all_te]
  # plot_stuff
  # Training:'
  plt.rcParams.update({'font.size': 30})
  vr = np.arange(1, n_views + 1)
  # r_tr_vals = [r_sub_tr[vi] for vi in vr]
  # i_tr_vals = [i_sub_tr[vi] for vi in vr]
  # c_tr_vals = [c_sub_tr[n_views - vi + 1] for vi in vr]
  # r_te_vals = [r_sub_te[vi] for vi in vr]
  # i_te_vals = [i_sub_te[vi] for vi in vr]
  # c_te_vals = [c_sub_te[n_views - vi + 1] for vi in vr]
  plt.plot(vr, r_te_vals, label="Robust MAE")
  plt.plot(vr, i_te_vals, label="Intersect MAE")
  plt.plot(vr, c_te_vals, label="Concat AE")
  plt.legend()
  plt.xlabel("Number of available views")
  plt.xticks(vr)
  plt.ylabel("Average reconstruction error")
  plt.title("4 Views: Test error vs. available views")
  plt.show()
  # plt.rcParams.update({'font.size': 22})


  # All views:
  
    

# Motivation:
# Why do multi-view AEs tend to go toward intersections?


_TEST_FUNCS = {
    0: test_3news,
    1: test_RMAE_synthetic_subset_redundancy,
}


if __name__ == "__main__":
  # import sys
  # drop_scale = True
  # zero_at_input = True
  # ndims_red = None
  # try:
  #   drop_scale = bool(sys.argv[1])
  #   zero_at_input = bool(sys.argv[2])
  #   ndims_red = int(sys.argv[2])
  # except:
  #   pass
  # print("Drop scale: %s" % drop_scale)
  # print("Zero at input: %s" % zero_at_input)
  # print("Reduced dims: %s" % ndims_red)

  # test_RMAE(ndims_red, drop_scale, zero_at_input)

  np.set_printoptions(linewidth=1000, precision=3, suppress=True)
  torch.set_printoptions(precision=3)
  options = [
      ("ndims_red", int, "Reduced dims for data", 50),
      ("nviews", int, "Number of views", 3),
      ("npts", int, "Number of points", 1000),
      ("s_dim", int, "View subset dimensions", 3),
      ("peps", float, "Perturb epsilon", 1e-3),
      ("noise_eps", float, "Noise epsilon", 1e-3),
      ("scale", float, "Scale of the data.", 1.),
      ("zero_at_input", bool, "RMAE: flag for zero at input (or code)", True),
      ("drop_scale", bool, "RMAE: drop-out like scaling for missing views", True),
      ("max_iters", int, "Number of iters for opt.", 1000),
      ("batch_size", int, "Batch size for opt.", 100),
      ]
  args = utils.get_args(options)

  func = _TEST_FUNCS.get(args.expt, test_RMAE_synthetic_subset_redundancy)
  func(args)
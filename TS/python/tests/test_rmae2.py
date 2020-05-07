# Testing multi-view autoencoder
import itertools
import numpy as np
import torch
from torch import nn

from dataprocessing import multiview_datasets
from models import robust_multi_ae, torch_models, multi_ae
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
        view_data, ndims=ndims, fill=False, rtn_proj_mean=True)
    if rtn_proj_mean:
      return proj_base_data, view_sizes, labels, proj, means
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
  kwargs = {'C': 1e2}
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


def test_RMAE(ndims_red=None, drop_scale=True, zero_at_input=True):
  if ndims_red is not None:
    data, v_sizes, labels, projs, means = load_3news(ndims_reds, rtn_proj_mean=True)
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
  max_iters = 1
  rmae_model = setup_RMAE(
      v_sizes, drop_scale, zero_at_input, max_iters=max_iters)
  # IPython.embed()
  imae_model = setup_intersection_mae(v_sizes, max_iters=max_iters)
  cmae_model = setup_cat_ae(v_sizes, max_iters=max_iters)
  # IPython.embed()
  rmae_model.fit(tr_data)
  imae_model.fit(tr_data)
  cmae_tr, cmae_te = {0: tr_cat}, {0: te_cat}
  cmae_model.fit(cmae_tr)

  model = imae_model
  IPython.embed()
  tr_x, _ = model.encode(tr_data, aggregate="mean")
  te_x, _ = model.encode(te_data, aggregate="mean")
  tr_y, te_y = labels[split_inds[0]], labels[split_inds[1]]
  tr_x, te_x = torch_utils.torch_to_numpy(tr_x), torch_utils.torch_to_numpy(te_x)

  tr_cat = multiview_datasets.fill_missing(tr_data, cat_dims=True)
  te_cat = multiview_datasets.fill_missing(te_data, cat_dims=True)

  print("RMAE")
  evaluate_downstream_task(tr_x, tr_y, te_x, te_y)

  print("simple CAT")
  evaluate_downstream_task(tr_cat, tr_y, te_cat, te_y)

  nviews = len(v_sizes)
  for vi in range(nviews):
    tr_vi = {vi:[x for x in tr_data[vi] if x is not None]}
    te_vi = {vi:[x for x in te_data[vi] if x is not None]}

    ntr = len(tr_vi[vi])
    nte = len(te_vi[vi])
    trv_x, _ = model.encode(tr_vi)
    trv_y = [l for i, l in enumerate(tr_y) if tr_data[vi][i] is not None]
    tev_x, _ = model.encode(te_vi)
    tev_y = [l for i, l in enumerate(te_y) if te_data[vi][i] is not None]
    trv_x, tev_x = torch_utils.torch_to_numpy(trv_x), torch_utils.torch_to_numpy(tev_x)

    print("View %i" % vi)
    print("RMAE")
    evaluate_downstream_task(trv_x, trv_y, tev_x, tev_y)

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



if __name__ == "__main__":
  import sys
  drop_scale = True
  zero_at_input = True
  ndims_red = None
  try:
    drop_scale = bool(sys.argv[1])
    zero_at_input = bool(sys.argv[2])
    ndims_red = int(sys.argv[2])
  except:
    pass
  print("Drop scale: %s" % drop_scale)
  print("Zero at input: %s" % zero_at_input)
  print("Reduced dims: %s" % ndims_red)

  test_RMAE(ndims_red, drop_scale, zero_at_input)
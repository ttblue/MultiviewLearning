# Testing multi-view autoencoder
import itertools
import numpy as np
import torch
from torch import nn

from models import robust_multi_ae
from synthetic import multimodal_systems as ms
from utils import torch_utils as tu
from utils import utils

try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


torch.set_default_dtype(torch.float32)


def default_data(npts=1000, nviews=3, ndim=9, peps=0.):
  scale = 1
  centered = True
  overlap = True
  gen_D_alpha = False
  perturb_eps = peps

  data, ptfms = ms.generate_redundant_multiview_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      overlap=overlap, gen_D_alpha=gen_D_alpha, perturb_eps=perturb_eps)

  return data, ptfms


def default_data2(npts=1000, nviews=4, ndim=12, peps=0.):
  scale = 1
  centered = True
  overlap = True
  perturb_eps = peps

  data, ptfms = ms.generate_local_overlap_multiview_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      perturb_eps=perturb_eps)

  return data, ptfms


def default_RMAE_config(v_sizes):
  n_views = len(v_sizes)
  hidden_size = 16
  joint_code_size = 32

  # Default Encoder config:
  output_size = hidden_size
  layer_units = [32] # [32, 64]
  use_vae = False
  activation = nn.ReLU  # nn.functional.relu
  last_activation = nn.Sigmoid  # functional.sigmoid
  encoder_params = {}
  for i in range(n_views):
    input_size = v_sizes[i]
    layer_types, layer_args = tu.generate_linear_types_args(
        input_size, layer_units, output_size)
    encoder_params[i] = tu.MNNConfig(
        input_size=input_size, output_size=output_size, layer_types=layer_types,
        layer_args=layer_args, activation=activation,
        last_activation=last_activation, use_vae=use_vae)

  input_size = joint_code_size
  layer_units = [32]  #[64, 32]
  use_vae = False
  last_activation = tu.Identity
  decoder_params = {}
  for i in range(n_views):
    output_size = v_sizes[i]
    layer_types, layer_args = tu.generate_linear_types_args(
        input_size, layer_units, output_size)
    decoder_params[i] = tu.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, use_vae=use_vae)

  input_size = hidden_size * len(v_sizes)
  output_size = joint_code_size
  layer_units = [64]  #[64, 64]
  layer_types, layer_args = tu.generate_linear_types_args(
      input_size, layer_units, output_size)
  use_vae = False
  joint_coder_params = tu.MNNConfig(
      input_size=input_size, output_size=output_size, layer_types=layer_types,
      layer_args=layer_args, activation=activation,
      last_activation=last_activation, use_vae=use_vae)

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


def split_data(xvs, n=10, split_inds=None):
  xvs = {vi:np.array(xv) for vi, xv in xvs.items()}
  npts = xvs[utils.get_any_key(xvs)].shape[0]
  if split_inds is None:
    split_inds = np.linspace(0, npts, n + 1).astype(int)
  else:
    split_inds = np.array(split_inds)
  start = split_inds[:-1]
  end = split_inds[1:]

  split_xvs = [
      {vi: xv[idx[0]:idx[1]] for vi, xv in xvs.items()}
      for idx in zip(start, end)
  ]
  return split_xvs, split_inds


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


def test_RMAE(
    nviews=4, dim=12, npts=1000, peps=0., drop_scale=True, zero_at_input=True):
  # fname = "./data/mv_dim_%i_data.npy" % nviews
  # if not os.path.exists(fname):
  #   data, ptfms = default_data(nviews=nviews, ndim=dim)
  #   np.save(fname, [data, ptfms])
  # else:
  #   data, ptfms = np.load(fname)
  data, ptfms = default_data(npts=npts, nviews=nviews, ndim=dim, peps=peps)
  v_sizes = [data[vi].shape[1] for vi in range(len(data))]
  config = default_RMAE_config(v_sizes)

  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}
  tr_frac = 0.8
  split_inds = [0, int(tr_frac * npts), npts]
  (tr_data, te_data), _ = split_data(data, split_inds=split_inds)

  config.drop_scale = drop_scale
  config.zero_at_input = zero_at_input
  config.max_iters = 10000

  # IPython.embed()
  model = robust_multi_ae.RobustMultiAutoEncoder(config)
  model.fit(tr_data)
  # vlens = [data[vi].shape[1] for vi in range(len(data))]
  # msplit_inds = np.cumsum(vlens)[:-1]
  IPython.embed()
  # plot_heatmap(model.nullspace_matrix(), msplit_inds


if __name__ == "__main__":
  import sys
  drop_scale = True
  zero_at_input = True
  try:
    drop_scale = bool(sys.argv[1])
    zero_at_input = bool(sys.argv[1])
  except:
    pass
  print("Drop scale: %s"%drop_scale)
  print("Zero at input: %s"%zero_at_input)

  nviews = 3
  dim = 9
  npts = 1000
  peps = 0.
  test_RMAE(nviews, dim, npts, peps)
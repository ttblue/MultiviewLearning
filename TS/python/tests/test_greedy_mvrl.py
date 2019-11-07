# Tests for some CCA stuff
import numpy as np
import os
import torch
from torch import nn

from models import greedy_multi_view_rl, greedy_single_view_rl, torch_models
from synthetic import multimodal_systems as ms


try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


np.set_printoptions(precision=5, suppress=True)


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


def plot_heatmap(mat, msplit_inds, misc_title=""):
  fig = plt.figure()
  hm = plt.imshow(mat)
  plt.title("Redundancy Matrix: %s" % misc_title, fontsize=20)
  cbar = plt.colorbar(hm)
  cbar.ax.tick_params(labelsize=15)
  for mind in msplit_inds:
    mind -= 0.5
    plt.axvline(x=mind, ls="--")
    plt.axhline(y=mind, ls="--")
  plt.axis('off')
  plt.show(block=True)


def default_GMVRL_config(sv_type="nn", as_dict=False):
  verbose = True

  parallel = True
  n_jobs = None

  inner_parallel = False
  inner_n_jobs = None

  regularizer = "L2"
  lambda_reg = 1e-1

  if sv_type in ["nn", "greedy_nn"]:
    input_size = 10  # Computed online
    # Default Encoder config:
    output_size = 10  # Computed online
    layer_units = []  #[32] # [32, 64]
    use_vae = False
    activation = nn.ReLU  # nn.functional.relu
    last_activation = torch_models.Identity  #nn.Sigmoid  # functional.sigmoid
    # layer_types = None
    # layer_args = None
    bias = False
    dropout_p = 0.
    layer_types, layer_args = torch_models.generate_linear_types_args(
          input_size, layer_units, output_size, bias)
    nn_config = torch_models.MNNConfig(
        input_size=input_size, output_size=output_size, layer_types=layer_types,
        layer_args=layer_args, activation=activation,
        last_activation=last_activation, dropout_p=dropout_p, use_vae=use_vae)

    batch_size = 32
    lr = 1e-3
    max_iters = 1000
    single_view_config = greedy_single_view_rl.GNNSConfig(
        nn_config=nn_config, regularizer=regularizer, lambda_reg=lambda_reg,
        batch_size=batch_size, lr=lr, max_iters=max_iters,
        parallel=inner_parallel, n_jobs=inner_n_jobs, verbose=verbose)
    if as_dict:
      single_view_config = single_view_config.__dict__
  else:
    raise ValueError("Single view type %s not implemented." % sv_type)

  single_view_solver_type = "greedy_nn" if sv_type == "nn" else sv_type

  config = greedy_multi_view_rl.GMVRLConfig(
      single_view_solver_type=single_view_solver_type,
      single_view_config=single_view_config, parallel=parallel, n_jobs=n_jobs,
      verbose=verbose)

  return config.__dict__ if as_dict else config


def test_mv_NGSRL_NN(dtype=1, nviews=4, dim=12, npts=1000, peps=0.):
  # fname = "./data/mv_dim_%i_data.npy" % nviews
  # if not os.path.exists(fname):
  #   data, ptfms = default_data(nviews=nviews, ndim=dim)
  #   np.save(fname, [data, ptfms])
  # else:
  #   data, ptfms = np.load(fname)
  peps = 0.0
  default_dfunc = default_data if dtype == 1 else default_data2
  data, ptfms = default_dfunc(npts=npts, nviews=nviews, ndim=dim, peps=peps)

  config = default_GMVRL_config(sv_type="nn")
  data_old = data
  # data_scales = np.array([0.86307, 0.72325, 0.60052, 1.11316])
  data_scales = np.array([0.98, 0.95, 1.03, 1.07])
  #np.random.rand(len(data))
  # data_scales /= data_scales.mean()
  # globals().update(locals())
  data = {i: s*data_scales[i] for i, s in data_old.items()}
  # globals().update(locals())
  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}
  config.single_view_config.lambda_reg = 1e-2
  config.single_view_config.regularizer = "L1"
  config.single_view_config.max_iters = 200

  # IPython.embed()
  config.parallel = False
  config.single_view_config.parallel = True
  # config.lambda_global = 0  #1e-1
  # config.lambda_group = 0 #0.5  #1e-1
  # config.sp_eps = 5e-5
  # config.n_solves = 1

  model = greedy_multi_view_rl.GreedyMVRL(config)
  model.fit(data)
  # globals().update(locals())
  vlens = [data[vi].shape[1] for vi in range(len(data))]
  msplit_inds = np.cumsum(vlens)[:-1]

  # model.compute_projections(ng)
  # projections = model.view_projections
  # for i in projections: 
  #   for j in projections[i]: 
  #     projections[i][j] = projections[i][j].T
  ng = 1
  for ng in range(1, 4):
    plot_heatmap(model.nullspace_matrix(None, ng), msplit_inds, "%i greedy views" % ng)
    plt.show()
  IPython.embed()
  # plot_heatmap(model.nullspace_matrix(), msplit_inds)


if __name__=="__main__":
  dtype = 2
  nviews = 4
  d_view = 3
  dim = nviews * d_view
  npts = 1000
  peps = 0.
  test_mv_NGSRL_NN(dtype, nviews, dim, npts, peps)
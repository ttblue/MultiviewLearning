# Tests for some CCA stuff
import numpy as np
import os

from models import embeddings, ovr_mcca_embeddings, naive_block_sparse_mvrl\
                   naive_single_view_rl
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


def default_config(as_dict=False):
  # Config
  ndim = 3
  info_frac = 0.8
  scale = True

  use_diag_cov = False

  regularizer = "Linf"
  tau_u = 1.
  tau_v = 1.
  tau_all = 1.

  lmbda = 1.
  mu = 1.

  opt_algorithm = "alt_proj"
  init = "auto"

  tol = 1e-9
  sp_tol = 1e-6
  max_inner_iter = 1000
  max_iter = 30

  name = None
  plot = True
  verbose = True

  config = embeddings.CCAConfig(
      ndim=ndim, info_frac=info_frac, scale=scale, use_diag_cov=use_diag_cov,
      regularizer=regularizer, tau_u=tau_u, tau_v=tau_v, tau_all=tau_all,
      lmbda=lmbda, mu=mu, opt_algorithm=opt_algorithm, init=init,
      max_inner_iter=max_inner_iter, max_iter=max_iter, tol=tol, sp_tol=sp_tol,
      plot=plot, verbose=verbose)

  return config.__dict__ if as_dict else config


def default_mcca_config(as_dict=False):
  cca_config_dict = default_config(as_dict=True)
  cca_config_dict["tol"] = 1e-8
  cca_config_dict["sp_tol"] = 1e-5
  cca_config_dict["max_iter"] = 10
  cca_config_dict["plot"] = False

  parallel = False
  n_processes = 4

  save_file = None
  verbose = True

  config = ovr_mcca_embeddings.OVRCCAConfig(
      cca_config_dict=cca_config_dict, parallel=parallel,
      n_processes=n_processes,save_file=save_file, verbose=verbose)

  return config.__dict__ if as_dict else config


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


def test_GSCCA():
  data, ptfms = default_data()
  X = data[0]
  Gx = [np.arange(X.shape[1])]
  Y = np.c_[data[1], data[2]]
  Gy = [
      np.arange(data[1].shape[1]),
      np.arange(data[2].shape[1]) + data[1].shape[1]]

  config = default_config()
  # Can change config values here.
  config.ndim = 3
  config.tau_u = 1e-2 # 0.1
  config.tau_v = 1e-2 # 0.1
  config.lmbda = 1e-2
  config.mu = 1.

  config.regularizer = "Linf"
  config.tol = 1e-6
  config.sp_tol = 1e-6
  config.max_iter = 10
  config.max_inner_iter = 1500

  config.plot = True

  print(config.regularizer)
  # IPython.embed()
  model = embeddings.GroupRegularizedCCA(config)
  model.fit(X, Y, Gx, Gy)

  IPython.embed()


_DATA_FILE = os.path.join(os.getenv("RESEARCH_DIR"), "tests/data/cca_data.npy")

def test_GSCCA_loaded(fl=_DATA_FILE):
  data = np.load(fl).tolist()
  X = data["x"]
  Y = data["y"]
  Gx = data["Gx"]
  Gy = data["Gy"]
  u_init = data["ux"]
  v_init = data["vy"]
  config = data["config"]

  if len(u_init.shape) == 1:
    u_init = u_init.reshape(-1, 1)
  if len(v_init.shape) == 1:
    v_init = v_init.reshape(-1, 1)
  # X, Y, Gx, Gy = default_data()
  # config = default_config()
  # # Can change config values here.
  # config.ndim = 2
  # config.tau_u = 1e-2 # 0.1
  # config.tau_v = 1e-2 # 0.1
  # config.lmbda = 1e-2
  # config.mu = 1.

  config.tol = 1e-6
  config.sp_tol = 1e-6
  config.max_iter = 10
  # # config.max_inner_iter = 200

  # config.plot = True

  # print(config.regularizer)
  # IPython.embed()
  model = embeddings.GroupRegularizedCCA(config)
  model.fit(X, Y, Gx, Gy, u_init, v_init)

  IPython.embed()


def test_mv_GSCCA():
  # data, ptfms = default_data()
  data, ptfms = np.load("tmp.npy")
  config = default_mcca_config()

  config.parallel = True  # True
  config.cca_config_dict["max_iter"] = 10
  config.cca_config_dict["max_inner_iter"] = 500
  config.cca_config_dict["verbose"] = not config.parallel
  config.cca_config_dict["tau_all"] = 0
  config.verbose = False
  mcca_model = ovr_mcca_embeddings.OneVsRestMCCA(config)

  mcca_model.fit(data)

  IPython.embed()


def default_NGSRL_config(as_dict=False, sv_type="opt"):
  if sv_type == "opt":
    group_regularizer = "inf"
    global_regularizer = "L1"
    lambda_group = 1e-1
    lambda_global = 1e-1

    sp_eps = 1e-5

    n_solves = 5
    lambda_group_init = 1e-5
    lambda_group_beta = 10

    resolve_change_thresh = 0.05
    n_resolve_attempts = 3

    solve_joint = True
    parallel = True
    n_jobs = None

    verbose = True

  config = naive_block_sparse_mvrl.NBSMVRLConfig(
    group_regularizer=group_regularizer, global_regularizer=global_regularizer,
    lambda_group=lambda_group, lambda_global=lambda_global, sp_eps=sp_eps,
    n_solves=n_solves, lambda_group_init=lambda_group_init,
    lambda_group_beta=lambda_group_beta,
    resolve_change_thresh=resolve_change_thresh,
    n_resolve_attempts=n_resolve_attempts, solve_joint=solve_joint,
    parallel=parallel, n_jobs=n_jobs, verbose_interval=verbose_interval,
    verbose=verbose)

  return config.__dict__ if as_dict else config


def test_mv_NGSRL(nviews=4, dim=12, npts=1000, peps=0.):
  # fname = "./data/mv_dim_%i_data.npy" % nviews
  # if not os.path.exists(fname):
  #   data, ptfms = default_data(nviews=nviews, ndim=dim)
  #   np.save(fname, [data, ptfms])
  # else:
  #   data, ptfms = np.load(fname)
  data, ptfms = default_data(npts=npts, nviews=nviews, ndim=dim, peps=peps)
  config = default_NGSRL_config()

  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}

  config.lambda_global = 0. #1e-3
  config.lambda_group = 1e-1
  config.sp_eps = 5e-5

  config.solve_joint = False

  config.n_solves = 30
  config.lambda_group_init = 1e-5
  config.lambda_group_beta = 3

  config.resolve_change_thresh = 0.05
  config.n_resolve_attempts = 15

  config.parallel = False
  # config.lambda_global = 0  #1e-1
  # config.lambda_group = 0 #0.5  #1e-1
  # config.sp_eps = 5e-5
  # config.n_solves = 1

  model = naive_block_sparse_mvrl.NaiveBlockSparseMVRL(config)
  model.fit(data)

  vlens = [data[vi].shape[1] for vi in range(len(data))]
  msplit_inds = np.cumsum(vlens)[:-1]
  IPython.embed()
  plot_heatmap(model.nullspace_matrix(), msplit_inds)


def test_mv_NGSRL2(nviews=4, dim=12, npts=1000, peps=0.):
  # fname = "./data/mv_dim_%i_data.npy" % nviews
  # if not os.path.exists(fname):
  #   data, ptfms = default_data(nviews=nviews, ndim=dim)
  #   np.save(fname, [data, ptfms])
  # else:
  #   data, ptfms = np.load(fname)
  data, ptfms = default_data2(npts=npts, nviews=nviews, ndim=dim, peps=peps)
  config = default_NGSRL_config()

  # if npts > 0:
  #   data = {vi: d[:npts] for vi, d in data.items()}

  config.lambda_global = 1e-3
  config.lambda_group = 1e-1
  config.sp_eps = 5e-5

  config.n_solves = 30
  config.lambda_group_init = 1e-5
  config.lambda_group_beta = 3

  config.resolve_change_thresh = 0.05
  config.n_resolve_attempts = 15

  model = naive_block_sparse_mvrl.NaiveBlockSparseMVRL(config)
  model.fit(data)

  vlens = [data[vi].shape[1] for vi in range(len(data))]
  msplit_inds = np.cumsum(vlens)[:-1]
  IPython.embed()
  plot_heatmap(model.nullspace_matrix(), msplit_inds)


if __name__=="__main__":
  # test_GSCCA()
  # test_GSCCA_loaded()
  # test_mv_GSCCA()
  nviews = 4
  dim = 12
  npts = 100
  peps = 0.
  test_mv_NGSRL(nviews, dim, npts, peps)
  # test_mv_NGSRL2(nviews, dim, npts, peps)
# import matplotlib.pyplot as plt
# plt.plot(primal_residual, color='r', label='primal residual')
# plt.plot(dual_residual, color='b', label='dual residual') 
# plt.legend()
# plt.show()
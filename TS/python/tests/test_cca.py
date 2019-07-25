# Tests for some CCA stuff
import numpy as np
import os

from models import embeddings, ovr_mcca_embeddings
from synthetic import multimodal_systems as ms

import IPython


np.set_printoptions(precision=5, suppress=True)


def default_data():
  npts = 1000
  nviews = 3
  ndim = 9
  scale = 1
  centered = True
  overlap = True
  gen_D_alpha = False
  perturb_eps = 0.5

  data, ptfms = ms.generate_redundant_multiview_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      overlap=overlap, gen_D_alpha=gen_D_alpha, perturb_eps=perturb_eps)

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

if __name__=="__main__":
  # test_GSCCA()
  # test_GSCCA_loaded()
  test_mv_GSCCA()

# import matplotlib.pyplot as plt
# plt.plot(primal_residual, color='r', label='primal residual')
# plt.plot(dual_residual, color='b', label='dual residual') 
# plt.legend()
# plt.show()
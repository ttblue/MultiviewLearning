# Tests for some CCA stuff
import numpy as np
import os

from models import embeddings
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

  data = ms.generate_redundant_multiview_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      overlap=overlap, gen_D_alpha=gen_D_alpha)

  X = data[0]
  Gx = [np.arange(X.shape[1])]
  Y = np.c_[data[1], data[2]]
  Gy = [
      np.arange(data[1].shape[1]),
      np.arange(data[2].shape[1]) + data[1].shape[1]]

  return X, Y, Gx, Gy


def default_config():
  # Config
  ndim = 3
  info_frac = 0.8
  scale = True

  use_diag_cov = False

  regularizer = "Linf"
  tau_u = 1.
  tau_v = 1.

  lmbda = 1.
  mu = 1.

  opt_algorithm = "alt_proj"
  init = "auto"

  tol = 1e-9
  max_inner_iter = 1000
  max_iter = 30

  plot = True
  verbose = True

  return embeddings.CCAConfig(
      ndim=ndim, info_frac=info_frac, scale=scale, use_diag_cov=use_diag_cov,
      regularizer=regularizer, tau_u=tau_u, tau_v=tau_v, lmbda=lmbda, mu=mu,
      opt_algorithm=opt_algorithm, init=init, max_inner_iter=max_inner_iter,
      max_iter=max_iter, tol=tol, plot=plot, verbose=verbose)


def test_GSCCA():
  X, Y, Gx, Gy = default_data()
  config = default_config()
  # Can change config values here.
  config.ndim = 2
  config.tau_u = 1e-2 # 0.1
  config.tau_v = 1e-2 # 0.1
  config.lmbda = 1e-2
  config.mu = 1.

  config.regularizer = "Linf"
  config.max_iter = 100
  # config.max_inner_iter = 200

  config.plot = True

  print(config.regularizer)
  # IPython.embed()
  model = embeddings.GroupRegularizedCCA(config)
  model.fit(X, Y, Gx, Gy)

  IPython.embed()


_DATA_FILE = os.path.join(os.getenv("RESEARCH_DIR"), "tests/cca_data.npy")

def test_GSCCA_loaded(fl=_DATA_FILE):
  data = np.load(fl).tolist()
  X = data["x"]
  Y = data["y"]
  Gx = data["Gx"]
  Gy = data["Gy"]
  u = data["u"]
  v = data["v"]
  config = data["config"]

  if len(u.shape) == 1:
    u = u.reshape(-1, 1)
  if len(v.shape) == 1:
    v = v.reshape(-1, 1)
  # X, Y, Gx, Gy = default_data()
  # config = default_config()
  # # Can change config values here.
  # config.ndim = 2
  # config.tau_u = 1e-2 # 0.1
  # config.tau_v = 1e-2 # 0.1
  # config.lmbda = 1e-2
  # config.mu = 1.

  # config.regularizer = "Linf"
  # config.max_iter = 100
  # # config.max_inner_iter = 200

  # config.plot = True

  # print(config.regularizer)
  # IPython.embed()
  model = embeddings.GroupRegularizedCCA(config)
  model.fit(X, Y, Gx, Gy, u, v)

  IPython.embed()


if __name__=="__main__":
  # test_GSCCA()
  test_GSCCA_loaded()

# import matplotlib.pyplot as plt
# plt.plot(primal_residual, color='r', label='primal residual')
# plt.plot(dual_residual, color='b', label='dual residual') 
# plt.legend()
# plt.show()
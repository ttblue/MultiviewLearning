# Tests for some CCA stuff
import numpy as np

from models import embeddings
from synthetic import multimodal_systems as ms

import IPython


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

  regularizer = "L1_inf"
  lmbda = 1.0
  gamma = 1.0

  opt_algorithm = "alt_proj"
  init = "auto"

  tol = 1e-6
  max_iter = 10

  verbose = True

  return embeddings.CCAConfig(
      ndim=ndim, info_frac=info_frac, scale=scale, use_diag_cov=use_diag_cov,
      regularizer=regularizer, lmbda=lmbda, gamma=gamma,
      opt_algorithm=opt_algorithm, init=init, max_iter=max_iter, tol=tol,
      verbose=verbose)


def test_GSCCA():
  X, Y, Gx, Gy = default_data()
  config = default_config()
  # Can change config values here.
  config.ndim = 1

  model = embeddings.GroupSparseCCA(config)
  model.fit(X, Y, Gx, Gy)

  IPython.embed()


if __name__=="__main__":
  test_GSCCA()
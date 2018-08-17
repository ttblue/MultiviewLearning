# Some general manifold learning approaches.

import numpy as np
import scipy.spatial.distance as ssd
import sklearn.utils.extmath as sue

import tfm_utils as tu


def rbf_gmatrix(xs, ys=None, gamma=1.0):
  dis = ssd.squareform(ssd.pdist(xs)) if ys is None else ssd.cdist(xs, ys)
  return np.exp(-gamma * (dis ** 2))


def laplacian_eigenmaps(
    xs, e_dim=10, kernel="gaussian", kernel_params={"gamma": 1.0},
    standardize=False, zero_thresh=1e-10):
  # zero_thresh -- threshold to consider eigen value as zero
  # standardize -- flag for returning standardized eigen vectors
  xs = np.atleast_2d(xs)
  if kernel == "gaussian":
    W = rbf_gmatrix(xs, gamma=kernel_params["gamma"])
  else:
    raise NotImplementedError("%s kernel not available." % kernel)

  S = np.diag(W.sum(0))
  L = S - W
  Evals, Evecs = np.linalg.eigh(L)

  valid_evals = Evals > zero_thresh
  # extract the valid eigen vectors, then take the first e_dim
  E = (Evecs[:, valid_evals])[:, :e_dim]
  if standardize:
    E = E / np.sqrt(((Evals[valid_evals])[:e_dim]))

  return E


def instrumental_eigenmaps(
    xs, ys, e_dims=10, gmatrix_type="rbf", gmatrix_params={"gamma": 1.0}):
    # TODO: maybe have different gmatrix params for xs and ys?
  if gmatrix_type == "rbf":
    Gx = rbf_gmatrix(xs, gamma=gmatrix_params["gamma"])
    Gy = rbf_gmatrix(ys, gamma=gmatrix_params["gamma"])
  else:
    raise NotImplementedError("%s gram matrix not available." % gmatrix_type)

  Cx = tu.center_gram_matrix(Gx)
  Cy = tu.center_gram_matrix(Gy)

  U, S, VT = sue.randomized_svd(
      Cx.dot(Cy), n_components=e_dims, random_state=None)
  sqrt_S = np.diag(np.sqrt(S))
  Ex = U.dot(sqrt_S)
  Ey = VT.T.dot(sqrt_S)

  return Ex, Ey
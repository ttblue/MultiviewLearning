# For now, simple version of robust autoencoder.

import numpy as np
import scipy as sp
import sklearn.cross_decomposition as scd

import math_utils as mu


class ModelException(Exception):
  pass


def PCA_embedding(Xs, ndim=None, info_frac=0.8):
  # Xs -- n x r data matrix
  _, S, VT = np.linalg.svd(Xs, full_matrices=False)

  if ndim is None:
    tot_sv = S.sum()
    ndim = mu.get_index_p_in_ordered_pdf(np.cumsum(S) / tot_sv, info_frac)

  S_proj = S[:ndim]
  proj = VT[:ndim].T
  Xs_p = Xs.dot(proj)

  return Xs_p, (S_proj, proj)


def CCA(Xs, Ys, ndim=None, info_frac=0.8, scale=True):

  Xs = np.array(Xs)
  Ys = np.array(Ys)

  if Xs.shape[0] != Ys.shape[0]:
    raise ModelException("Xs and Ys don't have the same number of data points.")

  # if ndim is not None:
  #   cca_model = scd.CCA(ndim, scale=scale)
  #   cca_model.fit(Xs, Ys)
  #   X_proj, Y_proj = cca_model.transform(Xs, Ys)
  #   Xw, Yw = cca_model.x_weights_, cca_model.y_weights_
  # else:
  Xs_data = mu.shift_and_scale(Xs, scale=scale)
  Ys_data = mu.shift_and_scale(Ys, scale=scale)

  X_centered = Xs_data[0]
  Y_centered = Ys_data
  n = X_centered.shape[0]
  X_cov = X_centered.T.dot(X_centered) / n
  Y_cov = Y_centered.T.dot(Y_centered) / n

  X_cov_sqrt_inv = mu.sym_matrix_power(X_cov, -0.5)
  Y_cov_sqrt_inv = mu.sym_matrix_power(Y_cov, -0.5)

  cov = X_centered.T.dot(Y_centered) / n
  normalized_cov = X_cov_sqrt_inv.dot(cov.dot(Y_cov_sqrt_inv))
  U, S, VT = np.linalg.svd(normalized_cov)

  if ndim is None:
    ndim = mu.get_index_p_in_ordered_pdf(S, info_frac)
  Wx = X_cov_sqrt_inv.dot(U[:, :ndim])
  Wy = Y_cov_sqrt_inv.dot(VT[:ndim].T)
  X_proj = X_centered.dot(Wx)
  Y_proj = Y_centered.dot(Wy)

  return (X_proj, Wx, Xs_data), (Y_proj, Wy, Ys_data)


def correlation_info(v_Xs):
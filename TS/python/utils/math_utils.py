import os

import numpy as np

# ==============================================================================
# Utility functions
# ==============================================================================

def rbf_fourierfeatures(d_in, d_out, gammak, sine=True):
  # Returns a function handle to compute random fourier features.
  if not sine:
    W = np.random.normal(0., np.sqrt(2 * gammak), (d_in, d_out))
    h = np.random.uniform(0., 2 * np.pi, (1, d_out))
    def rbf_ff(x):
      ff = np.cos(x.dot(W) + h) / np.sqrt(d_out / 2.)
      return ff
  else:
    W = np.random.normal(0., np.sqrt(2 * gammak), (d_in, d_out // 2))
    def rbf_ff(x):
      ff = np.c_[np.cos(x.dot(W)), np.sin(x.dot(W))] / np.sqrt(d_out / 2.)
      return ff

  return rbf_ff


def mm_rbf_fourierfeatures(d_in, d_out, a, file_name=None):
  # Returns a function handle to compute random fourier features.
  # Can load from/save to file if one exists.
  if file_name is not None:
    # Some simple checks for filename.
    basename = os.path.basename(file_name)
    if len(basename.split('.')) == 1:
      file_name = file_name + ".npy"

    if os.path.exists(file_name):
      W, h, a = np.load(file_name)
    else:
      W = np.random.normal(0., 1., (d_in, d_out))
      h = np.random.uniform(0., 2*np.pi, (1, d_out))
      np.save(file_name, [W, h, a])
  else:
    W = np.random.normal(0., 1., (d_in, d_out))
    h = np.random.uniform(0., 2*np.pi, (1, d_out))

  def mm_rbf(x):
    xhat = np.mean(np.cos((1/a)*x.dot(W)+h)/np.sqrt(d_out)*np.sqrt(2), axis=0)
    return xhat

  return mm_rbf


def get_index_p_in_ordered_pdf(pdf, p):
  cdf = np.cumsum(pdf) / pdf.sum()
  return np.searchsorted(cdf, p, side="right")


def convert_zeros_to_num(mat, num=1.0, eps=1e-6):
  return np.where(np.abs(mat) > eps, mat, num)


def shift_and_scale(Xs, scale=True):
  if not isinstance(Xs, list):
    Xs = [Xs]
  Xs = [np.array(X) for X in Xs]

  Xs_means = [X.mean(axis=0) for X in Xs]
  Xs_output = [(X - Xm) for X, Xm in zip(Xs, Xs_means)]
  if scale:
    Xs_std = [convert_zeros_to_num(X.std(axis=0, ddof=1)) for X in Xs_output]
    Xs_output = [(X / X_std) for X, X_std in zip(Xs_output, Xs_std)]
  else:
    Xs_std = [np.ones(X.shape[1]) for X in Xs_output]

  if len(Xs) == 1:
    Xs_output, Xs_means, Xs_std = Xs_output[0], Xs_means[0], Xs_std[0]

  return Xs_output, Xs_means, Xs_std


def sym_matrix_power(mat, exps=-1, eps=1e-6, pinv=True):
  if not np.allclose(mat, mat.T):
    raise ValueError("Matrix is not symmetric.")

  if not isinstance(exps, list):
    exps = [exps]
  exps = np.array(exps)

  evals, evecs = np.linalg.eigh(mat)
  evals = np.where(np.abs(evals) < eps, 0, evals)
  if np.any(evals < 0.) and np.any((exp - (exp).astype(int)) > eps):
    raise ValueError("Matrix has negative eigenvalues for root power.")
  elif not pinv and np.any(np.abs(evals) < eps) and np.any(exps < 0):
    raise ValueError("Matrix has zero eigenvalues for inversion.")

  mat_exps = []
  valid_inds = None
  for exp in exps:
    if pinv and exp < 0 and valid_inds is None:
      valid_inds = np.abs(evals) > eps
      evals_exp = evals
      evals_exp[valid_inds] = np.power(evals_exp[valid_inds], exp)
    else:
      evals_exp = np.power(evals, exp)
    mat_exps.append(evecs.T.dot(np.diag(evals_exp).dot(evecs)))

  return mat_exps[0] if len(mat_exps) == 1 else mat_exps


def random_unitary_matrix(dim=3):
  H = np.eye(dim)
  D = np.ones((dim,))
  for n in range(1, dim):
    x = np.random.normal(size=(dim-n+1,))
    D[n-1] = np.sign(x[0])
    x[0] -= D[n-1] * np.sqrt((x * x).sum())
    # Householder transformation
    Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
    mat = np.eye(dim)
    mat[n-1:, n-1:] = Hx
    H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
  D[-1] = -1 ** (1 - (dim % 2)) * D.prod()
  # Equivalent to np.dot(np.diag(D), H) but faster, apparently
  H = (D * H.T).T
  return H
# Toy data for flow related testing:
import numpy as np
import scipy.linalg as slg

from utils import math_utils


################################################################################
# Transform related toy data:

def coupled_scale_shift_tfm(X, bit_mask, W=None, gamma=0.1):
  bit_mask = bit_mask.astype("bool")
  ndim = bit_mask.shape[0]
  nfixed = bit_mask.sum()

  # If same size of fixed and not fixed, no need for dim correction
  if W is None and nfixed * 2 == ndim:
    W = np.eye(nfixed)
  W = np.random.randn(nfixed, ndim - nfixed) if W is None else W

  X_fixed = X[:, bit_mask]
  Y = np.empty_like(X)
  Y[:, bit_mask] = X_fixed
  X_scaling = np.exp(-gamma * (X[:, bit_mask].dot(W.T)))
  Y[:, ~bit_mask] = X[:, ~bit_mask] * X_scaling

  return Y, W


_DEFAULT_TFM_ORDER = ["scaleshift", "linear", "leakyrelu", "reverse"]
def simple_transform_data(npts=1000, ndim=10, tfm_types=[]):
  X = np.random.randn(npts, ndim)
  tfm_types = [ttype.lower() for ttype in tfm_types] or _DEFAULT_TFM_ORDER

  tfm_args = []
  Y = X
  gamma = 0.1
  # Coupled scaling.
  for ttype in tfm_types:
    if ttype == "scaleshift":
      bit_mask = np.zeros(ndim)
      # Set random half of bits to 0
      bit_mask[np.random.permutation(ndim)[:ndim//2]] = 1
      Y, W = coupled_scale_shift_tfm(Y, bit_mask, gamma=gamma)
      tfm_args.append((ttype, bit_mask, W, gamma))
    elif ttype == "linear":
      W = math_utils.random_unitary_matrix(ndim)
      D_scale = np.random.rand(ndim)
      D = np.diag(D_scale / D_scale.sum())
      P, L, U = slg.lu(W.dot(D))
      W = L.dot(U)
      Y = Y.dot(W.T)
      tfm_args.append((ttype, L, U))
    elif ttype == "leakyrelu":
      Y = np.where(Y > 0, Y, gamma * Y)
      tfm_args.append((ttype, gamma))
    elif ttype == "reverse":
      Y = Y[::-1]
      tfm_args.append((ttype,))
  return X, Y, tfm_args
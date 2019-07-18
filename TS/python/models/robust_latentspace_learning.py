# For now, simple version of robust autoencoder.

import cvxpy as cvx
import numpy as np
import scipy as sp
import sklearn.cross_decomposition as scd

from models import embeddings#, ovr_mcca_embeddings as ome
from utils import math_utils as mu

import IPython


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
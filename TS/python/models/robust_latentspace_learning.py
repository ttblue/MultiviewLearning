# For now, simple version of robust autoencoder.

import cvxpy as cvx
import numpy as np
import scipy as sp
import sklearn.cross_decomposition as scd

from utils import math_utils as mu
from synthetic import multimodal_systems as ms

import IPython


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



if __name__ == "__main__":
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

  cca_info = correlation_info(data)
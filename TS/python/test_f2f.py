# Testing function to function regression.
import itertools
import numpy as np

try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False

import func_to_func as ff
import func_utils as fu
import gaussian_random_features as grf
import synthetic.simple_systems as ss
import tfm_utils as tutils
import time_sync as tsync
import transforms as trns

import IPython


def sample_func(x):
  return 2*x + 3


def gen_double_func(nd, nt, tau=30, ntau=3, ninds=3, f=sample_func):
  basis = fu.SinusoidalBasis(cosine_basis=True)
  U = list(itertools.product(list(range(ninds)), repeat=ntau))
  pts = tsync.compute_td_embedding(np.linspace(0, 10, nt), tau, ntau)

  feval = basis.eval_basis(pts, U, eval_all=True)

  dA, dB = [], []
  for _ in range(nd):
    random_coeffs = np.random.randn(len(U))
    Aevals = feval.dot(random_coeffs)
    Bevals = f(Aevals)

    dA.append((pts, Aevals))
    dB.append((pts, Bevals))

  return dA, dB


def split_data(xs, ys, n=10, split_inds=None):
  xs = np.array(xs)
  ys = np.array(ys)

  if split_inds is None:
    split_inds = np.linspace(0, xs.shape[0], n + 1).astype(int)
  start = split_inds[:-1]
  end = split_inds[1:] + 1

  dset = [(xs[idx[0]:idx[1]], ys[idx[0]:idx[1]]) for idx in zip(start, end)]
  return dset, split_inds


def test_lorenz():

  visualize = False and MPL_AVAILABLE
  tmax = 100
  nt = 10000
  x, y, z = ss.generate_lorenz_system(tmax, nt)

  tau = 10
  ntau = 3
  xts = tsync.compute_td_embedding(x, tau, ntau)
  yts = tsync.compute_td_embedding(y, tau, ntau)

  num_U = 3
  num_V = 3

  U_inds = list(range(num_U))
  V_inds = list(range(num_V))

  U = list(itertools.product(U_inds, repeat=ntau))
  V = list(itertools.product(V_inds, repeat=ntau))

  k = None
  cosine_basis = False
  basis = fu.SinusoidalBasis(k, cosine_basis)
  rn = 1000
  gammak = 1e2
  fgen_args = {"rn":rn, "gammak":1.0, "sine":False}
  tb = ff.TripleBasisEstimator(U, V, basis, fgen_args)


  n = 30
  P, split_inds = split_data(xts[:-1], xts[1:, 0], n=n)
  Q, _ = split_data(yts[:-1], yts[1:, 0], split_inds=split_inds)

  ntr = 25
  Ptr, Qtr = P[:ntr], Q[:ntr]
  Pte, Qte = P[ntr:], Q[ntr:]
  tb.fit(Ptr, Ptr)

  Qx = [q[0] for q in Pte]
  IPython.embed()

  plt.plot()


def test_simple():

  visualize = False and MPL_AVAILABLE
  nd = 1000
  nt = 5000
  tau = 30
  ntau = 3
  ninds = 3
  f = sample_func

  P, Q = gen_double_func(nd, nt, tau, ntau, ninds, f)
  Q = P

  num_U = 5
  num_V = 5

  U_inds = list(range(num_U))
  V_inds = list(range(num_V))

  U = list(itertools.product(U_inds, repeat=ntau))
  V = list(itertools.product(V_inds, repeat=ntau))

  k = None
  cosine_basis = False
  basis = fu.SinusoidalBasis(k, cosine_basis)
  rn = 1000
  gammak = 1.0
  fgen_args = {"rn":rn, "gammak":1.0, "sine":False}
  reg = "ridge"
  verbose = True
  tb = ff.TripleBasisEstimator(U, V, basis, fgen_args, reg, verbose=verbose)

  ntr = int(0.8 * nd)
  Ptr, Qtr = P[:ntr], Q[:ntr]
  Pte, Qte = P[ntr:], Q[ntr:]
  tb.fit(Ptr, Qtr)

  Qx = [q[0] for q in Qte]
  pred = tb.predict(Pte, Qx)

  IPython.embed()
  plt.plot(pred[0], color='r'); plt.plot(Qte[0][1], color='b'); plt.show()


if __name__=="__main__":
  test_simple()

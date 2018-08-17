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


def gen_double_func(
    nd, nt, tau=30, ntau=3, ninds=3, cosine_basis=False, f=sample_func):
  # temp
  global feval, pts, U
  basis = fu.SinusoidalBasis(cosine_basis=cosine_basis)
  U = list(itertools.product(list(range(ninds)), repeat=ntau))
  if ntau == 1:
    pts = np.linspace(0, 1, nt).reshape(-1, 1)
  else:
    pts = tsync.compute_td_embedding(np.linspace(0, 1, nt), tau, ntau)

  feval = basis.eval_basis(pts, U, eval_all=True)

  dA, dB, coeffs = [], [], []
  for _ in range(nd):
    random_coeffs = np.random.randn(len(U))
    Aevals = feval.dot(random_coeffs)
    Bevals = f(Aevals)

    dA.append((pts, Aevals))
    dB.append((pts, Bevals))
    coeffs.append(random_coeffs)

  return dA, dB, coeffs, U


def split_data(xs, ys, n=10, split_inds=None):
  xs = None if xs is None else np.array(xs)
  ys = np.array(ys)

  if split_inds is None:
    split_inds = np.linspace(0, ys.shape[0], n + 1).astype(int)
  start = split_inds[:-1]
  end = split_inds[1:] + 1

  if xs is not None:
    dset = [(xs[idx[0]:idx[1]], ys[idx[0]:idx[1]]) for idx in zip(start, end)]
  else:
    split_ys = [ys[idx[0]:idx[1]]for idx in zip(start, end)]
    dset = [
        (np.linspace(0, 1, ys.shape[0]).reshape(-1, 1), ys) for ys in split_ys]
  return dset, split_inds


def test_lorenz(no_ds=True):

  visualize = False and MPL_AVAILABLE
  tmax = 10000
  nt = 1000000
  x, y, z = ss.generate_lorenz_system(tmax, nt, sig=0.1)
  if no_ds:
    ntau = 1
    xts, yts = x, z

    xts = xts - np.min(xts)
    yts = yts - np.min(yts)
    xts = xts / np.max(np.abs(xts))
    yts = yts / np.max(np.abs(yts))

    nb = 120
    num_U = nb
    num_V = nb
    U_inds = list(range(num_U))
    V_inds = list(range(num_V))
  else:
    x = x - np.min(x)
    y = y - np.min(y)
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(x))

    tau = 20
    ntau = 3
    xts = tsync.compute_td_embedding(x, tau, ntau)
    yts = tsync.compute_td_embedding(y, tau, ntau)

    num_U = 8
    num_V = 8
    U_inds = list(range(num_U))
    V_inds = list(range(num_V))

  # f([x(t), x(t-tau), x(t-2tau), ...]) -> x(t+1)
  # g([y(t), y(t-tau), y(t-2tau), ...]) -> y(t+1)

  U = list(itertools.product(U_inds, repeat=ntau))
  V = list(itertools.product(V_inds, repeat=ntau))

  k = None
  cosine_basis = True
  basis = fu.SinusoidalBasis(cosine_basis=cosine_basis)
  rn = 2000
  gammak = 5e-2
  fgen_args = {"rn":rn, "gammak":gammak, "sine":False}
  reg = "ridge"
  verbose = True
  tb = ff.TripleBasisEstimator(U, V, basis, fgen_args, reg, verbose=verbose)
  tb.lm = 3e-13

  n = 500
  if no_ds:
    P, split_inds = split_data(None, xts, n=n)
    Q, split_inds = split_data(None, yts, n=n)
  else:
    P, split_inds = split_data(xts[:-1], xts[1:, 0], n=n)
    Q, _ = split_data(yts[:-1], yts[1:, 0], split_inds=split_inds)

  trp = 0.8
  ntr = int(n * trp)
  Ptr, Qtr = P[:ntr], Q[:ntr]
  Pte, Qte = P[ntr:], Q[ntr:]
  tb.fit(Ptr, Qtr)

  Qx = [q[0] for q in Qte]
  pred = tb.predict(Pte, Qx)
  rec_p = [basis.func_approximation(q[0], q[1], U) for q in Pte]
  rec_q = [basis.func_approximation(q[0], q[1], V) for q in Qte]

  IPython.embed()

  pp = 500
  for idd in range(3):
    plt.plot(pred[idd][:pp], color='r', label='Pred Y')
    plt.plot(Qte[idd][1][:pp], color='b', label='True Y')
    plt.plot(rec_q[idd][:pp], color='g', label='Approx Y')
    plt.plot(Pte[idd][1][:pp], color='k', label='True X')
    plt.plot(rec_p[idd][:pp], color='y', label='Approx X')
    plt.legend()
    plt.show()


def test_simple():

  visualize = False and MPL_AVAILABLE
  nd = 1000
  nt = 5000
  tau = 30
  ntau = 1
  ninds = 5
  f = sample_func
  cosine_basis_input = False

  P, Q, coeffs, true_inds = gen_double_func(
      nd, nt, tau, ntau, ninds, cosine_basis_input, f)

  num_U = 6
  num_V = 6

  U_inds = list(range(num_U))
  V_inds = list(range(num_V))

  U = list(itertools.product(U_inds, repeat=ntau))
  V = list(itertools.product(V_inds, repeat=ntau))

  k = None
  cosine_basis = True
  basis = fu.SinusoidalBasis(cosine_basis=cosine_basis)
  rn = 250
  gammak = 1e-1
  fgen_args = {"rn":rn, "gammak":gammak, "sine":False}
  reg = "ridge"
  verbose = True
  tb = ff.TripleBasisEstimator(U, V, basis, fgen_args, reg, verbose=verbose)
  tb.lm = 0.

  ntr = int(0.8 * nd)
  Ptr, Qtr = P[:ntr], Q[:ntr]
  Pte, Qte = P[ntr:], Q[ntr:]
  tb.fit(Ptr, Qtr)

  Qx = [q[0] for q in Qte]
  pred = tb.predict(Pte, Qx)
  rec_q = [basis.func_approximation(q[0], q[1], V) for q in Qte]
  rec_p = [basis.func_approximation(q[0], q[1], U) for q in Pte]

  IPython.embed()
  for idd in range(10):
    plt.plot(pred[idd], color='r', label='P')
    plt.plot(Qte[idd][1], color='b', label='T')
    plt.plot(rec_q[idd], color='g', label='Aq')
    plt.plot(rec_p[idd], color='y', label='Ap')
    plt.legend()
    plt.show()

  idd = 0
  x0, y0 = P[idd]
  pinds = np.array(list(range(ninds))).reshape(-1, 1)
  px = basis.eval_basis(x0, pinds)
  pc = basis.project(x0, y0, pinds)
  ppr = px.dot(pc)
  plt.plot(y0, color='b', label="T"); plt.plot(ppr, color='r', label='P'); plt.legend(); plt.show()

if __name__=="__main__":
  test_lorenz()
  # test_simple()


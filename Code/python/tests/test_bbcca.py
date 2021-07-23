# Tests for multi bb cca

import numpy as np

try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False

import func_utils as fu
import gaussian_random_features as grf
import multi_bb_cca as mbc
import synthetic.simple_systems as ss
import tfm_utils as tutils
import time_sync as tsync

import IPython


def split_data(xs, n=10, split_inds=None):
  xs = np.array(xs)

  if split_inds is None:
    split_inds = np.linspace(0, xs.shape[0], n + 1).astype(int)
  else:
    split_inds = np.array(split_inds)
  start = split_inds[:-1]
  end = split_inds[1:]

  split_xs = [xs[idx[0]:idx[1]]for idx in zip(start, end)]
  return split_xs, split_inds


def test_lorenz(USE_EM=True):
  visualize = False and MPL_AVAILABLE

  no_ds = False
  tmax = 10
  nt = 1000
  x, y, z = ss.generate_lorenz_system(tmax, nt, sig=0.1)

  if no_ds:
    ntau = 1
    xts, yts, zts = x, y, z

    xts = xts - np.min(xts)
    yts = yts - np.min(yts)
    zts = zts - np.min(zts)
    xts = xts / np.max(np.abs(xts))
    yts = yts / np.max(np.abs(yts))
    zts = zts / np.max(np.abs(zts))
  else:
    x = x - np.min(x)
    y = y - np.min(y)
    z = z - np.min(z)
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    z = z / np.max(np.abs(z))

    tau = 20
    ntau = 3
    xts = tsync.compute_td_embedding(x, tau, ntau)
    yts = tsync.compute_td_embedding(y, tau, ntau)
    zts = tsync.compute_td_embedding(z, tau, ntau)

  nt = xts.shape[0]

  tr_frac = 0.8
  split_inds = [0, int(tr_frac * nt), nt]
  dsets = [split_data(ds, split_inds=split_inds)[0] for ds in [xts, yts, zts]]
  Vs_tr = [ds[0] for ds in dsets]
  Vs_te = [ds[1] for ds in dsets]
  rn = 100
  gammak = 5e-2
  sine = False
  n_embedding = 2
  fgen_args = {"rn":rn, "gammak":gammak, "sine":sine}
  fgen = grf.GaussianRandomFeatures(dim=ntau, **fgen_args)
  # IPython.embed()
  if not USE_EM:
    Es, Us = mbc.multiway_bb_si(
        Vs_tr, n_embedding=n_embedding, feature_func=fgen.computeRandomFeatures,
        solver="cobyla", use_iemaps=False)
  else:
    em_iter = 1000
    Es, Us, Z = mbc.multiway_bb_si_EM(
      Vs_tr, n_embedding=n_embedding, feature_func=fgen.computeRandomFeatures,
      em_iter=em_iter, use_iemaps=False)

  IPython.embed()



  feature_func = fgen.computeRandomFeatures
  phis = []
  for V in Vs_te:
    phis.append(tu.center_matrix(feature_func(V)))
  zte = [(phi.dot(u)).squeeze() for phi, u in zip(phis, [u[:, 1] for u in Us])]

  r = 1
  plt.plot(Es[0][:, r], color='g', label='x')
  plt.plot(Es[1][:, r], color='r', label='y')
  plt.plot(Es[2][:, r], color='b', label='z')
  plt.legend()
  plt.show()
  plt.plot(zte[0], color='g', label='x')
  plt.plot(zte[1], color='r', label='y')
  plt.plot(zte[2], color='b', label='z')
  plt.legend()
  plt.show()

if __name__ == "__main__":
  use_em = True
  test_lorenz(use_em)
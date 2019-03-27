# Tests for the SSLF
import matplotlib.pyplot as plt
import numpy as np

from models import structured_sparse_factorizer as sslf

import IPython


def generate_redundant_data(npts, nviews=3, ndim=15, scale=2):
  data = np.random.uniform(high=scale, size=(npts, ndim))

  n_per_view = ndim // nviews
  n_remainder = ndim - n_per_view * nviews
  view_groups = [
      (i * n_per_view + np.arange(n_per_view)).astype(int)
      for i in range(nviews)]
  remaining_data = (
      data[:, -n_remainder:] if n_remainder > 0 else np.empty((npts, 0)))

  view_data = {}
  for vi, vg in enumerate(view_groups):
    view_data[vi] = np.c_[data[:, vg], remaining_data]

  return view_data


class BasicDataset(object):
  def __init__(self, txs):
    self.v_txs = txs
    self.n_ts = txs[next(iter(txs.keys()))].shape[0]
    self.v_dims = {i: txs[i].shape[1] for i in txs}
    self.n_views = len(txs)
    self.synced = True

  def split(self, fracs, shuffle=True):
    fracs = np.array(fracs) / np.sum(fracs)

    num_split = (fracs * self.n_ts).astype(int)
    num_split[-1] = self.n_ts - num_split[:-1].sum()
    all_inds = (
        np.random.permutation(self.n_ts) if shuffle else np.arange(self.n_ts)
    )
    end_inds = np.cumsum(num_split).tolist()
    start_inds = [0] + end_inds[:-1]

    dsets = []
    for si, ei in zip(start_inds, end_inds):
      split_inds = all_inds[si:ei]
      split_txs = {vi:self.v_txs[vi][split_inds] for vi in self.v_txs}
      dsets.append(BasicDataset(split_txs))
    return dsets


def test_sslf_toy():
  npts = 1000 
  nviews = 3
  ndim = 9
  scale = 2

  data = generate_redundant_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale)
  all_dset = BasicDataset(data)

  tr_dset, te_dset = all_dset.split([0.8, 0.2])

  regularizer = "L1_inf"
  reduce_dict_every_iter = True
  D_init_var_frac = 0.95
  removal_thresh = 0.02
  lmbda = 1.0
  gamma = 1.0
  max_iters = 100
  verbose = True
  config = sslf.SSLFConfig(
      regularizer=regularizer, reduce_dict_every_iter=reduce_dict_every_iter,
      D_init_var_frac=D_init_var_frac, removal_thresh=removal_thresh,
      lmbda=lmbda, gamma=gamma, max_iters=max_iters, verbose=verbose)

  model = sslf.StructuredSparseLatentFactorizer(config)

  IPython.embed()
  model.fit(tr_dset)

  IPython.embed()

if __name__=="__main__":
  test_sslf_toy()
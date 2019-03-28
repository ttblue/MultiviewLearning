# Tests for the SSLF
import matplotlib.pyplot as plt
import numpy as np

from models import structured_sparse_latent_factorizer as sslf

import IPython


def flatten(list_of_lists):
  return [a for b in list_of_lists for a in b]


def padded_identity(n, m, idx):
  # Helper function -- 
  # Create an n x m block column matrix:
  # [0_{idx x m};
  #  I_{m x m};
  #  0_{(n-idx-m) x m}]
  return np.r_[np.zeros((idx, m)), np.identity(m), np.zeros(((n - idx - m), m))]

def generate_redundant_data(
      npts, nviews=3, ndim=15, scale=2, centered=True, overlap=True,
      gen_D_alpha=True):
  data = np.random.uniform(high=scale, size=(npts, ndim))

  if centered:
    data -= data.mean(axis=0)

  n_per_view = ndim // nviews
  n_remainder = ndim - n_per_view * nviews
  view_groups = [
      (i * n_per_view + np.arange(n_per_view)).astype(int)
      for i in range(nviews)]
  remaining_data = (
      data[:, -n_remainder:] if n_remainder > 0 else np.empty((npts, 0)))

  view_data = {}
  for vi, vg in enumerate(view_groups):
    view_inds = (
        # Exclude one view-group and give the rest
        flatten([vg for i, vg in enumerate(view_groups) if i != vi])
        # Or only use that one group.
        if overlap else view_groups[vi])
    view_data[vi] = np.c_[data[:, view_inds], remaining_data]

  # Trivial solution to check
  if gen_D_alpha:
    dim_per_view = n_per_view * (nviews - 1 if overlap else 1) + n_remainder
    alpha = data.T

    I_rem = padded_identity(
        dim_per_view, n_remainder, dim_per_view - n_remainder)
    D = {}
    for vi in range(nviews):
      if overlap:
        cols = []
        cidx = 0
        for i in range(nviews):
          v_col = (
              np.zeros((dim_per_view, n_per_view)) if i == vi else
              padded_identity(dim_per_view, n_per_view, idx * n_per_view))
          cidx += 1
          col.append(v_col)
      else:
        cols = [
            (padded_identity(dim_per_view, n_per_view, 0) if i == vi else
             np.zeros((dim_per_view, n_per_view)))
            for i in range(nviews)]
      cols.append(I_rem)
      D[vi] = np.concatenate(cols, axis=1)
    return view_data, D, alpha

  return view_data


class BasicDataset(object):
  def __init__(self, txs):
    self.v_txs = txs
    self.n_ts = txs[next(iter(txs.keys()))].shape[0]
    self.v_dims = {i: txs[i].shape[1] for i in txs}
    self.n_views = len(txs)
    self.synced = True

  def split(self, fracs, shuffle=True, get_inds=True):
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

    if get_inds:
      inds = [all_inds[si:ei] for si, ei in zip(start_inds, end_inds)]
      return dsets, inds
    return dsets


def simple_plot(lst_of_plts):
  for vals, color, lbl, ls in lst_of_plts:
    plt.plot(vals, color=color, label=lbl, linestyle=ls)
    plt.legend()
  plt.show()


def test_sslf_toy():
  npts = 1000 
  nviews = 3
  ndim = 9
  scale = 1
  centered = True
  overlap = False
  gen_D_alpha = True

  data = generate_redundant_data(
      npts=npts, nviews=nviews, ndim=ndim, scale=scale, centered=centered,
      overlap=overlap, gen_D_alpha=gen_D_alpha)

  D = None
  alpha = None
  if gen_D_alpha:
    data, D, alpha = data

  # IPython.embed()
  all_dset = BasicDataset(data)

  (tr_dset, te_dset), inds = all_dset.split([0.8, 0.2], get_inds=True)
  if gen_D_alpha:
    te_alpha = alpha[:, inds[1]]
    alpha = alpha[:, inds[0]]


  regularizer = "L1_inf"
  reduce_dict_every_iter = False
  D_init_var_frac = 1.0 # 0.95
  removal_thresh = 0.02
  lmbda = 2.0
  gamma = 2.0
  stopping_epsilon = 1e-3
  max_iters = 1000
  verbose = True
  config = sslf.SSLFConfig(
      regularizer=regularizer, reduce_dict_every_iter=reduce_dict_every_iter,
      D_init_var_frac=D_init_var_frac, removal_thresh=removal_thresh,
      lmbda=lmbda, gamma=gamma, stopping_epsilon=stopping_epsilon,
      max_iters=max_iters, verbose=verbose)

  model = sslf.StructuredSparseLatentFactorizer(config)

  # IPython.embed()
  model.fit(tr_dset, D, alpha)

  IPython.embed()
  loss_plots = [
      (model._loss_history["alpha"], 'r', 'alpha loss', '-'),
      (model._reg_loss_history["alpha"], 'r', 'alpha reg', '--'),
      (model._loss_history["D"], 'b', 'D loss', '-'),
      (model._reg_loss_history["D"], 'b', 'D reg', '--'),
      ]
  simple_plot(loss_plots)

if __name__=="__main__":
  test_sslf_toy()
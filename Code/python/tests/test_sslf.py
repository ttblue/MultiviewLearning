# Tests for the SSLF
import matplotlib.pyplot as plt
import numpy as np

from models import structured_sparse_latent_factorizer as sslf
from synthetic import multimodal_systems as ms

import IPython


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


def simple_plot(lst_of_plts, title=None):
  for vals, color, lbl, ls in lst_of_plts:
    plt.plot(vals, color=color, label=lbl, linestyle=ls)
    plt.legend()
  if title is not None:
    plt.title(title)
  plt.show()


def test_sslf_toy():
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
  lmbda = 1.0
  gamma = 1.0
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
  simple_plot(loss_plots, title="Sparse Structured Latent Factorization -- Loss")

if __name__=="__main__":
  test_sslf_toy()
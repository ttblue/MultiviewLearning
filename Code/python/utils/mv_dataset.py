# TODO: Probably need to fix these dataset classes:
# 1. Add abstract parent class and such

import numpy as np
import torch

from utils import dataset, utils

import IPython


DatasetException = dataset.DatasetException


class MultimodalTimeSeriesDataset:
  def __init__(self, txs, vis, t_length, shuffle=True, synced=False):
    # If synced, the assumption is that the time-series are concatenated
    # while maintaing the synchronized order.
    self.t_length = t_length
    self.synced = synced

    txs, inds = utils.split_txs_into_length(txs, t_length, ignore_end=False)
    vis = np.array(vis)[inds]
    self.views, v_counts = np.unique(vis, return_counts=True)
    if synced and not (v_counts == v_counts[0]).all():
      raise DatasetException(
          "Synced dataset has different number of data points for different"
          "views.")
    self.n_views = self.views.shape[0]
    self.n_ts = v_counts[0] if synced else v_counts.sum()

    self.v_txs = {vi:txs[(vis==vi).nonzero()[0]] for vi in self.views}
    self.v_dims = {vi:self.v_txs[vi].shape[1] for vi in self.views}

    if not synced:
      self.v_nts = {vi:vc for vi, vc in zip(self.views, v_counts)}
      self.v_fracs = np.array([self.v_nts[vi]/self.n_ts for vi in self.views])
    else:
      self.v_nts = {vi:self.n_ts for vi in self.views}
      frac = 1./ self.n_views
      self.v_fracs = {vi:frac for vi in self.views}

    self.epoch = 0
    self.v_idx = {vi:0 for vi in self.views}
    self.v_empty = {vi:False for vi in self.views}

    self._shuffle = shuffle
    self.shuffle_data()

    self._is_torch = False

  def toggle_shuffle(self, shuffle=None):
    self._shuffle = not self._shuffle if shuffle is None else shuffle

  def convert_to_torch(self):
    if self._is_torch:
      return

    self.v_txs = {
        vi:torch.from_numpy(self.v_txs[vi])
        for vi in self.views
    }
    self._is_torch = True

  def convert_to_numpy(self):
    if not self._is_torch:
      return

    self.v_txs = {
        vi:self.v_txs[vi].detach().numpy().transpose(permutation)
        for vi in self.views
    }
    self._is_torch = False

  def shuffle_data(self, rtn=False):
    if not self._shuffle:
      return

    if self.synced:
      shuffle_inds = np.random.permutation(self.n_ts)
      shuffled_txs = {vi:self.v_txs[vi][shuffle_inds] for vi in self.views}
    else:
      shuffle_inds = {
          vi:np.random.permutation(self.v_nts[vi]) for vi in self.v_nts
      }
      shuffled_txs = {vi:self.v_txs[vi][si] for vi, si in shuffle_inds.items()}
    
    if rtn:
      return shuffled_txs
    else:
      self.v_txs = shuffled_txs

  def is_empty(self):
    return np.all(list(self.v_empty.values()))

  def _get_view_batch(self, batch_size, vi, permutation):
    if self.v_empty[vi]:
      return (
          torch.empty((batch_size, 0, self.v_dims[vi])).permute(permutation)
          if self._is_torch else
          np.empty((batch_size, 0, self.v_dims[vi])).transpose(permutation)
      )

    start_idx = self.v_idx[vi]
    end_idx = start_idx + batch_size

    if end_idx >= self.v_nts[vi]:
      self.v_idx[vi] = 0
      self.v_empty[vi] = True
    else:
      self.v_idx[vi] = end_idx

    if self._is_torch:
      return self.v_txs[vi][start_idx:end_idx].permute(permutation)
    return self.v_txs[vi][start_idx:end_idx].transpose(permutation)

  def get_ts_batches(self, batch_size, permutation=(0, 1, 2)):
    self.v_empty = {vi:False for vi in self.views}
    self.shuffle_data()

    if self.synced:
      v_bsizes = np.ones(self.n_views).astype(int) * batch_size
    else:
      v_bsizes = (self.v_fracs * batch_size).astype(int)
      v_bsizes[-1] = batch_size - v_bsizes[:-1].sum()

    while not self.is_empty():
      batch = {
          vi:self._get_view_batch(bsize, vi, permutation)
          for vi, bsize in zip(self.views, v_bsizes)
      }
      yield batch

  def get_all_view_ts(self, vi, permutation=(0, 1, 2)):
    if self._is_torch:
      return self.v_txs[vi].permute(permutation)
    return self.v_txs[vi].transpose(permutation)

  def split(self, proportions):
    raise NotImplementedError("Split function not yet ready for this class.")
    # proportions = np.array(proportions)
    # if proportions.sum() != 1:
    #   proportions /= proportions.sum()

    # split_num = (proportions * self.num_ts).astype(int)
    # split_num[-1] = self.num_ts - split_num[:-1].sum()

    # x_shuffled, y_shuffled = self.shuffle_data(rtn=True)

    # end_inds = np.cumsum(split_num).tolist()
    # start_inds = [0] + end_inds[:-1]

    # dsets = []
    # for sind, eind in zip(start_inds, end_inds):
    #   x_split = x_shuffled[sind:eind]
    #   y_split = y_shuffled[sind:eind]
    #   dsets.append(
    #       MultimodalDSDataset(x_split, y_split, self._shuffle))

    # return dsets

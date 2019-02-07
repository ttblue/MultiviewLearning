# TODO: Probably need to fix these dataset classes:
# 1. Add abstract parent class and such

import numpy as np
import torch

import utils

import IPython


class DatasetException(Exception):
  pass


# Some utility functions
def create_batches(x, y, batch_size, num_steps):
  data_len = x.shape[0]
  epoch_size = data_len // (batch_size * num_steps)
  if epoch_size == 0:
    raise DatasetException(
        "epoch_size is 0. Decrease batch_size or num_steps")

  x_batches = x[0:batch_size * num_steps * epoch_size].reshape(
      [batch_size, num_steps * epoch_size, x.shape[1]])
  x_batches = np.split(x_batches, epoch_size, axis=1)

  if y is None:
    y_batches = None
  else:
    y = np.atleast_2d(y)
    y_batches = y[0:batch_size * num_steps * epoch_size].reshape(
        [batch_size, num_steps * epoch_size, y.shape[1]])
    y_batches.squeeze()
    y_batches = np.split(y_batches, epoch_size, axis=1)

  return x_batches, y_batches


class TimeseriesDataset:

  def __init__(self, xs, ys, shuffle=True, shift_scale=False):
    # xs, ys are a list of time-series data.
    self.xs = [np.array(x) for x in xs]
    self.ys = [np.array(y) for y in ys]

    if shift_scale:
      self.shift_and_scale()
    else:
      self.mu = np.zeros(self.xs[0].shape[1])
      self.sigma = np.ones(self.xs[0].shape[1])

    self.num_ts = len(xs)

    if self.num_ts != len(ys):
      raise DatasetException("")

    self._shuffle = shuffle
    self.shuffle_data()

    self.epoch = 0
    self.ts_idx = 0

  def toggle_shuffle(self, shuffle=None):
    self._shuffle = not self._shuffle if shuffle is None else shuffle

  def shuffle_data(self, rtn=False):
    if not self._shuffle:
      if not rtn:
        return
      else:
        return np.copy(self.xs).tolist(), np.copy(self.ys).tolist()

    shuffle_inds = np.random.permutation(self.num_ts)
    xs = [self.xs[i] for i in shuffle_inds]
    ys = [self.ys[i] for i in shuffle_inds]

    if rtn:
      return xs, ys
    else:
      self.xs = xs
      self.ys = ys

  def shift_and_scale(self, mu=None, sigma=None):
    # Every shift and scale is cumulative.
    # So only use this once, unless you know what you're doing.
    if mu is None or sigma is None:
      wts = np.array([x.shape[0] for x in self.xs]).astype("float")
      wts /= wts.sum()

      if mu is None:
        ts_mus = np.array([x.mean(axis=0) for x in self.xs])
        mu = ts_mus.T.dot(wts)

      if sigma is None:
        ts_sigmas_squared = np.array(
            [np.square(x-mu).mean(axis=0) for x in self.xs])
        sigma_squared = ts_sigmas_squared.T.dot(wts)
        sigma = np.sqrt(sigma_squared)

    self.mu = mu
    self.sigma = sigma

    self.xs = [(x - mu) / sigma for x in self.xs]

  def get_ts_batches(self, batch_size, num_steps):
    x = self.xs[self.ts_idx]
    y = self.ys[self.ts_idx]

    self.ts_idx += 1
    if self.ts_idx == self.num_ts:
      self.shuffle_data()
      self.ts_idx = 0
      self.epoch += 1

    return create_batches(x, y, batch_size, num_steps)

  def reset(self):
    self.ts_idx = 0
    self.epoch = 0
    self.shuffle_data()

  def split(self, proportions):
    proportions = np.array(proportions)
    if proportions.sum() != 1:
      proportions /= proportions.sum()

    split_num = (proportions * self.num_ts).astype(int)
    split_num[-1] = self.num_ts - split_num[:-1].sum()

    x_shuffled, y_shuffled = self.shuffle_data(rtn=True)

    end_inds = np.cumsum(split_num).tolist()
    start_inds = [0] + end_inds[:-1]

    dsets = []
    for sind, eind in zip(start_inds, end_inds):
      x_split = x_shuffled[sind:eind]
      y_split = y_shuffled[sind:eind]
      dsets.append(TimeseriesDataset(x_split, y_split, self._shuffle))

    return dsets


class DynamicalSystemDataset:

  def __init__(self, xs, dxs, ys, shuffle=True, shift_scale=True):
    # xs, ys are a list of time-series data.
    self.xs = [np.array(x) for x in xs]
    self.dxs = [np.array(dx) for dx in dxs]
    self.ys = [np.array(y) for y in ys]

    self._shift_scale = shift_scale
    self.num_ts = len(xs)

    if self.num_ts != len(ys) or self.num_ts != len(dxs):
      raise DatasetException("")

    self._shuffle = shuffle
    self.shuffle_data()

  def toggle_shuffle(self, shuffle=None):
    self._shuffle = not self._shuffle if shuffle is None else shuffle

  def shuffle_data(self, rtn=False):
    if not self._shuffle:
      if not rtn:
        return
      else:
        return (
            np.copy(self.xs).tolist(), np.copy(self.dxs).tolist(),
            np.copy(self.ys).tolist())

    shuffle_inds = np.random.permutation(self.num_ts)
    xs = [self.xs[i] for i in shuffle_inds]
    dxs = [self.dxs[i] for i in shuffle_inds]
    ys = [self.ys[i] for i in shuffle_inds]

    if rtn:
      return xs, dxs, ys
    else:
      self.xs = xs
      self.dxs = dxs
      self.ys = ys

  def get_samples(self, sample_length, num_per_ts=-1, channels=None, zero_frac=0.1):
    sample_xs = []
    sample_dxs = []
    sample_ys = []

    for idx in xrange(self.num_ts):
      xs, dxs, ys = self.xs[idx], self.dxs[idx], self.ys[idx]
      if channels is not None:
        xs, dxs = xs[:, channels], dxs[:, channels]

      start_inds = np.arange(0, xs.shape[0] - sample_length, sample_length)
      end_inds = start_inds + sample_length

      if num_per_ts > 0:
        rinds = np.random.permutation(start_inds.shape[0])[:num_per_ts]
      else:
        rinds = np.arange(start_inds.shape[0])

      for ridx in rinds:
        rxs = xs[start_inds[ridx]:end_inds[ridx]]
        if zero_frac is not None and zero_frac > 0.:
          if (rxs.sum(1) == 0).sum() > zero_frac * rxs.shape[0]:
            continue
        rdxs = dxs[start_inds[ridx]:end_inds[ridx]]
        rys = ys[start_inds[ridx]:end_inds[ridx]]
        if self._shift_scale:
          sigma = rxs[:, 0].std()
          rxs = (rxs - np.mean(rxs[:, 0])) / sigma
          rdxs = rdxs / sigma

        sample_xs.append(rxs)
        sample_dxs.append(rdxs)
        sample_ys.append(rys)

    return sample_xs, sample_dxs, sample_ys

  def split(self, proportions):
    proportions = np.array(proportions)
    if proportions.sum() != 1:
      proportions /= proportions.sum()

    split_num = (proportions * self.num_ts).astype(int)
    split_num[-1] = self.num_ts - split_num[:-1].sum()

    x_shuffled, dx_shuffled, y_shuffled = self.shuffle_data(rtn=True)

    end_inds = np.cumsum(split_num).tolist()
    start_inds = [0] + end_inds[:-1]

    dsets = []
    for sind, eind in zip(start_inds, end_inds):
      x_split = x_shuffled[sind:eind]
      dx_split = dx_shuffled[sind:eind]
      y_split = y_shuffled[sind:eind]
      dsets.append(
          DynamicalSystemDataset(x_split, dx_split, y_split, self._shuffle))

    return dsets


class MultimodalAsyncTimeSeriesDataset:
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
      self.v_nts = {vi:frac for vi in self.views}

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
          torch.empty((0, self.v_dims[vi])) if self._is_torch else
          np.empty((0, self.v_dims[vi]))
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

import numpy as np


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
    y_batches = y[0:batch_size * num_steps * epoch_size].reshape(
        [batch_size, num_steps * epoch_size])
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
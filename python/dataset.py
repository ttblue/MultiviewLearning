import numpy as np


class DatasetException(Exception):
  pass


class TimeseriesDataset:

  def __init__(self, xs, ys, shuffle=True):
    # xs, ys are a list of time-series data.
    self.xs = [np.array(x) for x in xs]
    self.ys = [np.array(y) for y in ys]
    self.num_ts = len(xs)
    self.num_features = self.xs[0].shape[1]

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

  def get_ts_batches(self, batch_size, num_steps):
    x = self.xs[self.ts_idx]
    y = self.ys[self.ts_idx]

    data_len = x.shape[0]
    epoch_size = data_len // (batch_size * num_steps)
    if epoch_size == 0:
      raise DatasetException(
          "epoch_size is 0. Decrease batch_size or num_steps")

    x_batches = x[0:batch_size * num_steps * epoch_size].reshape(
        [batch_size, num_steps * epoch_size, self.num_features])
    y_batches = y[0:batch_size * num_steps * epoch_size].reshape(
        [batch_size, num_steps * epoch_size])

    self.t_idx += 1
    if self.t_idx == self.num_ts:
      self.shuffle_data()
      self.t_idx = 0
      self.epoch += 1

    x_batches = np.split(x_batches, epoch_size, axis=1)
    y_batches = np.split(y_batches, epoch_size, axis=1)
    return x_batches, y_batches, epoch_size

  def reset(self):
    self.t_idx = 0
    self.epoch = 0
    self.shuffle_data()

  def split(self, proportions):

    proportions = np.array(proportions)
    if proportions.sum() != 1:
      proportions /= proportions.sum()

    split_num = (proportions * self.num_ts).astype(int)
    split_num[-1] = self.num_ts - split_num[:-1].sum()

    x_shuffled, y_shuffled = self.shuffle_data(rtn=True)

    end_inds = np.cumsum(split_inds).tolist()
    start_inds = [0] + end_inds[:-1].tolist()

    dsets = []
    for sind, eind in zip(start_inds, end_inds):
      x_split = x_shuffled[sind:eind]
      y_split = y_shuffled[sind:eind]
      dsets.append(TimeseriesDataset(x_split, y_split, self.shuffle))

    return dsets
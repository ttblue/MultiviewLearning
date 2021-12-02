import gzip
import numpy as np
import os
import pickle

from dataprocessing import multiview_datasets as md


import IPython


_DATA_DIR = os.getenv("DATA_DIR")
_MNIST_FILE = os.path.join(_DATA_DIR, "./MNIST/mnist.pkl.gz")

def load_original_mnist():
  with gzip.open(_MNIST_FILE, "rb") as fh:
    train_set, valid_set, test_set = pickle.load(fh, encoding="latin1")

  return train_set, valid_set, test_set


_mnist_h = 28
_mnist_w = 28
_mnist_dim = _mnist_h * _mnist_w
_mnist_idx_grid = np.arange(_mnist_h * _mnist_w).reshape(_mnist_h, _mnist_w)
# only 4 views for no w
def get_mnist_split_inds(n_views=4, shape="grid"):
  split_inds = None
  if shape == "grid" and n_views == 4:
    # Quadrants:
    _half_h = _mnist_h // 2
    _half_w = _mnist_w // 2
    split_inds = []
    for i in range(2):
      for j in range(2):
        h0, h1 = i * _half_h, (i + 1) * _half_h
        w0, w1 = j * _half_w, (j + 1) * _half_w
        view_inds = _mnist_idx_grid[h0:h1, w0:w1].reshape(1, -1).squeeze()
        split_inds.append(view_inds)
  else:
    # Simple split
    vdim = _mnist_dim // n_views
    vdims = np.ones(n_views) * vdim
    vdims[0] = _mnist_dim - vdims[1:].sum()
    end_idxs = np.cumsum(vdims).astype(int).tolist()
    start_idxs = [0] + end_idxs[:-1]
    split_inds = [
        np.arange(sidx, eidx) for (sidx, eidx) in zip(start_idxs, end_idxs)]

  return split_inds


def load_split_mnist(n_views=3, split_inds=None, shape="grid"):
  train_set, valid_set, test_set = load_original_mnist()
  train_x, train_y = train_set
  valid_x, valid_y = valid_set
  test_x, test_y = test_set

  if split_inds is None:
    split_inds = get_mnist_split_inds(n_views, shape=shape)

  train_xvs = {}
  valid_xvs = {}
  test_xvs = {}

  for vi, view_inds in enumerate(split_inds):
    train_xvs[vi] = train_x[:, view_inds]
    valid_xvs[vi] = valid_x[:, view_inds]
    test_xvs[vi] = test_x[:, view_inds]

  return (
      (train_xvs, train_y), (valid_xvs, valid_y), (test_xvs, test_y),
      split_inds)


if __name__=="__main__":
  train_set, valid_set, test_set = load_original_mnist()
  n_views = 3
  tr_dat, va_dat, te_dat = load_split_mnist(n_views=n_views)

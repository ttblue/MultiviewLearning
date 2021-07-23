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


def load_split_mnist(n_views=3):
  train_set, valid_set, test_set = load_original_mnist()
  train_x, train_y = train_set
  valid_x, valid_y = valid_set
  test_x, test_y = test_set

  dim = train_x.shape[1]
  vdim = dim // n_views
  vdims = np.ones(n_views) * vdim
  vdims[0] = dim - vdims[1:].sum()

  train_xvs = {}
  valid_xvs = {}
  test_xvs = {}

  end_idxs = np.cumsum(vdims).astype(int).tolist()
  start_idxs = [0] + end_idxs[:-1]
  for vi, (sidx, eidx) in enumerate(zip(start_idxs, end_idxs)):
    train_xvs[vi] = train_x[:, sidx:eidx]
    valid_xvs[vi] = valid_x[:, sidx:eidx]
    test_xvs[vi] = test_x[:, sidx:eidx]

  return (train_xvs, train_y), (valid_xvs, valid_y), (test_xvs, test_y)


if __name__=="__main__":
  train_set, valid_set, test_set = load_original_mnist()
  n_views = 3
  tr_dat, va_dat, te_dat = load_split_mnist(n_views=n_views)

import glob
import numpy as np
import os

from utils import math_utils

import IPython


PM_DATA_DIR = os.getenv("PM_DATA_DIR")


_CMAS_DIR = os.path.join(PM_DATA_DIR, "CMAPSSData")
_CMAS_DSET_TYPES = ["train", "test"]
def load_cmas(dset_type="all", normalize=True):
  dtypes = [dset_type] if dset_type in _CMAS_DSET_TYPES else _CMAS_DSET_TYPES
  data = {}

  for dtype in dtypes:
    dfiles = glob.glob(os.path.join(_CMAS_DIR, dtype + "*.txt"))
    ids, ts, feats, ys = [], [], [], []
    for dfile in dfiles:
      dnum = int(dfile.split(".")[-2][-3:])
      with open(dfile, 'r') as fh:
        dlines = fh.readlines()
      fl_data = np.array([
          [float(val) for val in dline.strip().split(" ")]
          for dline in dlines])
      units, locs, cts = np.unique(
          fl_data[:, 0], return_index=True, return_counts=True)

      # Extract labels -- time-index of failure
      if dtype == "train":
        # The time-index of failure is last t-stamp for training points
        ruls = fl_data[locs + cts - 1, 0].astype(int)
      else:
        # For test points, these are given in file.
        rul_file = os.path.join(_CMAS_DIR, "RUL_FD00%i.txt" % dnum)
        with open(rul_file, 'r') as fh:
          ruls = [int(r) for r in fh.read().strip().split('\n')]

      for unum, loc, ct, rul in zip(units, locs, cts, ruls):
        idxs = np.arange(loc, loc + ct)
        ids.append((dnum, int(unum)))
        ts.append(fl_data[idxs, 1])
        feats.append(fl_data[idxs, 2:])
        ys.append(rul)

    data[dtype] = {"ids": ids, "ts": ts, "features": feats, "y": ys}

  if normalize:
    dtype = "train" if "train" in data else "test"
    feats = np.concatenate(data[dtype]["features"], axis=0)
    _, means, stds = math_utils.shift_and_scale(feats, scale=True)
    for dtype, tdat in data.items():
      data[dtype]["features"] = [
          (feat - means) / stds for feat in tdat["features"]]
    misc = {"means": means, "stds": stds}
    return data, misc

  return data


if __name__ == "__main__":
  data = load_cmas("all")

import glob
import numpy as np
import os

import IPython


PM_DATA_DIR = os.getenv("PM_DATA_DIR")


_CMAS_DIR = os.path.join(PM_DATA_DIR, "CMAPSSData")
_CMAS_DSET_TYPES = ["train", "test"]
def load_cmas(dset_type="all"):
  data = (
      {dset_type: {}} if dset_type in _CMAS_DSET_TYPES else
      {dtype: {} for dtype in _CMAS_DSET_TYPES})
    
  for dtype in data:
    dfiles = glob.glob(os.path.join(_CMAS_DIR, dtype + "*.txt"))
    for dfile in dfiles:
      dnum = int(dfile.split(".")[-2][-3:])
      dnum_data = {}
      with open(dfile, 'r') as fh:
        dlines = fh.readlines()
      fl_data = np.array([
          [float(val) for val in dline.strip().split(" ")]
          for dline in dlines])
      units, locs, cts = np.unique(
          fl_data[:, 0], return_index=True, return_counts=True)
      for unum, loc, ct in zip(units, locs, cts):
        idxs = np.arange(loc, loc + ct)
        dnum_data[int(unum)] = {
            "ts": fl_data[idxs, 1], "features": fl_data[idxs, 2:]}
      data[dtype][dnum] = dnum_data

    if dtype == "train":
      for dnum in data[dtype].keys():
        for unum, dat in data[dtype][dnum].items():
          data[dtype][dnum][unum]["y"] = dat["ts"][-1]
    else:
      for dnum in data[dtype].keys():
        rul_file = os.path.join(_CMAS_DIR, "RUL_FD00%i.txt" % dnum)
        with open(rul_file, 'r') as fh:
          ruls = [int(r) for r in fh.read().strip().split('\n')]
        for unum, rul in zip(data[dtype][dnum].keys(), ruls):
          data[dtype][dnum][unum]["y"] = rul

  return data


if __name__ == "__main__":
  data = load_cmas("all")

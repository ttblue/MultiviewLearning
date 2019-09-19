import itertools
import numpy as np
from numpy import fft
import time

from models.model_base import ModelException, BaseConfig
from utils import time_series_utils as tsu, utils


import IPython


def fourier_featurize_windows(tvals, tstamps, window_size):
  w_tstamps, w_tvals = tsu.split_discnt_ts_into_windows(
    tvals, tstamps, window_size, ignore_rest=False, shuffle=True)

  w_fft = fft.rfft(w_tvals, axis=1)
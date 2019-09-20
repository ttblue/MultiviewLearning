import itertools
import numpy as np
from numpy import fft
import time

from models.model_base import ModelException, BaseConfig
from utils import time_series_utils as tsu, utils


import IPython


def _convert_dict_to_numpy(dc):
  op = {}
  if not isinstance(dc, dict):
    return np.array(dc)
  for key in dc:
    op[key] = _convert_dict_to_numpy(dc[key])
  return op


class FFConfig(BaseConfig):
  def __init__(self, ndim=5, use_imag=False, *args, **kwargs):
    super(FFConfig, self).__init__(*args, **kwargs)
    self.ndim = ndim
    self.use_imag = use_imag


class TimeSeriesFourierFeaturizer(object):
  def __init__(self, config):
    self.config = config
    self.trained = False

  def compute_embedding_basis(self, X):
    mu = X.mean(0)
    X_centered = X - mu

    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    basis = VT.T
    # S = S[:ndim]
    ndim = min(VT.shape[0], self.config.ndim)

    return basis, mu, S, ndim

  # def fourier_featurize_windows(tvals, tstamps):
  #   w_tstamps, w_tvals = tsu.split_discnt_ts_into_windows(
  #     tvals, tstamps, self.config.window_size, ignore_rest=False, shuffle=True)

  def fit(self, tvals, tstamps):
    # Not using tstamps for now -- since everything is assumed to be sampled
    # in the same frequency
    if len(tvals.shape) < 3:
      tvals = tvals.reshape(tvals.shape[0], tvals.shape[1], 1)

    w_fft = fft.rfft(tvals, axis=1)

    self._nchannels = w_fft.shape[2]
    self.basis = {i: {} for i in range(self._nchannels)}
    self.mu = {i: {} for i in range(self._nchannels)}
    self.S = {i: {} for i in range(self._nchannels)}
    self._ndims = {i: {} for i in range(self._nchannels)}

    # Compute complex embedding just in case, even if not being used.
    for i in range(self._nchannels):
      X = w_fft[:, :, i]
      X_components = {"re": X.real, "im": X.imag}

      for comp, X_val in X_components.items():
        basis, mu, S, ndim = self.compute_embedding_basis(X_val)
        self.basis[i][comp] = basis
        self.mu[i][comp] = mu
        self.S[i][comp] = S
        self._ndims[i][comp] = ndim

    self.trained = True

  def _encode_channel(self, fft_channel, ch_id, use_imag):
    # Can use S if we need to scale
    basis, mu, S, ndim = (
        self.basis[ch_id], self.mu[ch_id], self.S[ch_id], self._ndims[ch_id])
    ndim_re = min(self.config.ndim, ndim["re"])
    code_re = (fft_channel.real - mu["re"]).dot(basis["re"][:, :ndim_re])
    if use_imag:
      ndim_im = min(self.config.ndim, ndim["im"])
      code_im = (fft_channel.imag - mu["im"]).dot(basis["im"][:, :ndim_im])
      return np.c_[code_re, code_im]
    return code_re

  def encode(self, tvals, tstamps=None, use_imag=None):
    if len(tvals.shape) < 3:
      tvals = tvals.reshape(tvals.shape[0], tvals.shape[1], 1)

    use_imag = self.config.use_imag if use_imag is None else use_imag
    w_fft = fft.rfft(tvals, axis=1)
    codes = []
    for i in range(w_fft.shape[2]):
      X = w_fft[:, :, i]
      codes.append(self._encode_channel(X, i, use_imag))

    code = np.concatenate(codes, axis=1)
    return code

  def _decode_channel(self, code_channel, ch_id, use_imag):
    # Can use S if we need to scale
    basis, mu, S, ndim = (
        self.basis[ch_id], self.mu[ch_id], self.S[ch_id], self._ndims[ch_id])
    # Config might have changed
    ndim_re = min(self.config.ndim, ndim["re"])

    code_re, code_im = np.array_split(code_channel, [ndim_re], axis=1)
    wfft = code_re.dot(basis["re"].T[:ndim_re]) + mu["re"]
    if use_imag:
      ndim_im = min(self.config.ndim, ndim["im"])
      wfft = wfft + 1j * (
          code_re.dot(basis["im"].T[:ndim_im]) + mu["im"])
    return wfft

  def decode(self, codes, use_imag):
    use_imag = self.config.use_imag if use_imag is None else use_imag
    dim_factor = 2 if use_imag else 1
    dims = np.array(
        [min(self.config.ndim, self._ndims[i]["re"])
         for i in range(self._nchannels)])
    if use_imag:
      im_dims = np.array(
          [min(self.config.ndim, self._ndims[i]["im"])
          for i in range(self._nchannels)])
      dims += im_dims

    split_inds = np.cumsum(dims)[:-1]
    channel_codes = np.array_split(codes, split_inds, axis=1)

    tvals = []
    for i, ch_code in enumerate(channel_codes):
      wfft = self._decode_channel(ch_code, i, use_imag)
      tvals.append(np.fft.irfft(wfft))

    tvals = np.stack(tvals, axis=2)
    return tvals

  def _reset_ndims(self):
    self._ndims = {key: {} for key in range(self._nchannels)}
    for key, basis in self.basis.items():
      for comp, mat in basis.items():
        self._ndims[key][comp] = mat.shape[1]

  def reset(self, basis, mu, S, config=None):
    if config is not None:
      self.config = config
    self.basis = basis
    self.mu = mu
    self.S = S
    self._nchannels = len(basis)
    self._reset_ndims()

    self.trained = True
    return self

  def split_channels_into_models(self):
    models = []
    for i in range(self._nchannels):
      basis = {0: self.basis[i]}
      mu = {0: self.mu[i]}
      S = {0: self.S[i]}

      model = TimeSeriesFourierFeaturizer(self.config)
      models.append(model.reset(basis, mu, S))

    return models

  def save_to_file(self, fname):
    data = [self.basis, self.mu, self.S, self.config.__dict__]
    with open(fname, "b") as fh:
      np.save(fh, data)

  def load_from_file(self, fname):
    data = np.load(fname).tolist()
    [basis, mu, S] = map(_convert_dict_to_numpy, data[:-1])
    config = FFconfig(**data[-1])

    return self.reset(basis, mu, S, config)
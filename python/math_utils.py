import os

import numpy as np
# ==============================================================================
# Utility functions
# ==============================================================================

def rbf_fourierfeatures(d_in, d_out, gammak, sine=True):
  # Returns a function handle to compute random fourier features.
  if not sine:
    W = np.random.normal(0., np.sqrt(2 * gammak), (d_in, d_out))
    h = np.random.uniform(0., 2 * np.pi, (1, d_out))
    def rbf_ff(x):
      ff = np.cos(x.dot(W) + h) / np.sqrt(d_out / 2.)
      return ff
  else:
    W = np.random.normal(0., np.sqrt(2 * gammak), (d_in, d_out // 2))
    def rbf_ff(x):
      ff = np.c_[np.cos(x.dot(W)), np.sin(x.dot(W))] / np.sqrt(d_out / 2.)
      return ff

  return rbf_ff


def mm_rbf_fourierfeatures(d_in, d_out, a, file_name=None):
  # Returns a function handle to compute random fourier features.
  # Can load from/save to file if one exists.
  if file_name is not None:
    # Some simple checks for filename.
    basename = os.path.basename(file_name)
    if len(basename.split('.')) == 1:
      file_name = file_name + ".npy"

    if os.path.exists(file_name):
      W, h, a = np.load(file_name)
    else:
      W = np.random.normal(0., 1., (d_in, d_out))
      h = np.random.uniform(0., 2*np.pi, (1, d_out))
      np.save(file_name, [W, h, a])
  else:
    W = np.random.normal(0., 1., (d_in, d_out))
    h = np.random.uniform(0., 2*np.pi, (1, d_out))

  def mm_rbf(x):
    xhat = np.mean(np.cos((1/a)*x.dot(W)+h)/np.sqrt(d_out)*np.sqrt(2), axis=0)
    return xhat

  return mm_rbf

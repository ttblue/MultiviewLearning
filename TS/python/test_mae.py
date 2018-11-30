# Testing multi-view autoencoder
import numpy as np
import torch

try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False

import gaussian_random_features as grf
import multi_ae as mae
from synthetic import simple_systems as ss
import time_sync as tsync
import utils

import IPython


torch.set_default_dtype(torch.float32)


# Assuming < 5 views for now
_COLORS = ['r', 'g', 'b', 'y']
def plot_recon(true_vals, pred_vals, labels, title=None):
  if not MPL_AVAILABLE:
    print("Matplotlib not available.")
    return

  for tr, pr, l, c in zip(true_vals, pred_vals, labels, _COLORS):
    plt.plot(tr, color=c, label=l + " True")
    plt.plot(pr, color=c + ':', label=l + " Pred")
  plt.legend()
  if title:
    plt.title(title)
  plt.show()

def test_lorenz():
  visualize = False and MPL_AVAILABLE

  no_ds = False
  tmax = 10
  nt = 1000
  x, y, z = ss.generate_lorenz_system(tmax, nt, sig=0.1)

  if no_ds:
    ntau = 1
    xts, yts, zts = x, y, z

    xts = xts - np.min(xts)
    yts = yts - np.min(yts)
    zts = zts - np.min(zts)
    xts = xts / np.max(np.abs(xts))
    yts = yts / np.max(np.abs(yts))
    zts = zts / np.max(np.abs(zts))
  else:
    x = x - np.min(x)
    y = y - np.min(y)
    z = z - np.min(z)
    x = x / np.max(np.abs(x))
    y = y / np.max(np.abs(y))
    z = z / np.max(np.abs(z))

    tau = 20
    ntau = 3
    xts = tsync.compute_td_embedding(x, tau, ntau)
    yts = tsync.compute_td_embedding(y, tau, ntau)
    zts = tsync.compute_td_embedding(z, tau, ntau)

  nt = xts.shape[0]
  n_views = 3
  v_sizes = [ntau] * n_views

  hidden_size = 10
  input_size = ntau
  output_size = hidden_size
  layer_units = [10, 10]
  activation = torch.nn.functional.relu
  last_activation = torch.sigmoid
  is_encoder = True
  enc_config = mae.EDConfig(
    input_size=input_size, output_size=output_size,
    layer_units=layer_units, is_encoder=is_encoder,
    activation=activation, last_activation=last_activation)
  enc_params = {i:enc_config for i in range(n_views)}

  input_size = hidden_size
  output_size = ntau
  layer_units = [10, 10]
  is_encoder = False
  dec_config = mae.EDConfig(
    input_size=input_size, output_size=output_size,
    layer_units=layer_units, is_encoder=is_encoder,
    activation=activation, last_activation=last_activation)
  dec_params = {i:dec_config for i in range(n_views)}

  max_iters = 5000
  batch_size = 50
  lr = 1e-3
  verbose = True
  config = mae.MAEConfig(
      v_sizes=v_sizes,
      code_size=hidden_size,
      encoder_params=enc_params,
      decoder_params=dec_params,
      max_iters=max_iters,
      batch_size=batch_size,
      lr=lr,
      verbose=verbose)

  rn = 100
  gammak = 5e-2
  sine = False
  fgen_args = {"rn":rn, "gammak":gammak, "sine":sine}
  fgen = grf.GaussianRandomFeatures(dim=ntau, **fgen_args)
#   ffunc = fgen.computeRandomFeatures
  ffunc = lambda x: x
  tr_frac = 0.8
  split_inds = [0, int(tr_frac * nt), nt]
  dsets = [utils.split_data(ds, split_inds=split_inds)[0] for ds in [xts, yts, zts]]
  Vs_tr = [ffunc(ds[0]) for ds in dsets]
  Vs_te = [ffunc(ds[1]) for ds in dsets]
  Xtr = np.concatenate(Vs_tr, axis=1)
  Xte = np.concatenate(Vs_te, axis=1)

  # mae = reload(mae)
  code_learner = mae.MultiAutoEncoder(config)
  code_learner.fit(Xtr)

  IPython.embed()

if __name__ == "__main__":
  test_lorenz()
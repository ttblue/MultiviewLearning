# Testing multi-view autoencoder
import numpy as np
import torch

try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False

import dataset
import gaussian_random_features as grf
import multi_ae as mae
import multi_rnn_ae as mrae
from synthetic import simple_systems as ss
import time_sync as tsync
import torch_utils as tu
import utils

import IPython


torch.set_default_dtype(torch.float32)

# Assuming < 5 views for now
_COLORS = ['b', 'r', 'g', 'y']
def plot_recon(true_vals, pred_vals, labels, title=None):
  if not MPL_AVAILABLE:
    print("Matplotlib not available.")
    return

  for tr, pr, l, c in zip(true_vals, pred_vals, labels, _COLORS):
    plt.plot(tr[:, 0], color=c, label=l + " True")
    plt.plot(pr[:, 0], color=c, ls='--', label=l + " Pred")
  plt.legend()
  if title:
    plt.title(title)
  plt.show()


def plot_simple(tr, pr, l, title=None):
  plt.plot(tr[:, 0], color='b', label=l + " True")
  plt.plot(pr[:, 0], color='r', ls='--', label=l + " Pred")
  plt.legend()
  if title:
    plt.title(title)
  plt.show()


def test_lorenz_MAE():
  visualize = False and MPL_AVAILABLE

  no_ds = False
  tmax = 50
  nt = 5000
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

  hidden_size = 32
  input_size = ntau
  output_size = hidden_size
  layer_units = [64, 32]
  activation = torch.nn.ReLU
  last_activation = torch.nn.Sigmoid
  use_vae = True
  enc_config = tu.MNNConfig(
    input_size=input_size, output_size=output_size,
    layer_units=layer_units, activation=activation,
    last_activation=last_activation, use_vae=use_vae)
  enc_params = {i:enc_config for i in range(n_views)}

  input_size = hidden_size
  output_size = ntau
  layer_units = [32, 64]
  use_vae = False
  dec_config = tu.MNNConfig(
    input_size=input_size, output_size=output_size,
    layer_units=layer_units, activation=activation,
    last_activation=last_activation, use_vae=use_vae)
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

  # rn = 100
  # gammak = 5e-2
  # sine = False
  # fgen_args = {"rn":rn, "gammak":gammak, "sine":sine}
  # fgen = grf.GaussianRandomFeatures(dim=ntau, **fgen_args)
#   ffunc = fgen.computeRandomFeatures
  ffunc = lambda x: x
  tr_frac = 0.8
  split_inds = [0, int(tr_frac * nt), nt]
  dsets = [utils.split_data(ds, split_inds=split_inds)[0] for ds in [xts, yts, zts]]
  Vs_tr = [ffunc(ds[0]) for ds in dsets]
  Vs_te = [ffunc(ds[1]) for ds in dsets]
  Xtr = np.concatenate(Vs_tr, axis=1)
  Xte = np.concatenate(Vs_te, axis=1)

  true_ts = [xts, yts, zts] 
  labels = ['x', 'y', 'z'] 

  # mae = reload(mae)
  code_learner = mae.MultiAutoEncoder(config)
  code_learner.fit(Xtr)

  IPython.embed()

  recon_x = code_learner.predict([xts], [0], vi_out=[1, 2])
  recon_y = code_learner.predict([yts], [1], vi_out=[0, 2])
  recon_z = code_learner.predict([zts], [2], vi_out=[0, 1])

  plot_recon([yts, zts], recon_x, 'yz', title="X recon")
  plot_recon([xts, zts], recon_y, 'xz', title="Y recon")
  plot_recon([xts, yts], recon_z, 'xy', title="Z recon")



def test_lorenz_RNNMAE():
  visualize = False and MPL_AVAILABLE

  no_ds = False
  # tmax = 50
  # nt = 5000
  # t_len = 200
  tmax = 100
  nt = 10000
  t_len = t_length = 50
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

  # latent_size = 32
  # rnn_size = 20

  latent_size = 16
  rnn_size = 16
  l1_size = ntau

  num_layers = 1
  cell_type = torch.nn.LSTM

  # Not being used now.
  # input_size = ntau
  # output_size = l1_size
  # layer_units = [] # [32, 64]
  # activation = torch.nn.ReLU
  # last_activation = torch.nn.Sigmoid
  # use_vae = False
  # pre_enc_config = tu.MNNConfig(
  #     input_size=input_size, output_size=output_size,
  #     layer_units=layer_units, activation=activation,
  #     last_activation=last_activation, use_vae=use_vae)

  input_size = ntau
  hidden_size = rnn_size
  return_only_final = False
  return_only_hidden = True
  en_rnn_config = tu.RNNConfig(
      input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
      cell_type=cell_type, return_only_hidden=return_only_hidden,
      return_only_final=return_only_final)

  input_size = rnn_size
  output_size = latent_size
  layer_units = [32, 64]
  layer_types, layer_args = tu.generate_layer_types_args(
      input_size, layer_units, output_size)
  activation = torch.nn.ReLU
  last_activation = torch.nn.Sigmoid
  use_vae = False
  post_enc_config = tu.MNNConfig(
    input_size=input_size, output_size=output_size, layer_types=layer_types,
    layer_args=layer_args, activation=activation,
    last_activation=last_activation, use_vae=use_vae)

  en_funcs = [tu.RNNWrapper, tu.MultiLayerNN]
  en_configs = [en_rnn_config, post_enc_config]

  encoder_params = {
      vi: {"layer_funcs": en_funcs, "layer_config": en_configs}
      for vi in range(n_views)
  }

  input_size = latent_size
  output_size = rnn_size
  layer_units = [64, 32]
  layer_types, layer_args = tu.generate_layer_types_args(
      input_size, layer_units, output_size)
  activation = torch.nn.ReLU
  last_activation = lambda: tu._IDENTITY  # hack
  use_vae = False
  pre_dec_config = tu.MNNConfig(
    input_size=input_size, output_size=output_size, layer_types=layer_types,
    layer_args=layer_args, activation=activation,
    last_activation=last_activation, use_vae=use_vae)

  input_size = rnn_size
  hidden_size = ntau
  return_only_final = False
  return_only_hidden = True
  de_rnn_config = tu.RNNConfig(
      input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
      cell_type=cell_type, return_only_hidden=return_only_hidden,
      return_only_final=return_only_final)

  # Not being used now.
  # input_size = l1_size
  # output_size = ntau
  # layer_units = [] # [64, 32]
  # activation = torch.nn.ReLU
  # last_activation = lambda: tu._IDENTITY  # hack
  # post_dec_config = tu.MNNConfig(
  #     input_size=input_size, output_size=output_size,
  #     layer_units=layer_units, activation=activation,
  #     last_activation=last_activation, use_vae=use_vae)

  de_funcs = [tu.MultiLayerNN, tu.RNNWrapper]
  de_configs = [pre_dec_config, de_rnn_config]

  decoder_params = {
      vi: {"layer_funcs": de_funcs, "layer_config": de_configs}
      for vi in range(n_views)
  }

  use_vae = False
  max_iters = 2000
  batch_size = 100
  lr = 1e-3
  verbose = True

  config = mrae.MRNNAEConfig(
      n_views=n_views,
      encoder_params=encoder_params,
      decoder_params=decoder_params,
      use_vae=use_vae,
      max_iters=max_iters,
      batch_size=batch_size,
      lr=lr,
      verbose=verbose)

  # input_size = ntau
  # output_size = latent_size
  # layer_units = [16]
  # activation = torch.nn.ReLU
  # last_activation = torch.nn.Sigmoid
  # is_encoder = True
  # pre_enc_config = tu.MNNConfig(
  #   input_size=input_size, output_size=output_size,
  #   layer_units=layer_units, is_encoder=is_encoder,
  #   activation=activation, last_activation=last_activation)
  # enc_params = {i:enc_config for i in range(n_views)}

  # input_size = hidden_size
  # output_size = ntau
  # layer_units = [10, 10]
  # is_encoder = False
  # dec_config = tu.MNNConfig(
  #   input_size=input_size, output_size=output_size,
  #   layer_units=layer_units, is_encoder=is_encoder,
  #   activation=activation, last_activation=last_activation)
  # dec_params = {i:dec_config for i in range(n_views)}

  # max_iters = 5000
  # batch_size = 50
  # lr = 1e-3
  # verbose = True
  # config = mae.MAEConfig(
  #     v_sizes=v_sizes,
  #     code_size=hidden_size,
  #     encoder_params=enc_params,
  #     decoder_params=dec_params,
  #     max_iters=max_iters,
  #     batch_size=batch_size,
  #     lr=lr,
  #     verbose=verbose)

#   rn = 100
#   gammak = 5e-2
#   sine = False
#   fgen_args = {"rn":rn, "gammak":gammak, "sine":sine}
#   fgen = grf.GaussianRandomFeatures(dim=ntau, **fgen_args)
# #   ffunc = fgen.computeRandomFeatures
  ffunc = lambda x: x
  nts = int(np.ceil(nt / t_len))
  # dsets = [
  #     utils.split_data(ds, n=nts, const_len=True)[0] for ds in [xts, yts, zts]
  # ]
  dsets = [ds.astype(np.float32) for ds in [xts, yts, zts]]

  tr_frac = 0.8
  split_inds = [0, int(tr_frac * nt), nt]
  split_dsets = [utils.split_data(ds, split_inds=split_inds)[0] for ds in dsets]
  Tr_txs = [ffunc(ds[0]) for ds in split_dsets]
  Tr_vis = [0, 1, 2] # [[i] * len(ds[0]) for i, ds in enumerate(split_dsets)]
  Te_txs = [ffunc(ds[1]) for ds in split_dsets]
  Te_vis = [0, 1, 2] # [[i] * len(ds[1]) for i, ds in enumerate(split_dsets)]
  # IPython.embed()
  # Tvs_tr = [ffunc(ds[0]) for ds in split_dsets]
  # Tvs_te = [ffunc(ds[1]) for ds in split_dsets]

  # all_dsets = [d1 + d2 for d1, d2 in zip(Tvs_tr, Tvs_te)]

  dset_type = dataset.MultimodalAsyncTimeSeriesDataset
  Tr_dset = dset_type(Tr_txs, Tr_vis, t_len, shuffle=True, synced=True)
  Te_dset = dset_type(Te_txs, Te_vis, t_len, shuffle=True, synced=True)

  # mae = reload(mae)
  code_learner = mrae.MultiRNNAutoEncoder(config)
  code_learner.fit(Tr_dset)

  IPython.embed()

  # recon_x = code_learner.predict(Tvs_te[0], 0, vi_out=[1, 2])
  # recon_y = code_learner.predict(Tvs_te[1], 1, vi_out=[0, 2])
  # recon_z = code_learner.predict(Tvs_te[2], 2, vi_out=[0, 1])
  dsets3 = []
  for ds in dsets:
      dsets3.append(
          utils.split_txs_into_length([ds], t_length, ignore_end=False)[0])
  recon_x = code_learner.predict(dsets3[0], 0, vi_out=[1, 2])
  recon_y = code_learner.predict(dsets3[1], 1, vi_out=[0, 2])
  recon_z = code_learner.predict(dsets3[2], 2, vi_out=[0, 1])

  true_ts = [np.array(tx).reshape(-1, 3) for tx in dsets3]
  recon_x = [np.array(recon_x[i]).reshape(-1, 3) for i in [1, 2]]
  recon_y = [np.array(recon_y[i]).reshape(-1, 3) for i in [0, 2]]
  recon_z = [np.array(recon_z[i]).reshape(-1, 3) for i in [0, 1]]

  ttx = [true_ts[1], true_ts[2]]
  tty = [true_ts[0], true_ts[2]]
  ttz = [true_ts[0], true_ts[1]]
  plot_recon(ttx, recon_x, 'yz', title="X recon")
  plot_recon(tty, recon_y, 'xz', title="Y recon")
  plot_recon(ttz, recon_z, 'xy', title="Z recon")


if __name__ == "__main__":
  test_lorenz_RNNMAE()
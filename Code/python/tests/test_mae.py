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
import multiview_forecaster as mfor
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

  code_sample_noise_var = 0.5
  max_iters = 5000
  batch_size = 50
  lr = 1e-3
  verbose = True
  config = mae.MAEConfig(
      v_sizes=v_sizes,
      code_size=hidden_size,
      encoder_params=enc_params,
      decoder_params=dec_params,
      code_sample_noise_var=code_sample_noise_var,
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



def test_lorenz_RNNMAE(forecast=True):
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
  num_layers = 1
  cell_type = torch.nn.LSTM
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
  num_layers = 1
  cell_type = torch.nn.LSTM
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
  batch_size = 80  #100
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

  if forecast:
    input_size = latent_size
    hidden_size = latent_size
    num_layers = 1
    cell_type = torch.nn.LSTM
    return_only_final = False
    return_only_hidden = False
    forecaster_rnn_config = tu.RNNConfig(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        cell_type=cell_type, return_only_hidden=return_only_hidden,
        return_only_final=return_only_final)

    ls_type = "mrae"
    ls_learner_config = None
    layer_params = {
        "layer_funcs": [tu.RNNWrapper],
        "layer_config": [forecaster_rnn_config],
    }
    lr = 1e-3
    batch_size = 80  #100
    max_iters = 2000
    verbose = True
    forecast_config = mfor.MVForecasterConfig(
        ls_type=ls_type,
        ls_learner_config=ls_learner_config,
        layer_params=layer_params,
        lr=lr,
        batch_size=batch_size,
        max_iters=max_iters,
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

  dset_type = dataset.MultimodalTimeSeriesDataset
  Tr_dset = dset_type(Tr_txs, Tr_vis, t_len, shuffle=True, synced=True)
  Te_dset = dset_type(Te_txs, Te_vis, t_len, shuffle=True, synced=True)

  # mae = reload(mae)
  code_learner = mrae.MultiRNNAutoEncoder(config)
  code_learner.fit(Tr_dset)

  if forecast:
    forecaster = mfor.MVForecaster(forecast_config)
    forecaster.set_ls_learner(code_learner)

    forecaster.fit(Tr_dset)

  # recon_x = code_learner.predict(Tvs_te[0], 0, vi_out=[1, 2])
  # recon_y = code_learner.predict(Tvs_te[1], 1, vi_out=[0, 2])
  # recon_z = code_learner.predict(Tvs_te[2], 2, vi_out=[0, 1])

  dsets3 = []
  for ds in dsets:
      dsets3.append(
          utils.split_txs_into_length([ds], t_length, ignore_end=False)[0])

  recon_x = code_learner.predict(dsets3[0], 0, vi_out=[0, 1, 2])
  recon_y = code_learner.predict(dsets3[1], 1, vi_out=[0, 1, 2])
  recon_z = code_learner.predict(dsets3[2], 2, vi_out=[0, 1, 2])

  true_ts = [np.array(tx).reshape(-1, 3) for tx in dsets3]
  recon_x = [np.array(recon_x[i]).reshape(-1, 3) for i in recon_x]
  recon_y = [np.array(recon_y[i]).reshape(-1, 3) for i in recon_y]
  recon_z = [np.array(recon_z[i]).reshape(-1, 3) for i in recon_z]

  if forecast:
    half1_dsets3 = []
    half2_dsets3 = []
    ht_length = t_length // 2
    n_steps = t_length - ht_length
    for ds in dsets3:
      half1_dsets3.append(ds[:, :ht_length])
      half2_dsets3.append(ds[:, ht_length:])

    future_x = [tx for i, tx in forecaster.predict(half1_dsets3[0], 0, [0, 1, 2], n_steps).items()]
    future_y = [tx for i, tx in forecaster.predict(half1_dsets3[0], 1, [0, 1, 2], n_steps).items()]
    future_z = [tx for i, tx in forecaster.predict(half1_dsets3[0], 2, [0, 1, 2], n_steps).items()]

    arr_reshape = lambda x: np.array(x).reshape(-1, 3)
    arr_reshape1 = lambda x: np.array(x[:, :25]).reshape(-1, 3)
    arr_reshape2 = lambda x: np.array(x[:, 25:]).reshape(-1, 3)
    cat_map = lambda x: np.r_[x[0], x[1]]

    true1_ts = list(map(arr_reshape, half1_dsets3))
    future1_x = list(map(arr_reshape1, future_x))
    future1_y = list(map(arr_reshape1, future_y))
    future1_z = list(map(arr_reshape1, future_z))

    true2_ts = list(map(arr_reshape, half2_dsets3))
    future2_x = list(map(arr_reshape2, future_x))
    future2_y = list(map(arr_reshape2, future_y))
    future2_z = list(map(arr_reshape2, future_z))

    future_x = list(map(arr_reshape, future_x))
    future_y = list(map(arr_reshape, future_y))
    future_z = list(map(arr_reshape, future_z))

    rearr_x = list(map(cat_map, zip(future1_x, future2_x)))
    rearr_y = list(map(cat_map, zip(future1_y, future2_y)))
    rearr_z = list(map(cat_map, zip(future1_z, future2_z)))
    true_rearr = list(map(cat_map, zip(true1_ts, true2_ts)))
    # future2_x = []
    # for i, tx in future_x.items():
    #   future2_x.append(op(tx[:, ht_length:]))
    # future2_x = []
    # for i, tx in future_x.items():
    #   future2_x.append(op(tx[:, ht_length:]))
    # future2_x = []
    # for i, tx in future_x.items():
    #   future2_x.append(op(tx[:, ht_length:]))
    # future2_x = [np.array(future_x[i][:, ht_length:]).reshape(-1, 3) for i in future_x]
    # future2_y = [np.array(future_y[i][:, ht_length:]).reshape(-1, 3) for i in future_y]
    # future2_z = [np.array(future_z[i][:, ht_length:]).reshape(-1, 3) for i in future_z]

    # future_x = [np.array(future_x[i]).reshape(-1, 3) for i in future_x]
    # future_y = [np.array(future_y[i]).reshape(-1, 3) for i in future_y]
    # future_z = [np.array(future_z[i]).reshape(-1, 3) for i in future_z]

  IPython.embed()

  plot_recon(true_ts, recon_x, 'xyz', title="X recon")
  plot_recon(true_ts, recon_y, 'xyz', title="Y recon")
  plot_recon(true_ts, recon_z, 'xyz', title="Z recon")

  pred_map = {
      'x': [future_x, future1_x, future2_x, rearr_x],
      'y': [future_y, future1_y, future2_y, rearr_y],
      'z': [future_z, future1_z, future2_z, rearr_z],
  }
  true_map = {'x': 0, 'y': 1, 'z': 2}
  pl_idx = 1
  # if forecast:
  for ip in 'xyz':
    for op in 'xyz':
      plt.subplot(3, 3, pl_idx)
      pl_idx += 1

      idx = true_map[op]
      l = op.upper()
      title = "%s to %s" % (ip.upper(), op.upper())

      tr = true_rearr[idx][:, 0]
      pr = pred_map[ip][-1][idx]
      plt.plot(tr, color='b', label=l + " True")
      plt.plot(pr[:, 0], color='r', ls='--', label=l + " Pred")
      plt.legend()
      plt.title(title)
  plt.show()
          # plot_simple(true_rearr[idx], pred_map[ip][-1][idx], op.upper(), lbl)
      # plot_recon(true2_ts[1:2], future2_x[1:2], 'yz', title="X future")
      # plot_recon(true2_ts[:1], future2_y[:1], 'xz', title="Y future")
      # plot_recon(true2_ts[:-1], future2_z[:-1], 'xy', title="Z future")

      # plot_recon(true1_ts[1:2], future1_x[1:2], 'yz', title="X future")
      # plot_recon(true1_ts[:1], future1_y[:1], 'xz', title="Y future")
      # plot_recon(true1_ts[:-1], future1_z[:-1], 'xy', title="Z future")




if __name__ == "__main__":
  test_lorenz_RNNMAE()
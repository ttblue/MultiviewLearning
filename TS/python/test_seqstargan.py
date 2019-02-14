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
import seq_star_gan as ssg
from synthetic import simple_systems as ss
import time_sync as tsync
import torch_utils as tu
import utils

import IPython


_PARTY_ROCK = "IN THE HOUSE TONIGHT"
_EVERYBODY = "HAVE A GOOD TIME"


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


def test_lorenz_SSG():
  visualize = False and MPL_AVAILABLE

  no_ds = False
  tmax = 100
  nt = 10000
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

  # Encoder -- RNN + MNN
  # Encoder params:
  # MNN:
  latent_size = 16
  l1_size = ntau
  rnn_size = 16

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

  # RNN
  input_size = rnn_size
  output_size = latent_size
  layer_units = [32, 64]
  layer_types, layer_args = tu.generate_layer_types_args(
    input_size, layer_units, output_size)
  activation = torch.nn.ReLU
  last_activation = torch.nn.Sigmoid
  use_vae = False
  post_en_config = tu.MNNConfig(
    input_size=input_size, output_size=output_size, layer_types=layer_types,
    layer_args=layer_args, activation=activation,
    last_activation=last_activation, use_vae=use_vae)

  encoder_funcs = [tu.RNNWrapper, tu.MultiLayerNN]
  encoder_config = [en_rnn_config, post_en_config]
  encoder_params = {
      i: {
          "layer_funcs": encoder_funcs,
          "layer_config": encoder_config,
          }
      for i in range(n_views)
  }

  # Decoder -- MNN + RNN:
  # Decoder params:
  # MNN
  input_size = latent_size
  output_size = rnn_size
  layer_units = [64, 32]
  layer_types, layer_args = tu.generate_layer_types_args(
      input_size, layer_units, output_size)
  activation = torch.nn.ReLU
  last_activation = torch.nn.Sigmoid
  use_vae = False
  pre_de_config = tu.MNNConfig(
    input_size=input_size, output_size=output_size, layer_types=layer_types,
    layer_args=layer_args, activation=activation,
    last_activation=last_activation, use_vae=use_vae)

  # RNN
  input_size = rnn_size
  hidden_size = n_views
  num_layers = 1
  cell_type = torch.nn.LSTM
  return_only_final = False
  return_only_hidden = True
  de_rnn_config = tu.RNNConfig(
      input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
      cell_type=cell_type, return_only_hidden=return_only_hidden,
      return_only_final=return_only_final)

  decoder_funcs = [tu.MultiLayerNN, tu.RNNWrapper]
  decoder_config = [pre_de_config, de_rnn_config]
  decoder_params = {
      i: {
          "layer_funcs": decoder_funcs,
          "layer_config": decoder_config,
          }
      for i in range(n_views)
  }

  # Classifier -- MNN + RNN
  # Classifier params:
  # MNN 
  input_size = latent_size
  output_size = rnn_size
  layer_units = [64, 32]
  layer_types, layer_args = tu.generate_layer_types_args(
      input_size, layer_units, output_size)
  activation = torch.nn.ReLU
  last_activation = torch.nn.Sigmoid
  use_vae = False
  pre_cla_config = tu.MNNConfig(
    input_size=input_size, output_size=output_size, layer_types=layer_types,
    layer_args=layer_args, activation=activation,
    last_activation=last_activation, use_vae=use_vae)

  # RNN
  input_size = latent_size  # rnn_size
  hidden_size = n_views
  num_layers = 1
  return_only_final = True
  return_only_hidden = True
  cla_rnn_config = tu.RNNConfig(
      input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
      cell_type=cell_type, return_only_hidden=return_only_hidden,
      return_only_final=return_only_final)

  # --
  # classifier_funcs = [tu.MultiLayerNN, tu.RNNWrapper]
  # classifier_config = [pre_cla_config, cla_rnn_config]
  classifier_funcs = [tu.RNNWrapper]
  classifier_config = [cla_rnn_config]
  classifier_params = {
      "layer_funcs": classifier_funcs,
      "layer_config": classifier_config,
  }
  use_cla = True
  cla_iter = 5

  # Generator and Discriminator:
  # Not here for now.
  gen_dis_iter = 10
  generator_params = None
  discriminator_params = None

  # Overall config
  ae_dis_alpha = 0.1
  use_gen_dis = False
  t_length = 50

  ae_itrs = [1]
  cla_itrs = [4] if use_cla else []
  gen_itrs = [1] if use_gen_dis else []
  dis_itrs = [1] if use_gen_dis else []
  order = ["ae", "cla"]
  t_scheduler_config = {
      "ae_itrs": ae_itrs,
      "cla_itrs": cla_itrs,
      "gen_itrs": gen_itrs,
      "dis_itrs": dis_itrs,
      "order": order,
  }

  enable_cuda = False
  lr = 1e-3
  batch_size = 60
  max_iters = 5000

  verbose = True

  config = ssg.SSGConfig(
      n_views=n_views,
      encoder_params=encoder_params,
      decoder_params=decoder_params,
      classifier_params=classifier_params,
      generator_params=generator_params,
      discriminator_params=discriminator_params,
      use_cla=use_cla,
      ae_dis_alpha=ae_dis_alpha,
      use_gen_dis=use_gen_dis,
      t_length=t_length,
      t_scheduler_config=t_scheduler_config,
      enable_cuda=enable_cuda,
      lr=lr,
      batch_size=batch_size,
      max_iters=max_iters,
      verbose=verbose)

#   rn = 100
#   gammak = 5e-2
#   sine = False
#   fgen_args = {"rn":rn, "gammak":gammak, "sine":sine}
#   fgen = grf.GaussianRandomFeatures(dim=ntau, **fgen_args)
# #   ffunc = fgen.computeRandomFeatures
  # ffunc = lambda x: x
  # nts = int(np.ceil(nt / t_length))
  # dsets = [
  #     utils.split_data(ds, n=nts, const_len=True)[0] for ds in [xts, yts, zts]]

  # tr_frac = 0.8
  # split_inds = [0, int(tr_frac * nts), nts]
  # split_dsets = [utils.split_data(ds, split_inds=split_inds)[0] for ds in dsets]
  # Tr_txs = utils.flatten([ffunc(ds[0]) for ds in split_dsets])
  # Tr_vis = utils.flatten([[i] * len(ds[0]) for i, ds in enumerate(split_dsets)])
  # Te_txs = utils.flatten([ffunc(ds[1]) for ds in split_dsets])
  # Te_vis = utils.flatten([[i] * len(ds[1]) for i, ds in enumerate(split_dsets)])
  # Tr_dset = dataset.MultimodalAsyncTimeSeriesDataset(
  #     Tr_txs, Tr_vis, t_length, shuffle=True, synced=False)
  # Te_dset = dataset.MultimodalAsyncTimeSeriesDataset(
  #     Te_txs, Te_vis, t_length, shuffle=True, synced=False)
  # all_dsets = [d1 + d2 for d1, d2 in zip(Tvs_tr, Tvs_te)]

  # IPython.embed()
  # mae = reload(mae)
  ffunc = lambda x: x
  nts = int(np.ceil(nt / t_length))
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

  Tr_dset = dataset.MultimodalAsyncTimeSeriesDataset(
      Tr_txs, Tr_vis, t_length, shuffle=True, synced=False)
  Te_dset = dataset.MultimodalAsyncTimeSeriesDataset(
      Te_txs, Te_vis, t_length, shuffle=True, synced=False)

  code_learner = ssg.SeqStarGAN(config)
  code_learner.fit(Tr_dset)

  viouts = {
    vo: {vi: np.ones(Te_dset.v_nts[vi]) * vo for vi in range(n_views)}
    for vo in range(n_views)
  }
  preds = {}
  for vo in viouts: 
    preds[vo] = code_learner.predict(Te_dset.v_txs, vi_outs=viouts[vo])
  # Doing the long-winded way for easy copying into terminal
  # for vo in range(n_views):
  #   viouts[vo] = {}
  #   for vi in Te_dset.views: 
  #     viouts[vo][vi] = np.ones(Te_dset.v_nts[vi]) * vo 
  IPython.embed()
  plot_type = "all_to_one"
  pvi = 1  # Input view
  pvo = 2  # Output view
  pfi = 0  # Index of feature to plot
  L = {0:'X', 1: 'Y', 2: 'Z'}
  for i in range(5):
    l = L[pvi]
    l2 = L[pvo]
    if plot_type == "original":
      plt.plot(Te_dset.v_txs[0][i][:, pfi], color='b', label='X')
      plt.plot(Te_dset.v_txs[1][i][:, pfi], color='r', label='Y')
      plt.plot(Te_dset.v_txs[2][i][:, pfi], color='g', label='Z')
    elif plot_type == "single_compare":
      plt.plot(Te_dset.v_txs[pvi][i][:, pfi], color='b', label=l2)
      plt.plot(
          preds[pvo][pvi][i][:, pfi], color='r', label='P%s to %s' % (l, l2))
    elif plot_type == "all_to_one":
      plt.plot(Te_dset.v_txs[pvo][i][:, pfi], color='k', label=l2)
      plt.plot(
          preds[pvo][0][i][:, pfi], color='b', ls="--", label="PX to %s" % l2)
      plt.plot(
          preds[pvo][1][i][:, pfi], color='r', ls="--", label="PY to %s" % l2)
      plt.plot(
          preds[pvo][2][i][:, pfi], color='g', ls="--", label="PZ to %s" % l2)
    elif plot_type == "all_from_one":
      plt.plot(Te_dset.v_txs[0][i][:, pfi], color='b', label='X')
      plt.plot(Te_dset.v_txs[1][i][:, pfi], color='r', label='Y')
      plt.plot(Te_dset.v_txs[2][i][:, pfi], color='g', label='Z')
      plt.plot(
          preds[0][pvi][i][:, pfi], color='b', ls="--", label="P%s to X" % l)
      plt.plot(
          preds[1][pvi][i][:, pfi], color='r', ls="--", label="P%s to Y" % l)
      plt.plot(
          preds[2][pvi][i][:, pfi], color='g', ls="--", label="P%s to Z" % l)

    plt.legend()
    plt.show()

  # recon_x = code_learner.predict(Tvs_te[0], 0, vi_out=[1, 2])
  # recon_y = code_learner.predict(Tvs_te[1], 1, vi_out=[0, 2])
  # recon_z = code_learner.predict(Tvs_te[2], 2, vi_out=[0, 1])

  # recon_x = code_learner.predict(all_dsets[0], 0, vi_out=[1, 2])
  # recon_y = code_learner.predict(all_dsets[1], 1, vi_out=[0, 2])
  # recon_z = code_learner.predict(all_dsets[2], 2, vi_out=[0, 1])

  # true_ts = [np.array(tx).reshape(-1, 3) for tx in all_dsets]
  # recon_x = [np.array(tx).reshape(-1, 3) for tx in recon_x]
  # recon_y = [np.array(tx).reshape(-1, 3) for tx in recon_y]
  # recon_z = [np.array(tx).reshape(-1, 3) for tx in recon_z]

  # ttx = [true_ts[1], true_ts[2]]
  # tty = [true_ts[0], true_ts[2]]
  # ttz = [true_ts[0], true_ts[1]]
  # plot_recon(ttx, recon_x, 'yz', title="X recon")
  # plot_recon(tty, recon_y, 'xz', title="Y recon")
  # plot_recon(ttz, recon_z, 'xy', title="Z recon")


if __name__ == "__main__":
  test_lorenz_SSG()
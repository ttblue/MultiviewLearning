from __future__ import print_function, division

import gc
import glob
import sys
import os
import time

import multiprocessing

import collections
import numpy as np

import dataset
import featurize_pig_data as fpd
import lstm
# import multi_task_learning as mtl
# import time_series_ml as tsml
import L21_block_regression as lbr
import math_utils as mu
import nn_utils as nnu
import time_series_utils as tsu
import utils

import sklearn.svm as sksvm

import IPython


_PLOTTING = True
try:
  import matplotlib.pyplot as plt, matplotlib.cm as cm
  from mpl_toolkits.mplot3d import Axes3D
except ImportError:
  _PLOTTING = False
  pass

FREQUENCY = 250
VERBOSE = tsu.VERBOSE
DATA_DIR = os.getenv("PIG_DATA_DIR")
SAVE_DIR = os.getenv("PIG_FEATURES_DIR")

np.set_printoptions(suppress=True, precision=3)

################################################################################

def mt_krc_pigs_slow():
  # For now:
  # np.random.seed(1)

  all_data, _ = fpd.load_pig_features_and_labels_numpy(num_pigs=-1, ds=5, ws=30)

  pig_ids = all_data.keys()
  num_train, num_test = 1, 1
  # rand_inds = np.random.permutation(len(pig_ids))[:(num_train + num_test)]
  # train_inds = [pig_ids[i] for i in rand_inds[:num_train]]
  # test_inds = [pig_ids[i] for i in rand_inds[num_train:]]
  train_inds = [33]
  test_inds = [34]
  print("Train inds: %s\nTest inds: %s"%(train_inds, test_inds))

  binary_class = 1
  training_tasks = []
  channels = len(all_data[train_inds[0]]["features"])
  labels = np.array([y for idx in train_inds for y in all_data[idx]["labels"]])
  labels = (labels == binary_class).astype(int)
  for channel in xrange(channels):
    features = np.vstack([all_data[idx]["features"][channel] for idx in train_inds])
    training_tasks.append((features, labels))

  test_tasks = []
  for channel in xrange(channels):
    features = np.vstack([all_data[idx]["features"][channel] for idx in test_inds])
    test_tasks.append((features, ))

  IPython.embed()
  d_in = training_tasks[0][0].shape[1]
  d_out = 100
  a = 1.
  feature_gen = mu.rbf_fourierfeatures(d_in, d_out, a)
  omega = mtl.create_independent_omega(T=channels, lambda_s=1.)

  pred_y = mtl.mt_krc(training_tasks, test_tasks, omega, feature_gen)
  test_labels = np.array([y for idx in test_inds for y in all_data[idx]["labels"]])

  IPython.embed()

################################################################################

def cluster_slow_pigs(num_pigs=4, ws=30):
  all_data, _ = fpd.load_pig_features_and_labels_numpy(num_pigs=num_pigs, ds=5, ws=ws, category="both")
  class_names = [
      "Ground_Truth", "EKG", "Art_pressure_MILLAR", "Art_pressure_Fluid_Filled",
      "Pulmonary_pressure", "CVP", "Plethysmograph", "CCO", "SVO2", "SPO2",
      "Airway_pressure", "Vigeleo_SVV"]

  pig_ids = all_data.keys()
  # num_pigs = 4
  # rand_inds = np.random.permutation(len(pig_ids))[:num_pigs]
  # pig_inds = [pig_ids[idx] for idx in rand_inds]
  print("Pig inds: %s"%pig_ids)

  channels = len(all_data[pig_ids[0]]["features"])
  labels = np.array([y for idx in pig_ids for y in all_data[idx]["labels"]])
  all_features = []
  for channel in xrange(channels):
    features = np.vstack([all_data[idx]["features"][channel] for idx in pig_ids])
    all_features.append(features)

  mi_matrix = tsml.cluster_windows(all_features, labels, class_names)

################################################################################

def select_random_windows(data, num_per_window=10, num_per_pig=10, channel=None, pos_label=None):
  windows = {}
  labels = {}

  half_window_len = int(num_per_window/2)
  for idx in data:
    pig_features = data[idx]["features"]
    pig_labels = data[idx]["labels"]

    window_labels = [
        pig_labels[i+half_window_len]
        for i in range(0, pig_labels.shape[0]-num_per_window, num_per_window)
    ]
    if pos_label is None:
      choice_inds = np.random.permutation(len(window_labels))[:num_per_pig]
    else:
      window_labels = (np.array(window_labels) == pos_label).astype(int)
      # choice_inds = np.random.permutation(len(window_labels))[:num_per_pig]
      half_num_per_pig = int(num_per_pig/2)
      pos_inds = np.nonzero(window_labels)[0]
      neg_inds = np.nonzero(window_labels == 0)[0]
      np.random.shuffle(pos_inds)
      np.random.shuffle(neg_inds)
      choice_inds = pos_inds[:half_num_per_pig].tolist()
      choice_inds.extend(neg_inds[:num_per_pig-len(choice_inds)].tolist())

    if channel is None:
      window_split = [
          [f[i:i+num_per_window] for i in range(0, f.shape[0]-num_per_window, num_per_window)]
          for f in pig_features
      ]
      pig_windows = [
          [channel_windows[i] for i in choice_inds]
          for channel_windows in window_split
      ]
    elif isinstance(channel, list):
      window_split = [
          [pig_features[c][i:i+num_per_window] 
           for i in range(0, pig_features[c].shape[0]-num_per_window, num_per_window)]
          for c in channel
      ]
      pig_windows = [
          [channel_windows[i] for i in choice_inds]
          for channel_windows in window_split
      ]
    else:
      window_split = [
          pig_features[channel][i:i+num_per_window] 
          for i in range(0, pig_features[channel].shape[0]-num_per_window, num_per_window)
      ]
      pig_windows = [window_split[i] for i in choice_inds]

    windows[idx] = pig_windows
    labels[idx] = [window_labels[i] for i in choice_inds]

  return windows, labels


def pred_nn_slow_pigs(ws=30):
  # Only using a single channel for now.
  channel = 6
  pos_label = None
  num_per_window = 30
  num_per_pig = 50

  num_train_pigs = -1
  num_test_pigs = -1

  train_data, _ = fpd.load_pig_features_and_labels_numpy(num_pigs=num_train_pigs, ds=5, ws=ws, category="train")
  test_data, _ = fpd.load_pig_features_and_labels_numpy(num_pigs=num_test_pigs, ds=5, ws=ws, category="test")

  train_ids = train_data.keys()
  train_ts = [train_data[idx]["features"][channel] for idx in train_ids]
  train_labels = [np.array(train_data[idx]["labels"]) for idx in train_ids]
  if pos_label is not None:
    train_labels = [(lbls == pos_label).astype(int) for lbls in train_labels]

  train_windows, train_window_labels = select_random_windows(
      train_data, num_per_window=num_per_window, num_per_pig=num_per_pig,
      channel=channel, pos_label=pos_label)
  test_windows, test_window_labels = select_random_windows(
      test_data, num_per_window=num_per_window, num_per_pig=num_per_pig,
      channel=channel, pos_label=pos_label)

  # train_pred_labels = {}
  # train_pred_acc = {
  # pig_idx = 1
  # for idx in train_windows:
  #   t1 = time.time()
  #   train_pred_labels[idx] = tsml.predict_nn_dtw(train_ts, train_labels, train_windows[idx])
  #   train_pred_acc[idx] = np.sum(np.array(train_pred_labels[idx]) == np.array(train_window_labels[idx]))/num_per_pig
  #   ttaken = time.time() - t1
  #   print ("Train pig number: %i, ID: %i. Accuracy: %.2f. Time taken: %.2fs"%
  #           (pig_idx, idx, train_pred_acc[idx], ttaken))
  #   pig_idx += 1
  #   if pig_idx > 2:
  #     break
  # final_train_acc = np.mean(train_pred_acc.values())
  # print("Final training accuracy over %i pigs: %.2f"%
  #       (len(train_windows), final_train_acc))

  # print()

  test_pred_labels = {}
  test_pred_acc = {}
  pig_idx = 1
  for idx in test_windows:
    t1 = time.time()
    test_pred_labels[idx] = tsml.predict_nn_dtw(train_ts, train_labels, test_windows[idx])
    test_pred_acc[idx] = np.sum(np.array(test_pred_labels[idx]) == np.array(test_window_labels[idx]))/num_per_pig
    ttaken = time.time() - t1
    print ("Test pig number: %i, ID: %i. Accuracy: %.2f. Time taken: %.2fs"%
           (pig_idx, idx, test_pred_acc[idx], ttaken))
    pig_idx += 1
  final_test_acc = np.mean(test_pred_acc.values())
  print("Final test accuracy over %i pigs: %.2f"%
        (len(test_windows), final_test_acc))
  IPython.embed()

################################################################################

def pred_lstm_slow_pigs(ws=5):
  # Only using a single channel for now.
  channel = 6
  allowed_labels = [0, 1, 2]
  pos_label = None
  if pos_label not in allowed_labels:
    pos_label is None

  num_train_pigs = -1
  num_test_pigs = -1

  train_data, _ = fpd.load_pig_features_and_labels_numpy(num_pigs=num_train_pigs, ds=5, ws=ws, category="train")
  test_data, _ = fpd.oad_slow_pig_features_and_labels(num_pigs=num_test_pigs, ds=5, ws=ws, category="test")

  train_ids = train_data.keys()
  train_ts = [train_data[idx]["features"][channel] for idx in train_ids]
  train_labels = [np.array(train_data[idx]["labels"]) for idx in train_ids]
  if allowed_labels is not None:
    valid_inds = [[l in allowed_labels for l in lbls] for lbls in train_labels]
    train_ts = [ts[vi] for ts, vi in zip(train_ts, valid_inds)]
    train_labels = [lbls[vi] for lbls, vi in zip(train_labels, valid_inds)]
  if pos_label is not None:
    train_labels = [(lbls == pos_label).astype(int) for lbls in train_labels]

  test_ids = test_data.keys()
  test_ts = [test_data[idx]["features"][channel] for idx in test_ids]
  test_labels = [np.array(test_data[idx]["labels"]) for idx in test_ids]
  if allowed_labels is not None:
    valid_inds = [[l in allowed_labels for l in lbls] for lbls in test_labels]
    test_ts = [ts[vi] for ts, vi in zip(test_ts, valid_inds)]
    test_labels = [lbls[vi] for lbls, vi in zip(test_labels, valid_inds)]
  if pos_label is not None:
    test_labels = [(lbls == pos_label).astype(int) for lbls in test_labels]

  dset_train = dataset.TimeseriesDataset(train_ts, train_labels)
  dset_test = dataset.TimeseriesDataset(test_ts, test_labels)
  dset_test, dset_validate = dset_test.split([0.8, 0.2])
  # IPython.embed()
  # LSTM Config:
  if allowed_labels is not None:
    num_classes = len(allowed_labels) if pos_label is None else 2
  else:
    num_classes = 6 if pos_label is None else 2
  num_features = train_ts[0].shape[1]

  hidden_size = 600
  forget_bias = 0.5
  use_sru = False
  use_dynamic_rnn = True
  keep_prob = 1.0
  num_layers = 2
  init_scale = 0.1
  max_grad_norm = 5
  max_epochs = 5 ##
  max_max_epochs = 50 ##
  init_lr = 1.0
  lr_decay = 0.9
  batch_size = 20
  num_steps = 10
  verbose = True

  config = lstm.LSTMConfig(
      num_classes=num_classes, num_features=num_features, use_sru=use_sru,
      use_dynamic_rnn=use_dynamic_rnn, hidden_size=hidden_size,
      forget_bias=forget_bias, keep_prob=keep_prob, num_layers=num_layers,
      init_scale=init_scale, max_grad_norm=max_grad_norm, max_epochs=max_epochs,
      max_max_epochs=max_max_epochs, init_lr=init_lr, lr_decay=lr_decay,
      batch_size=batch_size, num_steps=num_steps, verbose=verbose)
  lstm_classifier = lstm.LSTM(config)
  lstm_classifier.fit(dset=dset_train, dset_v=dset_validate)
  IPython.embed()


def pred_lstm_slow_pigs_raw():
  np.random.seed(0)
  num_pigs = -1
  
  ds = 1
  ds_factor = 25
  columns = [0, 4, 6, 7]
  allowed_labels = [0, 1, 2]
  pos_label = None
  if pos_label not in allowed_labels:
    pos_label is None

  all_data, _ = fpd.load_pig_features_and_labels_numpy(
      num_pigs=num_pigs, ds=ds, ds_factor=ds_factor, feature_columns=columns,
      save_new=False)

  pig_ids = all_data.keys()
  all_ts = [all_data[idx]["features"] for idx in pig_ids]
  all_labels = [np.array(all_data[idx]["labels"]) for idx in pig_ids]
  if allowed_labels is not None:
    valid_inds = [
        np.array([l in allowed_labels for l in lbls]) for lbls in all_labels]
    all_ts = [ts[vi] for ts, vi in zip(all_ts, valid_inds)]
    all_labels = [lbls[vi] for lbls, vi in zip(all_labels, valid_inds)]
  if pos_label is not None:
    all_labels = [(lbls == pos_label).astype(int) for lbls in all_labels]

  all_dsets = dataset.TimeseriesDataset(all_ts, all_labels)
  # ttv_split = [0.6, 0.2, 0.2]
  # dset_train, dset_test, dset_validation = all_dsets.split(ttv_split)
  ttv_split = [0.8, 0.2]
  dset_train, dset_validation = all_dsets.split(ttv_split)
  dset_validation.toggle_shuffle(False)
  dset_train.shift_and_scale()
  dset_validation.shift_and_scale(dset_train.mu, dset_train.sigma)
  # IPython.embed()

  # LSTM Config:
  if allowed_labels is not None:
    num_classes = len(allowed_labels) if pos_label is None else 2
  else:
    num_classes = 6 if pos_label is None else 2
  num_features = all_ts[0].shape[1]

  use_sru = False
  use_dynamic_rnn = True

  hidden_size = 100
  forget_bias = 1.0
  keep_prob = .5
  num_layers = 1

  batch_size = 20
  num_steps = 50
  optimizer = "Adam"
  max_epochs = 100
  max_max_epochs = 100
  init_lr = 0.0001
  lr_decay = 1.0
  max_grad_norm = 5
  initializer = "xavier"
  init_scale = 0.1

  summary_log_path = None
  verbose = True

  config = lstm.LSTMConfig(
      num_classes=num_classes, num_features=num_features, use_sru=use_sru,
      use_dynamic_rnn=use_dynamic_rnn, hidden_size=hidden_size,
      forget_bias=forget_bias, keep_prob=keep_prob, num_layers=num_layers,
      batch_size=batch_size, num_steps=num_steps, optimizer=optimizer,
      max_epochs=max_epochs, max_max_epochs=max_max_epochs, init_lr=init_lr,
      lr_decay=lr_decay, max_grad_norm=max_grad_norm, initializer=initializer,
      init_scale=init_scale, summary_log_path=summary_log_path, verbose=verbose)
  lstm_classifier = lstm.LSTM(config)
  lstm_classifier.fit(dset=dset_train, dset_v=dset_validation)
  IPython.embed()


def pred_L21reg_slow_pigs_raw():
  np.random.seed(0)
  num_pigs = -1

  ds = 10
  ds_factor = 1
  columns = [0, 4, 6, 7]
  valid_labels = [0, 1, 2]

  all_data, _ = fpd.load_pig_features_and_labels_numpy(
      num_pigs=num_pigs, ds=ds, ds_factor=ds_factor, feature_columns=columns,
      save_new=False, valid_labels=valid_labels, use_derivs=True)

  pig_ids = all_data.keys()
  all_xs = [all_data[idx]["features"] for idx in pig_ids]
  all_dxs = [all_data[idx]["derivs"] for idx in pig_ids]
  all_ys = [np.array(all_data[idx]["labels"]) for idx in pig_ids]

  dt = fpd.FREQUENCY / (ds * ds_factor)
  tau = 15
  tde_d = 4
  # IPython.embed()
  # Using only pleth
  all_xs = [
      tsu.compute_time_delay_embedding(np.squeeze(xs[:, -1]), dt, tau, d=tde_d)
      for xs in all_xs
  ]
  all_dxs = [dxs[:xs.shape[0], -1] for xs, dxs in zip(all_xs, all_dxs)]
  all_ys = [ys[:xs.shape[0]] for xs, ys in zip(all_xs, all_ys)]

  all_dsets = dataset.DynamicalSystemDataset(
      all_xs, all_dxs, all_ys, shift_scale=True, tau=13)
  trdset, tedset = all_dsets.split([0.8, 0.2])

  sample_length_s = 30
  sample_length = int(dt * sample_length_s)
  # IPython.embed()

  tr_xs, tr_dxs, tr_ys = trdset.get_samples(sample_length, -1, channels=None)
  te_xs, te_dxs, te_ys = tedset.get_samples(sample_length, -1, channels=None)

  degree = 3
  tr_pxs = [lbr.generate_polynomials(xs, degree) for xs in tr_xs]
  te_pxs = [lbr.generate_polynomials(xs, degree) for xs in te_xs]

  IPython.embed()

  U0 = lbr.L21_block_regression(tr_dxs + te_dxs, tr_pxs + te_pxs, 300, max_iterations=20)
  tr_f = U0[:len(tr_pxs)]
  te_f = U0[len(tr_pxs):]

  IPython.embed()

  # tr_f = U0[:len(tr_pxs)]
  # te_f = U0[len(tr_pxs):]

  tr_wys = [collections.Counter(y).most_common(1)[0][0] for y in tr_ys]
  te_wys = [collections.Counter(y).most_common(1)[0][0] for y in te_ys]

  classifier = sksvm.SVC()
  classifier.fit(tr_f, tr_wys)

  pred_y = classifier.predict(te_f)
  acc = (pred_y == te_wys).sum() / pred_y.shape[0]

  # key = all_data.keys()[0]
  # X = all_data[key]["features"][:, [0]]
  # PX = lbr.generate_polynomials(X)
  # PXs = np.split(PX, [int(PX.shape[0]/2)])
  # DX = all_data[key]["derivs"][:, [0]]
  # DXs = np.split(DX, [int(DX.shape[0]/2)])
  IPython.embed()

  plt.plot(tr_accs, color='b', label="Training Accuracy")
  plt.plot(v_accs, color='r', label="Validation Accuracy")
  plt.title("SRU")
  plt.legend()
  plt.show()

  # pig_ids = all_data.keys()
  # all_ts = [all_data[idx]["features"] for idx in pig_ids]
  # all_labels = [np.array(all_data[idx]["labels"]) for idx in pig_ids]
  # if allowed_labels is not None:
  #   valid_inds = [
  #       np.array([l in allowed_labels for l in lbls]) for lbls in all_labels]
  #   all_ts = [ts[vi] for ts, vi in zip(all_ts, valid_inds)]
  #   all_labels = [lbls[vi] for lbls, vi in zip(all_labels, valid_inds)]
  # if pos_label is not None:
  #   all_labels = [(lbls == pos_label).astype(int) for lbls in all_labels]


################################################################################

def pred_nn_tde_slow_pigs_raw():
  # np.random.seed(0)
  num_pigs = -1
  ds = 1
  ds_factor = 10
  columns = [0, 7]
  valid_labels = [0, 1, 2, 3, 4, 5, 6]
  wsize = 30 # seconds
  dt = 1 / (fpd.FREQUENCY / (ds * ds_factor))

  all_data, _ = fpd.load_pig_features_and_labels_numpy(
      num_pigs=num_pigs, ds=ds, ds_factor=ds_factor, feature_columns=columns,
      save_new=False, valid_labels=valid_labels, use_derivs=False)

  pig_ids = all_data.keys()
  all_ts = [all_data[idx]["features"] for idx in pig_ids]
  all_labels = [np.array(all_data[idx]["labels"]) for idx in pig_ids]
  
  all_dsets = dataset.TimeseriesDataset(all_ts, all_labels)
  # ttv_split = [0.6, 0.2, 0.2]
  # dset_train, dset_test, dset_validation = all_dsets.split(ttv_split)
  ttv_split = [0.8, 0.2]
  dset_train, dset_test = all_dsets.split(ttv_split)
  dset_test.toggle_shuffle(False)
  dset_train.shift_and_scale()
  dset_test.shift_and_scale(dset_train.mu, dset_train.sigma)

  # recompute_taus = True
  # taus_file = os.path.join(SAVE_DIR, "waveform/slow/params", "taus_ds_%i_columns_%s.npy"%(ds, columns))
  # if recompute_taus or not os.path.exists(taus_file):
  #   if not recompute_taus and not os.path.exists(os.path.dirname(taus_file)):
  #     os.makedirs(os.path.dirname(taus_file))

  #   tau_s_to_search = 0.75
  #   M = int(tau_s_to_search / dt)

  # IPython.embed()
  #   all_taus = {}
  #   for i, channel in enumerate(columns[1:]):
  #     taus = []
  #     for xs in dset_test.xs:
  #       taus.append(tsu.compute_tau(xs[:, i] , M=M, show=False))
  #     all_taus[channel] = int(np.mean(taus))

  #   if not recompute_taus:
  #     np.save(taus_file, all_taus)

  # else:
  #   all_taus = np.load(taus_file).tolist()

  # print(all_taus)
  channel = 7
  tde_d = 4
  all_taus = {}
  all_taus[channel] = 13
  wlen = int(wsize / dt)
  half_wlen = int(wlen / 2)
  train_tde_windows = [
      tsu.compute_time_delay_embedding(xs, dt, all_taus[channel], d=tde_d)
      for xs in dset_train.xs]
  train_tde_windows = [
      np.split(window, np.arange(wlen, window.shape[0], wlen)[:-1])
      for window in train_tde_windows]
  train_labels = [
      [ys[i] for i in 
          (np.arange(0, ys.shape[0] - (tde_d - 1) * all_taus[channel], wlen)[:-1] + half_wlen)]
      for ys in dset_train.ys
  ]

  test_tde_windows = [
      tsu.compute_time_delay_embedding(xs, dt, all_taus[channel], d=tde_d)
      for xs in dset_test.xs]
  test_tde_windows = [
      np.split(window, np.arange(wlen, window.shape[0], wlen)[:-1])
      for window in test_tde_windows]
  test_labels = [
      [ys[i] for i in
        (np.arange(0, ys.shape[0] - (tde_d - 1) * all_taus[channel], wlen)[:-1] + half_wlen)]
      for ys in dset_test.ys
  ] 

  del dset_train, dset_test, all_dsets, all_data


  ############### TEMP
  # tw = train_tde_windows[1][15]
  # tw_new = train_tde_windows[1][16]
  # tw2 = train_tde_windows[00][150]

  nr = 1000
  sample_windows = []
  num_per_tw = 10
  for tws in train_tde_windows:
    sample_windows.extend(
        [tws[i] for i in np.random.permutation(len(tws))[:num_per_tw]])
  bandwidth = nnu._compute_kernel_bandwidth(sample_windows, c=1.0)
  dim = tde_d
  feature_gen = mu.rbf_fourierfeatures(dim, nr, bandwidth)

  # # tw3 = nnu.one_step_wn_forecast(tw, tw, feature_gen, dr="forward")
  # # tw3 = nnu.one_step_wn_forecast_RBF(tw, tw, bandwidth*10, dr="forward")
  # # tw4 = nnu.one_step_wn_forecast(tw, tw2, feature_gen, dr="forward")
  # twk2 = nnu.k_step_wn_forecast_RBF(np.atleast_2d(tw[-1]), tw, bandwidth, k=tw.shape[0]-1, dr="forward")
  # # twk = nnu.k_step_wn_forecast(np.atleast_2d(tw[0]), tw, feature_gen, k=tw.shape[0]-1, dr="forward")
  # # tw5 = np.concatenate(twk['forward'], axis=0)
  # tw6 = np.concatenate(twk2['forward'], axis=0)



  # # plot_td_attractor(tw)
  # IPython.embed()
  dr = "forward"
  leafsize = 10
  print("Computing KD Trees")
  train_kdtrees = [
      [nnu.ForecasterKDTree(twin, leafsize, dr) for twin in twindows]
      for twindows in train_tde_windows]

  bandwidth = 100#21571441786008  # As computed previously.
  n_jobs = 22
  nn = 1
  nw = 10
  num_steps = 10
  all_pred_labels = []
  all_pred_inds = []
  feature_gen = None
  first = True
  forecast_type = "knn"
  num_test = len(test_tde_windows)
  errors = 0
  t_start = time.time()
  print("Computing NN predictions:")
  for test_i, test_windows in enumerate(test_tde_windows):
    print("Test time series %i out of %i."%(test_i + 1, num_test))
    pred_labels = []
    pred_inds = []
    num_windows = len(test_windows)
    for test_wi, tw in enumerate(test_windows):
      # if test_labels[test_i][test_wi] == 0: continue
      t1 = time.time()
      print("\n\tTS %i: Window %i out of %i."%(test_i + 1, test_wi + 1, num_windows))#, end='\r')
      sys.stdout.flush()
      gc.collect()
      all_train_windows = train_kdtrees if first else None
      first = False

      try:
        ts_inds, w_inds, _ = nnu.find_nearest_windows_forecast_dist(
            tw, all_train_windows, num_steps, nw=nw, dr=dr,
            forecast_type=forecast_type, nn=10, n_jobs=n_jobs)

        nn_labels = [
            train_labels[ts_idx][w_idx] for ts_idx, w_idx in zip(ts_inds, w_inds)]
        pred_label = collections.Counter(nn_labels).most_common(1)[0][0]
        # nn_inds = [
        #     (ts_idx, w_idx) for ts_idx, w_idx in zip(ts_inds, w_inds)
        #     if train_labels[ts_idx][w_idx] == pred_label]

        # print("\tTS_idx: %i\t w_idx: %i"%(ts_idx, w_idx))
        pred_inds.append((ts_inds, w_inds))
        print("\t%s"%(nn_labels))
        print("\t%s"%(zip(ts_inds, w_inds)))
      except:
        pred_inds.append((None, None))
        errors += 1
        pred_label = -1
      # if pred_label == test_labels[test_i][test_wi]:
      #   tew = tw
      #   tw_nn = np.zeros(tw.shape)
      #   tw6 = {}
      #   ii = 0
      #   for ts_idx, w_idx in zip(ts_inds, w_inds):
      #     print('A')
      #     trw = train_kdtrees[ts_idx][w_idx]
      #     twk = nnu.k_step_wn_forecast(np.atleast_2d(tew[0]), trw, nn, k=tew.shape[0]-1, forecast_type="knn", dr="forward")
      #     tw6[ii] = np.concatenate([np.atleast_2d(tew[0]).tolist()] + twk['forward'], axis=0)
      #     ii += 1
      #     # IPython.embed()
      #   tw_nn = np.mean(np.array(tw6.values()), axis=0)
      #   IPython.embed()
        # tw3 = np.r_[np.atleast_2d(tew[0]).tolist(), nnu.one_step_wn_forecast_RBF(tew[:-1], trw, bandwidth, dr="forward")]
        # pass
      print("\tPred: %i\t Actual: %i"%(
          pred_label, test_labels[test_i][test_wi]))
      print("\tTime taken: %.2f"%(time.time() - t1))
      pred_labels.append(pred_label)


    # print("\tWindow %i."%(test_wi + 1))
    all_pred_labels.append(pred_labels)
    all_pred_inds.append(pred_inds)

  try:
    print("\n\nTotal time taken: %.2f"%(time.time() - t_start))
    np.save('tst_results.npy', [all_pred_labels, test_labels, all_pred_inds])
  except:
    pass
  IPython.embed()
  try:
    import sklearn.metrics as sm
    pls = np.concatenate([np.array(pl) for pl in all_pred_labels])
    tls = np.concatenate([np.array(tl) for tl in test_labels])
    cm = sm.confusion_matrix(tls, pls).astype(float)
  except:
    pass
  IPython.embed()



def plot_td_attractor(tw):
  plt.set_cmap('RdBu')
  colors = cm.RdBu(np.linspace(0, 1, tw.shape[0]))

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter3D(tw[:, 0], tw[:, 1], tw[:, 2], color=colors)
  plt.show()


def plt_ts(tws, labels=None):
  colors = ['b', 'r', 'g']
  if labels is None:
    labels = ["%i"%(i+1) for i in xrange(len(tws))]
  for i, tw, lbl in zip(xrange(len(tws)), tws, labels):
    plt.plot(tw[:, 0], color=colors[i%3], label=lbl)
  plt.legend()
  plt.show()

if __name__ == "__main__":
  # mt_krc_pigs_slow()
  # cluster_slow_pigs(10)
  # pred_nn_slow_pigs(ws=5)
  # pred_lstm_slow_pigs_raw()
  # for j in range(1, 11):
  #    cluster_slow_pigs(j)
  # pred_L21reg_slow_pigs_raw()
  expt_type = "knn"
  if len(sys.argv) > 1:
   try:
     expt_num = int(sys.argv[1])
     if expt_num == 1:
       expt_type = "lstm"
     elif expt_num == 2:
       expt_type = "l21"
   except:
     pass

  if expt_type == "lstm":
    pred_lstm_slow_pigs_raw()
  elif expt_type == "knn":  
    pred_nn_tde_slow_pigs_raw()
  elif expt_type == "l21":
    pred_L21reg_slow_pigs_raw()
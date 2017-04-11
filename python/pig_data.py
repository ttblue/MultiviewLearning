from __future__ import print_function, division

import glob
import sys
import os
import time

import multiprocessing

import numpy as np

import dataset
import featurize_pig_data as fpd
import lstm
import multi_task_learning as mtl
# import time_series_ml as tsml
import time_series_utils as tsu
import utils

import IPython

FREQUENCY = 250
VERBOSE = tsu.VERBOSE
DATA_DIR = os.getenv("PIG_DATA_DIR")
SAVE_DIR = os.getenv("PIG_FEATURES_DIR")

np.set_printoptions(suppress=True, precision=3)

################################################################################

def rbf_fourierfeatures(d_in, d_out, a):
  # Returns a function handle to compute random fourier features.
  W = np.random.normal(0., 1., (d_in, d_out))
  h = np.random.uniform(0., 2 * np.pi, (1, d_out))
  def rbf_ff(x):
    ff = np.cos((1 / a) * x.dot(W) + h) / np.sqrt(d_out) * np.sqrt(2)
    return ff
  return rbf_ff


def mt_krc_pigs_slow():
  # For now:
  # np.random.seed(1)

  all_data, _ = fpd.load_slow_pig_features_and_labels(num_pigs=-1, ds=5, ws=30)

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
  feature_gen = rbf_fourierfeatures(d_in, d_out, a)
  omega = mtl.create_independent_omega(T=channels, lambda_s=1.)

  pred_y = mtl.mt_krc(training_tasks, test_tasks, omega, feature_gen)
  test_labels = np.array([y for idx in test_inds for y in all_data[idx]["labels"]])

  IPython.embed()

################################################################################

def cluster_slow_pigs(num_pigs=4, ws=30):
  all_data, _ = fpd.load_slow_pig_features_and_labels(num_pigs=num_pigs, ds=5, ws=ws, category="both")
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

  train_data, _ = fpd.load_slow_pig_features_and_labels(num_pigs=num_train_pigs, ds=5, ws=ws, category="train")
  test_data, _ = fpd.load_slow_pig_features_and_labels(num_pigs=num_test_pigs, ds=5, ws=ws, category="test")

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

  train_data, _ = fpd.load_slow_pig_features_and_labels(num_pigs=num_train_pigs, ds=5, ws=ws, category="train")
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
  num_pigs = -1
  ds = 1
  ds_factor = 25
  columns = [0, 6, 7, 11]
  allowed_labels = [0, 1, 2]
  pos_label = None
  if pos_label not in allowed_labels:
    pos_label is None

  all_data, _ = fpd.load_slow_pig_features_and_labels_numpy(
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

  hidden_size = 256
  forget_bias = 1.0
  keep_prob = 1.0
  num_layers = 1

  batch_size = 20
  num_steps = 50
  optimizer = "Adam"
  max_epochs = 100
  max_max_epochs = 300
  init_lr = 0.001
  lr_decay = 0.99
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


if __name__ == "__main__":
  # mt_krc_pigs_slow()
  # cluster_slow_pigs(10)
  # pred_nn_slow_pigs(ws=5)
  pred_lstm_slow_pigs_raw()
  # for j in range(1, 11):
  #    cluster_slow_pigs(j)


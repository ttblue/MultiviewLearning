# Maltese Cross: Crossed Seq2Seq.

from __future__ import division, print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf
import scipy.integrate as si
# import recurrent.rnn_cell.sru as sru

import classifier
import dataset

import matplotlib.pyplot as plt

import IPython

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def generate_random_mixed_inds(num_A, num_B):
  choices = ["A"] * num_A + ["B"] * num_B
  np.random.shuffle(choices)
  return choices


class MalteseCrossConfig(classifier.Config):

  def __init__(self,
               dim_A,
               dim_B,
               use_sru,
               use_dynamic_rnn,
               hidden_size,
               latent_size,
               forget_bias,
               keep_prob,
               num_layers,
               batch_size,
               num_steps,
               optimizer,
               max_epochs,
               max_max_epochs,
               init_lr,
               lr_decay,
               max_grad_norm,
               initializer,
               nn_initializer,
               init_scale,
               summary_log_path=None,
               verbose=True):

    self.dim_A = dim_A
    self.dim_B = dim_B

    self.use_sru = use_sru
    self.use_dynamic_rnn = use_dynamic_rnn

    self.hidden_size = hidden_size
    self.latent_size = latent_size
    self.forget_bias = forget_bias
    self.keep_prob = keep_prob
    self.num_layers = num_layers

    self.batch_size = batch_size
    self.num_steps = num_steps
    self.optimizer = optimizer
    self.max_epochs = max_epochs
    self.max_max_epochs = max_max_epochs
    self.init_lr = init_lr
    self.lr_decay = lr_decay
    self.max_grad_norm = max_grad_norm
    self.initializer = initializer
    self.nn_initializer = nn_initializer
    self.init_scale = init_scale

    self.summary_log_path = summary_log_path
    self.verbose = verbose


class HiddenLayer(object):

  def __init__(
      self, name, x, d_in, d_out, W=None, b=None, activation=tf.nn.relu):

    self.input = x

    # Assuming that W and b will be reused.
    self.W = tf.get_variable(name + "_W", [d_in, d_out]) if W is None else W
    self.b = tf.get_variable(name + "_b", [d_out]) if b is None else b

    self.activation = activation
    self.output = activation(tf.nn.xw_plus_b(x, self.W, self.b))


class NNFeatureTransform(object):

  def __init__(self, x, config):

    self.input = x

    self.layers = []
    for layer_params in config:
      x_in = self.layers[-1].output if self.layers else x
      # IPython.embed()
      self.layers.append(HiddenLayer(x=x_in, **layer_params))

    self.output = self.layers[-1].output


class MCModel(object):

  def __init__(self, mc_config, nn_configs, is_training):
    self.config = mc_config
    self.nn_configs = nn_configs
    self.is_training = is_training

    # batch_size = input_.batch_size
    # num_steps = input_.num_steps
    # size = config.hidden_size
    # Create variable x, y for both streams
    self._xA = tf.placeholder(
        dtype=data_type(),
        shape=[self.config.batch_size, self.config.num_steps,
               self.config.dim_A],
        name="xA")
    self._yA = tf.placeholder(
        dtype=data_type(),
        shape=[self.config.batch_size, self.config.num_steps,
               self.config.dim_A],
        name="yA")
    self._xB = tf.placeholder(
        dtype=data_type() ,
        shape=[self.config.batch_size, self.config.num_steps,
               self.config.dim_B],
        name="xB")
    self._yB = tf.placeholder(
        dtype=data_type() ,
        shape=[self.config.batch_size, self.config.num_steps,
               self.config.dim_B],
        name="yB")

    self._create_encoder_NN()
    self._setup_cell()
    self._setup_RNN_output()
    self._setup_decoder_NN()
    self._setup_losses()
    if is_training:
      self._setup_optimizer()

  def _create_encoder_NN(self):
    if self.config.nn_initializer == "xavier":
      initializer = tf.contrib.layers.xavier_initializer()
    else:
      initializer = None

    with tf.variable_scope("Encoder", initializer=initializer):
      # Switch variables for relevant stream input
      self._use_inputA = tf.Variable(0.0, trainable=False, name="inputA")
      self._use_inputB = tf.Variable(0.0, trainable=False, name="inputB")

      # Convert tensors to the right shape
      self._xA_input = tf.reshape(self._xA, [-1, self.config.dim_A])
      self._xB_input = tf.reshape(self._xB, [-1, self.config.dim_B])

      # Create encoders for stream A and B
      self._encoderA = NNFeatureTransform(
          self._xA_input, self.nn_configs["A"]["encoder"])
      self._encoderB = NNFeatureTransform(
          self._xB_input, self.nn_configs["B"]["encoder"])

  def _lstm_cell(self):
    # Can be replaced with a different cell
    if self.config.use_sru:
      cell = sru.SimpleSRUCell(
          self.config.hidden_size, [0.0, 0.5, 0.9, 0.99, 0.999],
          self.config.hidden_size, 64)
    else:
      cell = tf.contrib.rnn.LSTMCell(
        self.config.hidden_size, forget_bias=self.config.forget_bias)
        # state_is_tuple=True)
    return cell

  def _attn_cell(self):
    if self.is_training and self.config.keep_prob < 1:
      return tf.contrib.rnn.DropoutWrapper(
          self._lstm_cell(), output_keep_prob=self.config.keep_prob)
    else:
      return self._lstm_cell()

  def _setup_cell(self):
    if self.config.num_layers <= 1:
      self._cell = self._attn_cell()
    else:
      self._cell = tf.contrib.rnn.MultiRNNCell(
          [self._attn_cell() for _ in xrange(self.config.num_layers)],
          state_is_tuple=True)

    self._initial_state = self._cell.zero_state(
        self.config.batch_size, data_type())

    with tf.device("/cpu:0"):
      self._inputA = tf.reshape(
          self._encoderA.output,
          [self.config.batch_size, self.config.num_steps, -1])
      self._inputB = tf.reshape(
          self._encoderB.output,
          [self.config.batch_size, self.config.num_steps, -1])
      if self.is_training and self.config.keep_prob < 1:
        self._inputA = tf.nn.dropout(self._inputA, self.config.keep_prob)
        self._inputB = tf.nn.dropout(self._inputB, self.config.keep_prob)

  def _setup_RNN_output(self):
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    with tf.variable_scope("LSTM") as scope:
      if self.config.use_dynamic_rnn or self.config.use_sru:
        outputA, stateA = tf.nn.dynamic_rnn(
            self._cell, self._inputA, initial_state=self._initial_state)
        scope.reuse_variables()
        outputB, stateB = tf.nn.dynamic_rnn(
          self._cell, self._inputB, initial_state=self._initial_state)
      else:
        inputs = tf.unstack(self._inputA, num=self.config.num_steps, axis=1)
        outputA, stateA = tf.contrib.rnn.static_rnn(
            self._cell, inputs, initial_state=self._initial_state)
        scope.reuse_variables()
        inputs = tf.unstack(self._inputB, num=self.config.num_steps, axis=1)
        outputB, stateB = tf.contrib.rnn.static_rnn(
            self._cell, inputs, initial_state=self._initial_state)

    self._RNN_outputA = tf.reshape(
        tf.concat(outputA, 1), [-1, self.config.latent_size])
    self._final_stateA = stateA
    self._RNN_outputB = tf.reshape(
        tf.concat(outputB, 1), [-1, self.config.latent_size])
    self._final_stateB = stateB

  def _setup_decoder_NN(self):
    if self.config.nn_initializer == "xavier":
      initializer = tf.contrib.layers.xavier_initializer()
    else:
      initializer = None

    # IPython.embed()
    with tf.variable_scope("Decoder", initializer=initializer):
      # Create decoders for stream A and B
      self._decoderA = NNFeatureTransform(
          self._RNN_outputA, self.nn_configs["A"]["decoder"])
      self._decoderB = NNFeatureTransform(
          self._RNN_outputB, self.nn_configs["B"]["decoder"])

      self._outputA = tf.reshape(
          self._decoderA.output,
          [self.config.batch_size, self.config.num_steps, -1])
      self._outputB = tf.reshape(
          self._decoderB.output,
          [self.config.batch_size, self.config.num_steps, -1])

  def _setup_losses(self):
    # Train over two separate loss functions
    # TODO: Change this to shift one forward
    lossA = 0
    # IPython.embed()
    for i in xrange(self.config.dim_A):
      _y = tf.slice(self._yA, [0, 0, i],
                    [self.config.batch_size, self.config.num_steps, 1])
      _Y = tf.slice(self._outputA, [0, 0, i],
                    [self.config.batch_size, self.config.num_steps, 1])
      lossA += tf.reduce_mean(tf.squared_difference(_y, _Y))
    self._lossA = lossA
    self._costA = tf.reduce_sum(self._lossA)
    self._errorA = tf.norm(tf.cast(self._outputA - self._yA, data_type()))

    lossB = 0
    for i in xrange(self.config.dim_B):
      _y = tf.slice(self._yB, [0, 0, i],
                    [self.config.batch_size, self.config.num_steps, 1])
      _Y = tf.slice(self._outputB, [0, 0, i],
                    [self.config.batch_size, self.config.num_steps, 1])
      lossB += tf.reduce_mean(tf.squared_difference(_y, _Y))
    self._lossB = lossB
    self._costB = tf.reduce_sum(self._lossB)
    self._errorB = tf.norm(tf.cast(self._outputB - self._yB, data_type()))

  def _setup_optimizer(self):
    # Keeping one global LR for now.
    self._lr = tf.Variable(0.0, trainable=False)
    # I don't think you need to separate the tvars
    # The gradient of the params of A w.r.t. costB should be 0 and vice versa.
    tvars = tf.trainable_variables()
    gradsA, _ = tf.clip_by_global_norm(tf.gradients(self._costA, tvars),
                                       self.config.max_grad_norm)
    gradsB, _ = tf.clip_by_global_norm(tf.gradients(self._costB, tvars),
                                       self.config.max_grad_norm)

    if self.config.optimizer == "Adam":
      optimizerA = tf.train.AdamOptimizer(self._lr)
      optimizerB = tf.train.AdamOptimizer(self._lr)
    elif self.config.optimizer == "Adagrad":
      optimizerA = tf.train.AdagradOptimizer(self._lr)
      optimizerB = tf.train.AdagradOptimizer(self._lr)
    else:
      optimizerA = tf.train.GradientDescentOptimizer(self._lr)
      optimizerB = tf.train.GradientDescentOptimizer(self._lr)

    self._trainA_op = optimizerA.apply_gradients(
        zip(gradsA, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    self._trainB_op = optimizerB.apply_gradients(
        zip(gradsB, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        data_type(), shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value}) 

  @property
  def xA(self):
    return self._xA

  @property
  def xB(self):
    return self._xB

  @property
  def yA(self):
    return self._yA

  @property
  def yB(self):
    return self._yB

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def errorA(self):
    return self._errorA

  @property
  def errorB(self):
    return self._errorB

  @property
  def costA(self):
    return self._costA

  @property
  def costB(self):
    return self._costB

  @property
  def final_stateA(self):
    return self._final_stateA

  @property
  def final_stateB(self):
    return self._final_stateB

  @property
  def lr(self):
    return self._lr

  @property
  def trainA_op(self):
    return self._trainA_op

  @property
  def trainB_op(self):
    return self._trainB_op


class MalteseCrossifier(classifier.Classifier):

  def __init__(self, config, nn_configs):
    self.nn_configs = nn_configs
    super(MalteseCrossifier, self).__init__(config)

  def _load_model(self, mtype="train"):
    if mtype == "train":
      self._model = self._train_model
    elif mtype == "validation":
      self._model = self._validation_model
    else:
      self._model = self._test_model

  def _run_epoch(self, dset_A, dset_B):
    if self.config.verbose:
      epoch_start_time = time.time()

    self._load_model("train")
    if self.config.lr_decay < 1:
      lr_decay = (self.config.lr_decay **
                  max(self._epoch_idx + 1 - self.config.max_epochs, 0.0))
      lr = self.config.init_lr * lr_decay
      self._model.assign_lr(self._session, lr)
    else: 
      lr = self.config.init_lr
      self._model.assign_lr(self._session, lr)

    if self.config.verbose:
      print("\n\nEpoch: %i\tLearning rate: %.5f"%(self._epoch_idx + 1, lr))

    # Produce a random order of inds
    random_order = generate_random_mixed_inds(dset_A.num_ts, dset_B.num_ts)

    dsets = {"A": dset_A, "B": dset_B}
    costs = {"A": 0., "B": 0.}
    idxs = {"A": 0, "B": 0}
    iters = {"A": 0, "B": 0}
    tot_steps = {"A": 0, "B": 0}
    error = {"A": 0., "B": 0.}

    num_ts = {"A": dset_A.num_ts, "B": dset_B.num_ts}
    tot_ts = dset_A.num_ts + dset_B.num_ts

    for ts_idx in xrange(tot_ts):
      if self.config.verbose:
        start_time = time.time()
      stream = random_order[ts_idx]
      idxs[stream] += 1

      # TODO: TEMPORARY
      x_batches, _ = dsets[stream].get_ts_batches( 
         self.config.batch_size, self.config.num_steps)
      ts_costs, ts_iters, ts_error = self._run_single_ts(
          x_batches, x_batches, stream,  training=True)
      costs[stream] += ts_costs
      iters[stream] += ts_iters
      epoch_size = len(x_batches)
      tot_steps[stream] += epoch_size
      error[stream] += ts_error * epoch_size

      if self.config.verbose:
        print("\tTrain TS: %i/%i (%s: %i/%i). \tError: %.3f\tTime: %.2fs."%
              (ts_idx + 1, tot_ts, stream, idxs[stream], num_ts[stream],
               ts_error, (time.time() - start_time)), end='\r')
        sys.stdout.flush()

    epoch_results = {}
    for stream in ["A", "B"]:
      error[stream] /= tot_steps[stream]
      s_cost = costs[stream] / iters[stream]
      epoch_results[stream] = {"error": error[stream], "costs": s_cost}
    self.train_results.append(epoch_results)

    if self.config.verbose:
      print("\tTrain TS: %i/%i (%s: %i/%i). \tError: %.3f\tTime: %.2fs."%
            (ts_idx + 1, tot_ts, stream, idxs[stream], num_ts[stream],
             ts_error, (time.time() - start_time)), end='\r')
      for stream in ["A", "B"]:
        print("\nStream %s -- \tTrain Costs: %.3f\n\tTrain Error: %.3f\n\t"%
              (stream, epoch_results[stream]["costs"], error[stream]))
      print("\nTime taken: %.2fs\n\n" % (time.time() - epoch_start_time))

      # start_time = time.time()
    # if dset_v is not None:
    #   self._load_model("valid")
    #   v_iters = 0  # TODO: Don't need.
    #   v_costs = 0
    #   v_error = 0
    #   v_tot_steps = 0
    #   for ts_idx in xrange(dset_v.num_ts):
    #     x_batches, y_batches = dset_v.get_ts_batches(
    #         self.config.batch_size, self.config.num_steps)
    #     ts_costs, ts_iters, ts_error = self._run_single_ts(
    #         x_batches, y_batches, training=False)
    #     v_costs += ts_costs
    #     v_iters += ts_iters
    #     epoch_size = len(x_batches)
    #     v_tot_steps += epoch_size
    #     v_error += ts_error * epoch_size

    #   v_error /= v_tot_steps
    #   print("\n\tValidation Costs: %.3f\n\t"
    #         "Validation Error: %.3f\n\tTime: %.2fs."%
    #         (v_costs / v_iters, v_error, (time.time() - start_time)))
    #   self.validation_results.append(
    #       {"error": v_error, "costs": v_costs / v_iters})

  def _run_single_ts(self, x, y, stream, training=True):
    if stream not in ["A", "B"]:
      raise classifier.ClassifierException("Stream must be A or B.")

    costs = 0.0
    iters = 0
    total_error = 0
    state = self._session.run(self._model.initial_state)

    cost = {"A": self._model.costA, "B": self._model.costB}[stream]
    final_state = {"A": self._model.final_stateA,
                   "B": self._model.final_stateB}[stream]
    error = {"A": self._model.errorA, "B": self._model.errorB}[stream]
    var_x = {"A": self._model.xA, "B": self._model.xB}[stream]
    var_y = {"A": self._model.yA, "B": self._model.yB}[stream]

    fetches = {
        "cost": cost,
        "final_state": final_state,
        "error": error,
    }
    if training:
      fetches["eval_op"] = {
          "A": self._model.trainA_op, "B": self._model.trainB_op}[stream]

    epoch_size = len(x)
    for step in xrange(epoch_size):
      feed_dict = {}
      feed_dict[var_x] = x[step]
      feed_dict[var_y] = y[step]
      if self.config.use_sru or self.config.num_layers <= 1:
        feed_dict[self._model.initial_state] = state
      else:
        for i, (c, h) in enumerate(self._model.initial_state):
          feed_dict[c] = state[i].c
          feed_dict[h] = state[i].h

      vals = self._session.run(fetches, feed_dict)
      cost = vals["cost"]
      state = vals["final_state"]
      error = vals["error"]

      costs += cost
      iters += self.config.num_steps
      total_error += error

    total_error /= epoch_size
    return costs, iters, total_error

  def _predict_single_ts(self, x, from_stream, to_stream):
    state = self._session.run(self._model.initial_state)

    rnn_output = {"A": self._model._RNN_outputA,
                  "B": self._model._RNN_outputB}[from_stream]
    dim_out = {"A": self.config.dim_A, "B": self.config.dim_B}[to_stream]
    pred_output = {"A": self._model._outputA,
                   "B": self._model._outputB}[to_stream]
    x, _ = dataset.create_batches(
        x, None, self.config.batch_size, self.config.num_steps)
    # pred = np.empty((self.config.batch_size, 0))
    var_x = {"A": self._model.xA, "B": self._model.xB}[from_stream]
    pred = np.empty((0, self.config.num_steps, dim_out))
    epoch_size = len(x)

    for step in xrange(epoch_size):
      feed_dict = {}
      feed_dict[var_x] = x[step]
      for i, (c, h) in enumerate(self._model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      rnn_pred = self._session.run(rnn_output, feed_dict)
      step_pred = self._session.run(pred_output, {rnn_output: rnn_pred})
      pred = np.concatenate((pred, step_pred), axis=0)
      # pred = np.c_[pred, step_pred]

    return np.squeeze(np.reshape(pred, (-1, dim_out)))

  def fit(self, dset_A, dset_B):
    self.train_results = []
    # self.validation_results = []
    # self._best_validation = 0.0

    with tf.Graph().as_default():
      if self.config.initializer == "xavier":
        initializer = tf.contrib.layers.xavier_initializer()
      else:
        initializer = tf.random_uniform_initializer(-self.config.init_scale,
                                                    self.config.init_scale)

      with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
          self._train_model = MCModel(
              mc_config=self.config, nn_configs=self.nn_configs,
              is_training=True)
        tf.summary.scalar("Training_A_Loss", self._train_model.costA)
        tf.summary.scalar("Training_B_Loss", self._train_model.costB)
        tf.summary.scalar("Learning_Rate", self._train_model.lr)

      with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          self._validation_model = MCModel(
              mc_config=self.config, nn_configs=self.nn_configs,
              is_training=False)
        tf.summary.scalar("Validation_A_Loss", self._validation_model.costA)
        tf.summary.scalar("Validation_B_Loss", self._validation_model.costB)

      with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          self._test_model = MCModel(
              mc_config=self.config, nn_configs=self.nn_configs,
              is_training=False)

      init_op = tf.global_variables_initializer()
      self._session = tf.Session()
      self._session.run(init_op)

      IPython.embed()
      for self._epoch_idx in xrange(self.config.max_max_epochs):
        self._run_epoch(dset_A, dset_B)

  def predict(self, xs, from_stream, to_stream):
    self._load_model("test")
    num_ts = len(xs)
    preds = []
    for ts_idx in xrange(num_ts):
      pred_idx = self._predict_single_ts(xs[ts_idx], from_stream, to_stream)
      preds.append(pred_idx)

    # IPython.embed()
    return preds


def burst_plots(trs, ts1, ts2, max_rows=5):

  nrows = np.minimum(max_rows, np.int(ts1.shape[1]/2))
  f, axs = plt.subplots(nrows, 2)#, sharex='col', sharey='row')

  for ch in xrange(nrows):
    axs[ch][0].plot(ts1[:, 2 * ch], color='r')
    axs[ch][0].plot(ts2[:, 2 * ch], color='b')
    axs[ch][0].plot(trs[:, 2 * ch], color='g')
    if ch == 0:
      axs[ch][1].plot(ts1[:, 2 * ch + 1], color='r', label="Pred")
      axs[ch][1].plot(ts2[:, 2 * ch + 1], color='b', label="True")
      axs[ch][1].plot(trs[:, 2 * ch + 1], color='g', label="Source")
      axs[ch][1].legend()
    else:
      axs[ch][1].plot(ts1[:, 2 * ch + 1], color='r')
      axs[ch][1].plot(ts2[:, 2 * ch + 1], color='b')
      axs[ch][1].plot(trs[:, 2 * ch + 1], color='g')

  plt.legend()
  plt.show()


def single_channel_plots(trs, ts1, ts2, ch):

  f, axs = plt.subplots(2, 1)#, sharex='col', sharey='row')
  axs[0].plot(ts1[:, ch], color='r', label="Pred")
  axs[0].plot(ts2[:, ch], color='b', label="True")
  axs[0].legend()

  axs[1].plot(trs[:, ch], color='g', label="Source")
  axs[1].legend()

  plt.show()


def lorenz(V, t, s=10, r=28, b=2.667):
    x, y, z = V
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def generate_lorenz_attractor(
    tmax, nt, x0=0., y0=1., z0=1.05, s=10, r=28, b=2.667):

  ts = np.linspace(0, tmax, nt)
  f = si.odeint(lorenz, (x0, y0, z0), ts, args=(s, r, b))
  return f.T


def main():
  from synthetic import multimodal_systems as ms

  dim_A = 1
  dim_B = 1

  use_sru = False
  use_dynamic_rnn = False

  hidden_size = 1
  latent_size = 1
  forget_bias = 0.0
  keep_prob = 1.0
  num_layers = 2
  init_scale = 0.1
  batch_size = 10
  num_steps = 10
  optimizer = "adam"
  initializer = None
  nn_initializer = "xavier"
  max_epochs = 10
  max_max_epochs = 10
  init_lr = 0.0001
  lr_decay = 1.0
  max_grad_norm = 5.
  
  # init_scale,
  summary_log_path = "./log/log_file.log"
  verbose = True

  config = MalteseCrossConfig(
      dim_A=dim_A,
      dim_B=dim_B,
      use_sru=use_sru,
      use_dynamic_rnn=use_dynamic_rnn,
      hidden_size=hidden_size,
      latent_size=latent_size,
      forget_bias=forget_bias,
      keep_prob=keep_prob,
      num_layers=num_layers,
      batch_size=batch_size,
      num_steps=num_steps,
      optimizer=optimizer,
      max_epochs=max_epochs,
      max_max_epochs=max_max_epochs,
      init_lr=init_lr,
      lr_decay=lr_decay,
      max_grad_norm=max_grad_norm,
      initializer=initializer,
      nn_initializer=nn_initializer,
      init_scale=init_scale,
      summary_log_path=summary_log_path,
      verbose=verbose)

  nn_configs = {
      "A": {
          "encoder": [{
              "name": "AEncoder",
              "d_in": dim_A,
              "d_out": latent_size,
          }],
          "decoder": [{
              "name": "ADecoder",
              "d_in": latent_size,
              "d_out": dim_A,
          }],
      },
      "B": {
          "encoder": [{
              "name": "BEncoder",
              "d_in": dim_B,
              "d_out": latent_size,
          }],
          "decoder": [{
              "name": "BDecoder",
              "d_in": latent_size,
              "d_out": dim_B,
          }],
      },
  }


  xs = [[[1],[1],[1]]*1000]
  dsetA = dataset.TimeseriesDataset(xs, xs)
  dsetB = dataset.TimeseriesDataset(xs, xs)

  # IPython.embed()
  s2s = MalteseCrossifier(config, nn_configs)
  s2s.fit(dset_A=dsetA, dset_B=dsetB)
  # IPython.embed()
  IPython.embed()

if __name__ == "__main__":
  main()

# LSTM classfifiers.
# Based on the RNN tutorial.

from __future__ import division, print_function

import sys
import time

import numpy as np
import tensorflow as tf

import recurrent.rnn_cell.sru as sru

import classifier
import dataset

import IPython

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class LSTMConfig(classifier.Config):

  def __init__(self,
               num_classes,
               num_features,
               use_sru,
               use_dynamic_rnn,
               hidden_size,
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
               init_scale,
               summary_log_path=None,
               verbose=True):

    self.num_classes = num_classes
    self.num_features = num_features

    self.use_sru = use_sru
    self.use_dynamic_rnn = use_dynamic_rnn

    self.hidden_size = hidden_size
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
    self.init_scale = init_scale

    self.summary_log_path = summary_log_path
    self.verbose = verbose

# class PTBInput(object):
#   """The input data."""

#   def __init__(self, config, data, name=None):
#     self.batch_size = batch_size = config.batch_size
#     self.num_steps = num_steps = config.num_steps
#     self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#     self.input_data, self.targets = reader.ptb_producer(
#         data, batch_size, num_steps, name=name)


class LSTMModel(object):

  def __init__(self, config, is_training):
    self.config = config
    self.is_training = is_training

    # batch_size = input_.batch_size
    # num_steps = input_.num_steps
    # size = config.hidden_size
    # vocab_size = config.vocab_size
    # Create variable x, y
    self._x = tf.placeholder(
        dtype=data_type(),
        shape=[self.config.batch_size, self.config.num_steps,
               self.config.num_features],
        name="x")
    self._y = tf.placeholder(
        dtype=tf.int32,
        shape=[self.config.batch_size, self.config.num_steps],
        name="y")

    self._setup_cell()
    self._setup_output()
    if is_training:
      self._setup_optimizer()

  def _lstm_cell(self):
    # Can be replaced with a different cell
    if self.config.use_sru:
      cell = sru.SimpleSRUCell(
          self.config.hidden_size, [0.0, 0.5, 0.9, 0.99, 0.999],
          self.config.hidden_size, 64)
    else:
      cell = tf.contrib.rnn.LSTMCell(
        self.config.hidden_size, forget_bias=self.config.forget_bias,
        state_is_tuple=True)
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

    with tf.device("/gpu:0"):
      if self.is_training and self.config.keep_prob < 1:
        self._inputs = tf.nn.dropout(self._x, self.config.keep_prob)
      else:
        self._inputs = self._x

  def _setup_output(self):
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
   #  IPython.embed()
    with tf.variable_scope("LSTM"):
      if self.config.use_dynamic_rnn or self.config.use_sru:
        outputs, state = tf.nn.dynamic_rnn(
            self._cell, self._inputs, initial_state=self._initial_state)
      else:
        inputs = tf.unstack(self._inputs, num=self.config.num_steps, axis=1)
        outputs, state = tf.contrib.rnn.static_rnn(
            self._cell, inputs, initial_state=self._initial_state)
    # IPython.embed()
    # outputs = []
    # state = self._initial_state
    # with tf.variable_scope("LSTM"):
    #   for time_step in range(num_steps):
    #     if time_step > 0: tf.get_variable_scope().reuse_variables()
    #     (cell_output, state) = cell(inputs[:, time_step, :], state)
    #     outputs.append(cell_output)
    # IPython.embed()

    output = tf.reshape(tf.concat(outputs, 1), [-1, self.config.hidden_size])
    # output = tf.stack(outputs, 1)
    # import IPython
    # print(output.get_shape())
    # print(len(outputs))
    self._softmax_w = tf.get_variable(
        "softmax_w", [self.config.hidden_size, self.config.num_classes],
        dtype=data_type())
    self._softmax_b = tf.get_variable(
        "softmax_b", [self.config.num_classes], dtype=data_type())
    self._logits = tf.reshape(
        tf.matmul(output, self._softmax_w) + self._softmax_b,
        [self.config.batch_size, self.config.num_steps,
         self.config.num_classes])
    self._loss = tf.contrib.seq2seq.sequence_loss(
        self._logits, self._y,
        tf.ones([self.config.batch_size, self.config.num_steps],
                dtype=data_type()))
    self._cost = tf.reduce_sum(self._loss) / self.config.batch_size
    self._final_state = state

    self._pred = tf.cast(tf.argmax(self._logits, axis=2), tf.int32)
    self._accuracy = tf.reduce_mean(
        tf.cast(tf.equal(self._pred, self._y), data_type()))

  def _setup_optimizer(self):
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      self.config.max_grad_norm)

    if self.config.optimizer == "Adam":
      optimizer = tf.train.AdamOptimizer(self._lr)
    elif self.config.optimizer == "Adagrad":
      optimizer = tf.train.AdagradOptimizer(self._lr)
    else:
      optimizer = tf.train.GradientDescentOptimizer(self._lr)

    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        data_type(), shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def x(self):
    return self._x

  @property
  def y(self):
    return self._y

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def pred(self):
    return self._pred

  @property
  def accuracy(self):
    return self._accuracy

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class LSTM(classifier.Classifier):

  def __init__(self, config):
    super(LSTM, self).__init__(config)

  def _load_model(self, mtype="train"):
    if mtype == "train":
      self._model = self._train_model
    elif mtype == "validation":
      self._model = self._validation_model
    else:
      self._model = self._test_model

  def _run_epoch(self, dset, dset_v):
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

    costs = 0
    iters = 0  # TODO: Don't need.
    accuracy = 0
    tot_steps = 0
    for ts_idx in xrange(dset.num_ts):
      if self.config.verbose:
        start_time = time.time()

      x_batches, y_batches = dset.get_ts_batches(
          self.config.batch_size, self.config.num_steps)
      ts_costs, ts_iters, ts_accuracy = self._run_single_ts(
          x_batches, y_batches, training=True)
      costs += ts_costs
      iters += ts_iters
      epoch_size = len(x_batches)
      tot_steps += epoch_size
      accuracy += ts_accuracy * epoch_size

      if self.config.verbose:
        print("\tTrain TS %i\tAccuracy: %.3f\tTime: %.2fs."%
              (ts_idx + 1, ts_accuracy, (time.time() - start_time)), end='\r')
        sys.stdout.flush()

    accuracy /= tot_steps
    self.train_results.append({"accuracy": accuracy, "costs": costs / iters})

    if self.config.verbose:
      print("\tTrain TS %i\tAccuracy: %.3f\tTime: %.2fs."%
            (ts_idx + 1, ts_accuracy, (time.time() - start_time)))
      print("\n\tTrain Costs: %.3f\n\tTrain Accuracy: %.3f\n\t"
            "Time: %.2fs."%
            (costs / iters, accuracy, (time.time() - epoch_start_time)))

      start_time = time.time()

    self._load_model("valid")
    v_iters = 0  # TODO: Don't need.
    v_costs = 0
    v_accuracy = 0
    v_tot_steps = 0
    for ts_idx in xrange(dset_v.num_ts):
      x_batches, y_batches = dset_v.get_ts_batches(
          self.config.batch_size, self.config.num_steps)
      ts_costs, ts_iters, ts_accuracy = self._run_single_ts(
          x_batches, y_batches, training=False)
      v_costs += ts_costs
      v_iters += ts_iters
      epoch_size = len(x_batches)
      v_tot_steps += epoch_size
      v_accuracy += ts_accuracy * epoch_size

    v_accuracy /= v_tot_steps
    print("\n\tValidation Costs: %.3f\n\t"
          "Validation Accuracy: %.3f\n\tTime: %.2fs."%
          (v_costs / v_iters, v_accuracy, (time.time() - start_time)))
    self.validation_results.append(
        {"accuracy": v_accuracy, "costs": v_costs / v_iters})

  def _run_single_ts(self, x, y, training=True):
    costs = 0.0
    iters = 0
    total_accuracy = 0
    state = self._session.run(self._model.initial_state)

    fetches = {
        "cost": self._model.cost,
        "final_state": self._model.final_state,
        "accuracy": self._model._accuracy,
    }
    if training:
      fetches["eval_op"] = self._model.train_op

    epoch_size = len(x)
    for step in xrange(epoch_size):
      feed_dict = {}
      feed_dict[self._model._x] = x[step]
      feed_dict[self._model._y] = y[step]
      if self.config.use_sru or self.config.num_layers <= 1:
        feed_dict[self._model.initial_state] = state
      else:
        for i, (c, h) in enumerate(self._model.initial_state):
          feed_dict[c] = state[i].c
          feed_dict[h] = state[i].h

      vals = self._session.run(fetches, feed_dict)
      cost = vals["cost"]
      state = vals["final_state"]
      accuracy = vals["accuracy"]

      costs += cost
      iters += self.config.num_steps
      total_accuracy += accuracy

    total_accuracy /= epoch_size
    return costs, iters, total_accuracy

  def _predict_single_ts(self, x):
    state = self._session.run(self._model.initial_state)

    x, _ = dataset.create_batches(
        x, None, self.config.batch_size, self.config.num_steps)
    pred = np.empty((self.config.batch_size, 0))
    epoch_size = len(x)
    for step in xrange(epoch_size):
      feed_dict = {}
      feed_dict[self._model._x] = x[step]
      for i, (c, h) in enumerate(self._model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      step_pred = self._session.run(self._model.pred, feed_dict)
      pred = np.c_[pred, step_pred]

    return np.squeeze(np.reshape(pred, (-1, 1)))

  def fit(self, xs=None, ys=None, dset=None, xs_v=None, ys_v=None, dset_v=None):
    if xs is None and dset is None:
      raise classifier.ClassifierException("No data is given.")

    if xs is not None:
      if ys is None:
        raise classifier.ClassifierException("Labels not given, but data is.")
      dset = dataset.TimeseriesDataset(xs, ys)

    if xs_v is not None:
      if ys_v is None:
        raise classifier.ClassifierException(
            "Validation labels not given, but data is.")
      dset_v = dataset.TimeseriesDataset(xs_v, ys_v)

    if xs_v is None and ys_v is None:
      if dset_v is None:
        dset, dset_v = dset.split([0.8, 0.2])

    self.train_results = []
    self.validation_results = []
    self._best_validation = 0.0

    with tf.Graph().as_default():
      if self.config.initializer == "xavier":
        initializer = tf.contrib.layers.xavier_initializer()
      else:
        initializer = tf.random_uniform_initializer(-self.config.init_scale,
                                                    self.config.init_scale)

      with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
          self._train_model = LSTMModel(config=self.config, is_training=True)
        tf.summary.scalar("Training Loss", self._train_model.cost)
        tf.summary.scalar("Learning Rate", self._train_model.lr)

      with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          self._validation_model = LSTMModel(config=self.config, is_training=False)
        tf.summary.scalar("Validation Loss", self._validation_model.cost)

      with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          self._test_model = LSTMModel(config=self.config, is_training=False)

      init_op = tf.global_variables_initializer()
      self._session = tf.Session()
      self._session.run(init_op)

      for self._epoch_idx in xrange(self.config.max_max_epochs):
        self._run_epoch(dset, dset_v)

  def predict(self, xs):
    self._load_model("test")
    num_ts = len(xs)
    preds = []
    for ts_idx in xrange(num_ts):
      preds.append(self._predict_single_ts(xs[ts_idx]))

    return preds


def create_simple_dataset(num_ts, dim=2, ts_len=5000, nc=4):
  xs = []
  ys = []
  for _ in xrange(num_ts):
    y = np.random.randint(nc, size=(ts_len, 1))
    x = np.tile(y, [1, dim])
    y = y.squeeze()

    xs.append(x)
    ys.append(y)

  return xs, ys

def create_counter_dataset(num_ts, ts_len=5000):
  xs = []
  ys = []
  for _ in xrange(num_ts):
    x = np.random.randint(2, size=(ts_len, 2))
    y = [x[0].sum()]
    for i in xrange(1, ts_len):
      y.append(x[i-1:i+1].sum())

    xs.append(x)
    ys.append(np.array(y))

  return xs, ys  

def main():
  num_classes = 4
  num_features = 2

  hidden_size = 600
  forget_bias = 0.5
  keep_prob = 1.0
  num_layers = 2
  init_scale = 0.1
  max_grad_norm = 5
  max_epochs = 5 ##
  max_max_epochs = 15 ##
  init_lr = 1.0
  lr_decay = 0.9
  batch_size = 20
  num_steps = 10
  verbose = True

  config = LSTMConfig(
      num_classes=num_classes, num_features=num_features,
      hidden_size=hidden_size, forget_bias=forget_bias, keep_prob=keep_prob,
      num_layers=num_layers, init_scale=init_scale, max_grad_norm=max_grad_norm,
      max_epochs=max_epochs, max_max_epochs=max_max_epochs, init_lr=init_lr,
      lr_decay=lr_decay, batch_size=batch_size, num_steps=num_steps,
      verbose=verbose)

  xs, ys = create_simple_dataset(100, num_features, 1000, num_classes)
  # xs, ys = create_counter_dataset(100, 1000)
  dset = dataset.TimeseriesDataset(xs, ys)
  dset_train, dset_valid, dset_test = dset.split([0.6, 0.2, 0.2])
  IPython.embed()

  lstm = LSTM(config)
  lstm.fit(dset=dset_train, dset_v=dset_valid)
  IPython.embed()

if __name__ == "__main__":
  main()

import tensorflow as tf
from recurrent.rnn_cell.linear import linear as _linear
import IPython

class SimpleSRUCell(tf.contrib.rnn.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    """

    def __init__(self, num_stats, mavg_alphas, output_dims, recur_dims,
                 summarize=True, learn_alphas=False, linear_out=False,
                 include_input=False, activation=tf.nn.relu,
                 output_input=False):
        self._num_stats = num_stats
        self._output_dims = output_dims
        self._recur_dims = recur_dims
        if learn_alphas:
            init_logit_alphas = -tf.log(1.0/mavg_alphas-1)
            logit_alphas = tf.get_variable(
                'logit_alphas', initializer=init_logit_alphas
            )
            self._mavg_alphas = tf.reshape(tf.sigmoid(logit_alphas), [1, -1, 1])
        else:
            self._mavg_alphas = tf.reshape(mavg_alphas, [1, -1, 1])
        self._nalphas = int(self._mavg_alphas.get_shape()[1])
        self._summarize = summarize
        self._linear_out = linear_out
        self._activation = activation
        self._include_input = include_input
        self._output_input = output_input
        # self._state_is_tupe = True

    @property
    def state_size(self):
        return int(self._nalphas * self._num_stats)

    @property
    def output_size(self):
        return self._output_dims

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Make statistics on input.
            # print 111111111, self._recur_dims
            if self._recur_dims > 0:
                # IPython.embed()
                recur_output = self._activation(_linear(
                    state, self._recur_dims, True, scope='recur_feats'
                ), name='recur_feats_act')
                # print '\t', state
                # print '\t', recur_output.get_shape()
                # print '\t', inputs.get_shape()
                # IPython.embed()
                stats = self._activation(_linear(
                    [inputs, recur_output], self._num_stats, True, scope='stats'
                ), name='stats_act')
            else:
                # IPython.embed()
                stats = self._activation(_linear(
                    inputs, self._num_stats, True, scope='stats'
                ), name='stats_act')
            # print 2222222222
            # import IPython
            # IPython.embed()
            # Compute moving averages of statistics for the state.
            with tf.variable_scope('out_state'):
                # IPython.embed()
                state_tensor = tf.reshape(
                    state, [-1, self._nalphas, self._num_stats], 'state_tensor'
                )
                stats_tensor = tf.reshape(
                    stats, [-1, 1, self._num_stats], 'stats_tensor'
                )
                out_state = tf.reshape(self._mavg_alphas*state_tensor +
                                       (1-self._mavg_alphas)*stats_tensor,
                                       [-1, self.state_size], 'out_state')
            # print 3333333333333
            # IPython.embed()
            # Compute the output.
            if self._include_input:
                output_vars = [out_state, inputs]
            else:
                output_vars = out_state
            # print 44444444444444
            # IPython.embed()
            output = _linear(
                output_vars, self._num_stats, True, scope='output'
            )
            # print 555555555555555
            # IPython.embed()
            # print '\t', output.get_shape()
            if not self._linear_out:
                output = self._activation(output, name='output_act')
            if self._output_input:
                output = tf.concat([output, inputs], 1, name='output_concat')
            # print 6666666666666
            # IPython.embed()
            # print '\t', output.get_shape()
        return (output, out_state)

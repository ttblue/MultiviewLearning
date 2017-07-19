"""
Implements a simple switching linear model with one discrete hidden variable and
some observable variables. The hidden variable indexes into the linear model.
"""
import numpy as np
import pandas as pd

import time_series_generator as tsg


def generateRandomTransitionMatrix(n, self_prob=None):
  """
  Generates a transition matrix: a column stochastic matrix with each entry >=0.
  All values are chosen uniformly at random and then normalized, expect the
  self-transition.

  Args:
    n: Number of states.
    self_prob: A number of vector of size n of self-transition probabilities.

  Returns:
    P: n x n column stochastic matrix with transition probabilities.
  """
  P = np.random.uniform(0., 1., size=(n, n))
  P = P / P.sum(0)

  if self_prob is not None:
    self_prob = np.squeeze(self_prob)

    # Some error checks
    if len(self_prob.shape) > 0:
      if len(self_prob.shape) > 1:
        tsg.TimeSeriesGenerationException(
            "Incorrect shape of self probabilities: %s"%(self_prob.shape))
      if self_prob.shape[0] != n:
        tsg.TimeSeriesGenerationException(
            "Incorrect length of self probabilities: %s"%(self_prob.shape[0]))

    scale = 1 - self_prob
    P = P * scale
    P[xrange(n), xrange(n)] = self_prob

  return P


def sampleTransition(P, curr_state):
  n = P.shape[0]

  if isinstance(curr_state, int):
    if curr_state >= n or curr_state <= 0:
      tsg.TimeSeriesGenerationException(
          "Invalid current state %i with %i possible states: %s"
          %(curr_state, n))

    dist = P[:, curr_state]
  else:
    curr_state = np.squeeze(curr_state)
    if len(curr_state.shape) != 1 or len(curr_state.shape[0]) != n:
      tsg.TimeSeriesGenerationException(
          "Invalid current state vector %s: "%(curr_state))
    if np.any(curr_state < 0) or np.allclose(sum(curr_state), 1.):
      tsg.TimeSeriesGenerationException(
          "Invalid current state distribution %s: "%(curr_state))

    dist = P.dot(curr_state)

  new_state = np.random.choice(n, p=dist)
  return new_state


def generateLinearDynamicsModel(n, min_sv=-1, max_sv=1, min_b=-1, max_b=1):
  """
  Generates a random linear dynamics model within the singular value range.
  The values are generated from a gaussian distribution.

  Args:
    n: Number of variables.
    min_sv: Minimum allowed singular value for transition matrix.
    max_sv: Maximum allowed singular value for transition matrix.
    min_b: Minimum allowed value for affine component.
    max_b: Maximum allowed value for affine component.

  Returns:
    A: n x n matrix representing dynamics model.
    b: n x 1 matrix representing the affine component of the model.
  """
  if min_sv is not None and max_sv is not None:
    if min_sv > max_sv:
      tsg.TimeSeriesGenerationException(
            "Minimum allowed singular value must be less than maximum allowed"
            " singular value.")

  base_sigma = 5.
  A = np.random.randn(n, n) * base_sigma

  U, S, V = np.linalg.svd(A)
  S = np.clip(S, min_sv, max_sv)
  A = U.dot(np.diag(S).dot(V))

  b = np.uniform(min_b, max_b)

  return A, b


class SwitchingLinearGenerator(tsg.TimeSeriesGenerator):
  """
  Creates a switching linear system, based on a single discrete hidden state.
  """

  def __init__(
      self, n_obs_vars, n_hidden, noise_sigma=0.5, min_steps_in_state=10,
      transition_matrix=None, linear_bank=None, init_obs=None,
      init_hidden=None):
    """
    Constructor.

    Args:
      n_obs_vars: Number of observable variables.
      n_hidden: Number of values the hidden state can take.
      noise_sigma: Parameter for added independent gaussian noise.
      min_steps_in_state: Minimum number of steps before the hidden state can
          transition.
      transition_matrix: The transition matrix for the hidden state. If None,
          this will be randomly generated.
      linear_bank: The linear models corresponding to the different values of
          the hidden state. If None, this will be randomly generated.
      init_obs: An initial observation for the observable variables.
      init_hidden: An initial hidden state.
    """

    self._n_obs_vars = n_obs_vars
    self._n_hidden = n_hidden
    self._noise_sigma = noise_sigma
    self._min_steps = 1 if min_steps_in_state is None else min_steps_in_state

    default_self_prob = 0.8
    self._P = (generateRandomTransitionMatrix(n_hidden, default_self_prob)
               if transition_matrix is None else transition_matrix)

    if linear_bank is None:
      default_min_sv = -1
      default_max_sv = 1

      self._linear_bank = [
          generateLinearDynamicsModel(
              n_obs_vars, default_min_sv, default_max _sv)
          for _ in xrange(n_hidden)]

    else:
      self._linear_bank = _linear_bank

    # Initial values:
    self._x = np.zeros(n_obs_vars) if init_obs is None init_obs
    self._hidden = (
        np.random.choice(self._n_hidden)
        if init_hidden is None else init_hidden)
    self._A, self._b = self._linear_bank[self._init_hidden]

    # Some book-keeping:
    self._t_since_state_change = 0
    self._init_obs = self._x
    self._init_hidden = self._hidden
    # Maintain a history of a certain number of tsteps
    self._history_size = 100
    self._hidden_history = [self._hidden]

  def sampleNext(self, return_hidden=False):
    x = self._A.dot(self._x) + self._b
    self._x = x

    if len(self._hidden_history) < self._history_size:
      self._hidden_history.append(self._hidden)
    else:
      self._hidden_history = self._hidden_history[1:] + [self._hidden]

    self._t_since_state_change += 1
    if self._t_since_state_change >= self._min_steps:
      transition_probs = self._P[:, self._hidden]
      hidden = np.choice(self._n_hidden, p=transition_probs)
      if hidden != self._hidden:
        self._t_since_state_change = 0
      self._hidden, hidden = hidden, self._hidden

    return x, hidden if return_hidden else x

  def sampleNSteps(self, n, return_hidden=False):
    output = [self.sampleNext(return_hidden) for _ in xrange(n)]

    if return_hidden:
      
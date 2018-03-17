"""
Implements a simple switching linear model with one discrete hidden variable and
some observable variables. The hidden variable indexes into the linear model.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate

import time_series_generator as tsg


def sigmoid(x):   
  return 1. / (1. + np.exp(-x))

def generatePulse(
    tlen, amp, expand=100, shrink=50, bias=0.075, fc=5, bw=0.5, thresh=1e-3):
  """
  Generates a pulse which rise up to amp and then drop down below 0.
  This is done by scaling and stretching a gaussian pulse.

  If any of these arguments don't make sense, take a look at the code.
  Args:
    tlen: Length of pulse.
    amp: Amplitude of pulse.
    expand: Expansion of tlen to generate initial pulse which is then selected
        and interpolated.
    shrink: Scale of sigmoid used to "select" part of gaussian pulse.
    bias: Bias of sigmoid used to "select" part of gaussian pulse.
    fc
  """
  t = np.linspace(-1, 1, tlen * expand, endpoint=False)
  amps = sigmoid(shrink * (t)) * sigmoid(shrink * (t[-1::-1] + bias))
  pulse = signal.gausspulse(t, fc=fc, bw=bw)

  pulse = pulse * amps

  nzinds = (np.abs(pulse) > thresh).nonzero()[0]
  pulse = pulse[nzinds[0]: nzinds[-1]]
  pulse = pulse / np.max(np.abs(pulse)) * amp

  t = t[nzinds[0]: nzinds[-1]]
  fnc = interpolate.interp1d(t, pulse)
  pulse = fnc(np.linspace(t[0], t[-1], tlen))

  return pulse


def generatePulseWaveform(
      tlen, locs=[0.5], amps=[1], widths=[0.1], mu=0., noise_sigma=0.05):
  """
  Generates pulse waveforms with variable peaks. Pulses rise up to maximum
  amplitude and then drop down below that.

  Args:
    tlen: Length of pulse.
    locs: Locations of pulses in fraction of length of waveform.
    amps: Amplitudes of pulses.
    widths: Widths of pulses in fraction of length of waveform.
    mu: Waveform mean.
    noise_sigma: 
  """
  pulse_waveform = (
      np.ones(tlen) * mu + np.random.normal(scale=noise_sigma, size=(tlen,)))

  for loc, width, amp in zip(locs, widths, amps):
    plen = np.round(tlen * width).astype(int)
    loc = np.round(tlen * loc).astype(int)
    pulse = generatePulse(plen, amp)

    hplen = int(plen // 2)
    print(loc, plen)
    pulse_waveform[loc - hplen: loc + plen - hplen] += pulse

  return pulse_waveform


class SimulatedBioGenerator(tsg.TimeSeriesGenerator):
  """
  Creates a simulated biological system with shifting means and frequency.

  The means and frequency change through a predetermined pattern.
  """

  def __init__(self, init_mean, init_period):
    """
    Constructor for Bio Generato.

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
      default_min_sb = -1
      default_max_sb = 1

      self._linear_bank = [
          generateLinearDynamicsModel(
              n_obs_vars, default_min_sv, default_max_sv, default_min_sb, 
              default_max_sb)
          for _ in xrange(n_hidden)]
    else:
      self._linear_bank = _linear_bank

    # Initial values:
    self._x = np.zeros(n_obs_vars) if init_obs is None else init_obs
    self._hidden = (
        np.random.choice(self._n_hidden)
        if init_hidden is None else init_hidden)
    self._A, self._b = self._linear_bank[self._hidden]

    # Some book-keeping:
    self._t_since_state_change = 0
    self._init_obs = self._x
    self._init_hidden = self._hidden
    # Maintain a history of a certain number of tsteps
    self._history_size = 100
    self._hidden_history = [self._hidden]

  def sampleNext(self, return_hidden=False):
    noise = np.random.normal(scale=self._noise_sigma, size=(self._n_obs_vars,))
    x = self._A.dot(self._x) + self._b + noise
    self._x = x

    if len(self._hidden_history) < self._history_size:
      self._hidden_history.append(self._hidden)
    else:
      self._hidden_history = self._hidden_history[1:] + [self._hidden]

    self._t_since_state_change += 1
    if self._t_since_state_change >= self._min_steps:
      transition_probs = self._P[:, self._hidden]
      hidden = np.random.choice(self._n_hidden, p=transition_probs)
      if hidden != self._hidden:
        self._t_since_state_change = 0
        self._A, self._b = self._linear_bank[hidden]
      self._hidden, hidden = hidden, self._hidden
    else:
      hidden = self._hidden

    return (x, hidden) if return_hidden else x

  def sampleNSteps(self, n, return_hidden=False):
    return [self.sampleNext(return_hidden) for _ in xrange(n)]


def test_stuff():
  import matplotlib.pyplot as plt
  no = 2
  sigma = 0.2
  nh = 3
  n = 500
  msis = 30

  sl = SwitchingLinearGenerator(
      no, nh, noise_sigma=sigma, min_steps_in_state=msis)
  ns = sl.sampleNSteps(500, True)
  xs = [x for x,h in ns]
  hs = [h for x,h in ns]

  for i in xrange(no):
    plt.subplot(no + 1, 1, i + 1)
    plt.plot([x[i] for x in xs])

  plt.subplot(no + 1, 1, no + 1)
  plt.plot(hs)
  plt.show()

  import IPython
  IPython.embed()

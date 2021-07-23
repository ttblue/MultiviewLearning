import numpy as np
import scipy.integrate as si


# Lorenz system:
def lorenz_deriv(V, t, s=10, r=28, b=2.667):
    x, y, z = V
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def generate_lorenz_system(
    tmax, nt, x0=0., y0=1., z0=1.05, s=10, r=28, b=2.667, sig=0.01):

  ts = np.linspace(0, tmax, nt)
  f = si.odeint(lorenz_deriv, (x0, y0, z0), ts, args=(s, r, b))
  return f.T + sig * np.random.randn(*f.T.shape)


# Simple harmonic oscillator
# d2x + w^2 x = 0
def sho_deriv(u, t, w):
  d1x, x = u
  return [-(w ** 2) * x, d1x]


def generate_simple_harmonic_oscillator(tmax, nt, x0=0., d1x0=1., w=1.):
  ts = np.linspace(0, tmax, nt)
  f = si.odeint(sho_deriv, (x0, d1x0), ts, args=(w,))
  return f.T


def diff_freq_sinusoids(tmax, nt, w1=1.0, w2=2.0):
  ts = np.linspace(0, tmax, nt)
  x1 = np.atleast_2d(2 * np.pi * ts * w1)
  x2 = np.atleast_2d(2 * np.pi * ts * w2)
  x = np.r_[x1, x2]
  f = np.sin(x)

  return f.T


if __name__ == "__main__":
  import matplotlib.pyplot as plt

  tmax = 20
  nt = 1000
  x0 = 0.
  d1x0 = 1.
  w = 1.

  f = generate_simple_harmonic_oscillator(tmax, nt, x0, d1x0, w)
  ts = np.linspace(0, tmax, nt)
  plt.plot(ts, f[0, :], color="b", label="x")
  plt.plot(ts, f[1, :], color="g", label="x_dot")
  plt.legend()
  plt.show()
import numpy as np
import scipy.integrate as si


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
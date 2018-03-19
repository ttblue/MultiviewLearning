import numpy as np
from pylds import models

import matplotlib.pyplot as plt


def generate_LDS_data_with_two_observation_models(
    n, T, D_latent, D_obs1, D_obs2=None, save_file=None):
  """
  Create a linear dynamical system with two observation models.

  Arguments:
    n -- Number of LDS data sequences to generate
    T -- Length of each sequence
    D_latent -- Number of latent states
    D_obs1 -- Number of observations in first obs. model
    D_obs2 -- Number of observations in second obs. model.
        If None, same as D_obs1
    save_file -- Name of file to save data in. Data is returned regardless.
  """

  data = []

  D_obs2 = D_obs1 if D_obs2 is None else D_obs2
  D_obs = D_obs1 + D_obs2

  C = None
  for i in xrange(n):
    lds = models.DefaultLDS(D_obs, D_latent, C=C)
    xs, _ = lds.generate(T)

    # Having two linear observation models is the same as having one and 
    # splitting into two.
    xs1, xs2 = np.split(xs, [D_obs1], axis=1)
    data.append((xs1, xs2, lds.A)) # Keep A for now.

    C = lds.C  # Keep the observation model.

  if save_file is not None:
    np.savez(save_file, data=data, C=C)

  return data, C


if __name__ == "__main__":
  n = 100
  T = 100
  D_latent = 10
  D_obs1 = 10
  save_file = None

  data, C = generate_LDS_data_with_two_observation_models(
      n=n, T=T, D_latent=D_latent, D_obs1=D_obs1, save_file=save_file)
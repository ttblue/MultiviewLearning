import numpy as np
from pylds import models, util

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


def generate_LDS_data_with_two_observation_models_train_test(
    n_tr, n_te, T, D_latent, D_obs1, D_obs2=None, theta_max_pert=0.1,
    save_file=None):
  """
  Create a train and test dataset for linear dynamical systems with two
  observation models.

  Arguments:
    n_tr -- Number of train LDS data sequences to generate
    n_te -- Number of test LDS data sequences to generate
    T -- Length of each sequence
    D_latent -- Number of latent states
    D_obs1 -- Number of observations in first obs. model
    D_obs2 -- Number of observations in second obs. model.
        If None, same as D_obs1
    theta_max_pert -- maximum to perturb observation model for test
    save_file -- Name of file to save data in. Data is returned regardless.
  """

  data_tr = []
  data_te = []


  D_obs2 = D_obs1 if D_obs2 is None else D_obs2
  D_obs = D_obs1 + D_obs2

  # Generate training data.
  C_train = None
  C = None
  for i in xrange(n_tr):
    lds = models.DefaultLDS(D_obs, D_latent, C=C)
    xs, _ = lds.generate(T)

    # Having two linear observation models is the same as having one and 
    # splitting into two.
    xs1, xs2 = np.split(xs, [D_obs1], axis=1)
    data_tr.append((xs1, xs2, lds.A)) # Keep A for now.

    if C_train is None:
      C_train = lds.C  # Keep the observation model.
      C = C_train # small_rot.dot(C_train)

  # Generate testing data.
  for i in xrange(n_te):
    small_theta = np.random.uniform(0, theta_max_pert)
    small_rot = util.random_rotation(D_obs, small_theta)
    C = small_rot.dot(C_train)
    lds = models.DefaultLDS(D_obs, D_latent, C=C)
    xs, _ = lds.generate(T)

    # Having two linear observation models is the same as having one and 
    # splitting into two.
    xs1, xs2 = np.split(xs, [D_obs1], axis=1)
    data_te.append((xs1, xs2, lds.A)) # Keep A for now.

  if save_file is not None:
    np.savez(save_file, data_tr=data_tr, data_te=data_te, C=C_train)

  return data_tr, data_te, C_train


if __name__ == "__main__":
  n = 100
  T = 100
  D_latent = 10
  D_obs1 = 10
  save_file = None

  data, C = generate_LDS_data_with_two_observation_models(
      n=n, T=T, D_latent=D_latent, D_obs1=D_obs1, save_file=save_file)
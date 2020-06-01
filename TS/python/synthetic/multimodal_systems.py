import numpy as np

import matplotlib.pyplot as plt

from utils import tfm_utils, math_utils

# Probably should come up with a better way to generate features, other than
# just random gaussian samples


def flatten(list_of_lists):
  return [a for b in list_of_lists for a in b]


def padded_identity(n, m, idx):
  # Helper function -- 
  # Create an n x m block column matrix:
  # [0_{idx x m};
  #  I_{m x m};
  #  0_{(n-idx-m) x m}]
  return np.r_[np.zeros((idx, m)), np.identity(m), np.zeros(((n - idx - m), m))]


def subset_redundancy_data(
    npts, n_views, subsets, s_dim, scale=1., noise_eps=1e-3, tfm_final=False,
    peps=1e-3, rtn_correspondences=True):
  # noise_eps -- for gaussian noise
  # peps (perturb_eps) -- for angle of rotation transformation

  view_data = {vi: [] for vi in range(n_views)}
  corrs = {vi: [] for vi in range(n_views)}

  for sub_i, subset in enumerate(subsets):
    sub_dat = np.random.randn(npts, s_dim) * scale
    for vi in subset:
      vi_sub_dat = sub_dat
      view_data[vi].append(vi_sub_dat)
      corrs[vi].extend([sub_i] * s_dim)

  for vi in range(n_views):
    view_data[vi] = np.concatenate(view_data[vi], axis=1)
    corrs[vi] = np.array(corrs[vi])

  ptfms = None
  if tfm_final:
    ptfms = {}
    for vi in range(n_views):
      v_dat = view_data[vi]
      v_dim = v_dat.shape[1]
      v_mu = vdat.mean(0)
      # center data -> rotate it -> translate back
      vdat_pert, R = tfm_utils.perturb_matrix(v_dat - v_mu, peps)
      view_data[vi] = (
          vdat_pert + v_mu + np.random.randn(npts, v_dim) * noise_eps)
      ptfms[vi] = R, v_mu

  if rtn_correspondences:
    return view_data, ptfms, corrs
  return view_data, ptfms


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
    small_rot = math_utils.random_rotation(D_obs, small_theta)
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


def generate_redundant_multiview_data(
      npts, nviews=3, ndim=15, scale=2, centered=True, overlap=True,
      gen_D_alpha=True, perturb_eps=1e-2):
  data = np.random.uniform(high=scale, size=(npts, ndim))

  if centered:
    data -= data.mean(axis=0)

  n_per_view = ndim // nviews
  n_remainder = ndim - n_per_view * nviews
  view_groups = [
      (i * n_per_view + np.arange(n_per_view)).astype(int)
      for i in range(nviews)]
  remaining_data = (
      data[:, -n_remainder:] if n_remainder > 0 else np.empty((npts, 0)))

  view_data = {}
  for vi, vg in enumerate(view_groups):
    view_inds = (
        # Exclude one view-group and give the rest
        flatten([vg for i, vg in enumerate(view_groups) if i != vi])
        # Or only use that one group.
        if overlap else view_groups[vi])
    view_data[vi] = np.c_[data[:, view_inds], remaining_data]

  perturb_tfms = None
  if perturb_eps > 0:
    perturb_tfms = {}
    perturb_tfms[0] = np.eye(view_data[0].shape[1])
    for vi in range(1, nviews):
      view_data[vi], perturb_tfms[vi] = tfm_utils.perturb_matrix(
          view_data[vi], perturb_eps)

  # Trivial solution to check
  if gen_D_alpha:
    dim_per_view = n_per_view * (nviews - 1 if overlap else 1) + n_remainder
    alpha = data.T

    I_rem = padded_identity(
        dim_per_view, n_remainder, dim_per_view - n_remainder)
    D = {}
    for vi in range(nviews):
      if overlap:
        cols = []
        cidx = 0
        for i in range(nviews):
          v_col = (
              np.zeros((dim_per_view, n_per_view)) if i == vi else
              padded_identity(dim_per_view, n_per_view, cidx * n_per_view))
          if i != vi:
            cidx += 1
          cols.append(v_col)
      else:
        cols = [
            (padded_identity(dim_per_view, n_per_view, 0) if i == vi else
             np.zeros((dim_per_view, n_per_view)))
            for i in range(nviews)]
      cols.append(I_rem)
      D[vi] = np.concatenate(cols, axis=1)
    return view_data, D, alpha

  return view_data, perturb_tfms


def generate_local_overlap_multiview_data(
      npts, nviews=4, ndim=16, scale=2, centered=True, perturb_eps=1e-2):
  data = np.random.uniform(high=scale, size=(npts, ndim))

  if centered:
    data -= data.mean(axis=0)

  n_per_view = ndim // nviews
  n_remainder = ndim - n_per_view * nviews
  view_groups = [
      (i * n_per_view + np.arange(n_per_view)).astype(int)
      for i in range(nviews)]
  split_inds = [i * n_per_view for i in range(1, nviews)]
  split_data = np.split(data, split_inds, axis=1)
  remaining_data = (
      data[:, -n_remainder:] if n_remainder > 0 else np.empty((npts, 0)))

  view_data = {}
  for vi in range(nviews):
    vj = vi + 1
    if vj >= nviews: vj = 0
    view_data[vi] = np.c_[split_data[vi], split_data[vj], remaining_data]

  perturb_tfms = None
  if perturb_eps > 0:
    perturb_tfms = {}
    perturb_tfms[0] = np.eye(view_data[0].shape[1])
    for vi in range(1, nviews):
      view_data[vi], perturb_tfms[vi] = tfm_utils.perturb_matrix(
          view_data[vi], perturb_eps)

  return view_data, perturb_tfms


if __name__ == "__main__":
  n = 100
  T = 100
  D_latent = 10
  D_obs1 = 10
  save_file = None

  data, C = generate_LDS_data_with_two_observation_models(
      n=n, T=T, D_latent=D_latent, D_obs1=D_obs1, save_file=save_file)
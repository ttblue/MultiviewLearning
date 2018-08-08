# Some tests
import numpy as np
import sklearn.preprocessing as skp

import dynamical_systems_learning as dsl
import feature_functions as ffuncs
# import nn_transform_learner as ntl
import ff_transform_learner as ftl
import synthetic.simple_systems as ss
import tfm_utils as tutils
import time_sync as tsync
import transforms as trns

import matplotlib.pyplot as plt
from matplotlib import cm

import IPython


def test_ntl_on_sho():
  visualize = True
  verbose = True

  tmax = 100
  nt = 1000
  x0 = 0.
  d1x0 = 1.
  w = 1.

  x, _ = ss.generate_simple_harmonic_oscillator(tmax, nt, x0, d1x0, w)

  tau = 10
  ntau = 2
  xts = tsync.compute_td_embedding(x, tau, ntau)

  # Transform
  theta = np.pi / 4
  new_xdir = np.atleast_2d([np.cos(theta), np.sin(theta)]).T
  new_ydir = np.atleast_2d([- np.sin(theta), np.cos(theta)]).T
  scaling = np.diag([2., 0.5])
  R_true_base = np.c_[new_xdir, new_ydir].astype(np.float64)
  R_true = R_true_base.dot(scaling)
  t_true = np.array([.25, .25], dtype=np.float64).reshape(-1, 1)
  yts = (R_true.dot(xts.T) + t_true).T

  if visualize:
    plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
    plt.plot(yts[:, 0], yts[:, 1], color="r", label="y")
    plt.legend()
    plt.show()

    plt.plot(xts[:,0], color="b", label="x")
    plt.plot(yts[:,0], color="r", label="y")
    plt.legend()
    plt.show()


  IPython.embed()
  gammak = 100
  lr = 1e-3
  max_iters = 1000
  verbose = True
  tlearner = ntl.NNTransformLearner(xts, dim=ntau, gammak=gammak)
  config = ntl.NNTrainerConfig(lr=lr, max_iters=max_iters, verbose=verbose)
  ttrainer = ntl.NNTransformTrainer(tlearner, config)
  print("Set up trainer.")
  print("Training.")
  ttrainer.fit(yts)

  IPython.embed()
  R, t = tlearner.R.detach().numpy(), tlearner.t.detach().numpy()
  yR = (R.dot(yts.T) + t).T

  ts_pred = tlearner.forward_simulate(yR[0], nt).detach().numpy()
  # plt.plot(xR[:, 0], xR[:, 1], color="g")
  plt.plot(yts[:, 0], yts[:, 1], color="g", label="y")
  plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
  plt.plot(yR[:, 0], yR[:, 1], color="r", label="f(y)")
  plt.plot(ts_pred[:, 0], ts_pred[:, 1], color="k", label="sim")
  plt.legend()
  plt.show()


def plot_forward_sims(alpha, dyn_func, nt=1000, nsims=50):
  from matplotlib import cm
  import IPython
  init_x = np.linspace(0.1, 1, nsims)
  colors = np.atleast_2d(cm.winter(init_x).squeeze())
  init_x = np.r_[[init_x] * alpha.shape[1]].T

  for idx in range(nsims):
    sim_vals = [init_x[idx]]
    print("Generating sim %i out of %i." % (idx + 1, nsims), end='\r')
    for _ in range(nt):
      sim_vals.append(dyn_func(sim_vals[-1]))
    sim_vals = np.array(sim_vals)
    # IPython.embed()
    plt.plot(sim_vals[:, 0], sim_vals[:, 1], color=colors[idx])

  print("Generating sim %i out of %i." % (idx + 1, nsims))
  plt.show()


def test_ff_on_sho():
  visualize = False
  verbose = True

  tmax = 100
  nt = 1000
  x0 = 0.
  d1x0 = 1.
  w = 1.

  x, _ = ss.generate_simple_harmonic_oscillator(tmax, nt, x0, d1x0, w)

  tau = 10
  ntau = 2
  xts = tsync.compute_td_embedding(x, tau, ntau)

  # Transform
  theta = np.pi / 4
  new_xdir = np.atleast_2d([np.cos(theta), np.sin(theta)]).T
  new_ydir = np.atleast_2d([- np.sin(theta), np.cos(theta)]).T
  scaling = np.diag([1.5, 0.75])
  R_true_base = np.c_[new_xdir, new_ydir].astype(np.float64)
  R_true = R_true_base.dot(scaling)
  t_true = np.array([0.25, -.25], dtype=np.float64).reshape(-1, 1)
  yts = (R_true.dot(xts.T) + t_true).T

  degree = 2
  include_bias = True
  poly_features = skp.PolynomialFeatures(
      degree=degree, include_bias=include_bias)
  feature_func = poly_features.fit_transform

  if visualize:
    plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
    plt.plot(yts[:, 0], yts[:, 1], color="r", label="y")
    plt.legend()
    plt.show()

    plt.plot(xts[:,0], color="b", label="x")
    plt.plot(yts[:,0], color="r", label="y")
    plt.legend()
    plt.show()

  # IPython.embed()
  affine = False
  rcond = 1e-10
  Ax, bx, errx = dsl.learn_linear_dynamics(xts, feature_func, affine, rcond)
  Ay, by, erry = dsl.learn_linear_dynamics(yts, feature_func, affine, rcond)

  tlearner = ftl.FFTransformLearner(
      dim=ntau, feature_func=poly_features, target_alpha=None, rcond=rcond,
      affine=affine, ff_type="poly")
  # ap2 = tlearner.forward_simulate(yts)
  # IPython.embed()
  print("Set up learner.")

  target_alpha = tlearner.forward_simulate(xts).detach().numpy()  # np.c_[A, b].T
  Axf, bxf = target_alpha.T[:, :-1], target_alpha.T[:, -1]
  alpha_y = tlearner.forward_simulate(yts).detach().numpy()  # np.c_[A, b].T
  Ayf, byf = alpha_y.T[:, :-1], alpha_y.T[:, -1]

  R_guess = np.eye(ntau)
  t_guess = np.zeros([ntau, 1])
  tlearner.update_target_alpha(target_alpha)
  tlearner.reset_transform(init_R=R_guess.copy(), init_t=t_guess.copy())
  # Visualize
  show_x = True
  show_y = True
  if visualize:
    xts_pred = [xts[0]]
    xts_predf = [xts[0]]
    for _ in range(len(xts) - 1):
      xts_pred.append(Ax.dot(feature_func(xts_pred[-1]).squeeze()) + bx)
      xts_predf.append(Axf.dot(feature_func(xts_predf[-1]).squeeze()) + bxf)
    xts_pred = np.array(xts_pred)
    xts_predf = np.array(xts_predf)

    yts_pred = [yts[0]]
    yts_predf = [yts[0]]
    for _ in range(len(yts) - 1):
      yts_pred.append(Ay.dot(feature_func(yts_pred[-1]).squeeze()) + by)
      yts_predf.append(Ayf.dot(feature_func(yts_predf[-1]).squeeze()) + byf)
    yts_pred = np.array(yts_pred)
    yts_predf = np.array(yts_predf)

    if show_x:
      plt.plot(xts[:, 0], xts[:, 1], color="b")
      # plt.plot(xts_pred[:, 0], xts_pred[:, 1], color="r")
      plt.plot(xts_predf[:, 0], xts_predf[:, 1], color="g")
      plt.show()
    if show_y:
      plt.plot(yts[:, 0], yts[:, 1], color="b")
      # plt.plot(yts_pred[:, 0], yts_pred[:, 1], color="r")
      plt.plot(yts_predf[:, 0], yts_predf[:, 1], color="g")
      plt.show()
    IPython.embed()

  lr = 1e-3
  max_iters = 500
  verbose = True
  config = ftl.FFTrainerConfig(lr=lr, max_iters=max_iters, verbose=verbose)
  ttrainer = ftl.FFTransformTrainer(tlearner, config)
  print("Set up trainer.")
  print("Training.")
  ttrainer.fit(yts)

  # global R, t, target_alpha, feature_func, Rinv, tinv
  R, t = tlearner.R.detach().numpy(), tlearner.t.detach().numpy()
  yR = (R.dot(yts.T) + t).T
  alpha2 = tlearner.forward(yts).detach().numpy().T
  A2, b2 = alpha2[:, :-1], alpha2[:, -1]


  Rt = tutils.make_homogeneous_tfm(R, t.squeeze())
  Rtinv = np.linalg.inv(Rt)
  Rinv, tinv = Rtinv[:ntau, :ntau], Rtinv[:-1, -1]

  f = lambda y: (R.dot(y.T) + t.squeeze()).T.squeeze()
  g = lambda x: target_alpha.T.dot(feature_func(x[None, :]).squeeze())
  finv = lambda x: (Rinv.dot(x.T) + tinv.squeeze()).T.squeeze()

  IPython.embed()

  finv_g_f_y = [yts[0]]
  for _ in range(len(yts) - 1):
    finv_g_f_y.append(finv(g(f(finv_g_f_y[-1]))))
    # yt = finv_g_f_y[-1]
    # fyt = (R.dot(yt.T) + t.squeeze()).T.squeeze()
    # gfyt = target_alpha.T.dot(feature_func(fyt[None, :]).squeeze())
    # finv_g_f_y.append((Rinv.dot(gfyt.T) + tinv.squeeze()).T.squeeze())
  finv_g_f_y = np.array(finv_g_f_y)

  plt.plot(yts[:, 0], color='b', label='y')
  plt.plot(fgfy[:,0], color='r', label='tfm_y')
  plt.legend()
  plt.show()

  ts0 = yR[0]
  ts_pred = [ts0]
  # for _ in range(len(yR) - 1):
  #   ts_pred.append(A2.dot(feature_func(ts_pred[-1]).squeeze()) + b2)
  # ts_pred = np.array(ts_pred)
  # plt.plot(xR[:, 0], xR[:, 1], color="g")
  plt.plot(yts[:, 0], yts[:, 1], color="g", label="y")
  plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
  plt.plot(yR[:, 0], yR[:, 1], color="r", label="f(y)")
  # plt.plot(ts_pred[:, 0], ts_pred[:, 1], color="k", label="sim")
  plt.legend()
  plt.show()

  plt.plot(xts[:, 0], color="b", label="x")
  plt.plot(yts[:, 0], color="g", label="y")
  plt.plot(yR[:, 0], color="r", label="f(y)")
  plt.legend()
  plt.show()


def transform_tde(ts, theta, t=None, tau=10, scaling=None):
  ts = np.atleast_2d(ts)
  ntau = ts.shape[1]
  if ntau < 2:
    ts_to_rot = tsync.compute_td_embedding(ts, tau, 2)
  else:
    ts_to_rot = ts[:, :2]

  new_xdir = np.atleast_2d([np.cos(theta), np.sin(theta)]).T
  new_ydir = np.atleast_2d([-np.sin(theta), np.cos(theta)]).T
  scaling = np.eye(2) if scaling is None else scaling

  R_base = np.c_[new_xdir, new_ydir].astype(np.float64)
  R = R_base.dot(scaling)
  t = np.zeros((2,1)) if t is None else t.reshape(2, 1)

  rot_ts = (R.dot(ts_to_rot.T) + t)[0, :].squeeze()
  rot_ts = (np.atleast_2d(rot_ts) if ntau == 1 else
            tsync.compute_td_embedding(rot_ts, tau, ntau))

  return rot_ts, R, t


def make_tfm_into_size(R, t, dim):
  dim_R = R.shape[0]
  assert dim_R == t.shape[0]

  if dim == dim_R:
    return R, t
  elif dim < dim_R:
    return R[:dim, :dim], t[:dim]
  else:
    R_new = np.eye(dim)
    R_new[:dim_R, :dim_R] = R
    t_new = np.zeros((dim, 1))
    t_new[:dim_R] = t
    return R_new, t_new


def test_ff_on_lorenz():
  visualize = False
  tmax = 100
  nt = 10000
  x, y, z = ss.generate_lorenz_system(tmax, nt)

  tau = 10
  ntau = 3
  xts = tsync.compute_td_embedding(x, tau, ntau)

  # # Transform
  theta = np.pi / 3
  # t = np.array([1.5, -0.5], dtype=np.float64)
  # scaling = np.eye(2)
  # yts, R_true, t_true = transform_tde(xts, theta, t, tau=10, scaling=scaling)
  new_xdir = np.atleast_2d([np.cos(theta), np.sin(theta)]).T
  new_ydir = np.atleast_2d([-np.sin(theta), np.cos(theta)]).T
  scaling = np.eye(2) # np.diag([1.5, 0.75])
  R_true_base = np.c_[new_xdir, new_ydir].astype(np.float64)
  R_true = R_true_base.dot(scaling)
  t_true = np.array([0.25, -.25], dtype=np.float64).reshape(-1, 1)
  R_true, t_true = make_tfm_into_size(R_true, t_true, ntau)
  yts = (R_true.dot(xts.T) + t_true).T

  degree = 2
  include_bias = True
  feature_func = ffuncs.PolynomialFeatures(degree, include_bias)

  if visualize:
    plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
    plt.plot(yts[:, 0], yts[:, 1], color="r", label="y")
    # plt.plot(zts[:, 0], zts[:, 1], color="g", label="z")
    plt.legend()
    plt.show()

    plt.plot(xts[:,0], color="b", label="x")
    plt.plot(yts[:,0], color="r", label="y")
    # plt.plot(zts[:,0], color="g", label="z")
    plt.legend()
    plt.show()

  # IPython.embed()
  affine = False
  rcond = 1e-5
  Ax, bx, errx = dsl.learn_linear_dynamics(
      xts, feature_func.numpy_features, affine, rcond)
  Ay, by, erry = dsl.learn_linear_dynamics(
      yts, feature_func.numpy_features, affine, rcond)

  g_one = lambda x, alpha: alpha.T.dot(feature_func(np.atleast_2d(x)).squeeze())
  g_many = lambda x, alpha: feature_func(np.atleast_2d(x)).dot(alpha)
  g = lambda x, alpha: (
      g_many(x, alpha) if len(x.squeeze().shape) > 1 else g_one(x, alpha))

  # R_guess = np.eye(ntau)
  # t_guess = np.zeros([ntau, 1])
  # transform = trns.AffineTransform(ntau, R_guess, t_guess)
  scale = 1.2
  ngrid = 5
  all_pts = np.r_[xts, yts]
  mins = scale * all_pts.min(0)
  maxs = scale * all_pts.max(0)
  ranges = []
  for mi, ma in zip(mins, maxs):
    ranges.append(np.linspace(mi, ma, ngrid))
  mesh = []
  for coord in np.meshgrid(*ranges):
    mesh.append(coord.reshape(-1, 1))
  base_grid = np.concatenate(mesh, 1)

  reg_coeff = 1e3
  transform = trns.ThinPlateSplineTransform(ntau, base_grid)
  regularizer = trns.TPSIntegralCost(transform, reg_coeff)

  tlearner = ftl.FFTransformLearner(
      transform, feature_func, None, regularizer, rcond)
  target_alpha = tlearner.forward_simulate(xts).detach().numpy()  # np.c_[A, b].T
  tlearner.update_target_alpha(target_alpha)
  alpha_y = tlearner.forward_simulate(yts).detach().numpy()  # np.c_[A, b].T
  print("Set up learner.")
  # ap2 = tlearner.forward_simulate(yts)
  # IPython.embed()

  # Visualize
  # show_x = True
  # show_y = True
  # if visualize:
  #   # xts_pred = [xts[0]]
  #   # xts_predf = [xts[0]]
  #   # for _ in range(len(xts) - 1):
  #   #   xts_pred.append(g(xts_pred[-1], Ax.T))
  #   #   xts_predf.append(g(xts_predf[-1], target_alpha))
  #   # xts_pred = np.array(xts_pred)
  #   # xts_predf = np.array(xts_predf)

  #   # yts_pred = [yts[0]]
  #   # yts_predf = [yts[0]]
  #   # for _ in range(len(yts) - 1):
  #   #   yts_pred.append(g(yts_pred[-1], Ay.T))
  #   #   yts_predf.append(g(yts_predf[-1], alpha_y))
  #   # yts_pred = np.array(yts_pred)
  #   # yts_predf = np.array(yts_predf)

  #   if show_x:
  #     plt.plot(xts[:, 0], xts[:, 1], color="b")
  #     # plt.plot(xts_pred[:, 0], xts_pred[:, 1], color="r")
  #     plt.plot(xts_predf[:, 0], xts_predf[:, 1], color="g")
  #     plt.show()
  #   if show_y:
  #     plt.plot(yts[:, 0], yts[:, 1], color="b")
  #     # plt.plot(yts_pred[:, 0], yts_pred[:, 1], color="r")
  #     plt.plot(yts_predf[:, 0], yts_predf[:, 1], color="g")
  #     plt.show()
  # IPython.embed()

  lr = 1e-3
  max_iters = 5000
  verbose = True
  config = ftl.FFTrainerConfig(lr=lr, max_iters=max_iters, verbose=verbose)
  ttrainer = ftl.FFTransformTrainer(tlearner, config)
  print("Set up trainer.")
  print("Training.")
  ttrainer.fit(yts)

  yR = transform.numpy_transform(yts)
  IPython.embed()
  # global R, t, target_alpha, feature_func, Rinv, tinv
  # tfm_params = transform.get_parameters()
  # R = tfm_params["R"]
  # t = tfm_params["t"]
  alpha2 = tlearner.forward_simulate(yts, use_tf=True).detach().numpy().T

  # Rt = tutils.make_homogeneous_tfm(R, t.squeeze())
  # Rtinv = np.linalg.inv(Rt)
  # Rinv, tinv = Rtinv[:ntau, :ntau], Rtinv[:-1, -1]

  # tfm_func = lambda y, R, t: (R.dot(y.T) + t).T.squeeze()
  # f = (lambda y: 
  #     tfm_func(y, R, t.reshape(-1, 1)) if len(y.squeeze().shape) > 1 else
  #     tfm_func(y, R, t.squeeze()))
  # finv = (lambda y: 
  #     tfm_func(y, Rinv, tinv.reshape(-1, 1)) if len(y.squeeze().shape) > 1 else
  #     tfm_func(y, Rinv, tinv.squeeze()))

  # IPython.embed()
  # finv_g_f_y = [yts[0]]
  # for j in range(len(yts) - 1):
  #   try:
  #     finv_g_f_y.append(finv(g(f(finv_g_f_y[-1]), target_alpha)))
  #   except:
  #     IPython.embed()
    # yt = finv_g_f_y[-1]
    # fyt = (R.dot(yt.T) + t.squeeze()).T.squeeze()
    # gfyt = target_alpha.T.dot(feature_func(fyt[None, :]).squeeze())
    # finv_g_f_y.append((Rinv.dot(gfyt.T) + tinv.squeeze()).T.squeeze())
  # finv_g_f_y = np.array(finv_g_f_y)

  # plt.plot(yts[:, 0], color='b', label='y')
  # plt.plot(finv_g_f_y[:,0], color='r', label='tfm_y')
  # plt.legend()
  # plt.show()

  # ts0 = yR[0]
  # ts_pred = [ts0]
  # for _ in range(len(yR) - 1):
  #   ts_pred.append(A2.dot(feature_func(ts_pred[-1]).squeeze()) + b2)
  # ts_pred = np.array(ts_pred)
  # plt.plot(xR[:, 0], xR[:, 1], color="g")
  plt.plot(yts[:, 0], yts[:, 1], color="g", label="y")
  plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
  plt.plot(yR[:, 0], yR[:, 1], color="r", label="f(y)")
  # plt.plot(ts_pred[:, 0], ts_pred[:, 1], color="k", label="sim")
  plt.legend()
  plt.title("Reg: %.3f" % reg_coeff)
  plt.show()

  plt.plot(xts[:, 0], color="b", label="x")
  plt.plot(yts[:, 0], color="g", label="y")
  plt.plot(yR[:, 0], color="r", label="f(y)")
  plt.legend()
  plt.show()


def test_ff_on_two_sins():
  visualize = False
  tmax = 30
  nt = 1000
  w1 = 1.0
  w2 = 2.0
  x, y = ss.diff_freq_sinusoids(tmax, nt, w1, w2).T

  tau = 10
  ntau = 3
  xts = tsync.compute_td_embedding(x, tau, ntau)
  yts = tsync.compute_td_embedding(y, tau, ntau)

  degree = 3
  include_bias = True
  feature_func = ffuncs.PolynomialFeatures(degree, include_bias)

  if visualize:
    plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
    plt.plot(yts[:, 0], yts[:, 1], color="r", label="y")
    plt.legend()
    plt.show()

    plt.plot(xts[:,0], color="b", label="x")
    plt.plot(yts[:,0], color="r", label="y")
    plt.legend()
    plt.show()

  # IPython.embed()
  affine = False
  rcond = 1e-5
  Ax, bx, errx = dsl.learn_linear_dynamics(
      xts, feature_func.numpy_features, affine, rcond)
  Ay, by, erry = dsl.learn_linear_dynamics(
      yts, feature_func.numpy_features, affine, rcond)

  ######
  g_one = lambda x, alpha: alpha.T.dot(feature_func(np.atleast_2d(x)).squeeze())
  g_many = lambda x, alpha: feature_func(np.atleast_2d(x)).dot(alpha)
  g = lambda x, alpha: (
      g_many(x, alpha) if len(x.squeeze().shape) > 1 else g_one(x, alpha))

  R_guess = np.eye(ntau)
  t_guess = np.zeros([ntau, 1])
  transform = trns.AffineTransform(ntau, R_guess, t_guess)

  tlearner = ftl.FFTransformLearner(
      transform, feature_func, None, rcond)
  target_alpha = tlearner.forward_simulate(xts).detach().numpy()  # np.c_[A, b].T
  tlearner.update_target_alpha(target_alpha)
  alpha_y = tlearner.forward_simulate(yts).detach().numpy()  # np.c_[A, b].T
  print("Set up learner.")
  ######

  # Visualize
  # show_x = True
  # show_y = True
  # if visualize:
  #   # xts_pred = [xts[0]]
  #   # xts_predf = [xts[0]]
  #   # for _ in range(len(xts) - 1):
  #   #   xts_pred.append(g(xts_pred[-1], Ax.T))
  #   #   xts_predf.append(g(xts_predf[-1], target_alpha))
  #   # xts_pred = np.array(xts_pred)
  #   # xts_predf = np.array(xts_predf)

  #   # yts_pred = [yts[0]]
  #   # yts_predf = [yts[0]]
  #   # for _ in range(len(yts) - 1):
  #   #   yts_pred.append(g(yts_pred[-1], Ay.T))
  #   #   yts_predf.append(g(yts_predf[-1], alpha_y))
  #   # yts_pred = np.array(yts_pred)
  #   # yts_predf = np.array(yts_predf)

  #   if show_x:
  #     plt.plot(xts[:, 0], xts[:, 1], color="b")
  #     # plt.plot(xts_pred[:, 0], xts_pred[:, 1], color="r")
  #     plt.plot(xts_predf[:, 0], xts_predf[:, 1], color="g")
  #     plt.show()
  #   if show_y:
  #     plt.plot(yts[:, 0], yts[:, 1], color="b")
  #     # plt.plot(yts_pred[:, 0], yts_pred[:, 1], color="r")
  #     plt.plot(yts_predf[:, 0], yts_predf[:, 1], color="g")
  #     plt.show()
  # IPython.embed()

  lr = 1e-3
  max_iters = 1000
  verbose = True
  config = ftl.FFTrainerConfig(lr=lr, max_iters=max_iters, verbose=verbose)
  ttrainer = ftl.FFTransformTrainer(tlearner, config)
  print("Set up trainer.")
  print("Training.")
  ttrainer.fit(yts)

  # global R, t, target_alpha, feature_func, Rinv, tinv
  tfm_params = transform.get_parameters()
  R = tfm_params["R"]
  t = tfm_params["t"]
  yR = transform.numpy_transform(yts)
  alpha2 = tlearner.forward_simulate(yts, use_tf=True).detach().numpy().T

  Rt = tutils.make_homogeneous_tfm(R, t.squeeze())
  Rtinv = np.linalg.inv(Rt)
  Rinv, tinv = Rtinv[:ntau, :ntau], Rtinv[:-1, -1]

  tfm_func = lambda y, R, t: (R.dot(y.T) + t).T.squeeze()
  f = (lambda y: 
      tfm_func(y, R, t.reshape(-1, 1)) if len(y.squeeze().shape) > 1 else
      tfm_func(y, R, t.squeeze()))
  finv = (lambda y: 
      tfm_func(y, Rinv, tinv.reshape(-1, 1)) if len(y.squeeze().shape) > 1 else
      tfm_func(y, Rinv, tinv.squeeze()))

  IPython.embed()
  # finv_g_f_y = [yts[0]]
  # for j in range(len(yts) - 1):
  #   try:
  #     finv_g_f_y.append(finv(g(f(finv_g_f_y[-1]), target_alpha)))
  #   except:
  #     IPython.embed()
  #   # yt = finv_g_f_y[-1]
  #   # fyt = (R.dot(yt.T) + t.squeeze()).T.squeeze()
  #   # gfyt = target_alpha.T.dot(feature_func(fyt[None, :]).squeeze())
  #   # finv_g_f_y.append((Rinv.dot(gfyt.T) + tinv.squeeze()).T.squeeze())
  # finv_g_f_y = np.array(finv_g_f_y)

  # plt.plot(yts[:, 0], color='b', label='y')
  # plt.plot(finv_g_f_y[:,0], color='r', label='tfm_y')
  # plt.legend()
  # plt.show()

  # ts0 = yR[0]
  # ts_pred = [ts0]
  # for _ in range(len(yR) - 1):
  #   ts_pred.append(A2.dot(feature_func(ts_pred[-1]).squeeze()) + b2)
  # ts_pred = np.array(ts_pred)
  # plt.plot(xR[:, 0], xR[:, 1], color="g")
  plt.plot(yts[:, 0], yts[:, 1], color="g", label="y")
  plt.plot(xts[:, 0], xts[:, 1], color="b", label="x")
  plt.plot(yR[:, 0], yR[:, 1], color="r", label="f(y)")
  # plt.plot(ts_pred[:, 0], ts_pred[:, 1], color="k", label="sim")
  plt.legend()
  plt.show()

  plt.plot(xts[:, 0], color="b", label="x")
  plt.plot(yts[:, 0], color="g", label="y")
  plt.plot(yR[:, 0], color="r", label="f(y)")
  plt.legend()
  plt.show()
    
    

if __name__ == "__main__":
  # test_ntl_on_sho()
  # test_ff_on_sho()
  test_ff_on_lorenz()
  # test_ff_on_two_sins()

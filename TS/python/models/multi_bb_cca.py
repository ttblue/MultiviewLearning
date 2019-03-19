# Multi-way Byron Boots CCA:
# An attempt to extend BBoots to more # observations

import cvxpy as cvx
import itertools
import numpy as np
import scipy as sc
import scipy.optimize as so
import sklearn.utils.extmath as sue

import manifold_learning as ml
import tfm_utils as tu

import IPython


def temp_stupid_Q(ns, s=None):
  # ns -- either a list of sizes of sub-matrices, or the number of sub-matrices
  # of size s.
  if not isinstance(ns, list):
    ns = [s] * ns

  tot_n = np.sum(ns)
  Q = np.eye(tot_n) #np.zeros((tot_n, tot_n))
  for n in np.cumsum(ns):
    Q += np.eye(tot_n, k=n) + np.eye(tot_n, k=-n)

  return Q


def multiway_bb_si(
    Vs, n_embedding=2, feature_func=None, lm=0.0, solver="cobyla",
    use_iemaps=True):
  # Byron Boots' multi way 2-manifold system identification
  # Vs -- list of views of the data.
  # For single embedding dimension, we solve:
  # max \sum_{i \neq j} u_i * G_{ij} u_j
  # such that: ||u_i|| \leq 1 for all i

  nV = len(Vs)
  if nV == 2 and use_iemaps:
    # linear gmatrix and not normalizing are temporary
    return ml.instrumental_eigenmaps(
        Vs[0], Vs[1], e_dims=n_embedding,
        gmatrix_type="linear", standardize=True, rtn_svals=False)

  # if n_embedding > 1:
  #   raise NotImplementedError("Embedding size > 1 for len(Vs) > 2 coming soon.")

  # Set feature function to identity -- same as linear kernel
  if feature_func is None:
    feature_func = lambda x: x

  # Vs = [tu.center_matrix(V) for V in Vs]
  # Center data + featurize:
  npts = Vs[0].shape[0]
  Phis = [tu.center_matrix(feature_func(V)) for V in Vs]
  d = Phis[0].shape[1]
#   us = [cvx.Variable(d) for _ in range(nV)]

#   Phi_us = [Phi * u for Phi, u in zip(Phis, us)]
#   Phi_stacked = cvx.vstack(*Phi_us)

  def obj_func(u):
    obj = 0
    us = np.array_split(u, nV)
    Phi_us = [Phi.dot(ui) for Phi, ui in zip(Phis, us)]
    for i in range(nV):
      # obj -= lm * cvx.norm(us[i])
      for j in range(i + 1, nV):
#         obj += cvx.sum_squares(Phi_us[i] - Phi_us[j])
        obj += Phi_us[i].dot(Phi_us[j])
    return -obj

  def ineq_cnts(u):
    return np.array([np.linalg.norm(ui) for ui in np.array_split(u, nV)])

  # Do some initializations using svd:
  # Initialize to right singular vectors of data to begin with 
  U0s = []
  for V in Phis:
    _, _, V0T = sue.randomized_svd(
        V, n_components=n_embedding, random_state=None)
    U0s.append(V0T.T)
  # IPython.embed()

  Us = [np.empty((d, 0)) for _ in range(nV)]
  Es = [np.empty((npts, 0)) for _ in range(nV)]
  for e_idx in range(n_embedding):
    if e_idx > 0:
      def eq_cnts(u):
        cons = np.empty((0, 1))
        us = np.array_split(u, nV)
        for i, ui in enumerate(us):
          cons = np.r_[cons, Us[i].T.dot(ui).reshape(-1, 1)]
        return cons.squeeze()

    # if solver.lower() == "trust-constr":
    #   NLCnt = so.NonlinearConstraint(ineq_cnts, lb=0., ub=1.)
    #   constraints = ineq_cnts
    if solver.lower() == "cobyla":
      norm_max = 1.
      constraints = {
          "type": "ineq",
          "fun": lambda u: norm_max - ineq_cnts(u),
      }
      if e_idx > 0:
        # adding equality constraints
        eps = 1e-3  # leeway for inequality violation
        constraints = [
            {
                "type": "ineq",
                "fun": lambda u: eq_cnts(u) + eps,
            }, {
                "type": "ineq",
                "fun": lambda u: -eq_cnts(u) + eps,
            }, constraints]
    else:
      constraints = []

    u0s_idx = [
        tu.get_independent_component(
            U0s[idx][:, e_idx], Us[idx], normalize=True).squeeze()
        for idx in range(nV)
    ]
    u0 = np.concatenate(u0s_idx)
    # eps = 1e-3
    # u0 = np.ones(nV * d) / np.sqrt(nV * d * (1 + eps))
    # u_sol = so.fmin_cobyla(obj_func, u0, cons=[ineq_cnts])
    sol = so.minimize(obj_func, u0, constraints=constraints, method=solver)
    u_sol = np.squeeze(sol.x)

    for idx, u in enumerate(np.array_split(u_sol, nV)):
      # Make sure basis continues being orthonormal:
      # TODO: maybe don't normalize?
      u = tu.get_independent_component(u, Us[idx], normalize=True)
      Us[idx] = np.c_[Us[idx], u.reshape(-1, 1)]
      Es[idx] = np.c_[Es[idx], Phis[idx].dot(u)]

    IPython.embed()
  # Q = temp_stupid_Q(nV, npts)
  # obj = cvx.quad_form(Phi_stacked, Q)
#   obj = 0
#   for i in range(nV):
#     obj -= lm * cvx.norm(us[i])
#     for j in range(i + 1, nV):
#       obj += cvx.sum_squares(Phi_us[i] - Phi_us[j])
#       # obj += Phi_us[i].T * Phi_us[j]
  
#   cts = [cvx.norm(u) <= 1. for u in us] # [u.T * u <= 1. for u in us]
#   prob = cvx.Problem(cvx.Minimize(obj), cts)
  # n = 3
  # s = 2
  # prob = cvx.Problem(cvx.Maximize(obj), cts)
#   result = prob.solve(verbose=True)

  IPython.embed()
  return Es, Us


def multiway_bb_si_EM(
    Vs, n_embedding=2, feature_func=None, em_iter=1000, thresh=1e-4,
    use_iemaps=True):
  # thresh -- threshold to stop searching
  # Alternate between finding latent feature and linear transform:
  # min_{U_i, Z} \sum_i||U_i^T X_i - Z||^2_F
  # s.t. U_i^T U_i = I \forall i

  nV = len(Vs)
  if nV == 2 and use_iemaps:
    # linear gmatrix and not normalizing are temporary
    return ml.instrumental_eigenmaps(
        Vs[0], Vs[1], e_dims=n_embedding,
        gmatrix_type="linear", standardize=True, rtn_svals=False)

  # if n_embedding > 1:
  #   raise NotImplementedError("Embedding size > 1 for len(Vs) > 2 coming soon.")

  # Set feature function to identity -- same as linear kernel
  if feature_func is None:
    feature_func = lambda x: x

  # Vs = [tu.center_matrix(V) for V in Vs]
  # Center data + featurize:
  npts = Vs[0].shape[0]
  Phis = [tu.center_matrix(feature_func(V)) for V in Vs]
  d = Phis[0].shape[1]

  # Do some initializations using svd:
  # Initialize to right singular vectors of data to begin with 
  Us = []
  for V in Phis:
    _, _, V0T = sue.randomized_svd(
        V, n_components=n_embedding, random_state=None)
    Us.append(V0T.T)
#   IPython.embed()

  for itr in range(em_iter):
    # Find latent space with bases constant
#     try:
    Z = np.mean([U.T.dot(Phi.T) for U, Phi in zip(Us, Phis)], axis=0)
#     except:
#       IPython.embed()

    # Recompute U's
    # IPython.embed()
    Uns = [np.linalg.lstsq(Z.T, Phi, rcond=None)[0].T for Phi in Phis]
    # Find closest orthonormal matrix
    # U = Un * (Un^T Un)^{-1/2}
#     IPython.embed()
#     Us = [Un.dot(sc.linalg.sqrtm(Un.T.dot(Un))) for Un in Uns]
    Us = [tu.gram_schmidt(Un) for Un in Uns]
#     IPython.embed()

  Es = [U.dot(Z) for U in Us]

#   IPython.embed()
  return Us, Es, Z

if __name__ == '__main__':
  np.set_printoptions(suppress=True, precision=3)
  nv = 2
  n = 1000
  d = 10
  thresh = 1e-4
  em_iter = 1000
  Vs = [tu.center_matrix(np.random.rand(n, d)) for _ in range(nv)]
  n_embedding = 10

  # # E1x, E1y, S1 = multiway_bb_si(Vs, use_iemaps=True)
  # E1xs, E1ys, S1s = ml.instrumental_eigenmaps(
  #     Vs[0], Vs[1], e_dims=n_embedding,
  #     gmatrix_type="linear", standardize=True, rtn_svals=True)
  # E1x, E1y, S1 = ml.instrumental_eigenmaps(
  #     Vs[0], Vs[1], e_dims=n_embedding,
  #     gmatrix_type="linear", standardize=False, rtn_svals=True)
  # xs, ys = Vs
  # Gx = ml.linear_gmatrix(xs)
  # Gy = ml.linear_gmatrix(ys)
  # Cx = tu.center_gram_matrix(Gx)
  # Cy = tu.center_gram_matrix(Gy)
  # CC = Cx.dot(Cy)
  # Syx = ys.T.dot(xs) / n
  # Sxy = xs.T.dot(ys) / n

  # v1, v1s = E1x[:, 0], E1xs[:, 0]
  # M, L = np.linalg.eig(CC.T / n**2)
  # M, L = np.real(M), np.real(L)
  # M2, L2 = np.linalg.eig(CC / n**2)
  # M2, L2 = np.real(M2), np.real(L2)
  # mx1, mx2 = L[:, 0], L[:, 1]
  # lx1, lx2 = M[0], M[1]
  # my1, my2 = L2[:, 0], L2[:, 1]
  # ly1, ly2 = M2[0], M2[1]
  # wx1 = xs.T.dot(mx1) / np.sqrt(lx1)
  # wx2 = xs.T.dot(mx2) / np.sqrt(lx2)
  # wy1 = ys.T.dot(my1) / np.sqrt(ly1)
  # wy2 = ys.T.dot(my2) / np.sqrt(ly2)
  # wx1n = wx1 / np.linalg.norm(wx1)
  # wx2n = wx2 / np.linalg.norm(wx2)
  # wy1n = wy1 / np.linalg.norm(wy1)
  # wy2n = wy2 / np.linalg.norm(wy2)

  # print(Es1[0])
  # Es2 = multiway_bb_si(Vs, n_embedding=n_embedding, use_iemaps=False)
  Us, Es, Z = multiway_bb_si_EM(
      Vs, n_embedding=n_embedding, feature_func=None, em_iter=em_iter,
      thresh=1e-4, use_iemaps=False)
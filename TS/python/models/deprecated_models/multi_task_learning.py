from __future__ import division

import cvxpy as cvx
import numpy as np
import scipy as sp

import IPython


def create_independent_omega(T, lambda_s):
  # Tasks are independent.
  omega = np.eye(T) * lambda_s
  return omega


def create_mean_offset_omega(T, lambda_m, lambda_o):
  # Task weight vectors are given by common mean + individual offset:
  #   w_t = w_c + v_t
  # Regularization is:
  #   \lambda_c ||w_c||^2 + \frac{\lambda_o}{T} \sum ||v_t||^2
  # Refer: Regularized Multi-Task Learning by Evgeniou and Pontil

  rho1 = lambda_o * lambda_c / (lambda_o + lambda_c) / T
  rho2 = lambda_o ** 2 / (lambda_o + lambda_c) / T

  I = np.eye(T)
  omega = I * (rho1 + rho2 * (1 - 1 / T)) - (1 - I) * rho2 / T
  return omega


def create_pairwise_penalty_omega(T, lambda_p, lambda_s):
  # Tasks have pairwise similarity as well as individual regularization.
  I = np.eye(T)
  omega = I * (lambda_s + lambda_p * (T - 1)) - (1 - I) * lambda_p
  return omega


def create_temporal_penalty_omega(T, lambda_p, lambda_s):
  # Adjacent tasks are penalized for deviation.
  omega = (np.diag([2 * lambda_p + lambda_s] * T) -
           np.diag([lambda_p] * (T - 1), -1) -
           np.diag([lambda_p] * (T - 1), -1))
  omega[0, 0] -= lambda_p
  omega[-1, -1] -= lambda_p
  return omega


def sigmoid(x):
  return 1 / (1 + np.exp(-x))



def mt_krc(training_tasks, test_tasks, omega, feature_gen):
  """
  Multi-task Kernel Ridge Classification.

  training_tasks: List of (feature vectors, labels) tuples, one for each training task.
  test_tasks: List of (feature vectors of points) for each task.
  omega: Matrix representing relationship between tasks.
  feature_gen: Feature generator.
  """
  T = len(training_tasks)
  nT = [task[0].shape[0] for task in training_tasks]

  Y = np.array([y for task in training_tasks for y in task[1]])
  Y = np.where(Y, 1, -1)  # Convert to -1 for optimization.
  phiT = [feature_gen(task[0]) for task in training_tasks]
  q = phiT[0].shape[1]

  phi = sp.linalg.block_diag(*phiT)
  L = np.kron(omega, np.eye(q))
  C = sp.linalg.block_diag(*[np.eye(nt) / nt for nt in nT])
  Linv = np.kron(np.linalg.inv(omega), np.eye(q))

  # K = phi.dot(Linv).dot(phi.T) + C
  # IPython.embed()
  # K_inv = np.linalg.inv(K)
  K_inv, K = np.load('temp.npy')

  IPython.embed()
  Z = cvx.Variable(rows=sum(nT), cols=1)
  exp_nYZ = cvx.exp(-cvx.mul_elemwise(Y, Z))

  cost = 0.5 * cvx.quad_form(Z, K_inv) + cvx.sum_entries(cvx.log1p(exp_nYZ))
  obj = cvx.Minimize(cost)
  prob = cvx.Problem(obj)
  prob.solve()

  if prob.status == "optimal":
    print (prob.status, prob.value)
  else:
    print("Error in optimization.")
    return None

  Z_opt = Z.value

  nT_test = [task[0].shape[0] for task in test_tasks]
  phiT_test = [feature_gen(task[0]) for task in test_tasks]
  phi_test = sp.linalg.block_diag(*phiT_test)
  K_cross = phi_test.dot(np.linalg.inv(L)).dot(phi.T)

  Z_star = K_cross.dot(K_inv).dot(Z_opt)
  Y_pred_stack = (Z_star >= 0).astype(int)
  split_inds = np.cumsum(nT_test)
  Y_pred = np.split(Y_pred_stack, split_inds[:-1])

  return Y_pred

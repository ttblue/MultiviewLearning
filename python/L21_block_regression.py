import time

import itertools
import numpy as np
import scipy.sparse as ss

import IPython

def L21_block_regression(
    Y, X, lmbda, tol=1e-6, max_iterations=100, verbose=True):
  # % input: Y - matrix, each column is a function
  # %        X - cell array of feature matrices
  # %        lmbda - regularization strength
  # %        tol - stoping criterion
  # %        max_iterations - maximum number of iterations
  # % output: W 
  # % Minimizes sum_i |Y(:,i)-X{i}*W(:,i)|_{2}^2 + lambda*|W|_{2,1}
  # %
  # % algorithm adapted from:
  # % @inproceedings{nie2010efficient,
  # %   title={Efficient and robust feature selection via joint L2, 1-norms minimization},
  # %   author={Nie, Feiping and Huang, Heng and Cai, Xiao and Ding, Chris H},
  # %   booktitle={Advances in neural information processing systems},
  # %   pages={1813--1821},
  # %   year={2010}
  # % }
  # %

  # if ~strcmp(class(Y),'double')
  #   disp(strcat(['ERROR: Invalid class for Y (',class(Y),'), expected class double.']));
  #   return;
  # end
  # if ~strcmp(class(X),'cell')
  #   disp(strcat(['ERROR: Invalid class for X (',class(X),'), expected class cell.']));
  #   return;
  # end
  # if length(X)~=size(Y,2) && length(X)~=1
  #   disp(strcat(['ERROR: length of cell array X should be 1 or size(Y,2).']));
  #   return;
  # end
  # for i=1:length(X)
  # if ~strcmp(class(X{i}),'double')
  #   disp(strcat(['ERROR: Invalid class for X{',num2str(i),'} (',class(X{i}),'), expected class double.']));
  #   return;
  # end
  # if size(X{i},1)~=size(Y,1)
  #   disp(strcat(['ERROR: Number of rows not consistent for Y and X{',num2str(i),'}.']));
  #   return;
  # end
  # if size(X{i},2)~=size(X{1},2)
  #   disp(strcat(['ERROR: Number of rows not consistent for X{',num2str(i),'} and X{1}.']));
  #   return;
  # end
  # end

  
  Y = [np.array(yval) for yval in Y]
  X = [np.array(xval) for xval in X]

  T = len(X)
  if len(Y) != T:
    raise ValueError("X and Y not of the same size.")

  # Assume all X's are the same shape.
  n, m = X[0].shape
  d = Y[0].shape[0]
  last_objective_value = np.inf
  D = ss.spdiags(np.ones(m), 0, m, m)
  if m < n:
    L = ss.spdiags((lmbda ** 2) * np.ones(m), 0, m, m)
    XX = [x.T.dot(x) for x in X]
    IPython.embed()
    XY = [X[((j - 1) % T)].T.dot(Y[j]) for j in xrange(d)]
    # XY = []
    # for i in xrange(T):
    #   XX.append(X[i].T.dot(X[i]))
    # for j in xrange(d):
    #   jdx = ((j - 1) % T)
    #   XY.append(X[jdx].T.dot(Y[j]) )

  else:
    L = ss.spdiags((lmbda ** 2) * np.ones(n), 0, n, n)

  U1 = np.zeros((m, d))
  U2 = np.zeros((n, d))

  for it in xrange(max_iterations):
    for j in xrange(d):
      jdx = ((j - 1) % T)
      if m < n:
        v = (1 / lmbda ** 2) * (Y[j] - X[jdx] * (
            np.linalg.pinv(L + D.dot(XX[jdx])).dot(D.dot(XY[j]))))
      else:
           v = np.pinv(L + X[jdx].dot(D.dot(X[jdx].T))).dot(Y[j])
      U1[:, j] = D.dot(X[jdx].T).dot(v)
      U2[:, j] = lmbda*v

    D = ss.spdiags(2 * np.sqrt(np.sum(U1**2, 1)), 0, m, m)
    objective_value = D.sum()
    delta = last_objective_value - objective_value
    if np.isnan(delta):
      break

    last_objective_value = objective_value
    if delta < tol:
      break

    if verbose:
      print("Iteration: %i: %.3f" % (it + 1, delta))

  if np.isnan(delta):
    print("ERROR: Something is terribly wrong here.")

  if it == max_iterations:
    print("WARNING: Max iterations %i reached with tol: %.3f,"
          " final delta: %.3f." % (max_iterations, tol, delta))

  return U1


def generate_polynomials(X, max_degree=3, include_zero=True):
  X = np.array(X)
  if len(X.shape) < 2:
    X = np.atleast_2d(X).T

  n, d = X.shape
  PX = np.ones((n, 1)) if include_zero else np.empty((n,0))
  idxs = range(d)
  for degree in xrange(1, max_degree + 1):
    d_combs = itertools.combinations_with_replacement(idxs, degree)
    for dc in d_combs:
      xcomb = np.ones(n)
      for cidx in dc:
        xcomb *= X[:, cidx]
      PX = np.c_[PX, xcomb.reshape(-1, 1)]

  return PX
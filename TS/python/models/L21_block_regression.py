import time

import itertools
import numpy as np
import scipy.sparse as ss

import IPython


from numpy import inf, square, sqrt, zeros
from numpy.linalg import solve
from scipy.sparse import diags





def L21_block_regression(Y, X, lam,tol=1e-6,max_iterations=100,verbose=0):
  # input: Y - list of output vectors
  #        X - list of feature matrices
  #        lam - regularization strength
  #        tol - stoping criterion
  #        max_iterations - maximum number of iterations
  # output: W 
  # Minimizes sum_i |Y{i}-X{i}*W(:,i)|_{2}^2 + lambda*|W|_{2,1}
  #
  # algorithm adapted from:
  # @inproceedings{nie2010efficient,
  #   title={Efficient and robust feature selection via joint L2, 1-norms minimization},
  #   author={Nie, Feiping and Huang, Heng and Cai, Xiao and Ding, Chris H},
  #   booktitle={Advances in neural information processing systems},
  #   pages={1813--1821},
  #   year={2010}
  # }
  #
  X, Y = np.array(X), np.array(Y)
  lam = lam ** 2
  T = len(X)
  m = X[0].shape[1]
  last_objective_value = inf
  D = diags([1]*m)
  XX, XY = [], []
  n = [X[i].shape[0] for i in range(T)]
  for i in range(T): 
    XX.append(X[i].T*X[i])
    XY.append(X[i].T*Y[i])
  U1 = zeros((m,T));
  for it in range(max_iterations):
    for i in range(T):
      if m<n[i]: v = (1./(lam*n[i]))*( Y[i]-X[i]*solve(diags([lam*n[i]]*m)+D*XX[i],D*XY[i])  )
      else: v = solve(diags([lam*n[i]]*n[i])+X[i]*D*X[i].T,Y[i])
      U1[:,i] = (D*X[i].T*v).flat
    D = diags(2.*sqrt(square(U1).sum(1)))
    objective_value = D.sum()
    delta = abs( last_objective_value-objective_value )
    last_objective_value = objective_value
    if verbose: print(str(it+1)+": "+str(delta))
    if delta < tol: break
  if it+1==max_iterations: print("WARNING: max ("+str(max_iterations)+") iterations reached with tol: "+str(tol)+". Final delta: "+str(delta))
  return U1

if __name__=="__main__":
  import numpy as np
  I,w,X,Y,W = 5,4,[],[],[]
  D = diags([v*1 for v in (np.random.rand(w,1)>0.5).flat])
  for i in range(I):
    #n = np.random.randint(100,10000)
    n = 100
    X.append( np.asmatrix(np.random.rand(n,w)) )
    W.append( D*np.asmatrix(np.random.rand(w,1)) )
    Y.append( np.asmatrix( X[i]*W[i]+np.asmatrix(np.random.normal(scale=0.02,size=(n,1))) ) )
  W_ = L21_block_regression(Y,X,1e-1,verbose=1)
  print(D.todense())
  print(W)
  print(W_)


# def matrix_squeeze(mat):
#   return np.array(mat).squeeze()


# def L21_block_regression(
#     Y, X, lmbda, tol=1e-6, max_iterations=100, verbose=True):
#   # % input: Y - matrix, each column is a function
#   # %        X - cell array of feature matrices
#   # %        lmbda - regularization strength
#   # %        tol - stoping criterion
#   # %        max_iterations - maximum number of iterations
#   # % output: W 
#   # % Minimizes sum_i |Y(:,i)-X{i}*W(:,i)|_{2}^2 + lambda*|W|_{2,1}
#   # %
#   # % algorithm adapted from:
#   # % @inproceedings{nie2010efficient,
#   # %   title={Efficient and robust feature selection via joint L2, 1-norms minimization},
#   # %   author={Nie, Feiping and Huang, Heng and Cai, Xiao and Ding, Chris H},
#   # %   booktitle={Advances in neural information processing systems},
#   # %   pages={1813--1821},
#   # %   year={2010}
#   # % }
#   # %

#   # if ~strcmp(class(Y),'double')
#   #   disp(strcat(['ERROR: Invalid class for Y (',class(Y),'), expected class double.']));
#   #   return;
#   # end
#   # if ~strcmp(class(X),'cell')
#   #   disp(strcat(['ERROR: Invalid class for X (',class(X),'), expected class cell.']));
#   #   return;
#   # end
#   # if length(X)~=size(Y,2) && length(X)~=1
#   #   disp(strcat(['ERROR: length of cell array X should be 1 or size(Y,2).']));
#   #   return;
#   # end
#   # for i=1:length(X)
#   # if ~strcmp(class(X{i}),'double')
#   #   disp(strcat(['ERROR: Invalid class for X{',num2str(i),'} (',class(X{i}),'), expected class double.']));
#   #   return;
#   # end
#   # if size(X{i},1)~=size(Y,1)
#   #   disp(strcat(['ERROR: Number of rows not consistent for Y and X{',num2str(i),'}.']));
#   #   return;
#   # end
#   # if size(X{i},2)~=size(X{1},2)
#   #   disp(strcat(['ERROR: Number of rows not consistent for X{',num2str(i),'} and X{1}.']));
#   #   return;
#   # end
#   # end

  
#   Y = [np.array(yval) for yval in Y]
#   X = [np.array(xval) for xval in X]

#   T = len(X)
#   if len(Y) != T:
#     raise ValueError("X and Y not of the same size.")

#   lmbda = float(lmbda)
#   # Assume all X's are the same shape.
#   n, m = X[0].shape
#   d = len(Y)
#   last_objective_value = np.inf
#   D = ss.spdiags(np.ones(m), 0, m, m)
#   if m < n:
#     L = ss.spdiags((lmbda ** 2) * np.ones(m), 0, m, m)
#     XX = [x.T.dot(x) for x in X]
#     XY = [X[j % T].T.dot(Y[j]) for j in xrange(d)]
#     # XY = []
#     # for i in xrange(T):
#     #   XX.append(X[i].T.dot(X[i]))
#     # for j in xrange(d):
#     #   jdx = ((j - 1) % T)
#     #   XY.append(X[jdx].T.dot(Y[j]) )

#   else:
#     L = ss.spdiags((lmbda ** 2) * np.ones(n), 0, n, n)

#   U1 = np.zeros((m, d))
#   U2 = np.zeros((n, d))
#   # IPython.embed()

#   for it in xrange(max_iterations):
#     for j in xrange(d):
#       jdx = j % T
#       if m < n:
#         # IPython.embed()
#         v = (1 / lmbda ** 2) * (Y[j] - X[jdx].dot(
#           # np.linalg.solve(L + D.dot(XX[jdx]), D.dot(XY[j]))))
#             matrix_squeeze(np.linalg.inv(L + D.dot(XX[jdx])).dot(D.dot(XY[j])))))
#       else:
#         # v = np.linalg.pinv(L + X[jdx].dot(D.dot(X[jdx].T))).dot(Y[j])
#         v = np.linalg.solve(L + X[jdx].dot(D.dot(X[jdx].T)), Y[j])
#       # IPython.embed()
#       U1[:, j] = matrix_squeeze(D.dot(X[jdx].T).dot(v))
#       U2[:, j] = matrix_squeeze(lmbda*v)

#     D = ss.spdiags(2 * np.sqrt(np.sum(U1**2, 1)), 0, m, m)
#     objective_value = D.sum()
#     delta = last_objective_value - objective_value

#     if verbose:
#       print("Iteration: %i: %.3f" % (it + 1, delta))
#     # IPython.embed()

#     if np.isnan(delta):
#       break

#     last_objective_value = objective_value
#     if np.abs(delta) < tol:
#       break

#   if np.isnan(delta):
#     print("ERROR: Something is terribly wrong here.")

#   if it == max_iterations:
#     print("WARNING: Max iterations %i reached with tol: %.3f,"
#           " final delta: %.3f." % (max_iterations, tol, delta))

#   U1 = [u.squeeze() for u in np.split(U1, np.arange(1, d), axis=1)]
#   return U1


# def generate_polynomials(X, max_degree=3, include_zero=True):
#   X = np.array(X)
#   if len(X.shape) < 2:
#     X = np.atleast_2d(X).T

#   n, d = X.shape
#   PX = np.ones((n, 1)) if include_zero else np.empty((n,0))
#   idxs = range(d)
#   for degree in xrange(1, max_degree + 1):
#     d_combs = itertools.combinations_with_replacement(idxs, degree)
#     for dc in d_combs:
#       xcomb = np.ones(n)
#       for cidx in dc:
#         xcomb *= X[:, cidx]
#       PX = np.c_[PX, xcomb.reshape(-1, 1)]

#   return PX
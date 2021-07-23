# Defining various rigid and non-rigid transforms with torch support

import numpy as np
import scipy.spatial.distance as ssd
import torch
import torch.nn as nn


torch.set_default_dtype(torch.float64)


# Abstract class for transforms
class Transform(object):
  def __init__(self):
    self.params = {}

  def __call__(self, X):
    raise NotImplementedError

  def numpy_transform(self, X):
    raise NotImplementedError    

  def reset_transform(self, **params):
    raise NotImplementedError

  def parameter_generator(self):
    # For torch params as a generator
    return (p for p in self.params.values())

  def get_parameters(self, as_numpy=True):
    # For numpy params or torch
    op_params = {}
    for param in self.params:
      op_params[param] = (
          self.params[param].detach().numpy() if as_numpy else
          self.params[param])
    return op_params


# Abstract cost or regularizer class
class CostOrRegularizer(object):
  def __init__(self, transform, coeff):
    self.transform = transform
    self.coeff = coeff

  def __call__(self):
    raise NotImplementedError


################################################################################
# Basic rigid transform
class AffineTransform(Transform):

  def __init__(self, dim, R=None, t=None):
    super(AffineTransform, self).__init__()
    self.dim = dim
    self.reset_transform(R, t)

  def reset_transform(self, R=None, t=None):
    R = torch.eye(self.dim) if R is None else torch.from_numpy(R)
    t = torch.zeros(self.dim, 1) if t is None else torch.from_numpy(t)

    self.R = nn.Parameter(R, requires_grad=True)
    self.t = nn.Parameter(t, requires_grad=True)

    self.params["R"] = self.R
    self.params["t"] = self.t

  def __call__(self, X):
    if not isinstance(X, torch.Tensor):
      # Assuming it is numpy arry
      X = torch.from_numpy(X)
    return (torch.mm(self.R, X.t()) + self.t).t()

  def numpy_transform(self, X):
    R = self.R.detach().numpy()
    t = self.t.detach().numpy()
    return (R.dot(X.T) + t).T


################################################################################
def pairwise_euclidean(x, y=None, sq=False):
    """
    Input: 
      x -- Nxd torch tensor
      y -- optional Mxd torch tensor
    Output:
      dist -- NxM torch tensor where dist[i,j] is the square norm between
          x[i,:] and y[j,:]
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    
    Note:Iif y is not given then use y=x.

    From - 
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        if len(y.shape) < 2:
          y = y.unsqueeze(0)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist_final = torch.clamp(dist, 0.0, np.inf)
    if not sq:
      return torch.sqrt(dist_final)
    else:
      return dist_final
    # Ensure diagonal is zero if x=y for numerical stability issues
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    # IPython.embed()


def tps_kernel_dim3(distmat):
  return - distmat  


class ThinPlateSplineTransform(Transform):

  def __init__(self, dim, base_grid, A=None, B=None, c=None):
    # Base grid is the grid of points to act as basis for warping
    super(ThinPlateSplineTransform, self).__init__()
    self.dim = dim
    self.base_grid = None
    self.reset_transform(dim, base_grid, A, B, c)

  def kernel(self, distmat):
    if self.dim != 3:
      raise NotImplementedError("Only dim 3 implemented.")

    if self.dim == 3:
      return tps_kernel_dim3(distmat)

  def reset_transform(self, dim=None, base_grid=None, A=None, B=None, c=None):
    if base_grid is None and self.base_grid is None:
      raise ValueError("Please provide base grid for warping.")

    self.dim = self.dim if dim is None else dim
    if base_grid is not None:
      # We want our coefficients A to be in the null-space of the columns of
      # base grid; otherwise our TPS integral loss is not guaranteed to be
      # finite
      self.base_grid = torch.from_numpy(base_grid)
      constraints = np.c_[base_grid, np.ones((base_grid.shape[0], 1))]
      *_, vh = np.linalg.svd(constraints.T)
      self.bg_null_space = torch.from_numpy(vh.T[:, constraints.shape[1]:])

    self._n = self.bg_null_space.shape[1]

    A = torch.zeros(self._n, self.dim) if A is None else torch.from_numpy(A)
    B = torch.eye(self.dim) if B is None else torch.from_numpy(B)
    c = torch.zeros(self.dim, 1) if c is None else torch.from_numpy(c)

    self.A = nn.Parameter(A, requires_grad=True)
    self.B = nn.Parameter(B, requires_grad=True)
    self.c = nn.Parameter(c, requires_grad=True)

    self.params["A"] = self.A
    self.params["B"] = self.B
    self.params["c"] = self.c

  def __call__(self, X):
    if not isinstance(X, torch.Tensor):
      # Assuming it is numpy arry
      X = torch.from_numpy(X)

    distmat = pairwise_euclidean(X, self.base_grid)
    K = self.kernel(distmat)
    return (torch.mm(K, torch.mm(self.bg_null_space, self.A)) +
            torch.mm(X, self.B.t()) + self.c.t())

  def numpy_transform(self, X):
    A = self.A.detach().numpy()
    B = self.B.detach().numpy()
    c = self.c.detach().numpy()
    bgNS = self.bg_null_space.detach().numpy()

    distmat = ssd.cdist(X, self.base_grid.numpy())
    K = self.kernel(distmat)
    return (K.dot(bgNS.dot(A)) + X.dot(B.T) + c.T)


# Abstract cost or regularizer class
class TPSIntegralCost(CostOrRegularizer):
  def __init__(self, transform, coeff):
    super(TPSIntegralCost, self).__init__(transform, coeff)
    self.K = None

  def __call__(self):
    if self.K is None:
      base_grid = self.transform.base_grid
      distmat = pairwise_euclidean(base_grid, base_grid)
      self.K = self.transform.kernel(distmat)

    A_ns = self.transform.A
    bg_ns = self.transform.bg_null_space
    A = torch.mm(bg_ns, A_ns)

    return self.coeff * torch.trace(torch.mm(torch.mm(A.t(), self.K), A))
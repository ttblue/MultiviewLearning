# Code to find transforms and stuff
import numpy as np


def make_homogeneous_tfm(R, t, dtype=np.float64):
  d = R.shape[0]
  t = np.atleast_2d(t).T
  T = np.r_[np.c_[R, t], np.atleast_2d([0] * d + [1])].astype(dtype)
  return T


def generate_2d_rot(theta):
  new_x = np.atleast_2d([np.cos(theta), np.sin(theta)]).T
  new_y = np.atleast_2d([- np.sin(theta), np.cos(theta)]).T
  R = np.c_[new_x, new_y].astype(np.float64)
  return R


def make_right_handed(R):
  # Makes coordinate system right-handed, keeping the first column the same.
  R = np.array(R)
  dim = R.shape[0]
  if dim not in [2, 3]:
    # Can use: http://article.sapub.org/10.5923.j.ajcam.20170702.04.html
    raise NotImplementedError("Not implemented for dim %i." % dim)

  if dim == 3:
    x, y, _ = np.split(R, [1, 2], axis=1)
    x = x.squeeze() / np.linalg.norm(x)
    y = y.squeeze() / np.linalg.norm(y)
    z = np.cross(x, y)
    y = np.cross(z, x)
    return np.c_[x, y, z]
  else:
    x = R[:, 0]
    x = x / np.linalg.norm(x)
    thetax = np.arctan2(x[1], x[0])
    thetay = thetax + np.pi / 2
    y = np.array([np.cos(thetay), np.sin(thetay)])
    return np.c_[x, y]


def find_rotation_between_two_vectors(v1, v2, eps=1e-6):
  # Find R such that R*v1 = v2 in terms of directions
  # eps -- threshold for determining if two vectors are the same or negated.

  v1 = np.array(v1)
  v2 = np.array(v2)
  dim = v1.shape[0]
  assert v2.shape[0] == dim

  if dim not in [2, 3]:
    # Can use: http://article.sapub.org/10.5923.j.ajcam.20170702.04.html
    raise NotImplementedError("Not implemented for dim %i." % dim)

  if dim == 3:
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    x = v1
    z = np.cross(v1, v2)
    znorm = np.linalg.norm(z)
    while znorm < eps: # If they're aligned, rotate them about an arbitrary z
      y = np.random.randn(3)
      z = np.cross(v1, y)
      znorm = np.linalg.norm(z)
    z = z / znorm
    y = np.cross(z, x)

    R = np.c_[x, y, z]
    vy = R.T.dot(v2)
    theta = np.arctan2(vy[1], vy[0])
    Rtheta = np.eye(3)
    Rtheta[:2, :2] = generate_2d_rot(theta)

    return R.dot(Rtheta.dot(R.T))

  else:
    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])
    theta1 = 2 * np.pi + theta1 if theta1 < 0 else theta1
    theta2 = 2 * np.pi + theta2 if theta2 < 0 else theta2

    theta = theta2 - theta1
    return generate_2d_rot(theta)


def guess_best_transform(pts1, pts2):
  # Find a transform such that R * pts1 + t = pts2
  # Do PCA on both and align the axes.
  pts1 = np.asarray(pts1)
  pts2 = np.asarray(pts2)
  dim = pts1.shape[1]
  assert pts2.shape[1] == dim

  pm1 = np.mean(pts1, axis=0)
  pm2 = np.mean(pts2, axis=0)
  pts1_centered = pts1 - pm1
  pts2_centered = pts2 - pm2

  _, _, v1 = np.linalg.svd(pts1_centered, full_matrices=False)
  _, _, v2 = np.linalg.svd(pts2_centered, full_matrices=False)
  # IPython.embed()

  if dim in [2, 3]:
    v1 = make_right_handed(v1)
    v2 = make_right_handed(v2)

  T1_to_W = make_homogeneous_tfm(np.eye(dim), -pm1)
  R1_to_W = make_homogeneous_tfm(v1, np.zeros(dim))
  TW_to_2 = make_homogeneous_tfm(np.eye(dim), pm2)
  RW_to_2 = make_homogeneous_tfm(v2.T, np.zeros(dim))

  RT1_to_W = R1_to_W.dot(T1_to_W)
  RTW_to_2 = TW_to_2.dot(RW_to_2)

  RT1_to_2 = RTW_to_2.dot(RT1_to_W)
  R, t = RT1_to_2[:dim, :dim], RT1_to_2[:dim, -1]

  return R, t


def CM(dim):
  # centering matrix
  return np.eye(dim) - 1. / dim * np.ones((dim, dim))


def center_matrix(X):
  X = np.atleast_2d(X)
  return X - X.mean(0)


def center_gram_matrix(G):
  H = CM(G.shape[0])
  return H.dot(G).dot(H)
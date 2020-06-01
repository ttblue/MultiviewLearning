import itertools
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(mat, classes,
                title="Mutual Information",
                kill_diag=True,
                cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if kill_diag:
      mat[xrange(mat.shape[0]), xrange(mat.shape[0])] = 0.
  plt.imshow(mat, interpolation="nearest", cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  # print("Matrix:")
  # print(mt)

  thresh = mat.max() / 2.
  for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
    if kill_diag and i == j: continue
    plt.text(j, i, "%.2f"%mat[i, j],
             horizontalalignment="center",
             color="white" if mat[i, j] > thresh else "black")

  plt.tight_layout()


def plot_spaghet(
    x, ys, plot_mean=True, confidence_sigma=-1, color=None, show=False):
  """
  Plot utility for spaghetti plots.
  """
  color = 'b' if color is None else color

  # fig, ax = plt.subplots()
  # ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)

  for y in ys:
    plt.plot(x, ys, color=color)

  if plot_mean:
    y_mu = ys.mean(0)
    plt.plot(x, y_mu, label="Mean", linestyle="--", linewidth=10)
    if confidence_sigma > 1:
      y_std = ys.std(0)
      y_ub = y_mu + confidence_sigma * y_std
      y_lb = y_mu - confidence_sigma * y_std
      plt.fill_between(x, y_lb, y_ub, color=color, alpha=0.1)

  if show:
    plt.show()

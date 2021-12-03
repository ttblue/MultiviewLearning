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
      mat[range(mat.shape[0]), rrange(mat.shape[0])] = 0.
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
    plt.plot(x, y, color=color)

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


def mv_bar_plots(mv_accs, n_views, title="", ymin=None, ymax=None):
  view_range = list(range(n_views))

  colors = ["g", "b", "r", "k"]

  x_ticks = []
  x_labels = []
  y_vals = []
  b_colors = []

  x_curr = 1.

  b_size = 0.9
  x_space = 2.

  for i, nv in enumerate(range(n_views, 0, -1)):
    view_subsets = itertools.combinations(view_range, nv)
    for sub in view_subsets:
      x_ticks.append(x_curr)
      x_labels.append(sub)
      y_vals.append(mv_accs[sub])
      b_colors.append(colors[i])
      x_curr += 1

    x_curr += x_space

  pad_frac = 0.2

  if ymin is None or ymax is None:
    ymin, ymax = np.min(y_vals), np.max(y_vals)
    ydiff = ymax - ymin
    ymin -= ydiff * pad_frac
    ymax += ydiff * pad_frac

    ymin = max(0, ymin)
    ymax = min(1, ymax)

  y_ticks = np.linspace(ymin, ymax, 5).round(2)

  ax = plt.subplot()
  ax.bar(x_ticks, height=y_vals, width=b_size, color=b_colors)
  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_labels, fontdict={'fontsize': 15})
  ax.set_ylim(ymin, ymax)
  ax.set_xlabel("Available views", fontsize=20)
  ax.set_yticks(y_ticks)
  ax.set_yticklabels(y_ticks, fontdict={'fontsize': 15})
  ax.set_ylabel("Sleep stage classification accuracy", fontsize=20)

  if title:
    plt.title(title, fontsize=30)
  # plt.legend([], ["0: ECG", "1: BP", "2: EEG"])

  plt.show()

  return ymin, ymax


def mv_bar_plots2(
    tr_accs, te_accs, n_views, sub_names={}, title="", ymin=None, ymax=None):
  view_range = list(range(n_views))

  colors = ["g", "b", "r", "k"]

  x1_vals = []
  x2_vals = []
  x_ticks = []
  x_labels = []
  y1_vals = []
  y2_vals = []
  # b_colors = []

  x_curr = 1.

  b_size = 1.0
  x_space = 2.

  for vi in view_range:
    if vi not in sub_names:
      sub_names[vi] = str(vi)

  ax = plt.subplot()
  xtext_offset = - 0.1
  ytext_offset = + 0.01
  for i, nv in enumerate(range(n_views, 0, -1)):
    view_subsets = itertools.combinations(view_range, nv)
    for sub in view_subsets:
      x1_vals.append(x_curr)
      y1_vals.append(tr_accs[sub])
      ax.text(
          x_curr + xtext_offset, tr_accs[sub] + ytext_offset, "%.2f" % tr_accs[sub],
          va="center", fontweight="bold", fontsize=10)
      # b_colors.append("b")
      x_curr += 1

      x2_vals.append(x_curr)
      y2_vals.append(te_accs[sub])
      ax.text(
          x_curr + xtext_offset, te_accs[sub] + ytext_offset, "%.2f" % te_accs[sub],
          va="center", fontweight="bold", fontsize=10)
      # b_colors.append("r")

      sub_lbl = " + ".join([sub_names[vi] for vi in sub])
      x_ticks.append(x_curr - 0.5)
      x_labels.append(sub_lbl)
      x_curr += 1.2

    x_curr += x_space

  pad_frac = 0.2

  if ymin is None or ymax is None:
    y_vals = y1_vals + y2_vals
    ymin, ymax = np.min(y_vals), np.max(y_vals)
    ydiff = ymax - ymin
    ymin -= ydiff * pad_frac
    ymax += ydiff * pad_frac

    ymin = max(0, ymin)
    ymax = min(1, ymax)

  y_ticks = np.linspace(ymin, ymax, 5).round(2)

  ax.bar(x1_vals, height=y1_vals, width=b_size, color="b", label="Train")
  ax.bar(x2_vals, height=y2_vals, width=b_size, color="r", label="Test")

  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_labels, fontdict={'fontsize': 15})
  ax.set_ylim(ymin, ymax)
  ax.set_xlabel("Available views", fontsize=20)
  ax.set_yticks(y_ticks)
  ax.set_yticklabels(y_ticks, fontdict={'fontsize': 15})
  ax.set_ylabel("Sleep stage classification accuracy", fontsize=20)

  plt.legend(fontsize=20)
  if title:
    plt.title(title, fontsize=30)

  plt.show()

  return ymin, ymax

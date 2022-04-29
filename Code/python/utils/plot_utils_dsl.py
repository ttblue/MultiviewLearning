import itertools
import matplotlib, matplotlib.pyplot as plt
import numpy as np


_SUB_COLOR_MAP = {
    1: "r",
    2: "g",
    3: "b",
}
def plot_dsl_accs(sub_accs, nv_accs, title=""):
  matplotlib.rcParams["ytick.labelsize"] = 15
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)

  xticks = [0, 1, 2, 3]
  xvals = None
  for sub, sub_acc in sub_accs.items():
    if xvals is None:
      xvals = np.sort(list(sub_acc.keys()))

    yvals = [sub_acc[x] for x in xvals]
    color = _SUB_COLOR_MAP[len(sub)]
    ax.plot(xticks, yvals, linewidth=1, alpha=0.5, color=color)

  for nv, nv_acc in nv_accs.items():
    yvals = [nv_acc[x][0] for x in xvals]
    color = _SUB_COLOR_MAP[nv]
    lbl = "%i views available" % (4 - nv)
    ax.plot(xticks, yvals, linewidth=3, alpha=1, label=lbl, color=color)

  # ax.set_xscale("log")
  ax.set_xticks(xticks, fontsize=15)
  ax.set_xticklabels(xvals, fontsize=15)
  # ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
  ax.set_xlabel("DSL trade-off coefficient", fontsize=30)
  ax.set_ylabel("Benchmark MNIST classification accuracy", fontsize=30)

  ax.legend(fontsize=15)

  if title:
    plt.suptitle(title, fontsize=30)

  plt.show()
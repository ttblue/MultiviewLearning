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
#!/usr/bin/env python
import itertools
import numpy as np
import os
import pickle
import scipy
from sklearn import manifold, decomposition
import time
import torch
from torch import nn, functional
import umap

from dataprocessing import split_single_view_dsets as ssvd
from utils import math_utils, torch_utils, utils

from matplotlib import patches, pyplot as plt, tri
from mpl_toolkits.mplot3d import Axes3D

from tests.test_ac_flow_jp import get_sampled_cat, plot_digit, plot_many_digits

import IPython

from sklearn import ensemble, kernel_ridge, linear_model, neural_network, svm


_mnist_w = _mnist_h = 28
_mnist_w_2 = _mnist_h_2 = 14
_mnist_v4_inds = ssvd.get_mnist_split_inds(n_views=4, shape="grid")
def get_sampled_cat_grid(sample_vs, base_dat):
  sampled_cat = base_dat.copy()
  for vi, svdat in sample_vs.items():
    vi_inds = _mnist_v4_inds[vi]
    sampled_cat[:, vi_inds] = svdat
  return np.clip(sampled_cat, 0., 1.)


_rect_args = {
    "TL": ((0, 0), _mnist_w_2, _mnist_h_2),
       0: ((0, 0), _mnist_w_2, _mnist_h_2),
    "TR": ((_mnist_w_2, 0), _mnist_w_2, _mnist_h_2),
       1: ((_mnist_w_2, 0), _mnist_w_2, _mnist_h_2),
    "BL": ((0, _mnist_h_2), _mnist_w_2, _mnist_h_2),
       2: ((0, _mnist_h_2), _mnist_w_2, _mnist_h_2),
    "BR": ((_mnist_w_2, _mnist_h_2), _mnist_w_2, _mnist_h_2),
       3: ((_mnist_w_2, _mnist_h_2), _mnist_w_2, _mnist_h_2),
}
_view_map = {
    0: "TL", 1: "TR", 2: "BL", 3: "BR",
}


def get_rect_args(perm, dig_nums):
  if not isinstance(dig_nums, list):
    dig_nums = [dig_nums]

  locs = [_view_map[v] for v in perm]
  base_rect_args = [_rect_args[loc] for loc in locs]
  all_rect_args = []

  for dnum in dig_nums:
    _w_offset = dnum * (_mnist_w + 1) + (dnum == 2)
    d_rect_args = [
      ((xy[0] + _w_offset - 0.5, xy[1] - 0.5), w, h)
      for xy, w, h in base_rect_args  
    ]
    all_rect_args.extend(d_rect_args)

  return all_rect_args


def plot_many_digits(
    pred_digits, true_digits, perms=[], first=None, grid_size=(10, 10),
    rcolors="r", title="", show_ids=False):

  if not isinstance(pred_digits, list):
    pred_digits = [pred_digits]
  pred_digits = [np.copy(pdigs) for pdigs in pred_digits]
  nsets = len(pred_digits)

  # print(pred_digits.dtype)
  # pred_digits = [dig.reshape(28, 28) for dig in pred_digits]
  pred_digits = [pdigs.reshape(-1, _mnist_h, _mnist_w) for pdigs in pred_digits]
  if true_digits is not None:
    # true_digits = [dig.reshape(28, 28) for dig in true_digits]
    true_digits = true_digits.reshape(-1, _mnist_h, _mnist_w)
  ndigs = len(pred_digits[0])
  dignums = list(range(nsets + 1))

  if not isinstance(rcolors, list):
    rcolors = [rcolors] * ndigs

  nrows, ncols = grid_size
  if true_digits is None:
    fig, axs = plt.subplots(
        nrows, ncols, gridspec_kw = {'wspace':0.05, 'hspace':0.05})
  else:
    fig, axs = plt.subplots(nrows, ncols)

  # IPython.embed()
  # plt.subplots_adjust(wspace=0, hspace=0)
  # rect_locs = []
  # all_rect_args = [_rect_args[loc] for loc in rect_locs]
  white_line = np.ones((pred_digits[0][0].shape[1], 1))
  dig_idx = 0
  plotted_first = (first is None)
  for ri in range(nrows):
    # if dig_idx >= ndigs:
    #   break
    for ci in range(ncols):
      if ncols == 1:
        ax = axs[ri]
      elif nrows == 1:
        ax = axs[ci]
      else:
        ax = axs[ri, ci]
      if dig_idx >= ndigs:
        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_visible(False)
        ax.yaxis.set_ticks([])
        ax.axis("off")
        dig_idx += 1
        continue

      rcolor = rcolors[dig_idx]
      facecolor = (1, 0, 0, 0.15) if rcolor == "r" else (0, 1, 0, 0.15)
      if plotted_first:
        print("Plotting digit %i" % (dig_idx + 1), end="\r")
        # ax.set_aspect('equal')
        plot_dig = [pdigs[dig_idx] for pdigs in pred_digits]
        # IPython.embed()
        plot_dig_set = [plot_dig[0]]
        for pdig in plot_dig[1:]:
          plot_dig_set.extend([white_line, pdig])

        # pred_dig[pred_dig < 0] = 0
        # pred_dig[pred_dig > 1] = 1

        if true_digits is not None:
          true_dig = true_digits[dig_idx]
          plot_dig_set.extend([white_line, white_line, true_dig])
        plot_dig = np.concatenate(plot_dig_set, axis=1)
        
        plot_dig[plot_dig < 0] = 0
        plot_dig[plot_dig > 1] = 1
        # else:
        #   plot_dig = pred_dig
        rect_args = get_rect_args(perms[dig_idx], dignums) if perms else []
        dig_idx += 1

        # fig, ax = plt.subplots(1, 1)
        ax.imshow(plot_dig, cmap="gray")
        # rect
        for xy, h, w in rect_args:
          rect = patches.Rectangle(
              xy, w, h, linewidth=0.5, edgecolor=rcolor, facecolor=facecolor)
          ax.add_patch(rect)

        # yellow box
        if true_digits is not None:
          xy = ((_mnist_w + 1) * nsets + 0.5, -0.5)
          rect = patches.Rectangle(
              xy, 28, 28, linewidth=3, edgecolor='y', facecolor='none')
          ax.add_patch(rect)

        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_visible(False)
        ax.yaxis.set_ticks([])

      else:
        plotted_first = True
        first = first.reshape(28, 28)
        ax.imshow(first, cmap="gray")
        rect = patches.Rectangle(
            (0, 0), 27, 27, linewidth=1, edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        ax.xaxis.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_visible(False)
        ax.yaxis.set_ticks([])
      if show_ids:
        fig_id = ri * ncols + ci
        ax.set_title("%i" % fig_id)


  # IPython.embed()
  plt.tight_layout()
  # if true_digits is None:
  if title:
    fig.suptitle(title, fontsize=30, color="w", x=0.46, y=1.03)
  plt.show()
  # plt.pause(10.)
  # IPython.embed()


def get_all_sampled_digits(mv_samples, base_dat, n_views):

  mv_cat_grid = {}
  view_set = np.arange(n_views)
  for nv in range(1, n_views):
    for perm in itertools.combinations(view_set, nv):
      print("Computing digits for view subset %s" % (perm, ))
      unobs_views = [vi for vi in view_set if vi not in perm]
      sample_dat = {
          vi: mv_samples[vi][perm] for vi in unobs_views
      }
      mv_cat_grid[perm] = get_sampled_cat_grid(sample_dat, base_dat)

  return mv_cat_grid


def plot_many_digits_perm(
    perms, pred_digits, true_digits, first=None, grid_size=(10, 10),
    title=""):

  n_views = 4
  if isinstance(perms, tuple):
    perms = [perms] * len(true_digits)

  plot_perms = []
  rcolors = []
  for perm in perms:    
    if len(perm) == 3:
      pred_view = [vi for vi in range(n_views) if vi not in perm]
      plot_perms.append(tuple(pred_view))
      # rect_locs = [_view_map[perm[0]]]
      rcolors.append("g")
    else:
      # pred_views = [vi for vi in range(n_views) if vi not in perm]
      plot_perms.append(perm)
      # rect_locs = [_view_map[vi] for vi in pred_views]
      rcolors.append("r")

  # pos = [_view_map[vi] for vi in perm]
  # title = title + " -- Available Views: %s" % (pos, )
  plot_many_digits(
      pred_digits, true_digits, plot_perms, first, grid_size, rcolors, title)


def create_composite_nv_image(
    mv_digits, base_data, nv=1, tot_views=4, n_per_perm=8,
    *args, **kwargs):
  nv_perms = [perm for perm in mv_digits.keys() if len(perm) == nv]
  num_perms = len(nv_perms)

  tot_n_digs = mv_digits[nv_perms[0]].shape[0]
  # dig_ids = np.random.permutation(tot_n_digs)[:n_per_perm]
  dig_ids = np.arange(n_per_perm)

  white_col_size = 2
  white_row_size = 3
  white_cols = np.ones((n_per_perm, _mnist_h, white_col_size))
  white_row = np.ones((
      white_row_size, n_per_perm * (_mnist_w + white_col_size) - white_col_size))

  def pad_composite_row(dig_set, add_row=True):
    dig_imgs = dig_set.reshape(-1, _mnist_h, _mnist_w)
    dig_imgs_padded = np.concatenate([dig_imgs, white_cols], axis=2)
    # don't need last white line
    dig_row = np.concatenate(dig_imgs_padded, axis=1)[:, :-white_col_size]
    if add_row:
      dig_row = np.concatenate([dig_row, white_row], axis=0)
    return dig_row

  # true_digs = base_data[dig_ids].reshape(-1, 28, 28)
  # white_cols = np.ones((true_digs.shape[0], 1, 28))
  # true_digs_padded = np.concatenate([true_digs, white_cols], axis=1)
  # padded_true_row = pad_composite_row(base_data[dig_ids])
  # IPython.embed()
  true_row = pad_composite_row(base_data[dig_ids], add_row=True)
  comp_image = [true_row]
  comp_rect_args = []
  legend_rect_args = []
  # left-side legend column:

  for idx, perm in enumerate(nv_perms):
    perm_digs = mv_digits[perm][dig_ids]
    add_row = (idx < num_perms - 1)
    perm_row = pad_composite_row(mv_digits[perm][dig_ids], add_row=add_row)
    comp_image.append(perm_row)
    # rects:

    base_y = (idx + 1) * (_mnist_h + white_row_size)
    rect_views = (
        [vi for vi in range(tot_views) if vi not in perm] if nv > 1 else
        [perm[0]])
    rcolor = "g" if nv == 1 else "r"
    combined_rect = False
    if nv == 2:
      rview1, rview2 = rect_views
      x_offset1, y_offset1 = _rect_args[rview1][0]
      x_offset2, y_offset2 = _rect_args[rview2][0]
      if x_offset1 == x_offset2:
        x_offset = x_offset1
        y_offset = min(y_offset1, y_offset2)
        r_height = _mnist_h_2
        r_width = _mnist_w
        combined_rect = True
      elif y_offset1 == y_offset2:
        x_offset = min(x_offset1, x_offset2)
        y_offset = y_offset1
        r_height = _mnist_h
        r_width = _mnist_w_2
        combined_rect = True
    if combined_rect:
      for i in range(n_per_perm):
        base_x = (i + 1) * (_mnist_w + white_col_size)
        xyhw = (
            (base_x + x_offset, base_y + y_offset),
            r_width, r_height, rcolor)
        comp_rect_args.append(xyhw)
    else:
      for rview in rect_views:
        # if rview in ignored_rect_views and nv > 1:
        #   continue
        x_offset, y_offset = _rect_args[rview][0]
        for i in range(n_per_perm):
          base_x = (i + 1) * (_mnist_w + white_col_size)
          xyhw = (
              (base_x + x_offset, base_y + y_offset),
              _mnist_h_2, _mnist_w_2, rcolor)
          comp_rect_args.append(xyhw)

    # legend rect:
    border_rect_args = ((0, base_y), _mnist_h, _mnist_w, "k", "w")
    legend_rect_args.append(border_rect_args)
    for view in perm:
      x_offset, y_offset = _rect_args[view][0]
      v_rect_args = (x_offset, base_y + y_offset), _mnist_h_2, _mnist_w_2, "g", "g"
      legend_rect_args.append(v_rect_args)

  # First row stuff
  legend_rect_args.append(((0, 0), _mnist_h, _mnist_w, "k", "g"))
  comp_image = np.concatenate(comp_image, axis=0)
  legend_col = np.ones((comp_image.shape[0], _mnist_w + white_col_size))
  comp_image = np.concatenate([legend_col, comp_image], axis=1)

  return comp_image, comp_rect_args, legend_rect_args


def show_composite_image(
    img, rect_args, legend_rect_args, title="", savename=None):
  fig, ax = plt.subplots(1, 1)
  ax.imshow(img, cmap="gray")
  pixel_offset = 0.5
  for xy, h, w, color in rect_args:
    xy = (xy[0] - pixel_offset, xy[1] - pixel_offset)
    rect = patches.Rectangle(
        xy, w, h, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

  for xy, h, w, bcolor, fcolor in legend_rect_args:
    xy = (xy[0] - pixel_offset, xy[1] - pixel_offset)
    rect = patches.Rectangle(
        xy, w, h, linewidth=1, edgecolor=bcolor, facecolor=fcolor)
    ax.add_patch(rect)
  ax.xaxis.set_visible(False)
  ax.xaxis.set_ticks([])
  ax.yaxis.set_visible(False)
  ax.yaxis.set_ticks([])
  ax.set_axis_off()
  plt.title(title, fontsize=20, color="w")
  # if savename:
  #   plt.savefig(savename)
  # else:
  plt.show()


def plot_nv(
    mv_digits, base_data, nv=1, tot_views=4, n_per_perm=8,
    ignored_rect_views=[], title=""):

  comp_img, rect_args, legend_rect_args = create_composite_nv_image(
      mv_digits, base_data, nv, tot_views, n_per_perm)
  if nv == 1:
    title = title + " [1 view available]"
  else:
    title = title + " [%i views available]" % nv
  show_composite_image(comp_img, rect_args, legend_rect_args, title=title)


def evaluate_mnist():
  load_start_time = time.time()
  n_views = 4
  split_shape = "grid"
  all_tr_data, all_va_data, all_te_data, split_inds = ssvd.load_split_mnist(
      n_views=n_views, shape=split_shape)
  (tr_data, y_tr) = all_tr_data
  (va_data, y_va) = all_va_data
  (te_data, y_te) = all_te_data
  print("Time taken to load MNIST: %.2fs" % (time.time() - load_start_time))

  tot_views = 4
  n_per_perm = 10
  for dtype in ["tr", "te"]:
    for nv in range(1, 4):
      plot_nv(mv_digits[dtype], mv_cat[dtype], nv, title=tnames[dtype])
  # tr_idxs = np.array(tr_idxs, dtype=int)
  # n_tr = tr_data[0].shape[0]
  # n_va = va_data[0].shape[0]
  # n_te = te_data[0].shape[0]
  # all_tr_subset = {vi:all_tr_data[0][vi][tr_idxs] for vi in all_tr_data[0]}
  # # cat_tr = np.concatenate([all_tr_data[0][vi][tr_idxs] for vi in range(len(all_tr_data[0]))], axis=1)
  # # cat_va = np.concatenate([all_va_data[0][vi] for vi in range(len(all_va_data[0]))], axis=1)
  # # cat_te = np.concatenate([all_te_data[0][vi] for vi in range(len(all_te_data[0]))], axis=1)
  # cat_tr = get_sampled_cat_grid(all_tr_subset, np.zeros((n_tr, 784)))
  # cat_va = get_sampled_cat_grid(all_va_data[0], np.zeros((n_va, 784)))
  # cat_te = get_sampled_cat_grid(all_te_data[0], np.zeros((n_te, 784)))
  # # cat_te = np.concatenate([all_te_data[0][vi] for vi in range(len(all_te_data[0]))], axis=1)
  # samples_tr = np.clip(get_sampled_cat_grid({main_view:s_orig_tr}, cat_tr), 0, 1)
  # samples_te = np.clip(get_sampled_cat_grid({main_view:s_orig_te}, cat_te), 0, 1)

  svd_model_file = "./saved_models/mnist/tsvd_v%i/mdl.pkl" % n_views
  with open(svd_model_file, "rb") as fh:
    svd_models = {vi:pickle.load(fh) for vi in range(n_views)}

  tr_data = {
     vi: smdl.inverse_transform(smdl.transform(tr_data[vi])) for vi, smdl in svd_models.items()
  }
  va_data = {
     vi: smdl.inverse_transform(smdl.transform(va_data[vi])) for vi, smdl in svd_models.items()
  }
  te_data = {
     vi: smdl.inverse_transform(smdl.transform(te_data[vi])) for vi, smdl in svd_models.items()
  }

  tr_idxs = np.load("./data/mnist/aov/mnist_tr_inds.npy").astype(int)
  tr_data = {vi:vdat[tr_idxs] for vi, vdat in tr_data.items()}
  y_tr = y_tr[tr_idxs]

  n_tr = aov_tr_data[0].shape[0]
  n_va = va_data[0].shape[0]
  n_te = te_data[0].shape[0]

  cat_tr = get_sampled_cat_grid(tr_data, np.zeros((n_tr, 784)))
  cat_va = get_sampled_cat_grid(va_data, np.zeros((n_va, 784)))
  cat_te = get_sampled_cat_grid(te_data, np.zeros((n_te, 784)))
  # cat_tr = np.concatenate([tr_data[vi] for vi in range(len(tr_data))], axis=1)
  # cat_va = np.concatenate([va_data[vi] for vi in range(len(va_data))], axis=1)
  # cat_te = np.concatenate([te_data[vi] for vi in range(len(te_data))], axis=1)

  # Load stuff
  n_views = 4
  aov_digits_tr = {}
  aov_digits_te = {}
  aov_file_name = "./data/mnist/aov/aov_samples_v%i.npy"
  for vi in range(n_views):
    aov_samples_tr, aov_samples_te = np.load(
        aov_file_name % vi, allow_pickle=True).tolist()
    aov_digits_tr[vi] = get_sampled_cat_grid({vi: aov_samples_tr}, cat_tr)
    aov_digits_te[vi] = get_sampled_cat_grid({vi: aov_samples_te}, cat_te)

  mv_tr_idxs = np.load("./data/mnist/mv/mnist_mv_tr_inds.npy").astype(int)
  mv_tr_data = {vi:vdat[mv_tr_idxs] for vi, vdat in tr_data.items()}
  mv_y_tr = all_tr_data[1][mv_tr_idxs]

  mv_n_tr = mv_tr_data[0].shape[0]
  n_va = va_data[0].shape[0]
  n_te = te_data[0].shape[0]

  mv_cat_tr = get_sampled_cat_grid(mv_tr_data, np.zeros((mv_n_tr, 784)))
  # cat_va = get_sampled_cat_grid(va_data, np.zeros((n_va, 784)))
  # cat_te = get_sampled_cat_grid(te_data, np.zeros((n_te, 784)))
  mv_samples_tr = {}
  mv_samples_te = {}
  mv_file_name = "./data/mnist/mv/mv_samples_v%i.npy"
  for vi in range(n_views):
    mv_vi_tr, mv_vi_te = np.load(
        mv_file_name % vi, allow_pickle=True).tolist()
    mv_samples_tr[vi] = mv_vi_tr
    mv_samples_te[vi] = mv_vi_te

    # mv_digits_tr[vi] = get_sampled_cat_grid({vi: mv_samples_tr}, cat_tr)
    # mv_digits_te[vi] = get_sampled_cat_grid({vi: mv_samples_te}, cat_te)
  mv_digits_tr = get_all_sampled_digits(mv_samples_tr, mv_cat_tr, n_views)
  mv_digits_te = get_all_sampled_digits(mv_samples_te, cat_te, n_views)

  n_tr = cat_tr.shape[0]
  n_te = cat_te.shape[0]
  mnist_classifier = svm.SVC()
  mnist_classifier.fit(cat_tr, y_tr)

  base_tr_pred = mnist_classifier.predict(cat_tr)
  base_te_pred = mnist_classifier.predict(cat_te)
  base_tr_acc = (base_tr_pred == mv_y_tr).sum()/n_tr
  base_te_acc = (base_te_pred == y_te).sum()/n_te

  aov_tr_accs = {}
  aov_te_accs = {}
  aov_tr_preds = {}
  aov_te_preds = {}
  aov_tr_consistent_wrong = {}
  aov_te_consistent_wrong = {}
  cmats_tr = {}
  cmats_te = {}
  for vi in range(n_views):
    print("Evaluating view %i (AOV)" % vi)
    aov_tr_vi = aov_digits_tr[vi]
    aov_te_vi = aov_digits_te[vi]
    tr_preds_vi = mnist_classifier.predict(aov_tr_vi)
    te_preds_vi = mnist_classifier.predict(aov_te_vi)
    aov_tr_preds[vi] = tr_preds_vi
    aov_te_preds[vi] = te_preds_vi
    aov_tr_accs[vi] = (tr_preds_vi == y_tr).sum()/n_tr
    aov_te_accs[vi] = (te_preds_vi == y_te).sum()/n_te
    tr_preds_vi = aov_tr_preds[vi]
    te_preds_vi = aov_te_preds[vi]

    wrong_inds_tr = (tr_preds_vi != y_tr).nonzero()[0]
    wrong_inds_te = (te_preds_vi != y_te).nonzero()[0]
    base_correct_num_tr = (base_tr_pred[wrong_inds_tr] == y_tr[wrong_inds_tr]).sum()
    base_correct_num_te = (base_te_pred[wrong_inds_te] == y_te[wrong_inds_te]).sum()
    consistent_tr = (base_tr_pred[wrong_inds_tr] == tr_preds_vi[wrong_inds_tr]).sum()
    consistent_te = (base_te_pred[wrong_inds_te] == te_preds_vi[wrong_inds_te]).sum()
    aov_tr_consistent_wrong[vi] = {
        "base_correct": base_correct_num_tr,
        "base_consistent": consistent_tr,
        "frac": consistent_tr / wrong_inds_tr.shape[0],
        "total": wrong_inds_tr.shape[0],
    }
    aov_te_consistent_wrong[vi] = {
        "base_correct": base_correct_num_te,
        "base_consistent": consistent_te,
        "frac": consistent_te / wrong_inds_te.shape[0],
        "total": wrong_inds_te.shape[0]
    }
    cmats_tr[vi] = sm.confusion_matrix(tr_preds_vi[wrong_inds_tr], y_tr[wrong_inds_tr])
    cmats_te[vi] = sm.confusion_matrix(te_preds_vi[wrong_inds_te], y_te[wrong_inds_te])

  mv_mnist_classifier = svm.SVC()
  mv_mnist_classifier.fit(mv_cat_tr, mv_y_tr)

  base_mv_tr_acc = (mv_mnist_classifier.predict(mv_cat_tr) == mv_y_tr).sum()/mv_n_tr
  base_mv_te_acc = (mv_mnist_classifier.predict(cat_te) == y_te).sum()/n_te

  mv_tr_accs = {}
  mv_te_accs = {}
  for perm in mv_digits_tr:
    print("Evaluating views %s (MV)" % (perm,))
    mv_tr_vi = mv_digits_tr[perm]
    mv_te_vi = mv_digits_te[perm]
    mv_tr_accs[perm] = (mv_mnist_classifier.predict(mv_tr_vi) == mv_y_tr).sum()/mv_n_tr
    mv_te_accs[perm] = (mv_mnist_classifier.predict(mv_te_vi) == y_te).sum()/n_te

  IPython.embed()


def get_mcnemar_matrix(preds, ref_preds, truevals):
  rc_inds = (ref_preds == truevals).nonzero()[0]
  rw_inds = (ref_preds != truevals).nonzero()[0]

  rcpc_num = (preds[rc_inds] == truevals[rc_inds]).sum()
  rcpw_num = rc_inds.shape[0] - rcpc_num

  rwpc_num = (preds[rw_inds] == truevals[rw_inds]).sum()
  rwpw_num = rw_inds.shape[0] - rwpc_num

  results_matrix = np.array([[rcpc_num, rwpc_num], [rcpw_num, rwpw_num]])
  return results_matrix


def get_mcnemar_mats_all(preds, ref_preds, truevals):
  all_mats = {}
  for k, pvals in preds.items():
    rpvals = ref_preds[k]
    all_mats[k] = get_mcnemar_matrix(pvals, rpvals, truevals)

  return all_mats


def get_all_preds(imgs, mnist_mdl):
  preds = {}
  for k, ki in imgs.items():
    preds[k] = mnist_mdl.predict_imgs(ki)
  return preds


def get_all_preds_over_dsl_coeff(imgs, mnist_mdl):
  all_preds = {}
  for c, imgs_c in imgs.items():
    all_preds[c] = get_all_preds(imgs_c, mnist_mdl)
  return all_preds


def get_all_accs_dsl_coeffs(preds, ys):
  sub_accs = {}
  nv_accs = {}
  npts = ys.shape[0]

  def _get_accs(pred, y):
    return (pred == y).sum() / y.shape[0]

  for c, preds_c in preds.items():
    for sub, preds_c_sub in preds_c.items():
      nv_s = len(sub)
      if sub not in sub_accs:
        sub_accs[sub] = {}
      if nv_s not in nv_accs:
        nv_accs[nv_s] = {}

      acc = _get_accs(preds_c_sub, ys)
      sub_accs[sub][c] = acc

      if c in nv_accs[nv_s]:
        acc_so_far, num_subs = nv_accs[nv_s][c]
        new_acc = (acc_so_far * num_subs + acc) / (num_subs + 1)
        nv_accs[nv_s][c] = (new_acc, num_subs + 1)
      else:
        nv_accs[nv_s][c] = (acc, 1)

  return sub_accs, nv_accs

def sum_nv_mats(mat_set):
  nv_mats = {}
  for k, m in mat_set.items():
    nv = len(k)
    if nv not in nv_mats:
      nv_mats[nv] = m
    else:
      nv_mats[nv] += m

  return nv_mats


def plot_double_mcnemar(mat1, mat2, perm, ax=None, show_xyticks=True):
  mat1 = np.array(mat1).astype(int)
  mat2 = np.array(mat2).astype(int)

  # def get_vals_in_order(mat):
  #   mat = np.array(mat)
  #   val_mat = np.zeros((3, 2))
  #   col_sums = mat.sum(axis=0)

  #   val_mat[:2, :2] = mat
  #   val_mat[2, :2] = col_sums
  #   val_order = val_mat[::-1].ravel()
  #   return val_order

  get_vals_in_order = lambda mat: (
      np.array(mat.sum(axis=0).tolist() + mat[::-1].ravel().tolist()))

  m1_vals = get_vals_in_order(mat1)
  m2_vals = get_vals_in_order(mat2)

  ax = plt if ax is None else ax
  # fig, ax = plt.subplots()
  ax.set_xlim(0., 3.)
  ax.set_ylim(0., 3.)

  # visual grid:
  edge_segs = []
  # border first
  edge_segs += [np.array([[xi, 0], [xi, 3]]) for xi in range(4)]
  edge_segs += [np.array([[0, yi], [3, yi]]) for yi in range(4)]
  # diags
  edge_segs += [np.array([[0, yi], [2, yi+2]]) for yi in range(-1, 4)]
  border_lc = LineCollection(edge_segs, colors="k", linewidths=0.5)
  ax.add_collection(border_lc)
  # ax.set_xticks([])
  # ax.set_yticks([])
  # edge_segs = [
  #     np.array([[tr1.x[idx0], tr1.y[idx0]], [tr1.x[idx1], tr1.y[idx1]]])
  #     for idx0, idx1 in tr1.edges
  # ] + [
  #     np.array([[tr2.x[idx0], tr2.y[idx0]], [tr2.x[idx1], tr2.y[idx1]]])
  #     for idx0, idx1 in tr2.edges
  # ]

  # plt.show()
  # m1_fracs, m2_fracs = m1_vals / m1_sum, m2_vals / m2_sum
  # m1_fracs, m2_fracs = m1_vals, m2_vals
  # cmap = plt.get_cmap('RdBu')
  # img1 = ax.tripcolor(tr1, m1_fracs, cmap=cmap, vmax=1., vmin=0.)
  # img2 = ax.tripcolor(tr2, m2_fracs, cmap=cmap, vmax=1., vmin=0.)
  # cbar_ticks = np.arange(11) / 10.
  # ax.colorbar(img1, ticks=cbar_ticks)
  # ax.colorbar(img2, ticks=cbar_ticks, pad=-0.05)
  # 
  text_coords = [(i, j) for i in range(K) for j in range(K-1)]
  # del text_coords

  # top triangles
  base_offset = 0.5
  relative_offset = 0.2
  x_pixel_offset = base_offset + relative_offset
  y_pixel_offset = base_offset - relative_offset
  for (xcoord, ycoord), val in zip(text_coords, m1_vals):
    # print(xcoord, ycoord)
    xcoord += x_pixel_offset
    ycoord += y_pixel_offset
    ax.text(ycoord, xcoord, val, color="b", ha="center", va="center", fontsize=20)

  x_pixel_offset, y_pixel_offset = y_pixel_offset, x_pixel_offset
  for (xcoord, ycoord), val in zip(text_coords, m2_vals):
    # print(xcoord, ycoord)
    xcoord += x_pixel_offset
    ycoord += y_pixel_offset
    ax.text(ycoord, xcoord, val, color="r", ha="center", va="center", fontsize=20)

  row_sums = mat1.sum(axis=1).tolist()# + [mat1.sum()]
  ycoord = 2 + base_offset
  for i, val in enumerate(row_sums):
    xcoord = 2 - i + base_offset
    ax.text(ycoord, xcoord, val, color="k", ha="center", va="center", fontsize=20)

  view_split_segs = [
      [[2, 0.5], [3, 0.5]], [[2.5, 0], [2.5, 1]]]
  view_split_lc = LineCollection(view_split_segs, linestyles="dashed", colors="k")
  ax.add_collection(view_split_lc)
  rect_offset = {
       0: (0, 0),
       1: (0.5, 0),
       2: (0, 0.5),
       3: (0.5, 0.5),
  }
  base_x = 2
  base_y = 0
  rect_args = []
  rect_h = rect_w = 0.5
  for view in perm:
    x_offset, y_offset = rect_offset[view]
    xy = (base_x + x_offset, base_y + y_offset)
    rect = patches.Rectangle(
        xy, rect_w, rect_h, linewidth=1, edgecolor="g", facecolor="g")
    ax.add_patch(rect)

  if show_xyticks:
    ax.set_yticks([1.5, 2.5])
    ax.set_yticklabels(["Mdl.\nWrong", "Mdl.\nCorrect"],fontsize=15)#, rotation=90)

    ax.set_xticks([0.5, 1.5])#, minor=True)
    ax.set_xticklabels(["Ref. Correct", "Ref. Wrong"], fontsize=15)#, minor=True)
    ax.xaxis.set_ticks_position("top")

    ax.text(2.5, -0.2, "Available views", color="k", ha="center", fontsize=15)
  else:
    ax.set_xticks([])
    ax.set_yticks([])

  # plt.show()

def plot_mcnemar_nv(mats1, mats2, nv=1, mnames=None, title="", save_file=None):
  nv_perms = [perm for perm in mats1.keys() if len(perm) == nv]

  nrows = 2
  ncols = len(nv_perms) // nrows

  subplot_to_inches = 5
  padding = 0.5

  fig_w = (ncols + padding) * subplot_to_inches
  fig_h = (nrows + padding) * subplot_to_inches

  fig, axs = plt.subplots(nrows, ncols)
  fig.set_size_inches(fig_w, fig_h)
  axs = axs.ravel()

  tot_pts = None
  for i, perm in enumerate(nv_perms):
    mat1, mat2 = mats1[perm], mats2[perm]
    if tot_pts is None:
      tot_pts = mat1.sum()
    ax = axs[i]
    show_xyticks = (i == 0)
    plot_double_mcnemar(mat1, mat2, perm, ax=ax, show_xyticks=show_xyticks)

  if mnames is not None:
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
    ax.plot([], color="b", label=mnames[0])
    ax.plot([], color="r", label=mnames[1])
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=15)

  ann_text = "# Views:  %i\n# Points: %i" % (nv, tot_pts)
  axs[-ncols].text(0.5, -0.65, ann_text, va="bottom", fontsize=20)

  # title += ": %i view%s available" % (nv, "" if nv == 1 else "s")
  title = "McNemar Test: " + title
  plt.suptitle(title, fontsize=30)
  if save_file:
    plt.savefig(save_file)
  else:
    plt.show()


def save_all_mcnemar(mcnemar_mats):
  mztr, mzte, mgtr, mgte = mcnemar_mats
  mnames = ["Full Digit", "Zero Imputed"]
  save_dir = "./results/mnist/"

  title = "Training"
  fname = "mcnemar_tr_%i_views.png"
  for nv in range(1, 4):
    save_file = save_dir + fname % nv
    plot_mcnemar_nv(
        mgtr, mztr, nv=nv, title=title, mnames=mnames, save_file=save_file)

  title = "Testing"
  fname = "mcnemar_te_%i_views.png"
  for nv in range(1, 4):
    save_file = save_dir + fname % nv
    plot_mcnemar_nv(
        mgte, mzte, nv=nv, title=title, mnames=mnames, save_file=save_file)


def misc_acflow_pred():
  tr_nv_dat, _ = make_missing_dset(
      base_tr_data, main_view, False, separate_nv=True)


if __name__ == "__main__":
  # evaluate_mnist()
  pass
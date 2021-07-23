# Some datasets for multi-view expts

import numpy as np
import os
import scipy.io as sio
import yaml


import IPython


DATA_DIR = os.getenv("DATA_DIR")
#DATA_DIR = '/usr0/home/sibiv/Research/Data/MultiviewLearning/'


# 3 sources dataset
THREE_SOURCE_NEWS_DIR = os.path.join(DATA_DIR, "3SourceNews")
def load_3sources_dataset():
  news_sources = ["bbc", "guardian", "reuters"]

  def docs_file_reader(fl):
    with open(fl) as fh:
      dat = [int(line) for line in fh.readlines()]
    return np.array(dat)

  def mtx_file_reader(fl, sparse=False):
    dat = sio.mmread(fl)
    if not sparse:
      dat = dat.todense().A.T
    return dat

  def terms_file_reader(fl):
    with open(fl) as fh:
      dat = fh.readlines()
    return dat

  def clist_file_reader(fl):
    with open(fl) as fh:
      clist_dict = yaml.Loader(fh).get_data()
    clist_dict = {
        lbl: np.array([int(doc) for doc in lbl_list.split(',')])
        for lbl, lbl_list in clist_dict.items()
    }
    return clist_dict

  news_file_strs = {
      "docs": "3sources_%s.docs",
      "mtx": "3sources_%s.mtx",
      "terms": "3sources_%s.terms",
  }

  news_file_readers = {
      "docs": docs_file_reader,
      "mtx": mtx_file_reader,
      "terms": terms_file_reader,
  }

  news_data = {}
  for ftype, fstr in news_file_strs.items():
    news_data[ftype] = {}
    for src in news_sources:
      fl = os.path.join(THREE_SOURCE_NEWS_DIR, fstr % src)
      news_data[ftype][src] = news_file_readers[ftype](fl)
  mtx_dat = news_data["mtx"]
  docs_dat = news_data["docs"]
  terms_dat = news_data["terms"]

  overlap_file = os.path.join(THREE_SOURCE_NEWS_DIR, "3sources.overlap.clist")
  overlap_docs = clist_file_reader(overlap_file)
  disjoint_file = os.path.join(THREE_SOURCE_NEWS_DIR, "3sources.disjoint.clist")
  disjoint_docs = clist_file_reader(disjoint_file)

  # The largest index is the number of documents (indexed starting frorm 1)
  n_pts = max([max(doc_ids) for doc_ids in overlap_docs.values()])
  # Store view features:
  view_data = {}
  view_sizes = {}
  for vi, src in enumerate(news_sources):
    src_feats = [None] * n_pts
    for i, doc_id in enumerate(docs_dat[src]):
      src_feats[doc_id - 1] = mtx_dat[src][i]
    view_data[vi] = src_feats
    view_sizes[vi] = mtx_dat[src].shape[1]

  # Extract single and overlapping labels
  all_labels = list(overlap_docs.keys())
  n_classes = len(all_labels)
  disjoint_labels = np.empty(n_pts).astype("int")
  overlap_labels = np.zeros((n_pts, n_classes)).astype("int")
  for idx, lbl in enumerate(all_labels):
    disjoint_labels[disjoint_docs[lbl]-1] = idx
    overlap_labels[overlap_docs[lbl]-1, idx] = 1

  return view_data, view_sizes, disjoint_labels, overlap_labels, news_sources,\
         terms_dat


# NUS-WIDE-LITE
# https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
NWL_DIR = os.path.join(DATA_DIR, "NUS-WIDE-LITE")
def make_label_files():
  l_dir = os.path.join(NWL_DIR, "NUS-WIDE-Lite_groundtruth")
  with open(os.path.join(NWL_DIR, "Concepts81.txt")) as fh:
    concepts = [c.strip() for c in fh.readlines()]
  # # Train:
  # tr_labels = np.loadtxt(os.path.join(l_dir, "Lite_GT_Train.txt"), dtype=int)
  # np.save(os.path.join(l_dir, "tr_labels"), tr_labels)

  for ttype in ["Train", "Test"]:
    t_labels = None
    for concept in concepts:
      cfile = os.path.join(l_dir, "Lite_Labels_%s_%s.txt" % (concept, ttype))
      c_labels = np.loadtxt(cfile).reshape(-1, 1)
      t_labels = c_labels if t_labels is None else np.c_[t_labels, c_labels]
    np.save(os.path.join(l_dir, "%s_labels" % ttype), t_labels)


def load_nus_wide_lite():
  f_dir = os.path.join(NWL_DIR, "NUS-WIDE-Lite_features")
  l_dir = os.path.join(NWL_DIR, "NUS-WIDE-Lite_groundtruth")
  f_types = ["CH", "CM55", "CORR", "EDH", "WT"]

  # train_f = {}
  # test_f = {}
  # for i, f_type in enumerate(f_types):
  #   f_base_name = "Normalized_%s_Lite_" % f_type + "%s.dat"
  #   tr_file_name = os.path.join(f_dir, f_base_name % ("Train"))
  #   te_file_name = os.path.join(f_dir, f_base_name % ("Test"))
  #   train_f[i] = np.loadtxt(tr_file_name)
  #   test_f[i] = np.loadtxt(te_file_name)
  # feats = {"train": train_f, "test": test_f}
  # np.save(os.path.join(f_dir, "all_feats"), feats)

  fts = np.load(os.path.join(f_dir, "all_feats.npy"), allow_pickle=True).tolist()

  with open(os.path.join(NWL_DIR, "Concepts81.txt")) as fh:
    concepts = [c.strip() for c in fh.readlines()]

  labels = {}
  for ttype in ["Train", "Test"]:
    labels[ttype] = np.load(os.path.join(l_dir, "%s_labels.npy" % ttype))
  view_sizes = {i: fts["train"][i].shape[1] for i in range(len(f_types))}
  return fts, labels, view_sizes, concepts, f_types


# n-MNIST
# https://csc.lsu.edu/~saikat/n-mnist/
NMNIST_DIR = os.path.join(DATA_DIR, "NoisyMnist")
def load_nmnist():
  all_file = os.path.join(NMNIST_DIR, "all_nmnist.npy")
  if os.path.exists(all_file):
    data, labels, n_types = np.load(all_file, allow_pickle=True).tolist()
    view_sizes = {vi: data["train"][vi].shape[1] for vi in data["train"]}
    return data, view_sizes, labels, n_types

  awgn_file = os.path.join(NMNIST_DIR, "mnist-with-awgn.mat")
  mb_file = os.path.join(NMNIST_DIR, "mnist-with-motion-blur.mat")
  rcawgn_file = os.path.join(NMNIST_DIR, "mnist-with-reduced-contrast-and-awgn.mat")

  n_types = ["awgn", "mb", "rcawgn"]
  fls = [awgn_file, mb_file, rcawgn_file]
  data = {"train": {}, "test": {}}
  labels = None
  for i, fl in enumerate(fls):
    fl_data = sio.loadmat(fl)
    train_x = fl_data["train_x"]
    test_x = fl_data["test_x"]
    train_y = fl_data["train_y"]
    test_y = fl_data["test_y"]
    data["train"][i] = train_x
    data["test"][i] = test_x
    if labels is None:
      labels = {"train": train_y, "test": test_y}

  np.save(all_file, [data, labels, n_types])
  return data, labels, n_types

################################################################################
# Data utilities:
def cat_views(view_data):
  nviews = len(view_data)
  cat_data = np.concatenate(
      [view_data[vi] for vi in range(nviews)], axis=1)
  return cat_data


def fill_missing(view_data, fill=0., cat_dims=False):
  nviews = len(view_data)
  view_data_filled = {}
  for vi in range(nviews):
    vdat = view_data[vi]
    # Search for first non-None element to find dim
    vdim = next(np.array(ft) for ft in vdat if ft is not None).shape[0]
    v_filler = np.ones(vdim) * fill
    filled_view = np.array(
        [(v_filler if ft is None else np.array(ft)) for ft in vdat])
    view_data_filled[vi] = filled_view

  return cat_views(view_data_filled) if cat_dims else view_data_filled


def naive_clean(view_data):
  view_data_cleaned = {
      vi: np.array([ft for ft in vdat if ft is not None])
      for vi, vdat in view_data.items()
  }
  return view_data_cleaned


def dim_reduce(view_data, ndims=50, fill=True, rtn_proj_mean=False):
  cleaned_data = naive_clean(view_data)
  view_proj = {}
  view_means = {}
  for vi, vdat in cleaned_data.items():
    vdims = min(ndims, min(vdat.shape))

    vmean = vdat.mean(0)
    _, _, VT = np.linalg.svd(vdat - vmean)
    view_proj[vi] = VT[:vdims]
    view_means[vi] = vmean

  view_data_proj = {}
  for vi in view_data:
    vdat = view_data[vi]
    VT = view_proj[vi]
    vmean = view_means[vi]
    view_data_proj[vi] = [
        (None if ft is None else VT.dot(ft - vmean)) for ft in vdat]
  view_data_proj = (
      fill_missing(view_data_proj, cat_dims=False) if fill else view_data_proj)

  if rtn_proj_mean:
    return view_data_proj, view_proj, view_means
  return view_data_proj


# More utils
def split_data(view_data, fracs=[0.8, 0.2], shuffle=True, get_inds=True):
  fracs = np.array(fracs) / np.sum(fracs)

  nviews = len(view_data)
  npts = len(view_data[0])
  num_split = (fracs * npts).astype(int)
  num_split[-1] = npts - num_split[:-1].sum()
  all_inds = np.random.permutation(npts) if shuffle else np.arange(npts)
  end_inds = np.cumsum(num_split).tolist()
  start_inds = [0] + end_inds[:-1]

  dsets = []
  for si, ei in zip(start_inds, end_inds):
    split_inds = all_inds[si:ei]
    # IPython.embed()
    split_data = {
        vi:[view_data[vi][i] for i in split_inds] for vi in range(nviews)
    }
    dsets.append(split_data)

  if get_inds:
    inds = [all_inds[si:ei] for si, ei in zip(start_inds, end_inds)]
    return dsets, inds
  return dsets


def stratified_sample(view_data, labels, frac=0.2, get_inds=True):
  ltypes = np.unique(labels)
  idx_set = []
  for l in ltypes:
    linds = (labels == l).nonzero()[0]
    num_l = len(linds)
    num_sampled_l = int(np.round(frac * num_l))
    idx_set += [linds[i] for i in np.random.permutation(num_l)[:num_sampled_l]]

  np.random.shuffle(idx_set)

  labels_sampled = labels[idx_set]
  view_data_sampled = {vi: vdat[idx_set] for vi, vdat in view_data.items()}

  if get_inds:
    return view_data_sampled, labels_sampled, idx_set
  return view_data_sampled, labels_sampled


def get_view_subdatasets(view_data, labels, split_inds=None):
  npts = len(view_data[0])
  if split_inds is None:
    split_inds = [np.arange(npts)]

  view_dsets = []
  for split_subset in split_inds:
    split_dset = {}
    for vi, vdat in view_data.items():
      vdat_split = [vdat[i] for i in split_subset]
      valid_vdat = np.array([ft for ft in vdat_split if ft is not None])
      valid_inds = [i for i in range(len(vdat_split)) if vdat_split[i] is not None]
      valid_labels = labels[valid_inds]
      split_dset[vi] = (valid_vdat, valid_labels)
    view_dsets.append(split_dset)
  if len(view_dsets) == 1:
    return view_dsets[0]
  return view_dsets
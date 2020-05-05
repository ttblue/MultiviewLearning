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
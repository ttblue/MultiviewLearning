# Functions to load pig-video data + combined with other data.
import h5py
import os
import numpy as np
import scipy.io as sio
import time

# from dataprocessing.featurize_pig_data import \
#     create_label_timeline, convert_tstamps_to_labels
from utils import utils

import IPython


DATA_DIR = os.getenv("DATA_DIR")
VIDEO_DIR = os.path.join(DATA_DIR, "PigVideos/videos")
FEAT_DIR = os.path.join(
    DATA_DIR, "PigData/extracted/waveform/slow/numpy_arrays")
WS_FEAT_DIR = os.path.join(
    DATA_DIR, "PigData/extracted/waveform/slow/")
FEAT_ANNO_DIR = os.path.join(DATA_DIR, "PigData/raw/annotation/slow")
# PIG_DATA_DIR = os.getenv("PIG_DATA_DIR")
# FEAT_SAVE_DIR = os.getenv("PIG_FEATURES_DIR")

ALL_VIDEO_FILES = None
ALL_FEAT_FILES = None
VFILE_MAP = {}
PNAME_VFILE_MAP = {}
FFILE_MAP = {}
COMMON_PNUMS = []

VERBOSE = True

# Hardcoded assuming video files follow same format.
def extract_pnum_and_phase(fname):
  basename = os.path.basename(fname)
  try:
    split = basename.split("_")
    pnum = int(split[1][3:])
    tphase = (split[2], int(split[3]))
    phase_name = split[4]
  except Exception as e:
    print(e)
    IPython.embed()
  return pnum, tphase, phase_name


def load_all_filenames():
  global ALL_VIDEO_FILES, ALL_FEAT_FILES, VFILE_MAP, FFILE_MAP, COMMON_PNUMS,\
         PNAME_VFILE_MAP
  ALL_VIDEO_FILES = sorted([
      fl for fl in os.listdir(VIDEO_DIR)
      if os.path.isfile(os.path.join(VIDEO_DIR, fl))])
  ALL_FEAT_FILES = [
      fl for fl in os.listdir(FEAT_DIR)
      if os.path.isfile(os.path.join(FEAT_DIR, fl))]

  VFILE_MAP = {}
  all_phase_names = []
  for fl in ALL_VIDEO_FILES:
    if fl[-4:] != ".mat": continue
    pnum, tphase, pname = extract_pnum_and_phase(fl)
    if pnum not in VFILE_MAP:
      VFILE_MAP[pnum] = {}  #ftype: {} for ftype in ["anno", "feat", "spf", "mat"]}
    if pname not in VFILE_MAP[pnum]:
      VFILE_MAP[pnum][pname] = {}
    if pname not in all_phase_names:
      all_phase_names.append(pname)

    ftype = None
    fname = os.path.join(VIDEO_DIR, fl)
    if fl[-9:] == ".anno.mat": ftype = "anno"
    elif fl[-9:] == ".feat.mat": ftype = "feat"
    elif fl[-8:] == ".spf.mat": ftype = "spf"
    elif fl[-4:] == ".mat": ftype = "mat"

    if ftype is not None:
      VFILE_MAP[pnum][pname][ftype] = fname

  pnums_to_del = []
  for pnum in VFILE_MAP:
    if not any([lt for lt in VFILE_MAP[pnum].values()]):
      pnums_to_del.append(pnum)
  for pnum in pnums_to_del: del VFILE_MAP[pnum]

  PNAME_VFILE_MAP = {
      pname: {
          pnum: VFILE_MAP[pnum][pname]
          for pnum in VFILE_MAP if pname in VFILE_MAP[pnum]
      } for pname in all_phase_names
  }

  FFILE_MAP = {}
  col_identifier = "[0, 6, 7, 11]"
  for fl in ALL_FEAT_FILES:
    if col_identifier not in fl or fl[-4:] != ".npy":
      continue
    pnum = int(fl.split("_")[0])
    FFILE_MAP[pnum] = fl

  COMMON_PNUMS = sorted([pnum for pnum in VFILE_MAP if pnum in FFILE_MAP])

load_all_filenames()


def create_label_timeline(labels):  
  label_dict = {i:lbl for i,lbl in enumerate(labels)}
  label_dict[len(labels)] = labels[-1]

  return label_dict


def convert_tstamps_to_labels(
    tstamps, critical_times, label_dict, window_length_s=None):
  if window_length_s is not None:
    tstamps = [t + window_length_s / 2.0 for t in tstamps]
  segment_inds = np.searchsorted(critical_times, tstamps)
  labels = np.array([label_dict[idx] for idx in segment_inds])

  return labels


ALL_FEATURE_COLUMNS = [0, 3, 4, 5, 6, 7, 11]
def load_pig_features_and_labels(
    pig_list=COMMON_PNUMS, ds=1, ds_factor=10,
    feature_columns=ALL_FEATURE_COLUMNS, save_new=False, valid_labels=None):
  # Relevant data directories
  features_dir = FEAT_DIR
  #os.path.join(SAVE_DIR, "waveform/%s/numpy_arrays"%pig_type)
  ann_dir = FEAT_ANNO_DIR
  #os.path.join(DATA_DIR, "raw/annotation/%s"%pig_type)

  # Feature columns
  feature_columns = sorted(feature_columns)
  if 0 not in feature_columns:
    feature_columns = [0] + feature_columns

  # Finding file names from directories
  str_pattern = str(feature_columns) # [1:-1]
  wild_card_str = "*_numpy_ds_%i_columns*%s*"%(ds, str_pattern)

  fdict = utils.create_number_dict_from_files(
      features_dir, wild_card_str=wild_card_str)
  using_all = False

  # If specific feature columns not found, load the data with all columns
  if not fdict:
    all_columns = ALL_FEATURE_COLUMNS
    str_pattern = str(all_columns)# [1:-1]
    wild_card_str = "*_numpy_ds_%i_columns*%s*"%(ds, str_pattern)

    fdict = utils.create_number_dict_from_files(
        features_dir, wild_card_str=wild_card_str)
    using_all = True
  adict = utils.create_number_dict_from_files(ann_dir, wild_card_str="*.xlsx")

  # Find common pigs between annotations and data.
  common_keys = np.intersect1d(list(fdict.keys()), list(adict.keys())).tolist()
  # Create list of unused pigs for bookkeeping.
  _check_key = lambda key: key in common_keys and key in COMMON_PNUMS
  pig_list = COMMON_PNUMS if pig_list is None else pig_list
  used_pigs = [key for key in pig_list if _check_key(key)]
  np.random.shuffle(used_pigs)

  # IPython.embed()
  unused_pigs = [p for p in pig_list if p not in used_pigs]
  if VERBOSE:
    print("Not using pigs %s. One of anns, data or video missing." %
          (unused_pigs))

  # Extract data.
  all_data = {}
  curr_unused_pigs = len(unused_pigs)
  num_selected = 0

  finds = ([all_columns.index(idx) for idx in feature_columns]
            if using_all else None)

  # IPython.embed()
  for key in used_pigs:
    if VERBOSE:
      t_start = time.time()

    ann_time, ann_text = utils.load_xlsx_annotation_file(adict[key])
    critical_anns, ann_labels = utils.create_annotation_labels(ann_text, False)
    if critical_anns is None or ann_labels is None:
      print("Something went wrong with pig %i"%key)
      unused_pigs.append(key)
      continue

    pig_data = np.load(fdict[key])
    if using_all:
      pig_data = pig_data[:, finds]

    tstamps = pig_data[:, 0]
    features = pig_data[:, 1:]

    if save_new:
      new_file = os.path.join(
          features_dir, "%i_numpy_ds_%i_columns_%s.npy" %
              (key, ds, feature_columns))
      if not os.path.exists(new_file):
        np.save(new_file, pig_data)

    if ds_factor > 1:
      tstamps = tstamps[::ds_factor]
      features = features[::ds_factor]

    del pig_data

    critical_times = [ann_time[idx] for idx in critical_anns]
    critical_text = {idx:ann_text[idx] for idx in critical_anns}
    label_dict = create_label_timeline(ann_labels)
    labels = convert_tstamps_to_labels(
        tstamps, critical_times, label_dict, None)

    if valid_labels is None:
      valid_inds = (labels != -1)
    else:
      valid_inds = np.zeros(labels.shape).astype("bool")
      for vl in valid_labels:
        valid_inds = np.logical_or(valid_inds, labels==vl)

    # IPython.embed()
    features = features[valid_inds, :]
    labels = labels[valid_inds]

    if len(features.shape) == 1:
      features = features.reshape(-1, 1)

    all_data[key] = {
        "tstamps": tstamps,
        "features": features,
        "labels": labels,
        "ann_text": critical_text
    }

    if VERBOSE:
      print("Time taken to load pig %i: %.2f"%(key, time.time() - t_start))
    # if num_pigs > 0:
    #   num_selected += 1
    #   if num_selected >= num_pigs:
    #     break

  # new_unused_pigs = len(unused_pigs) - curr_unused_pigs
  # if num_pigs > 0:
  #   print("Not using pigs %s. Already have enough."%(pig_ids[num_selected+new_unused_pigs:]))
  # unused_pigs.extend(pig_ids[num_selected+new_unused_pigs:])

  return all_data#, unused_pigs


def load_tdPCA_featurized_slow_pigs(
    pig_list=COMMON_PNUMS, ds=5, ws=30, nfeats=3, view_subset=None,
    valid_labels=None):
  # Relevant data directories
  features_dir = WS_FEAT_DIR
  #os.path.join(SAVE_DIR, "waveform/%s/numpy_arrays"%pig_type)
  ann_dir = FEAT_ANNO_DIR
  #os.path.join(DATA_DIR, "raw/annotation/%s"%pig_type)
  
  fdict = utils.create_number_dict_from_files(
      features_dir, wild_card_str="*_ds_%i_ws_%i.npy"%(ds, ws))
  adict = utils.create_number_dict_from_files(ann_dir, wild_card_str="*.xlsx")

  common_keys = np.intersect1d(list(fdict.keys()), list(adict.keys())).tolist()
  _check_key = lambda key: key in common_keys and key in COMMON_PNUMS
  pig_list = COMMON_PNUMS if pig_list is None else pig_list
  used_pigs = [key for key in pig_list if _check_key(key)]
  np.random.shuffle(used_pigs)

  unused_pigs = [p for p in pig_list if p not in used_pigs]
  if VERBOSE:
    print("Not using pigs %s. Either annotations or data missing."%(unused_pigs))

  all_data = {}
  curr_unused_pigs = len(unused_pigs)
  num_selected = 0

  for key in used_pigs:
    if VERBOSE:
      t_start = time.time()

    ann_time, ann_text = utils.load_xlsx_annotation_file(adict[key])
    critical_anns, ann_labels = utils.create_annotation_labels(ann_text, False)
    critical_times = [ann_time[idx] for idx in critical_anns]
    if critical_anns is None or ann_labels is None:
      print("Something went wrong with pig %i"%key)
      unused_pigs.append(key)
      continue

    pig_data = np.load(fdict[key], encoding="bytes").tolist()
    tstamps = pig_data[b"tstamps"]
    features = pig_data[b"features"]

    critical_times = [ann_time[idx] for idx in critical_anns]
    critical_text = {idx:ann_text[idx] for idx in critical_anns}
    label_dict = create_label_timeline(ann_labels)
    labels = convert_tstamps_to_labels(tstamps, critical_times, label_dict, ws)

    if valid_labels is None:
      valid_inds = (labels != -1)
    else:
      valid_inds = np.zeros(labels.shape).astype("bool")
      for vl in valid_labels:
        valid_inds = np.logical_or(valid_inds, labels==vl)
    if view_subset is not None:
      features = [
          c_f[valid_inds, :nfeats] for i, c_f in enumerate(features)
          if i in view_subset]
    else:
      features = [c_f[valid_inds, :nfeats] for c_f in features]
    labels = labels[valid_inds]

    # lvals, counts = np.unique(labels, False, False, True)

    # # Debugging and data-quality checks:
    # # print(counts, counts.astype(float)/counts.sum())
    # # if (counts.astype(float)/counts.sum() < 0.05).any(): IPython.embed()
    # # if (labels == 0).sum() > 70: IPython.embed()
    # # if len(counts) < 6: IPython.embed()
    # # if (labels == 0).sum() < 50: IPython.embed()
    # # if key == 18: IPython.embed()

    # # Something weird happened with the data:
    # # 1. Too label types are missing
    # # 2. The stabilization period is too small
    # if len(lvals) < 5:
    #   if VERBOSE:
    #     print("Not using pig %i. Missing data from some phases."%key)
    #   unused_pigs.append(key)
    #   continue
    # if counts[0] < 10:
    #   if VERBOSE:
    #     print("Not using pig %i. Stabilization period is too small."%key)
    #   unused_pigs.append(key)
    #   continue
    all_data[key] = {
        "tstamps": tstamps,
        "features": features,
        "labels": labels,
        "ann_text": critical_text
    }

    if VERBOSE:
      print("Time taken to load pig %i: %.2f"%(key, time.time() - t_start))

  return all_data#, unused_pigs


def load_hdf(fl):
  dat = h5py.File(fl, 'r')
  loaded_dat = {
      k: dat[k][()] for k in dat.keys() if isinstance(dat[k], h5py.Dataset)
  }
  return loaded_dat


def clean_vdat(vdat, ftype):
  for key in ["__header__", "__version__", "__globals__"]:
    if key in vdat:
      del vdat[key]

  if ftype == "feat":
    del vdat["labels"]

  return vdat


def load_video_data(pnums=None, ftypes=["feat", "anno"]):
  if pnums is None:
    pnums = COMMON_PNUMS

  _loadmat = lambda fl, ftype: clean_vdat(sio.loadmat(fl), ftype)
  _loadhdf = lambda fl, ftype: load_hdf(fl)
  _load_func = {
      "feat": _loadmat, "anno": _loadmat, "spf": _loadhdf, "mat": _loadhdf}

  vdata = {}
  for pnum in pnums:
    if pnum not in VFILE_MAP:
      continue
    vdata[pnum] = {}
    # print(VFILE_MAP[pnum])
    for phase in VFILE_MAP[pnum]:
      vdata[pnum][phase] = {}
      for ftype in ftypes:
        vdata[pnum][phase][ftype] = _load_func[ftype](
            VFILE_MAP[pnum][phase][ftype], ftype)

  IPython.embed()
  return vdata


def load_common_vidfeat_data(pnums=None, ftypes=["feat", "anno"], nfiles=-1):
  pnums = (
      COMMON_PNUMS if pnums is None else 
      [pnum for pnum in pnums if pnum in COMMON_PNUMS])

  if nfiles > 0:
    pnums = pnums[:nfiles]

  vdata = load_video_data(pnums, ftypes)
  fdata = load_pig_features_and_labels(pnums)

  # Check common keys again.
  common_pnums = [key for key in vdata.keys() if key in fdata]
  vdata = {key: vdata[key] for key in common_pnums}
  fdata = {key: fdata[key] for key in common_pnums}
  return vdata, fdata


# Labels are:
# 0: Stabilization
# 1: Bleeding
# 2: Between bleeds
# 3: Hextend Resuscitation
# 4: Other Resuscitation
# 5: After Hextend (between resuscitation events)
# 6: After other resuscitation (between resuscitation events)
# 7: Post resuscitation
# -1: None

# Phases are:
# - EndBaseline
# - EndBleed
# - AfterBleed
# - BeforeResusc
# - EndHextend
# - AfterHextend

def detect_change_points(arr):
  arr = np.array(arr)
  diff = arr[:-1] - arr[1:]
  change_inds = np.r_[0, (diff.nonzero()[0] + 1)]
  change_vals = np.r_[arr[0] + arr[change_inds]]
  return change_vals, change_inds


# Detect continuous pair of values in array
# If an element of pair is None, then any value is allowed for element.
def detect_cts_pairs(arr, pair, rtn_pairs=True):
  # Hack: assuming last index is never selected.
  assert (pair[0] is not None or pair[1] is not None)
  arr = np.array(arr)

  if pair[0] is None:
    p2_inds = (arr == pair[1]).nonzero()[0]
    if p2_inds.shape[0] == 0:
      return []

    if p2_inds[0] == 0:
      p2_inds = p2_inds[1:]
    return [[idx - 1, idx] for idx in p2_inds] if rtn_pairs else p2_inds

  p1_inds = (arr == pair[0]).nonzero()[0]
  if p1_inds.shape[0] == 0:
    return np.array([])

  if pair[1] != None:
    pcheck = lambda idx: (
        (idx + 1 < arr.shape[0]) and (arr[idx + 1] == pair[1]))
    p1_inds = list(filter(pcheck, p1_inds))
  return [[idx, idx + 1] for idx in p1_inds] if rtn_pairs else p1_inds


# Given single video and feature data, time-sync and return as views.
def sync_single_vid_feat_data(vdat, fdat):
  phase_data = {}
  features = fdat["features"]
  tstamps = fdat["tstamps"]
  labels = fdat["labels"]

  change_vals, change_idxs = detect_change_points(labels)

  for phase in vdat:  # Stabilization
    if phase == "EndBaseline":  # Intervals of label 0
      change_idx_pairs = detect_cts_pairs(change_vals, (0, 1))
      loc = "end"
    elif phase == "EndBleed":  # Intervals of label 1
      change_idx_pairs = detect_cts_pairs(change_vals, (1, 2))
      loc = "end"
    elif phase == "AfterBleed":  # Intervals of label 2
      change_idx_pairs = detect_cts_pairs(change_vals, (2, None))
      loc = "start"
    elif phase == "BeforeResusc":
      # Both correspond to resuscitation:
      change_idx_pairs = detect_cts_pairs(change_vals, (None, 3))
      change_idx_pairs.extend(detect_cts_pairs(change_vals, (None, 4)))
      loc = "end"
    elif phase == "EndHextend":
      change_idx_pairs = detect_cts_pairs(change_vals, (3, None))
      loc = "end"
    elif phase == "AfterHextend":
      change_idx_pairs = detect_cts_pairs(change_vals, (5, None))
      loc = "start"

    if change_idx_pairs:
      # IPython.embed()
      idx_pairs = [change_idxs[pair] for pair in change_idx_pairs]
      phase_data[phase] = {
          "video": vdat[phase]["feat"]["features"],
          "feat": {
              "data": [(tstamps[sidx:eidx], features[sidx:eidx])
                       for sidx, eidx in idx_pairs],
              "loc": loc,
          }
      }

  return phase_data


def load_synced_vidfeat_data(pnums=None, ftypes=["feat", "anno"], nfiles=-1):
  vdata, fdata = load_common_vidfeat_data(pnums, ftypes, nfiles)

  synced_data = {
      key: sync_single_vid_feat_data(vdata[key], fdata[key])
      for key in vdata
  }

  return synced_data


if __name__ == "__main__":
  # vdata, fdata = load_common_vidfeat_data(nfiles=2)
  ftypes = ["feat", "mat"]
  synced_data = load_synced_vidfeat_data(ftypes=ftypes, nfiles=2)
  IPython.embed()
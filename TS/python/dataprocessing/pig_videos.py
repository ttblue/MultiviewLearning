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


def load_video_data(pnums=None, ftypes=["feat", "anno"]):
  if pnums is None:
    pnums = COMMON_PNUMS

  _loadmat = lambda fl: sio.loadmat(fl)
  _loadhdf = lambda fl: h5py.File(fl, 'r')
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
            VFILE_MAP[pnum][phase][ftype])

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

# (start label, end label, parts of interval)
PHASE_MAP = {
    "EndBaseline": (0, 1, ["end"]),
    "EndBleed": (1, 2, ["end"]),
    "AfterBleed": (2, 1, ["start"]),
    "BeforeResusc": 
}
# Given single video and feature data, time-sync and return as views.
def sync_single_vid_feat_data(vdat, fdat):

  phase_data = {}
  labels = fdat["labels"]
  # Use simple array-change detection method.
  # Extract "EndBaseline"
  if "EndBaseline" in vdat:
    # Interval of 0s ending with 1.
    z_minus_o = np.where(labels == 0, 1, 0) - np.where(labels == 1, 1, 0)
    change = z_minus_o[:-1] + z_minus_o[1:]
    # First 0 is one + index where change == 1
    start_idx = (change == 1).nonzero()[0][0]
    end_idx 
    pass
  if "EndBleed" in vdat::
    pass
  if "AfterBleed" in vdat:
    pass
  if "BeforeResusc" in vdat:
    pass
  if "EndHextend" in vdat:
    pass
  if "AfterHextend" in vdat:
    pass





# def load_synced_vidfeat_data(pnums=None, ftypes=["feat", "anno"], nfiles=-1):
#   vdata, fdata = load_common_vidfeat_data(pnums, ftypes, nfiles)




if __name__ == "__main__":
  vdata, fdata = load_common_vidfeat_data(nfiles=2)
  IPython.embed()
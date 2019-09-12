# Functions to load pig-video data + combined with other data.
import h5py
import os
import numpy as np
import scipy.io as sio
import time

# from dataprocessing.featurize_pig_data import \
#     create_label_timeline, convert_tstamps_to_labels
from utils import utils

import matplotlib.pyplot as plt

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
VFILE_PNUMS = []
FFILE_PNUMS = []
COMMON_PNUMS = []
PHASE_MAP = {}

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


def load_globals():
  global ALL_VIDEO_FILES, ALL_FEAT_FILES, VFILE_MAP, FFILE_MAP, COMMON_PNUMS,\
         PNAME_VFILE_MAP, VFILE_PNUMS, FFILE_PNUMS, PHASE_MAP
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

  VFILE_PNUMS = sorted(list(VFILE_MAP.keys()))
  FFILE_PNUMS = sorted(list(FFILE_MAP.keys()))
  COMMON_PNUMS = sorted([pnum for pnum in VFILE_MAP if pnum in FFILE_MAP])

  PHASE_MAP = {
    0: "EndBaseline",
    1: "EndBleed",
    2: "AfterBleed",
    # ?: "BeforeResuc",
    3: "EndHextend",
    5: "AfterHextend",
  }
  PHASE_MAP.update({v:k for k, v in PHASE_MAP.items()})

load_globals()


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


def complete_view_feature_sets(vf_sets, all_feats):
  found = {i:False for i in all_feats}
  vf_set_completed = []
  for vset in vf_sets:
    vset = [i for i in vset if i in found]
    found.update({i: True for i in vset})
    vf_set_completed.append(vset)
  remaining_vset = [i for i, fnd in found.items() if not fnd]
  if remaining_vset:
    vf_set_completed.append(remaining_vset)

  return vf_set_completed


ALL_FEATURE_COLUMNS = [0, 3, 4, 5, 6, 7, 11]
def load_pig_features_and_labels(
    pig_list=FFILE_PNUMS, ds=1, ds_factor=10,
    feature_columns=ALL_FEATURE_COLUMNS, view_feature_sets=None,
    save_new=False, valid_labels=None):
  # Relevant data directories
  features_dir = FEAT_DIR
  #os.path.join(SAVE_DIR, "waveform/%s/numpy_arrays"%pig_type)
  ann_dir = FEAT_ANNO_DIR
  #os.path.join(DATA_DIR, "raw/annotation/%s"%pig_type)

  # Feature columns
  if feature_columns is None: feature_columns = ALL_FEATURE_COLUMNS 
  feature_columns = sorted(feature_columns)
  if 0 not in feature_columns:
    feature_columns = [0] + feature_columns

  # Finding file names from directories
  str_pattern = "\\" + str(feature_columns)[:-1] + "\\]"  # [1:-1]
  wild_card_str = "_numpy_ds_%i_columns(.)*%s(.)*"%(ds, str_pattern)

  fdict = utils.create_number_dict_from_files(
      features_dir, wild_card_str=wild_card_str)
  using_all = False

  # If specific feature columns not found, load the data with all columns
  if not fdict:
    all_feature_columns = ALL_FEATURE_COLUMNS
    str_pattern = str(all_feature_columns)# [1:-1]
    wild_card_str = "_numpy_ds_%i_columns(.)*%s(.)*"%(ds, str_pattern)

    fdict = utils.create_number_dict_from_files(
        features_dir, wild_card_str=wild_card_str)
    using_all = True
  adict = utils.create_number_dict_from_files(ann_dir, wild_card_str=".xlsx")

  if view_feature_sets is not None:
    view_feature_sets = complete_view_feature_set(
        view_feature_sets, feature_columns)
    view_findex_sets = [
        [nz_feature_cols.index(vi) for vi in vset]
        for vset in view_feature_sets
    ]
    # view_feature_sets = [vi for vi in feature_columns if vi != 0]
  # Find common pigs between annotations and data.
  common_keys = np.intersect1d(list(fdict.keys()), list(adict.keys())).tolist()
  # Create list of unused pigs for bookkeeping.
  _check_key = lambda key: key in common_keys and key in FFILE_PNUMS
  pig_list = FFILE_PNUMS if pig_list is None else pig_list
  used_pigs = [key for key in pig_list if _check_key(key)]
  np.random.shuffle(used_pigs)

  unused_pigs = [p for p in pig_list if p not in used_pigs]
  if VERBOSE:
    print("Not using pigs %s. One of anns, data or video missing." %
          (unused_pigs))

  # Extract data.
  all_data = {}
  curr_unused_pigs = len(unused_pigs)
  num_selected = 0

  finds = ([all_feature_columns.index(idx) for idx in feature_columns]
            if using_all else None)

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

    features = features[valid_inds, :]
    features = (
        [features] if view_feature_sets is None else
        [features[:, vf_set] for vf_set in view_findex_sets]
    )
    for i in range(len(features)):
      if len(features[i].shape) == 1:
        features[i] = features[i].reshape(-1, 1)
    labels = labels[valid_inds]
    tstamps = tstamps[valid_inds]

    all_data[key] = {
        "tstamps": tstamps,
        "features": features,  # Treating as a single view
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
    pig_list=FFILE_PNUMS, ds=5, ws=30, nfeats=3, view_subset=None,
    valid_labels=None):
  # Relevant data directories
  features_dir = WS_FEAT_DIR
  #os.path.join(SAVE_DIR, "waveform/%s/numpy_arrays"%pig_type)
  ann_dir = FEAT_ANNO_DIR
  #os.path.join(DATA_DIR, "raw/annotation/%s"%pig_type)
  
  fdict = utils.create_number_dict_from_files(
      features_dir, wild_card_str="_ds_%i_ws_%i.npy"%(ds, ws))
  adict = utils.create_number_dict_from_files(ann_dir, wild_card_str=".xlsx")

  common_keys = np.intersect1d(list(fdict.keys()), list(adict.keys())).tolist()
  _check_key = lambda key: key in common_keys and key in FFILE_PNUMS
  pig_list = FFILE_PNUMS if pig_list is None else pig_list
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
  return h5py.File(fl, 'r')
  # dat = h5py.File(fl, 'r')
  # loaded_dat = {
  #     k: dat[k][()] for k in dat.keys() if isinstance(dat[k], h5py.Dataset)
  # }
  # return loaded_dat


def clean_vdat(vdat, ftype):
  for key in ["__header__", "__version__", "__globals__"]:
    if key in vdat:
      del vdat[key]

  if ftype == "feat":
    del vdat["labels"]

  return vdat


def extract_vessels(mat):
  ves_ref = [mat[v][()] for v in mat["vessels"][()][0]]
  ves_pix = [[mat[v][()] for v in v1[0]] for v1 in ves_ref]
  return ves_pix


def extract_velocities(mat, v_min=-1e3, v_max=1e3):
  vel_ref = [mat[v][()].squeeze() for v in mat["velocities"][()][0]]
  vel_ref = [vr[np.bitwise_and(vr >= v_min, vr <= v_max)] for vr in vel_ref]
  return vel_ref


def plot_vessels(ves_pix, img_size=(720, 480), loop=20):
  nves = len(ves_pix)
  # img = np.zeros(img_size)
  loop = 1 if loop <= 0 else loop
  idx = 0
  for idx in range(loop):
    print("Loop %i/%i" % (idx + 1, loop))
    for i, vp in enumerate(ves_pix):
      print("Vessel %i/%i" % (i + 1, nves))
      nvp = len(vp)
      # plt.figure()
      img = np.zeros(img_size)
      for j, (a, b) in enumerate(vp):
        print("Coord set %i/%i" % (j + 1, nvp), end='\r')
        img[a, b] = 1
        # for p1, p2 in zip(a, b): 
        #  img[p1, p2] = 1
      print("Coord set %i/%i" % (j + 1, nvp))
      plt.clf()
      plt.imshow(img)
      plt.pause(0.2)
        # plt.show()


def load_video_data(pnums=None, ftypes=["feat", "anno"], phases=None):
  if pnums is None:
    pnums = VFILE_PNUMS

  if phases is not None and not isinstance(phases, list):
    phases = [phases]

  loadmat = lambda fl, ftype: clean_vdat(sio.loadmat(fl), ftype)
  loadhdf = lambda fl, ftype: load_hdf(fl)
  load_func = {
      "feat": loadmat, "anno": loadmat, "spf": loadhdf, "mat": loadhdf}

  vdata = {}
  for pnum in pnums:
    if pnum not in VFILE_MAP:
      continue
    p_vdat = {}
    # print(VFILE_MAP[pnum])
    for phase in VFILE_MAP[pnum]:
      if phases is not None and phase not in phases:
        continue
      p_vdat[phase] = {}
      for ftype in ftypes:
        p_vdat[phase][ftype] = load_func[ftype](
            VFILE_MAP[pnum][phase][ftype], ftype)
      if p_vdat:
        vdata[pnum] = p_vdat

  return vdata


def load_common_vidfeat_data(
    pnums=None, ftypes=["feat", "anno"], phases=None, f_kwargs={}, nfiles=-1,
    vs_ftype="tdPCA "):
  pnums = (
      COMMON_PNUMS if pnums is None else 
      [pnum for pnum in pnums if pnum in COMMON_PNUMS])

  if nfiles > 0:
    pnums = pnums[:nfiles]

  vdata = load_video_data(pnums, ftypes, phases)
  fdata = (
      load_tdPCA_featurized_slow_pigs(pnums, **f_kwargs)
      if vs_ftype == "tdPCA" else
      load_pig_features_and_labels(pnums, **f_kwargs))

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


_FLOW_HIST_BINS = np.arange(1, 6) + 0.5
def extract_video_hist_features(
      vdat, vel_min=-100, vel_max=100, n_bins=20, aggregate=False):
  mat = vdat["mat"]
  anno = vdat["anno"]

  vel_ref = extract_velocities(mat, vel_min, vel_max)
  flow_types = anno["labels"].reshape(1, -1)
  flow_types = flow_types[flow_types > 1]

  # Hisogram of velocities
  vel_hist = [
    np.histogram(vr, bins=n_bins, range=(vel_min, vel_max))[0] / vr.shape[0]
    for vr in vel_ref]
  flow_hist = np.histogram(flow_types, bins=_FLOW_HIST_BINS)[0] / flow_types.shape[0]
  if aggregate:
    vel_hist = np.mean(vel_hist, axis=0)
    ft = np.r_[vel_hist, flow_hist].reshape(1, -1)
    return [ft]
  else:
    fts = np.concatenate(
        [np.r_[vh, flow_hist].reshape(1, -1) for vh in vel_hist], 0)
    return [fts]


# Given single video and feature data, time-sync and return as views.
def sync_single_vid_feat_data(vdat, fdat, phases=None):
  phase_data = {}
  features = fdat["features"]
  num_f = len(features) 

  tstamps = fdat["tstamps"]
  labels = fdat["labels"]

  change_vals, change_idxs = detect_change_points(labels)

  phases = vdat.keys() if phases is None else phases
  for phase in phases:  # Stabilization
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
      idx_pairs = [change_idxs[pair] for pair in change_idx_pairs]
      ts = np.concatenate([tstamps[sidx:eidx] for sidx, eidx in idx_pairs])      
      fts = {
          i: np.concatenate([c_f[sidx:eidx] for sidx, eidx in idx_pairs], 0)
          for i, c_f in enumerate(features)
      }

      phase_data[phase] = {
          "video": extract_video_hist_features(vdat[phase], aggregate=False),
          "feat": {
              "tstamps": ts,
              "data": fts,
              "loc": loc,
          }
      }

  return phase_data


def prod_aggregation(vfeats, ffeats):
  # Have a data point for each pair of vfeat and ffeat.
  # So total num points = num(vfeats) * num(ffeats)
  nviews_f = len(ffeats)
  nviews_v = len(vfeats)
  npts_f = len(ffeats[0])
  npts_v = len(vfeats[0])

  all_feats = {i: [] for i in range(nviews_f + nviews_v)}
  for vidx in range(npts_v):
    for i in range(nviews_v):
      all_feats[i].append(np.tile(vfeats[i][vidx].reshape(1, -1), (npts_f, 1)))
    for i in range(nviews_f):
      all_feats[i + nviews_v].append(ffeats[i])

  agg_data = {
      i: np.concatenate(f, axis=0) for i, f in all_feats.items()
  }
  return agg_data


_AGG_FUNCS = {
    "prod": prod_aggregation,
}
def generate_mv_data(phase_data, agg="prod"):
  mv_data = {}
  if agg not in _AGG_FUNCS:
    raise NotImplementedError("Aggregation %s not available." % agg)
  agg_func = _AGG_FUNCS[agg]
  for ph, pdat in phase_data.items():
    ffeats = pdat["feat"]["data"]
    vfeats = pdat["video"]
    mv_data[ph] = agg_func(vfeats, ffeats)

  return mv_data


def convert_synced_data_to_mv_dataset(synced_data):
  mv_dset = {ph: None for ph in synced_data[utils.get_any_key(synced_data)]}
  for key, sdat in synced_data.items():
    mv_dat = generate_mv_data(sdat)
    for ph, pdat in mv_dat.items():
      if not mv_dset[ph]:
        mv_dset[ph] = {i: [] for i in pdat}
      for i, vdat in pdat.items():
        mv_dset[ph][i].extend(vdat)

  return mv_dset


def load_synced_vidfeat_data(
    num_pigs=-1, phases=None, vs_ftype="tdPCA", f_kwargs={}, nfiles=-1):
  ftypes = ["feat", "anno", "mat"]

  pnums = None
  vdata, fdata = load_common_vidfeat_data(
      pnums, ftypes, phases, f_kwargs, nfiles, vs_ftype)

  num_pigs = len(vdata) if num_pigs < 0 else num_pigs
  synced_data = {
      key: sync_single_vid_feat_data(vdata[key], fdata[key], phases=phases)
      for i, key in enumerate(vdata) if i < num_pigs
  }
  return convert_synced_data_to_mv_dataset(synced_data)


if __name__ == "__main__":
  # vdata, fdata = load_common_vidfeat_data(nfiles=2)
  ftypes = ["mat", "anno"] #, "feat", "spf"]
  pnums = 1
  vdata = load_video_data(VFILE_PNUMS[:pnums], ftypes)

  vdat = vdata[utils.get_any_key(vdata)]["EndBaseline"]
  mat = vdat["mat"]
  anno = vdat["anno"]
  # spf = vdat["spf"]
  feat = vdat["feat"]

  ves_pix = extract_vessels(mat)
  plot_vessels(ves_pix)
  # synced_data = load_synced_vidfeat_data(ftypes=ftypes, nfiles=2)
  IPython.embed()

# Functions to load pig-video data + combined with other data.
import h5py
import os
import numpy as np
import scipy.io as sio


import IPython


DATA_DIR = os.getenv("DATA_DIR")
VIDEO_DIR = os.path.join(DATA_DIR, "PigVideos/videos")
FEAT_DIR = os.path.join(
  DATA_DIR, "PigData/extracted/waveform/slow/numpy_arrays")

ALL_VIDEO_FILES = None
ALL_FEAT_FILES = None
VFILE_MAP = {}
PNAME_VFILE_MAP = {}
FFILE_MAP = {}
COMMON_PNUMS = []


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




IPython.embed()

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
    vdata[pnum] = {ftype: [] for ftype in ftypes}
    for ftype in ftypes:
      for fl in VFILE_MAP[pnum][ftype]:
        vdata[pnum][ftype].append(_load_func[ftype](fl))

  return vdata


def load_feat_data(pnums=None):
  if pnums is None:
    pnums = COMMON_PNUMS

  fdata = {}
  for pnum in pnums:
    if pnum not in FFILE_MAP:
      continue
    fdata[pnum] = np.load(FFILE_MAP[pnum])

  return fdata


def load_common_vidfeat_data(pnums=None, ftypes=["feat", "anno"], nfiles=-1):
  pnums = (
      COMMON_PNUMS if pnums is None else 
      [pnum for pnum in pnums if pnum in COMMON_PNUMS])

  if nfiles > 0:
    pnums = pnums[:nfiles]

  vdata, fdata = load_video_data(pnums, feats), load_feat_data(pnums)
  return vdata, fdata


# Given single video and feature data, time-sync and return as views.
# def sync_single_vid_feat_data(vdata, fdata):



# def load_synced_vidfeat_data(pnums=None, ftypes=["feat", "anno"], nfiles=-1):
#   vdata, fdata = load_common_vidfeat_data(pnums, ftypes, nfiles)




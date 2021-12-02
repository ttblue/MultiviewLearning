# Utilities for working with data from physionet.
import numpy as np
import os
from scipy import fft
import wfdb


from utils import utils


import IPython
 

_DATA_DIR = os.path.join(os.getenv("DATA_DIR"), "Physionet")
_CODE_DIR = os.path.join(os.getenv("RESEARCH_DIR"), "tests")


_MITBIH_DIR = os.path.join(_DATA_DIR, "mit-bih")
def get_mitbih_records():
  record_fname = os.path.join(_MITBIH_DIR, "RECORDS")
  with open(record_fname, "r") as fh:
    record_names = fh.readlines()

  # subject_names = []
  subject_map = {}
  records = {}
  for rec in record_names:
    rec = rec.strip()
    if rec[-1] == "a":
      sname = rec[:-1]
    elif rec[-1] != "b":
      sname = rec

    # subject_names.append(sname)
    # if sname not in subject_map:
    #   subject_map[sname] = []
    subject_map[rec] = sname

    records[rec] = [
        wfdb.rdrecord(os.path.join(_MITBIH_DIR, rec)),
        wfdb.rdann(os.path.join(_MITBIH_DIR, rec), "st")
    ]

    # rfiles = {
    #   ftype: os.path.join(_MITBIH_DIR, rec + "." + ftype)
    #   for ftype in ["dat", "ecg", "st", "hea", "st-"]
    # }
    # record_files[rec] = rfiles

  return subject_map, records


_sleep_label_map = {
    '1': 1, '2': 2, '3': 3, '4': 4, 'W': 0, 'R': 5,
}
_ANN_DT_S = 30
def get_sleep_and_apnea_labels(ann, ann_dt, tot_num_anns):
  sleep_lbls = []
  apnea_lbls = []
  ann_lbls = ann.aux_note
  for lbl in ann_lbls:
    if lbl[-1] == "\x00":
      lbl = lbl[:-1]
    lsplit = lbl.split(' ')
    l0 = lsplit[0]
    l1 = ''.join(lsplit[1:])

    l0 = _sleep_label_map.get(l0, -1)
    sleep_lbls.append(l0)
    apnea_lbls.append(l1)

  samp_idxs = (ann.sample / ann_dt).astype(int)

  all_sleep_lbls = -np.ones(tot_num_anns)
  all_sleep_lbls[samp_idxs] = sleep_lbls
  all_apnea_lbls = -np.ones(tot_num_anns, dtype=object)
  all_apnea_lbls[samp_idxs] = apnea_lbls

  return all_sleep_lbls, all_apnea_lbls


def get_mitbih_data(
    fft_size=20, signal_ids=[0,1,2], normalize=True, shuffle=True):
  subject_map, records = get_mitbih_records()
  waveform_data = {rname: rec[0].p_signal for rname, rec in records.items()}
  ann_data = {rname: rec[1] for rname, rec in records.items()}

  freq = records[utils.get_any_key(records)][0].fs
  ann_dt = _ANN_DT_S * freq

  # IPython.embed()
  view_map = {vi: sig_id for vi, sig_id in enumerate(signal_ids)}
  # if not mus:
  #   mus = {sig_id: 0. for vi in signal_ids}
  # if not stds:
  #   stds = {sig_id: 1. for vi in signal_ids}

  x_vs_rec = {}
  y_rec = {}
  y_apnea_rec = {}
  for rname, r_wfd in waveform_data.items():
    r_ann = ann_data[rname]
    tot_num_anns = int(np.ceil(r_wfd.shape[0] / ann_dt))
    slbls, albls = get_sleep_and_apnea_labels(r_ann, ann_dt, tot_num_anns)

    mus = {sig_id: r_wfd[:, sig_id].mean() for sig_id in signal_ids}
    stds = {sig_id: r_wfd[:, sig_id].std() for sig_id in signal_ids}
    r_vdat = {
        vi: ((r_wfd[:, sig_id] - mus[sig_id]) / stds[sig_id]).reshape(
            -1, ann_dt)
        for vi, sig_id in view_map.items()
    }
    # r_vdat = {
    #     vi: ((r_wfd[:, sig_id] - mus[sig_id]) / stds[sig_id]).reshape(
    #         -1, ann_dt)
    #     for vi, sig_id in view_map.items()
    # }

    r_fft_dat = {
        vi: np.abs(fft.fft(sdat, n=fft_size, axis=1))
        for vi, sdat in r_vdat.items()
    }

    x_vs_rec[rname] = r_fft_dat
    y_rec[rname] = np.array(slbls)
    y_apnea_rec[rname] = np.array(albls)

  x_vs_subj = {}
  y_subj = {}
  y_apnea_subj = {}
  for rname, r_x in x_vs_rec.items():
    sname = subject_map[rname]
    r_y = y_rec[rname]
    ra_y = y_apnea_rec[rname]

    # Remove bad labels:
    valid_inds = (r_y > -1)
    globals().update(locals())
    r_x_valid = {vi: r_xi[valid_inds] for vi, r_xi in r_x.items()}
    r_y_valid = r_y[valid_inds]
    ra_y_valid = ra_y[valid_inds]
    globals().update(locals())

    if sname in x_vs_subj:
      x_vs_subj[sname] = {
          vi: np.concatenate([xs_vi, r_x_valid[vi]], axis=0)
          for vi, xs_vi in x_vs_subj[sname].items()
      }
      y_subj[sname] = np.concatenate([y_subj[sname], r_y_valid])
      y_apnea_subj[sname] = np.concatenate([y_apnea_subj[sname], ra_y_valid])
    else:
      x_vs_subj[sname] = r_x_valid
      y_subj[sname] = r_y_valid
      y_apnea_subj[sname] = ra_y_valid
    globals().update(locals())

  # if shuffle:
  #   subjects = list(x_vs_subj.keys())
  #   for sname in subjects:
  #     s_x = x_vs_subj[sname]
  #     s_y = y_subj[sname]
  #     sa_y = y_apnea_subj[sname]
  #     npts = s_x.shape[0]
  #     shuffle_inds = np.random.permutation(npts)
  #     x_vs_subj[sname] = s_x[shuffle_inds]
  #     y_subj[sname] = s_y[shuffle_inds]
  #     y_apnea_subj[sname] = ss_y[shuffle_inds]

  return x_vs_subj, y_subj, y_apnea_subj


_mitbih_file = os.path.join(_CODE_DIR, "data/mitbih/polysom_normalized.npy")
_mitbih_stats_file = os.path.join(_CODE_DIR, "data/mitbih/tr_stats.npy")
def get_mv_mitbih_split(tr_frac=0.8, center_shift=True, shuffle=True):
  x_vs_subj, y_subj, y_apnea_subj = np.load(
      _mitbih_file, allow_pickle=True).tolist()

  subjects = list(x_vs_subj.keys())
  num_subjs = len(subjects)
  num_tr = int(num_subjs * tr_frac)

  # shuffled_ids = np.random.permutation(num_subjs)
  # tr_ids, te_ids = shuffled_ids[:num_tr], shuffled_ids[num_tr:]
  [tr_mus, tr_stds, tr_ids] = np.load(_mitbih_stats_file).tolist()
  te_ids = [i for i in range(num_subjs) if i not in tr_ids]

  tr_x, te_x = {}, {}
  tr_y, te_y = [], []
  tr_ya, te_ya = [], []

  for idx in tr_ids:
    sname = subjects[idx]
    x, y, ya = x_vs_subj[sname], y_subj[sname], y_apnea_subj[sname]
    if not tr_x:
      tr_x = {vi: [] for vi in x}
    for vi, xvi in x.items():
      tr_x[vi].append(xvi)
    tr_y.append(y)
    tr_ya.append(ya)

  for idx in te_ids:
    sname = subjects[idx]
    x, y, ya = x_vs_subj[sname], y_subj[sname], y_apnea_subj[sname]
    if not te_x:
      te_x = {vi: [] for vi in x}
    for vi, xvi in x.items():
      te_x[vi].append(xvi)
    te_y.append(y)
    te_ya.append(ya)

  tr_x = {vi: np.concatenate(xvi, axis=0) for vi, xvi in tr_x.items()}
  te_x = {vi: np.concatenate(xvi, axis=0) for vi, xvi in te_x.items()}
  tr_y = np.concatenate(tr_y)
  te_y = np.concatenate(te_y)
  tr_ya = np.concatenate(tr_ya)
  te_ya = np.concatenate(te_ya)

  # if center_shift:
  #   tr_mu_std = {vi: xvi.mean(0)}
  if shuffle:
    shuffle_inds = np.random.permutation(tr_y.shape[0])
    tr_x = {vi: xvi[shuffle_inds] for vi, xvi in tr_x.items()}
    tr_y = tr_y[shuffle_inds]
    tr_ya = tr_ya[shuffle_inds]

    shuffle_inds = np.random.permutation(te_y.shape[0])
    te_x = {vi: xvi[shuffle_inds] for vi, xvi in te_x.items()}
    te_y = te_y[shuffle_inds]
    te_ya = te_ya[shuffle_inds]

  return (tr_x, tr_y, tr_ya, tr_ids), (te_x, te_y, te_ya, te_ids)
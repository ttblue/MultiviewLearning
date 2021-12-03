# Utilities for working with data from physionet.
import eeglib
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
from scipy import fft
import time
import torch
from torch import nn, optim
import wfdb

from models import torch_models
from models.model_base import BaseConfig
from utils import torch_utils, utils


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


def process_ecg(ecg_data, freq, window_size):
  ecg_sig, info = nk.ecg_process(ecg_data, sampling_rate=freq)

  events = np.arange(0, ecg_data.shape[0], window_size)
  epochs_start = 0
  epochs_end = window_size / freq
  epochs = nk.epochs_create(
      ecg_sig, events=events, sampling_rate=freq, epochs_start=epochs_start,
      epochs_end=epochs_end)
  ecg_ft = nk.ecg_intervalrelated(epochs, sampling_rate=freq)
  ecg_ft_np = ecg_ft.iloc[:, 1:].to_numpy()
  return ecg_ft_np


def process_eeg(eeg_data, freq, window_size, psd_nperseg=20):
  # Maybe do something else to downsample
  if ds_factor > 1:
    eeg_data = eeg_data.reshape(-1, ds_factor).mean(axis=1)
    freq /= ds_factor
    window_size /= ds_factor

  eeg_data = np.atleast_2d(eeg_data)
  hp = eeglib.helpers.Helper(
      eeg_data, sampleRate=freq, windowSize=window_size)
  wrap = wrapper.Wrapper(hp)
  wrap.addFeature.bandPower()  #4
  wrap.addFeature.DFA()  #1
  wrap.addFeature.HFD()  #1
  wrap.addFeature.hjorthActivity()  #1
  wrap.addFeature.hjorthMobility()  #1
  wrap.addFeature.hjorthComplexity()  #1
  wrap.addFeature.LZC()  #1
  wrap.addFeature.PFD()  #1
  wrap.addFeature.PSD(nperseg=psd_nperseg)  #variable
  wrap.addFeature.sampEn()  #1

  eeg_ft = wrap.getAllFeatures()
  return eeg_ft


def process_rsp(rsp_data, freq, window_size):
  rsp_sig, info = nk.rsp_process(rsp_data, sampling_rate=freq)

  events = np.arange(0, rsp_data.shape[0], window_size)
  epochs_start = 0
  epochs_end = window_size / freq
  epochs = nk.epochs_create(
      rsp_sig, events=events, sampling_rate=freq, epochs_start=epochs_start,
      epochs_end=epochs_end)

  rsp_ft = []
  invalid_inds = []
  dim = None
  for idx in range(len(events)):
    epoch = epochs[str(idx + 1)]
    try:
      e_ft = nk.rsp_intervalrelated(epoch, sampling_rate=freq)
      e_ft_np = e_ft.iloc[:, 1:].to_numpy()
      rsp_ft.append(e_ft_np.squeeze())
      if dim is None:
        dim = e_ft_np.shape[1]
    except:
      rsp_ft.append(None)
      invalid_inds.append(idx)

  dummy_ft = np.ones(dim) * np.nan
  for idx in invalid_inds:
    rsp_ft[idx] = dummy_ft

  rsp_ft = np.concatenate(rsp_ft, axis=0)
  return rsp_ft


def process_bp(bp_data, freq, window_size):
  pass


def get_sig_col_ids(record, sig_names):
  col_ids = []
  _base_sigs = {"ECG": 0, "BP": 1, "EEG": 2}
  for sig in sig_names:
    if sig in _base_sigs:
      col_ids.append(_base_sigs[sig])
    elif sig not in record.sig_name:
      return None
    else:
      col_ids.append(record.sig_name.index(sig))
  return col_ids


_DEFAULT_SIG_NAMES = ["ECG", "EEG", "Resp (nasal)"]
def get_mitbih_data(
    fft_size=20, signal_names=None, ds_factor=10,
    normalize=True, shuffle=True):

  signal_names = signal_names or _DEFAULT_SIG_NAMES
  subject_map, records = get_mitbih_records()
  num_records = len(records)
  # waveform_data = {rname: rec[0].p_signal for rname, rec in records.items()}
  # ann_data = {rname: rec[1] for rname, rec in records.items()}

  base_freq = records[utils.get_any_key(records)][0].fs
  base_ann_dt = _ANN_DT_S * base_freq
  freq = base_freq / ds_factor
  window_size = base_ann_dt / ds_factor
  # IPython.embed()
  # view_map = {vi: sig_id for vi, sig_id in enumerate(signal_ids)}
  # if not mus:
  #   mus = {sig_id: 0. for vi in signal_ids}
  # if not stds:
  #   stds = {sig_id: 1. for vi in signal_ids}
  x_vs_rec = {}
  y_rec = {}
  y_apnea_rec = {}
  valid_records = []
  r_idx = 1

  invalid_cols = {}
  col_dims = {}
  for rname, (rec, ann) in waveform_data.items():
    print("\nExtracting record %s. (%i/%i)" % (rname, r_idx, num_records))
    col_ids = get_sig_col_ids(record, signal_names)
    if not col_ids:
      print("  Record %s does not have all signals." % rname)
      continue

    valid_records.append(rname)
    r_wfd = rec.p_signal
    tot_num_anns = int(np.ceil(r_wfd.shape[0] / base_ann_dt))
    slbls, albls = get_sleep_and_apnea_labels(ann, base_ann_dt, tot_num_anns)
    r_vdat = {}

    r_invalid_rows = []
    for sidx, sig_name in enumerate(signal_names):
      t_start = time.time()
      cidx = col_ids[sidx]

      sig_data = r_wfd[:, sidx]
      if ds_factor > 1:
        sig_data = sig_data.reshape(-1, ds_factor).mean(1)

      if sig_name == "ECG":
        sig_f = process_ecg(sig_data, freq, window_size)
      elif sig_name == "EEG":
        sig_f = process_eeg(sig_data, freq, window_size)
      elif sig_name == "Resp (nasal)":
        sig_f = process_rsp(sig_data, freq, window_size)
      else:
        raise ValueError("Cannot extract signal %s" % sig_name)

      if sidx not in col_dims:
        col_dims[sidx] = sig_f.shape[1]

      nan_check = np.isnan(sig_f)
      bad_cols = nan_check.any(0)
      good_cols = np.nonzero(bad_cols == False)[0]
      bad_rows = nan_check[:, good_cols].any(1)

      r_invalid_rows.extend(np.nonzero(bad_rows)[0].tolist())

      if bad_cols.any():
        if sidx not in invalid_cols:
          invalid_cols[sidx] = []
        invalid_cols[sidx].extend(list(np.nonzero(bad_cols)[0]))

      r_vdat[sidx] = sig_f
      t_diff = time.time() - t_start
      print("  Record %s -- Extracted %s in %.2fs." % (rname, sig_name, t_diff))

    r_invalid_rows = np.unique(r_invalid_rows)
    n_pts = r_vdat[0].shape[0]
    valid_flag = np.ones(n_pts).astype("bool")
    valid_flag[r_invalid_rows] = False

    r_vdat = {sidx: rvi[valid_flag] for sidx, rvi in r_vdat.items()}
    slbls = np.array(slbls)[valid_flag]
    albls = np.array(albls)[valid_flag]

    x_vs_rec[rname] = r_vdat
    y_rec[rname] = slbls
    y_apnea_rec[rname] = albls
    print("  Record %s -- cleanup..." % rname)
    # r_vdat = {
    #     vi: ((r_wfd[:, sig_id] - mus[sig_id]) / stds[sig_id]).reshape(
    #         -1, ann_dt)
    #     for vi, sig_id in view_map.items()
    # }
    # r_vdat = {
    #     vi: ((r_wfd[:, sig_id] - mus[sig_id]) / stds[sig_id]).reshape(
    #         -1, ann_dt)
    #     for vi, sig_id in view_map.items()
    # }

    # r_fft_dat = {
    #     vi: fft.fft(sdat, n=fft_size, axis=1)
    #     for vi, sdat in r_vdat.items()
    # }
    # r_fft_dat = {
    #     vi: np.concatenate([np.real(fdat), np.imag(fdat)], axis=1)
    #     for vi, fdat in r_fft_dat.items()
    # }

  invalid_cols = {
      sidx: np.unique(icols) for sidx, icols in invalid_cols.items()}
  col_valid_flags = {
      sidx:np.ones(cdim).astype("bool") for sidx, cdim in col_dims.items()}

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


_mitbih_file = os.path.join(_CODE_DIR, "data/mitbih/polysom_normalized_full_fft.npy")
_mitbih_stats_file = os.path.join(_CODE_DIR, "data/mitbih/tr_stats.npy")
def get_mv_mitbih_split(tr_frac=0.8, center_shift=True, shuffle=True):
  x_vs_subj, y_subj, y_apnea_subj = np.load(
      _mitbih_file, allow_pickle=True).tolist()

  subjects = list(x_vs_subj.keys())
  num_subjs = len(subjects)
  num_tr = int(num_subjs * tr_frac)

  shuffled_ids = np.random.permutation(num_subjs)
  tr_ids, te_ids = shuffled_ids[:num_tr], shuffled_ids[num_tr:]
  # [tr_mus, tr_stds, tr_ids] = np.load(_mitbih_stats_file).tolist()
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


class PolysomConfig(BaseConfig):
  def __init__(
      self, nn_config, lr=1e-3, batch_size=50, max_iters=1000,
      grad_clip=5., verbose=True, *args, **kwargs):

    self.nn_config = nn_config

    self.lr = lr
    self.batch_size = batch_size
    self.max_iters = max_iters
    self.grad_clip = grad_clip

    self.verbose = verbose


# Simple FC NN to classify sleep cycle
NUM_SLEEP_LABELS = 6
class PolysomClassifier(nn.Module):
  def __init__(self, config):
    self.config = config
    super(PolysomClassifier, self).__init__()
    self._setup_net()

  def _setup_net(self):
    nn_config = self.config.nn_config
    self._classifier_net = torch_models.MultiLayerNN(nn_config)
    self._pre_logit = nn.Linear(
        nn_config.output_size, NUM_SLEEP_LABELS, bias=True)

    self.cross_ent_loss = nn.CrossEntropyLoss()
    self.frozen = False

  def _cat_mv_data(self, x_vs):
    nviews = len(x_vs)
    cat_xs = torch.cat([x_vs[vi] for vi in range(nviews)], dim=1)
    return cat_xs

  def get_pre_logits(self, xs):
    net_output = self._classifier_net(xs)
    pre_logits = self._pre_logit(net_output)
    return pre_logits

  def forward(self, x_vs, y, *args, **kwargs):
    cat_xs = self._cat_mv_data(x_vs)
    x_pre_logit = self.get_pre_logits(cat_xs)
    loss_val = self.cross_ent_loss(x_pre_logit, y.long())
    return loss_val

  def freeze(self):
    for param in self.parameters():
      param.requires_grad = False
    self.frozen = True

  def pre_train(self, xs, y, *args, **kwargs):
    if self.config.verbose:
      all_start_time = time.time()

    xs = torch_utils.numpy_to_torch(xs)
    y = torch_utils.numpy_to_torch(y)
    y = y.long()

    self.opt = optim.Adam(self.parameters(), self.config.lr)

    max_iters = self.config.max_iters
    try:
      itr = -1
      for itr in range(max_iters):
        if self.config.verbose:
          itr_start_time = time.time()

        self.opt.zero_grad()
        x_pre_logit = self.get_pre_logits(xs)
        loss_val = self.cross_ent_loss(x_pre_logit, y)
        loss_val.backward()

        nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
        self.opt.step()

        if self.config.verbose:
          itr_diff_time = time.time() - itr_start_time
          lval = float(loss_val.detach())
          print("Iteration %i out of %i (in %.2fs). Loss: %.5f. " %
                (itr + 1, max_iters, itr_diff_time, lval), end='\r')

      if self.config.verbose and itr >= 0:
        print("Iteration %i out of %i (in %.2fs). Loss: %.5f. " %
              (itr + 1, max_iters, itr_diff_time, lval))
    except KeyboardInterrupt:
      print("Pre-Training interrupted. Quitting now.")

    print("Pre-Training finished in %0.2f s." % (time.time() - all_start_time))
    return self

  def predict(self, xs, rtn_torch=False):
    xs = torch_utils.numpy_to_torch(xs)
    x_pre_logit = self.get_pre_logits(xs)

    preds = torch.argmax(x_pre_logit, dim=1)
    return preds if rtn_torch else torch_utils.torch_to_numpy(preds)


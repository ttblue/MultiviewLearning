# Utility functions for processing data and such.
from __future__ import print_function

import csv
import glob
import os
import re
import xlrd

import numpy as np

import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def load_csv(filename, delim='\t', downsample=1):
  with open(filename, 'r') as fh:
    reader = csv.reader(fh)
    data = []

    # First line is the name of the columns
    row = reader.next()[0]
    col_names = row.strip(delim).split(delim)
    col_len = len(col_names)

    idx = 0
    for row in reader:
      if idx % downsample == 0:
        data_row = [float(v) for v in row[0].strip(delim).split(delim)]
        # Assuming these are few and far between:
        if len(data_row) != col_len:
          continue
        data.append(data_row)
      idx += 1

  return col_names, np.array(data)


def load_annotation_file(ann_file):

  with open(ann_file, 'r') as fh:
    ann = fh.readlines()
  
  k = 0
  ann_idx = {}
  ann_text = {}
  for s in ann:
    s = s.strip()
    s_split = s.split('\t')
    if len(s_split) == 1: continue
    ann_idx[k] = float(s_split[0])
    ann_text[k] = ' '.join(s_split[1:]).lower().strip()
    k += 1

  return ann_idx, ann_text


def load_xlsx_annotation_file(ann_file, convert_to_s=True):

  wb = xlrd.open_workbook(ann_file)
  sh = wb.sheet_by_name('Annotations')

  conversion_factor = 86400. if convert_to_s else 1

  k = 0
  missing_inds = []
  ann_time = {}
  ann_text = {}
  # First two rows are not useful.
  for ridx in xrange(2, sh.nrows):
    row = sh.row(ridx)

    if row[0].ctype != xlrd.XL_CELL_EMPTY:
      tstamp = row[0].value * conversion_factor
      ann_time[k] = tstamp
    else:
      missing_inds.append(k)
    text = row[1].value
    # Remove trailing, leading and consecutive spaces; make everything lower case.
    ann_text[k] = ' '.join(text.strip().lower().split()) 
    k += 1

  # Interpolate existing times to get the missing values.
  xp = ann_time.keys()
  fp = [ann_time[i] for i in xp]
  f = np.interp(missing_inds, xp, fp)
  for i,t in zip(missing_inds, f):
    ann_time[i] = t

  return ann_time, ann_text


def create_data_feature_filenames(data_dir, save_dir, suffix):
  if data_dir[-5] != '*.csv':
    data_dir = os.path.join(data_dir, '*.csv')
  data_files = glob.glob(data_dir)

  features_files = []
  for data_file in data_files:
    data_file = os.path.basename(data_file)
    dfile_split = data_file.split('.')
    features_file = '.'.join(dfile_split[:-1]) + suffix
    features_file = os.path.join(save_dir, features_file)
    features_files.append(features_file)

  sorted_inds = np.argsort([int(os.path.basename(dfile).split('.')[0]) for dfile in data_files])
  data_files = [data_files[i] for i in sorted_inds]
  features_files = [features_files[i] for i in sorted_inds]

  return data_files, features_files


def create_annotation_labels(ann_text):
  # Labels are:
  # 0: Stabilization
  # 1: Bleeding
  # 2: Between bleeds
  # 3: Resuscitation
  # 4: Between resuscitation events
  # 5: Post resuscitation
  # -1: None

  stabilization_start_re = re.compile("30 min stabilization period$")
  stabilization_end_re = re.compile("30 min stabilization period (completed|ended)$")
  bleed_start_re = re.compile("^bleed \# [1-9](?! stopped| temporarily stopped)")
  bleed_end_re = re.compile("^bleed \# [1-9] (stopped|temporarily stopped)$")
  resuscitation_start_re = re.compile("^((begin |_begin )*resuscitation with hextend( \(bag \#[1-9]\))*|cpr|co started|dobutamine started|lr started|(na|sodium) nitrite( started)*)$")
  resuscitation_end_re = re.compile("^(co stopped|cpr (completed|stopped)|dobutamine stopped|lr (complete|stopped)|resuscitation (\(hextend\) )*complete[d]*|(na|sodium) nitrite stopped)$")

  lbl = -1
  critical_anns = []
  ann_labels = []

  for idx, text in ann_text.iteritems():
    new_lbl = lbl
    if stabilization_start_re.search(text) is not None:
      # Everything starts with stabilization. If there are multiple, only
      # the last one matters.
      critical_anns = []
      ann_labels = []
      new_lbl = 0
    # Ignore portion right after stabilization before first bleed.
    elif stabilization_end_re.search(text) is not None:
      new_lbl = -1
    elif bleed_start_re.search(text) is not None:
      new_lbl = 1 
    elif bleed_end_re.search(text) is not None:
      new_lbl = 2
    elif resuscitation_start_re.search(text) is not None:
      new_lbl = 3
    elif resuscitation_end_re.search(text) is not None:
      new_lbl = 4

    if new_lbl != lbl:
      critical_anns.append(idx)
      ann_labels.append(lbl)
      lbl = new_lbl

  if critical_anns[-1] != idx:
    critical_anns.append(idx)
    ann_labels.append(5)

  return critical_anns, ann_labels


if __name__ == '__main__':
  import IPython
  ann_idx, ann_text = load_xlsx_annotation_file('/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/33.xlsx')
  critical_anns, ann_labels = create_annotation_labels(ann_text)
  IPython.embed()
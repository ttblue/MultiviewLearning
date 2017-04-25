# Utility functions for processing data and such.
import csv
import glob
import os
import re
import sys
import xlrd

import numpy as np

import subprocess

class UtilsException(Exception):
  pass


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


def create_data_feature_filenames(data_dir, save_dir, suffix, extension=".csv"):
  import os
  import glob
  if data_dir[-5] != '*' + extension:
    data_dir = os.path.join(data_dir, '*' + extension)
  all_data_files = glob.glob(data_dir)

  data_file_dict = {}
  features_file_dict = {}
  for data_file in all_data_files:
    data_file_basename = os.path.basename(data_file)
    dfile_split = data_file_basename.split('.')
    dfile_name_split = dfile_split[0].split("_")
    dfile_idx = int(dfile_name_split[0])
    dfile_sub_idx = 0 if len(dfile_name_split) == 1 else int(dfile_name_split[1])

    if dfile_idx not in data_file_dict:
      data_file_dict[dfile_idx] = {}
      features_file = str(dfile_idx) + suffix
      features_file = os.path.join(save_dir, features_file)
      features_file_dict[dfile_idx] = features_file

    data_file_dict[dfile_idx][dfile_sub_idx] = data_file

  data_files = []
  features_files = []
  sorted_keys = sorted(data_file_dict.keys())

  for idx in sorted_keys:
    features_files.append(features_file_dict[idx])

    dfile_dict = data_file_dict[idx]
    dfile_sorted_inds = sorted(dfile_dict.keys())
    data_files.append([dfile_dict[k] for k in dfile_sorted_inds])

  # sorted_inds = np.argsort([int(os.path.basename(dfile).split('.')[0]) for dfile in data_files])
  # data_files = [data_files[i] for i in sorted_inds]
  # features_files = [features_files[i] for i in sorted_inds]

  return data_files, features_files


def create_number_dict_from_files(data_dir, wild_card_str=None, extension=".npy"):
  # Creates a dictionary from first number in filename to name of file.
  if wild_card_str is None:
    if extension[0] != '.': extension = '.' + extension
    wild_card_str = '*' + extension

  if data_dir[-len(wild_card_str):] != wild_card_str:
    data_dir = os.path.join(data_dir, wild_card_str)
  data_files = glob.glob(data_dir)

  file_dict = {}
  for fl in data_files:
    fname = '.'.join(os.path.basename(fl).split('.')[:-1])
    fnum = None
    
    try:
      fnum = int(fname)
    except ValueError:
      pass
    if fnum is None:
      for separator in ['.', '_', '-']:
        try:
          fnum = int(fname.split(separator)[0])
        except ValueError:
          continue
        break

    if fnum is None:
      continue
    file_dict[fnum] = fl

  return file_dict


def create_annotation_labels(ann_text, console=False):
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

  for idx in range(len(ann_text)):
    text = ann_text[idx]
    new_lbl = lbl
    if stabilization_start_re.search(text) is not None:
      # Everything starts with stabilization. If there are multiple, only
      # the last one matters.
      critical_anns = []
      ann_labels = []
      lbl = -1
      new_lbl = 0
    # Ignore portion right after stabilization before first bleed.
    elif stabilization_end_re.search(text) is not None:
      new_lbl = -1
    elif bleed_start_re.search(text) is not None:
      new_lbl = 1
    elif bleed_end_re.search(text) is not None:
      new_lbl = 2
    elif resuscitation_start_re.search(text) is not None:
      # TODO: Not sure about this. The last "between bleeds" is actually right
      # before resuscitation. So perhaps it should be "between resuscitation
      # events."
      if lbl == 2: lbl = 4
      new_lbl = 3
    elif resuscitation_end_re.search(text) is not None:
      new_lbl = 4

    if new_lbl != lbl:
      critical_anns.append(idx)
      ann_labels.append(lbl)
      lbl = new_lbl

  # TODO: Currently removing all resuscitation events before bleed.
  # I don't know what the right thing to do here is.
  # if console:
  #   import IPython
  #   IPython.embed()
  last_bleed_idx = (np.array(ann_labels) == 1).nonzero()[0][-1]
  between_bleed_inds = (np.array(ann_labels[:last_bleed_idx]) > 2).nonzero()[0]
  # valid_inds = ((np.array(ann_labels[:last_bleed_idx]) <= 2).tolist()
  #               + [True] * (len(ann_labels) - last_bleed_idx))
  for bidx in between_bleed_inds:
    ann_labels[bidx] = 2
  # critical_anns = np.array(critical_anns)[valid_inds].tolist()
  # if console:
  #   import IPython
  #   IPython.embed()
  # This next check is unnecessary. If the last label is anything but 3, that's
  # worrisome.
  # The last label should be "post resuscitation".
  while ann_labels[-1] == 4:
    del ann_labels[-1], critical_anns[-1]
  if ann_labels[-1] != 3:
    import IPython
    IPython.embed()
    raise UtilsException("The last label should be 3, not %i"%ann_labels[-1])

  if critical_anns[-1] != idx:
    critical_anns.append(idx)
    ann_labels.append(5)

  return critical_anns, ann_labels


def convert_csv_to_np(csv_files, out_file, downsample=1, columns=None):
  if not isinstance(csv_files, list):
    _, data = load_csv(csv_files)
    if columns is not None:
      data = data[:, columns]
    if downsample is not None and downsample > 1:
      data = data[::downsample]
  else:
    data = []
    for csv_file in csv_files:
      _, file_data = load_csv(csv_file)
      if columns is not None:
        file_data = file_data[:, columns]
      if downsample is not None and downsample > 1:
        file_data = file_data[::downsample]
      data.append(file_data)
    data = np.concatenate(data, axis=0)

  np.save(out_file, data)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


# if __name__ == '__main__':
#   import IPython
#   ann_idx, ann_text = load_xlsx_annotation_file('/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/33.xlsx')
#   critical_anns, ann_labels = create_annotation_labels(ann_text)
#   IPython.embed()
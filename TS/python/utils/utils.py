# Utility functions for processing data and such.
import argparse
import csv
import glob
import numpy as np
import os
import re
import subprocess
import sys
import xlrd

import IPython


class UtilsException(Exception):
  pass


def get_args(options=[]):
  # Arguments from command line using parser (usually for testing)
  # options is a list of (<name>, <type>, <help>, <default>) tuples
  parser = argparse.ArgumentParser(description="Default parser")
  parser.add_argument(
      "--expt", type=int, help="Experiment to be run.", default=0)
  for aname, atype, ahelp, adefault in options:
    if aname[:2] != "--":
      aname = "--" + aname
    if atype is bool:
      action = "store_false" if adefault else "store_true"
      parser.add_argument(aname, help=ahelp, default=adefault, action=action)
    else:
      parser.add_argument(aname, type=atype, help=ahelp, default=adefault)
  return parser.parse_args()


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
  try:
    if "Annotations" in wb.sheet_names():
      sh = wb.sheet_by_name("Annotations")
    else:
      sh = wb.sheet_by_name("Sheet1")
  except:
    IPython.embed()

  conversion_factor = 86400. if convert_to_s else 1

  k = 0
  missing_inds = []
  ann_time = {}
  ann_text = {}
  # First two rows are not useful.
  for ridx in range(2, sh.nrows):
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
  xp = list(ann_time.keys())
  fp = [ann_time[i] for i in xp]
  # IPython.embed()
  f = np.interp(missing_inds, xp, fp)
  for i,t in zip(missing_inds, f):
    ann_time[i] = t

  return ann_time, ann_text


def create_data_feature_filenames(data_dir, save_dir, suffix, extension=".csv"):
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
    wild_card_str = extension

  # if data_dir[-len(wild_card_str):] != wild_card_str:
  #   data_dir = os.path.join(data_dir, wild_card_str)
  # data_files = glob.glob(data_dir)
  pattern = re.compile(wild_card_str)
  file_dict = {}
  for fl in os.listdir(data_dir):
    if not re.search(wild_card_str, fl):
      continue

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
    file_dict[fnum] = os.path.join(data_dir, fl)

  return file_dict


def create_annotation_labels(ann_text, console=False):
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

  stabilization_start_re = re.compile("min stabilization period$")
  stabilization_end_re = re.compile("min stabilization period (completed|ended)$")
  bleed_start_re = re.compile("^bleed \#[ ]*[1-9](?! stopped| temporarily stopped)")
  bleed_end_re = re.compile("^bleed \#[ ]*[1-9] (stopped|temporarily stopped)$")
  resuscitation_start_re = re.compile("^((begin |_begin )*resuscitation( with hextend( \(bag \#[1-9]\))*)*|cpr|co started|dobutamine started|lr started|(na|sodium) nitrite( started)*|resuscitation resumed)$")
  resuscitation_end_re = re.compile("^(co stopped|cpr (completed|stopped)|dobutamine stopped|lr (complete|stopped)|resuscitation (\(hextend\) )*(complete[d]*|stopped)|(na|sodium) nitrite stopped|stop resuscitation|resuscitation paused)$")
  
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
      # Currently treating the time after last bleed and first resuscitation as 
      # Between bleeds, since it should have the same characteristics.
      # if lbl == 2: lbl = 4
      new_lbl = 3 if "hextend" in text else 4
    elif resuscitation_end_re.search(text) is not None:
      # TODO: Don't know about this:
      # if lbl not in [3, 4]: continue  # If resuscitation stops without start
      new_lbl = 5 if ("hextend" in text or lbl == 3) else 6

    if new_lbl != lbl:
      critical_anns.append(idx)
      ann_labels.append(lbl)
      lbl = new_lbl

  # IPython.embed()
  # TODO: Currently removing all resuscitation events before bleed.
  # I don't know what the right thing to do here is.
  if 0 not in ann_labels:
    IPython.embed()
  try:
    last_bleed_idx = (np.array(ann_labels) == 1).nonzero()[0][-1]
    between_bleed_inds = (np.array(ann_labels[:last_bleed_idx]) > 2).nonzero()[0]
  except:
    print("No bleeds found.")
    return None, None
  # valid_inds = ((np.array(ann_labels[:last_bleed_idx]) <= 2).tolist()
  #               + [True] * (len(ann_labels) - last_bleed_idx))
  for bidx in between_bleed_inds:
    ann_labels[bidx] = 2
  # critical_anns = np.array(critical_anns)[valid_inds].tolist()
  # if console:
  #   IPython.embed()
  # This next check is unnecessary. If the last label is anything but 3, that's
  # worrisome.
  # The last label should be "post resuscitation".
  while ann_labels[-1] in [5, 6]:
    del ann_labels[-1], critical_anns[-1]
  # if ann_labels[-1] != 3:
  #   IPython.embed()
  #   raise UtilsException("The last label should be 3, not %i"%ann_labels[-1])

  if critical_anns[-1] != idx:
    critical_anns.append(idx)
    ann_labels.append(7)

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


# Misc. useful utilities

def flatten(list_of_lists):
  return [a for b in list_of_lists for a in b]


def is_valid_partitioning(G, dim):
  if np.sum([len(g) for g in G]) != dim:
    return False

  flat_G = flatten(G)
  if len(flat_G) != dim:
    return False

  d_range = list(range(dim))
  for g in flat_G:
    if g not in d_range:
      return False
  return True


def get_any_key(dict_var):
  return list(dict_var.keys())[0]


# Data utils:
def split_data(Xs, fracs=[0.8, 0.2], shuffle=True, get_inds=True):
  fracs = np.array(fracs) / np.sum(fracs)

  npts = len(Xs)
  num_split = (fracs * npts).astype(int)
  num_split[-1] = npts - num_split[:-1].sum()
  all_inds = np.random.permutation(npts) if shuffle else np.arange(npts)
  end_inds = np.cumsum(num_split).tolist()
  start_inds = [0] + end_inds[:-1]

  dsets = []
  for si, ei in zip(start_inds, end_inds):
    split_inds = all_inds[si:ei]
    split_xs = Xs[split_inds]
    dsets.append(split_xs)

  if get_inds:
    inds = [all_inds[si:ei] for si, ei in zip(start_inds, end_inds)]
    return dsets, inds
  return dsets


# if __name__ == '__main__':
#   ann_idx, ann_text = load_xlsx_annotation_file('/usr0/home/sibiv/Research/Data/TransferLearning/PigData/extracted/33.xlsx')
#   critical_anns, ann_labels = create_annotation_labels(ann_text)
#   IPython.embed()


# Utilities for working with data from physionet.
import numpy as np
import os
import wfdb


from utils import utils
 

_DATA_DIR = os.path.join(os.getenv("DATA_DIR"), "Physionet")


_MITBIH_DIR = os.path.join(_DATA_DIR, "mit-bih")
def get_mitbih_records():
  record_fname = os.path.join(_MITBIH_DIR, "RECORDS")
  with open(record_fname, "r") as fh:
    record_names = fh.readlines()

  subject_names = []
  subject_map = {}
  records = {}
  for rec in record_names:
    if rec[-1] == "a":
      sname = rec[:-1]
    elif rec[-1] != "b":
      sname = rec

    subject_names.append(sname)
    if sname not in subject_map:
      subject_map[sname] = []
    subject_map[sname].append(rec)

    records[rec] = [
        wfdb.rdrecord(os.path.join(_MITBIH_DIR, rec)),
        wfdb.rdann(os.path.join(_MITBIH_DIR, rec), "st")
    ]

    # rfiles = {
    #   ftype: os.path.join(_MITBIH_DIR, rec + "." + ftype)
    #   for ftype in ["dat", "ecg", "st", "hea", "st-"]
    # }
    # record_files[rec] = rfiles

  return subject_names, subject_map, records


def get_mitbih_data():
  subject_names, subject_map, records = get_mitbih_records()
  waveform_data = {rname: rec[0].p_signal for rname, rec in records.items()}
  freq = records[utils.get_any_key(records)][0].fs


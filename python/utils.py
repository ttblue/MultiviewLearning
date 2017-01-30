# Utility functions for processing data and such.
import csv
import xlrd

import numpy as np


def load_csv(filename, delim='\t', downsample=1):
  with open(filename, 'r') as fh:
    reader = csv.reader(fh)
    data = []

    # First line is the name of the columns
    row = reader.next()[0]
    col_names = row.strip(delim).split(delim)

    idx = 0
    for row in reader:
      if idx % downsample == 0:
        data.append([float(v) for v in row[0].strip(delim).split(delim)])
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
    ann_text[k] = ' '.join(s_split[1:])
    k += 1

  return ann_idx, ann_text


def load_xlsx_annotation_file(ann_file):

  wb = xlrd.open_workbook(ann_file)
  sh = wb.sheet_by_name('Annotations')

  k = 0
  ann_idx = {}
  ann_text = {}
  for s in ann:
    s = s.strip()
    s_split = s.split('\t')
    if len(s_split) == 1: continue
    ann_idx[k] = float(s_split[0])
    ann_text[k] = ' '.join(s_split[1:])
    k += 1

  return ann_idx, ann_text


def create_annotation_file_from_xlsx(xlsx_file, ann_file):
  pass
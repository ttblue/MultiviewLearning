# Utility functions for processing data and such.
import csv

import numpy as np

def load_csv(filename, delim='\t', downsample=1):
  with open(filename, 'r') as fh:
    reader = csv.reader(fh)
    data = []

    # First line is the labels of the columns
    row = reader.next()[0]
    labels = row.strip(delim).split(delim)

    idx = 0
    for row in reader:
      if idx % downsample == 0:
        data.append([float(v) for v in row[0].strip(delim).split(delim)])
      idx += 1

  return labels, np.array(data)
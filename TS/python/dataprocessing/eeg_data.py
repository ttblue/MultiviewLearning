import ast
import numpy as np
import os
import pandas as pd
import scipy
import wfdb


# PTB-XL data
_PTBXL_DIR = os.path.join(
    "/usr0/home/sibiv/Research/TransferLearning/Data/PTB-XL/",
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/")
_SAMPLING_RATE = 100

def load_raw_data(df, sampling_rate=_SAMPLING_RATE, path=_PTBXL_DIR):
  if sampling_rate == 100:
    data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
  else:
    data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
  data = np.array([signal for signal, meta in data])
  return data


def load_ptbxl_annotations(path=_PTBXL_DIR):
  Y = pd.read_csv(
    os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id")
  Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
  return Y


_PTBXL_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
def convert_ptbxl_labels_to_numpy(y):
  y_vals = y.values

  label_checker = lambda lst, val: val in lst
  y_output = []
  for i, lbl in enumerate(labels):
    y_lbl = np.where(map(lambda lst: label_checker(lst, lbl), y_vals)).astype(int)
    y_output.append(y_lbl.reshape(1, -1))

  y_output = np.concatenate(y_output, axis=1)
  return y_output, labels


def load_ptbxl(
    view_index_groups=None, fft_size=20, sampling_rate=_SAMPLING_RATE,
    n_frac=0.8, path=_PTBXL_DIR):
  if view_index_groups is None:
    view_index_groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8, 9, 10, 11)]
  # Annotations

  Y = load_ptbxl_annotations(path)
  # Load raw signal data
  X = load_raw_data(Y, sampling_rate, path)

  # Load scp_statements.csv for diagnostic aggregation
  agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
  agg_df = agg_df[agg_df.diagnostic == 1]

  def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

  Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

  # FFT featurization:
  # Taking the absolute value instead of real
  X = np.abs(scipy.fft(X, n=fft_size, axis=1))

  # Split data into train and test
  test_fold = 10
  # Train
  X_train = X[np.where(Y.strat_fold != test_fold)]
  y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
  # Test
  X_test = X[np.where(Y.strat_fold == test_fold)]
  y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

  Xtr_mv = {}
  Xte_mv = {}
  n_tr, n_te = X_train.shape[0], X_test.shape[0]
  for vi, idx_grp in enumerate(view_index_groups):
    Xtr_mv[vi] = X_train[:, :, idx_grp].reshape(n_tr, -1)
    Xte_mv[vi] = X_test[:, :, idx_grp].reshape(n_te, -1)

  return (Xtr_mv, y_train), (Xte_mv, y_test), Y


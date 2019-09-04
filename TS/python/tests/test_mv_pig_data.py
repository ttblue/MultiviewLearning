# Tests for some CCA stuff on pig data.
import numpy as np
import os
import torch
from torch import nn

from dataprocessing import pig_videos
from models import embeddings, ovr_mcca_embeddings, naive_multi_view_rl, \
                   naive_single_view_rl
from synthetic import multimodal_systems as ms
from utils import torch_utils as tu


try:
  import matplotlib.pyplot as plt
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False


import IPython


np.set_printoptions(precision=5, suppress=True)


# Feature names:
# 0:  time (absolute time w.r.t. start of experiment)
# 1:  x-value (relative time w.r.t. first vital sign reading)
# 2:  EKG
# 3:  Art_pressure_MILLAR
# 4:  Art_pressure_Fluid_Filled
# 5:  Pulmonary_pressure
# 6:  CVP
# 7:  Plethysmograph
# 8:  CCO (continuous cardiac output)
# 9:  SVO2
# 10: SPO2
# 11: Airway_pressure
# 12: Vigeleo_SVV

# Using 3, 4, 5, 6, 7, 11
def test_vitals_only(num_pigs=3):
  pnums = pig_videos.COMMON_PNUMS
  if num_pigs > 0:
    pnums = pnums[:num_pigs]

  ds = 1
  ds_factor = 10
  vs_data = pig_videos.load_pig_features_and_labels(
      pig_list=pnums, ds=ds, ds_factor=ds_factor)


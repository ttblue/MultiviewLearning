# Utility functions for dealing with EEGLab data.   
import os

from oct2py import octave

# Set up functions from EEGLab
_HOME_DIR = os.getenv("HOME")
_EEGLAB_BASE_DIR = os.path.join(_HOME_DIR, "opt/EEGLab/eeglab14_1_2b")
_EEGLAB_FUNCTIONS_DIR = os.path.join(_EEGLAB_BASE_DIR, "functions")
octave.addpath(os.path.join(_EEGLAB_FUNCTIONS_DIR, "guifunc"))
octave.addpath(os.path.join(_EEGLAB_FUNCTIONS_DIR, "popfunc"))
octave.addpath(os.path.join(_EEGLAB_FUNCTIONS_DIR, "adminfunc"))
octave.addpath(os.path.join(_EEGLAB_FUNCTIONS_DIR, "sigprocfunc"))
octave.addpath(os.path.join(_EEGLAB_FUNCTIONS_DIR, "miscfunc"))


def load_set(file_name):
  EEG = octave.pop_loadset(file_name)
  return EEG
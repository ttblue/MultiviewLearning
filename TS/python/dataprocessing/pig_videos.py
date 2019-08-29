# Functions to load pig-video data + combined with other data.
import os
import numpy as np

DATA_DIR = os.getenv("DATA_DIR")
VIDEO_DIR = os.path.join(DATA_DIR, )

def get_numbered_files():
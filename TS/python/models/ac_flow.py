# Based on: https://arxiv.org/abs/1909.06319

import numpy as np
import torch

from models import robust_multi_ae, flow_pipeline,\
    flow_transforms, flow_likelihood


class MACFlowConfig(flow_pipeline.MFTConfig):
  def __init__(self, *args, **kwargs):
    super(MACFlowConfig, self).__init__(*args, **kwargs)


class MultiviewACFlow(flow_pipeline.MultiviewFlowTrainer):
  def __init__(self, config):
    super(MultiviewACFlow, self).__init__(config)

  def initialize(self, init_data):
    pass

  def fit(self, Xs):
    pass

  def predict(self, Xs, vout=None):
    pass
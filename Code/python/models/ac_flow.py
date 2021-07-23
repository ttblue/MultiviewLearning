# Based on: https://arxiv.org/abs/1909.06319
# Loss is almost exactly the same as before -- just need to only select 
# terms for available views
# Add discriminator?


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

  def sample_x(self, n):
    pass

  def sample_z(self, n):
    pass

  def _loss(self, ):
    pass

  def forward(self, x, b):
    pass

  def fit(self, Xs):
    pass

  def predict(self, Xs, vout=None):
    pass

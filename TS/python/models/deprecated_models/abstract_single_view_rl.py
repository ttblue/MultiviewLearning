# Abstract classes for single view RL solvers
import torch

from models.model_base import BaseConfig


class AbstractSVSConfig(BaseConfig):
  pass


# Simple abstract base class 
class AbstractSingleViewSolver(torch.nn.Module):
  def __init__(self, view_id, config):
    self.view_id = view_id
    self.config = config

    self._has_data = False
    self.projections = None

    self.has_history = False

    super(AbstractSingleViewSolver, self).__init__()

  def set_data(self, data):
    raise NotImplemented("Abstract class method.")

  def initialize(self):
    raise NotImplemented("Abstract class method.")

  def fit(self):
    raise NotImplemented("Abstract class method.")

  def compute_projections(self):
    raise NotImplemented("Abstract class method.")

  def get_objective(self, obj_type):
    raise NotImplemented("Abstract class method.")
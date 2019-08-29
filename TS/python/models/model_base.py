# Base stuff for different models.
import copy


class ModelException(Exception):
  pass


class BaseConfig(object):
  def __init__(self, verbose=True, *args, **kwargs):
    self.verbose = verbose

  def copy(self, deep=True):
    return copy.deepcopy(self) if deep else copy.copy(self)
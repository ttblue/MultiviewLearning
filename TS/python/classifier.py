# Abstract classes for classifiers and their parameters.


class ClassifierException(Exception):
  pass


class Config(object):
  pass


class Classifier(object):

  def __init__(self, config):
    self.config = config

  def fit(self, x, y):
    raise NotImplementedError()

  def predict(self, x, y):
    raise NotImplementedError()

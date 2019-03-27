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

  def _print_if_verbose(self, lines, **kwargs):
    if not self.config.verbose:
      return

    lines = lines if isinstance(lines, list) else [lines]
    for line in lines:
      print(line, **kwargs)
"""
Abstract class for generating synthetic data. Provides high level functionality
expected from the generated data.
"""


class TimeSeriesGenerationException(Exception):
  pass


class TimeSeriesGenerator(object):
  """
  Abstract class for generating synthetic time series.
  """

  def sampleNext(self):
    """
    Sample the next value.
    """
    raise NotImplementedError()

  def sampleNSteps(self, n):
    """
    Sample n steps into the future.
    """
    raise NotImplementedError()
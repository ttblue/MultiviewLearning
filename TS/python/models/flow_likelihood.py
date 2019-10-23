
################################################################################
# Auto-regressive models

class ARMConfig(BaseConfig):
  def __init__(self, *arg, **kwargs):
    super(ARMConfig, self).__init__(*args, **kwargs)


class BaseARM(object):
  def __init__(self, config):
    self.config = config

  def sample(self):
    raise NotImplementedError("Abstract class method")

  def log_likelihood(self, x):
    raise NotImplementedError("Abstract class method")


class LinearARM(BaseARM):
  def __init__(self, config):
    super(LinearARM, self).__init__(config)

  

class RecurrentARM(BaseARM):
  def __init__(self, config):
    super(RecurrentARM, self).__init__(config)

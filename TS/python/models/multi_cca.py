import copy
import multiprocessing as mp
import numpy as np

from models import embeddings


class OVRCCAConfig(object):
  def __init__(
      self, cca_config_dict={}, parallel=True, n_processes=4, save_file=None,
      verbose=True):

    self.cca_config_dict = cca_config_dict

    self.parallel = parallel
    self.n_processes = n_processes

    self.save_file = save_file
    self.verbose = verbose


# For parallelizing:
def learn_ovr_embedding(view_data, vi, cca_config):
  model = embeddings.GroupRegularizedCCA(cca_config)

  X = view_data[vi]
  Gx = [np.arange(X.shape[1])]

  rest_of_views = [vdata for i, vdata in view_data.items() if i != vi]
  Y = np.concatenate(rest_of_views, axis=1)
  # generate groupings based on views
  view_dims = [vdata.shape[1] for vdata in rest_of_views]
  dims_cs = [0] + np.cumsum(view_dims).tolist()
  Gy = [np.arange(lb, ub) for lb, ub in zip(dims_cs[:-1], dims_cs[1:])]

  return vi, model.fit(X, Y, Gx, Gy)


class OneVsRestCCA(object):
  def __init__(self, config):
    self.config = config

  def _initialize(self):
    self._view_configs = {}
    base_config = self.config.cca_config_dict
    if self.config.parallel:
      base_config["verbose"] = False
      base_config["plot"] = False

    for i in range(self._nviews):
      ndim = self._view_data[i].shape[1]
      config = copy.deepcopy(base_config)
      config["ndim"] = ndim
      config["name"] = "view %i" % (i + 1)
      self._view_configs[i] = embeddings.CCAConfig(**config)

  def _learn_all_embeddings(self):
    if self.config.parallel:
      # TODO: Fix memory-copying across processes
      if self.config.verbose:
        print("Starting parallel training for each view.")

      pool = mp.Pool(self.config.n_processes)
      data = self._view_data
      args = [(data, vi, self._view_configs[vi]) for vi in range(self._nviews)]
      results = pool.starmap(learn_ovr_embedding, args)
      self.view_models = {vi: model for vi, model in results}

    else:
      self.view_models = {}
      for vi in range(self._nviews):
        if self.config.verbose:
          print("\nLearning embedding for view %i." % (vi + 1))
        config = self._view_configs[vi]
        _, model = learn_ovr_embedding(self._view_data, vi, config)
        self.view_models[vi] = model

  def fit(self, view_data):
    self._view_data = view_data
    self._nviews = len(view_data)

    self._initialize()

    if self.config.verbose:
      print("Starting one-vs-rest training for all views.")
    self._learn_all_embeddings()

    return self
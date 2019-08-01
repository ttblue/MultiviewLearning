import copy
import multiprocessing as mp
import numpy as np

from models import embeddings
from models.model_base import ModelException


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


class OneVsRestMCCA(object):
  def __init__(self, config):
    self.config = config
    self._training = True

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
    self._training = False

    return self

  def save_to_file(self, fname, misc=None):
    data = {}
    for vi, model in self.view_models.items():
      data[vi] = {
          'x':model._X, 'ux':model._ux, 'vy':model._vy, 'y':model._Y,
          'Gx':model._Gx, 'Gy':model._Gy, 'config':model.config
      }
    data = data if misc is None else [data, misc]
    np.save(fname, data)


class OMLSLConfig(object):
  def __init__(self, sv_thresh=1e-6):
    self.sv_thresh = sv_thresh


class OVRMCCALatentSpaceLearner(object):
  def __init__(self, config, mcca_learner=None):
    self.config = config
    self.mcca_learner = mcca_learner
    self._training = True if mcca_learner is None else mcca_learner._training

  def set_mcca_learner(self, learner, mcca_config=None):
    if learner is None:
      learner = OneVsRestMCCA(mcca_config)

    self.mcca_learner = learner
    self._training = learner._training

  def extract_latent_space(self):
    if self.mcca_learner is None or self._training is True:
      raise ModelException(
          "Model has not been trained or OVRMCCA learner has not been set.")

    # To simplify variables and code
    learner = mcca_learner
    data = learner._view_data
    models = learner.view_models
    nviews = len(data)

    # For computing padding and such when all projections are put together.
    view_dims = [data[i].shape[1] for i in range(nviews)]
    dims_cs = [0] + np.cumsum(view_dims).tolist()
    all_dim = dims_cs[-1]

    # Here, for each view's "rest" projection from the OVR CCA computation,
    # we're padding 0's for the view's position, since it is not included in
    # the "rest." This is so that the "rest" projections from all the views
    # can be concatenated together.
    padded_Pis = []
    for vi in range(nviews):
      Pi = models[i]._vy
      Pi_start, Pi_end = Pi[:dims_cs[vi]], Pi[dims_cs[vi]:]
      zero_pad = np.zeros((view_dims[vi], Pi.shape[1]))
      padded_Pis.append(np.concatenate([Pi_start, zero_pad, Pi_end], axis=0))

    all_P = np.concatenate(padded_Pis, axis=1)
    U, S, VT = np.linalg.svd(all_P.T, full_matrices=True)
    true_dim = np.sum(S > self.config.sv_thresh)

    # Combination of all the padded "rest" projections from each view's OVR CCA
    # computation. Columns of all_P are the individual projections
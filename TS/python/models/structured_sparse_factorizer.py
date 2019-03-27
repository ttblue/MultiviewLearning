# Structured Sparse Latent Factorizer as taken from:
# https://papers.nips.cc/paper/3953-factorized-latent-spaces-with-structured\
# -sparsity.pdf
#
import cvxpy as cvx
import numpy as np
import time

from models import classifier
from utils import math_utils as mu

import IPython


_REGULARIZERS = {
  "L1": None,
  "Linf": None,
  "L1_inf": (lambda mat: cvx.mixed_norm(mat, "inf", 1)),
  "Linf_1": None,
}


class SSLFConfig(classifier.Config):
  def __init__(
      self, regularizer="L1_inf", reduce_dict_every_iter=True,
      D_init_var_frac=0.95, removal_thresh=0.02, lmbda=1.0, gamma=1.0,
      max_iters=100, verbose=True):

    self.regularizer = regularizer
    self.reduce_dict_every_iter = reduce_dict_every_iter
    self.D_init_var_frac = D_init_var_frac
    self.removal_thresh = removal_thresh

    self.lmbda = lmbda
    self.gamma = gamma

    self.max_iters = max_iters

    self.verbose = verbose


class StructuredSparseLatentFactorizer(classifier.Classifier):
  def __init__(self, config):
    super(StructuredSparseLatentFactorizer, self).__init__(config)

    if self.config.regularizer not in _REGULARIZERS:
      raise classifier.ClassiferException(
          "Regularizer %s not available." % self.config.regularizer)
    self._r_func = _REGULARIZERS[self.config.regularizer]

    self.alpha = None
    self.D = None
    self._default_key = None

    self.is_trained = False

  def get_dictionary_size(self):
    if self.D is None:
      raise classifier.ClassiferException("Training has not been done yet.")
    if self._default_key is None:
      self._default_key = next(iter(self.D.keys()))
    return self.D[self._default_key].shape[1]

  def _optimize_alpha(self):
    D = self.D
    txs = self._dset.v_txs
    n_ts = self._dset.n_ts

    IPython.embed()
    alpha = cvx.Variable((self.get_dictionary_size(), self._n_pts))
    recon_obj = 0.
    for vi in range(self._n_views):
      residual = D[vi] * alpha - txs[vi].T
      recon_obj += cvxpy.norm(residual, "fro")
    recon_obj /= n_ts
    alpha_reg = self._r_func(alpha)

    obj = cvx.Minimize(recon_obj + self.config.gamma * alpha_reg)
    prob = cvx.Problem(obj)

    prob.solve(verbose=self.config.verbose)
    self.alpha = alpha.value
    self._itr_alpha_loss = prob.value

  def _optimize_D(self):
    alpha = self.alpha
    txs = self._dset.v_txs
    dims = self._dset.v_dims
    n_ts = self._dset.n_ts
    D_size = self.get_dictionary_size()

    D = {vi:cvx.Variable((dims[vi], D_size)) for vi in range(self._n_views)}

    recon_obj = 0.
    D_reg = 0.
    for vi in D:
      residual = D[vi] * alpha - txs[vi].T
      recon_obj += cvxpy.norm(residual, "fro")
      D_reg += self._r_func(D[vi].T)
    recon_obj /= n_ts

    obj = cvx.Minimize(recon_obj + self.config.lmbda * D_reg)
    prob = cvx.Problem(obj)

    prob.solve(verbose=self.config.verbose)
    self.D = {vi:D[vi].value for vi in range(self._n_views)}
    self._itr_D_loss = prob.value

  def _remove_unneeded_elements_in_D(self):
    # Removing dictionary elements which are below threshold in each view.
    valid_D_cols = np.vstack(
        [(np.linalg.norm(self.D[vi], axis=0) >= self.config.removal_thresh)
            for vi in self.D])
    valid_inds = np.all(valid_D_cols, axis=0)
    self.D = {vi:self.D[vi][:, valid_inds] for vi in self.D}

    return int(valid_inds.shape[0] - valid_inds.sum())

  def _get_svd_D_alpha(self, txs):
    u, s, vt = np.linalg.svd(txs)
    conf_ind = mu.get_index_p_in_ordered_pdf(s, self.config.D_init_var_frac)
    D = vt[:conf_ind].T
    alpha = np.diag(s[:conf_ind]).dot(u.T[:conf_ind])
    return D, alpha

  def _initialize_params(self):
    txs = self._dset.v_txs
    concat_data = np.concatenate(list(txs.values()), axis=0)

    D_concat, alpha_concat = self._get_svd_D_alpha(concat_data)
    all_Ds = [D_concat]
    all_alphas = [D_concat]
    for vi in txs:
      D, alpha = self._get_svd_D_alpha(txs[vi])
      all_Ds.append(D)
      all_alphas.append(alpha)

    D = np.concatenate(all_Ds, axis=1)
    init_D = {vi:D for vi in range(self._n_views)}
    init_alpha = np.concatenate(all_alphas, axis=0)
    return init_D, init_alpha

  def _train_step(self):
    self._itr_alpha_loss = self._itr_D_loss = 0

    self._print_if_verbose("Optimizing alpha...")
    self._optimize_alpha()
    self._print_if_verbose("Optimizing D...")
    self._optimize_D()

    if self.config.reduce_dict_every_iter:
      self._print_if_verbose("Removing unneeded elements in D...")
      num_removed = self._remove_unneeded_elements_in_D()
      self._print_if_verbose(
          "%i elements with norm <= %.2f removed." %
          (num_removed, self.config.removal_thresh))

  def fit(self, dset, init_alpha=None, init_D=None):
    if not dset.synced:
      raise classifier.ClassiferException("Dataset must be synced.")

    # For keeping track of things
    all_start_time = time.time()

    self._dset = dset
    self._n_views, self._n_pts = dset.n_views, dset.n_ts
    if init_alpha is None or init_D is None:
      D, alpha = self._initialize_params()
      init_alpha = alpha if init_alpha is None else init_alpha
      init_D = D if init_D is None else init_D

    self.alpha, self.D = init_alpha, init_D

    try:
      for itr in range(self.config.max_iters):
        itr_start_time = time.time()
        self._print_if_verbose(
            "Epoch %i out of %i." % (itr + 1, self.config.max_iters))

        self._train_step()

        itr_duration = time.time() - itr_start_time

        self._print_if_verbose(
            "Loss for Alpha step: %.2f" % float(self._itr_alpha_loss))
        self._print_if_verbose(
            "Loss for D step: %.2f" % float(self._itr_D_loss))
        self._print_if_verbose(
            "Epoch %i took %0.2fs.\n" % (itr + 1, itr_duration))

    except KeyboardInterrupt:
      self._print_if_verbose("Training interrupted. Qutting now.")

    self.is_trained = True
    self._train_time = time.time() - all_start_time
    self._print_if_verbose("Training finished in %.2fs." % self._train_time)

  def predict(self, txs, v_out=None):
    if v_out is None:
      v_out = list(range(self._n_views))
    if not isinstance(v_out, list):
      v_out = [v_out]

    n_pts = txs[next(iter(txs.keys()))].shape[0]

    D = self.D
    alpha = cvx.Variable((self.get_dictionary_size(), n_pts))
    recon_obj = 0.
    for vi in txs:
      residual = D[vi] * alpha - txs[vi]
      recon_obj += cvxpy.norm(residual, "fro")
    recon_obj /= n_pts
    alpha_reg = cvx.norm(alpha, 1)

    obj = cvx.Minimize(recon_obj + self.config.gamma * alpha_reg)
    prob = cvx.Problem(obj)

    prob.solve(verbose=self.config.verbose)
    alpha = alpha.value

    self._print_if_verbose("Loss for fitting alpha: %.2f" % prob.value)
    preds = {}
    for vi in v_out:
      preds[vi] = (D[vi].dot(alpha)).T

    return preds
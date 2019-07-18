# Uses a synchronous or asynchronous module to forecast forward

import numpy as np
import time
import torch
from torch import autograd
from torch import nn
from torch import optim

from models.model_base import ModelException
from models import multi_rnn_ae as mrae
from models import seq_star_gan as ssg
from utils import torch_utils as tu
from utils.torch_utils import _DTYPE, _TENSOR_FUNC
from utils import utils

import IPython


class MVModelException(ModelException):
  pass


# Learning a two-step forecaster which first learns latent space and then learns
# RNN over latent state.
# TODO: Warm start over learned params in latent state training.

# Currently only MV RNN AE supported.
_LS_LEARNERS = {
    "mrae": mrae.MultiRNNAutoEncoder,
    # "ssg": ssg.SSGConfig,
}


class MVForecasterConfig(object):
  def __init__(
      self, ls_type, ls_learner_config, layer_params, lr, batch_size, max_iters,
      verbose):

    if ls_type is not None and ls_type not in _LS_LEARNERS:
      raise MVModelException("Invalid LS learner type: %s" % ls_type)

    self.ls_type = ls_type
    self.ls_learner_config = ls_learner_config
    self.given_ls_info = (ls_type is not None and ls_learner_config is not None)

    self.layer_params = layer_params

    self.lr = lr
    self.batch_size = batch_size
    self.max_iters = max_iters

    self.verbose = verbose


# Assumes RNN also returns both state (c_t) and hidden (h_t)
class MVForecaster(nn.Module):
  def __init__ (self, config):
    super(MVForecaster, self).__init__()
    self.config = config

    self.ls_learner = None
    self.has_ls = False
    self.ls_learner_trained = False
    if self.config.given_ls_info:
      self.ls_learner = _LS_LEARNERS[self.config.ls_type](
          self.config.ls_learner_config)
      self.has_ls = True

    self._initialize_layers()
    self._setup_optimizer()

    self.recon_criterion = nn.MSELoss(reduction="elementwise_mean")

  def _initialize_layers(self):
    layer_funcs = self.config.layer_params["layer_funcs"]
    layer_configs = self.config.layer_params["layer_config"]

    if len(layer_configs) > 1:
      all_ops = []
      for lfunc, lconfig in zip(layer_funcs, layer_configs):
        layer = lfunc() if lconfig is None else lfunc(lconfig)
        all_ops.append(layer)
      self._module_op = nn.Sequential(*all_ops)
    else:
      lfunc, lconfig = layer_funcs[0], layer_configs[0]
      self._module_op = lfunc() if lconfig is None else lfunc(lconfig)

  def _setup_optimizer(self):
    self.opt = optim.Adam(self.parameters(), self.config.lr)

  def set_ls_learner(self, ls_learner):
    self.ls_learner = ls_learner
    self.config.ls_learner_config = ls_learner.config
    ls_type = None
    for lname, ltype in _LS_LEARNERS.items():
      if isinstance(ls_learner, ltype):
        ls_type = ltype
        break
    if ls_type is None:
      raise MVModelException("Invalid LS learner type. Expecting one of %s." %
          list(_LS_LEARNERS.keys()))

    self.config.ls_type = ls_type
    self.ls_learner_trained = ls_learner.trained
    self.config.given_ls_info = True
    self.has_ls = True

  def fit_ls_learner(self, dset):
    # Calls fit for the LS learner
    if not self.has_ls:
      raise MVModelException(
          "LS learner has not been set. Call set_ls_learner first.")

    self.ls_learner.fit(dset)
    self.ls_learner_trained = True

  def forward(self, zs, n_steps=0):
    if not isinstance(zs, torch.Tensor):
      zs = torch.from_numpy(np.asarray(txs[vi])).type(_DTYPE)
    
    zs_next, cs_next = self._module_op(zs)
    if n_steps > 0:
      z_last = zs_next[[-1]]
      c_last = cs_next[1][[-1]]

      zs_future = []
      for _ in range(n_steps):
        z_last, (_, c_last) = self._module_op(z_last, (z_last, c_last))
        zs_future.append(z_last)

      # zs_future = torch.cat(zs_future, 0)
      zs_next = torch.cat([zs_next] + zs_future, 0)

    return zs_next # zs_pred, z_pred_mean, z_next

  def loss(self, zs_pred, z_pred_mean, z_next):
    obj = self.recon_criterion(z_pred_mean, z_next)
    for vi in zs_pred:
      obj += self.recon_criterion(zs_pred[vi], z_next)

    # TODO: Any additional loss terms
    return obj

  def _train_loop(self, dset):
    dset.shuffle_data()
    self.itr_loss = 0.
    for tx_batch in dset.get_ts_batches(
        self.config.batch_size, permutation=(1, 0, 2)):
      zs = self.ls_learner.encode(tx_batch)
      z_mean = torch.mean(torch.stack(list(zs.values())), 0)

      z_pred_mean = self.forward(z_mean[:-1])
      zs_pred = {vi: self.forward(zs[vi][:-1]) for vi in zs}
      z_next = z_mean[1:]

      self.opt.zero_grad()
      loss_val = self.loss(zs_pred, z_pred_mean, z_next)
      loss_val.backward()
      self.opt.step()
      self.itr_loss += loss_val

  def fit(self, dset):
    # dset: dataset.MultimodalTimeSeriesDataset
    if not self.ls_learner_trained:
      raise MVModelException(
          "LS learner has not been trained yet. Call ls_ts_learner first.")

    self._n_ts = dset.n_ts
    dset.convert_to_torch()

    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("Epoch %i out of %i." % (itr + 1, self.config.max_iters))
        self._train_loop(dset)

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          print("Loss: %.5f" % float(self.itr_loss.detach()))
          print("Epoch %i took %0.2fs.\n" % (itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
    print("Training finished in %0.2f s." % (time.time() - all_start_time))

    self.trained = True
    print("Training finished in %0.2f s." % (time.time() - all_start_time))

  def predict(self, txs, vi_in, vi_out=None, n_steps=100, rtn_torch=False):
    # txs: List of size n_views with n_ts x n_steps x n_features(view)
    # TODO: Fix this to work with multi-view input.
    if isinstance(vi_in, list):
      raise Exception("Not yet ready for multiview input.")

    if vi_out is None:
      vi_out = np.arange(self._n_views).tolist()

    if not isinstance(txs, torch.Tensor):
      txs = torch.from_numpy(np.asarray(txs)).type(_DTYPE).requires_grad_(False)
    txs = txs.permute(1, 0, 2)

    z = self.ls_learner._encode_view(txs, vi_in)
    z_next = self.forward(z, n_steps)

    # Retranspose dimensions after decoding
    preds = {
        vi: pr.permute(1, 0, 2)
        for vi, pr in self.ls_learner.decode(z_next, vi_out).items()
    }
    if not rtn_torch:
      preds = {
        vi: pr.detach().numpy() for vi, pr in preds.items()
    }

    return preds
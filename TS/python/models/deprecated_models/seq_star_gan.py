"""
Multi-view SeqGAN based off of this paper:
https://arxiv.org/pdf/1804.02812.pdf

Simple form (Stage 1):
Encoder (A) -- encodes different inputs into latent space
Decoder (B) -- decodes latent space + view label into original data
Classifier (C) -- classifies latent space into views

Additional components (Stage 2):
Generator (D) -- generates residual from (B) to get original data
Discriminator (E) -- discriminates between real data and (D)-generated data

For us, we have a separate versions of most components for each view.
"""
import numpy as np
import time

import torch
from torch import autograd
from torch import nn
from torch import optim

import torch_utils as tu
from torch_utils import _DTYPE, _TENSOR_FUNC
import utils

import IPython


_DTYPE = torch.float32
_TENSOR_FUNC = torch.FloatTensor
torch.set_default_dtype(_DTYPE)


_MODULES = ["ae", "cla", "dis", "gen"]


class GANModule(nn.Module):
  def __init__(self, config):
    super(GANModule, self).__init__()
    self.config = config
    self._initialize_layers()

  def _initialize_layers(self):
    layer_funcs = self.config["layer_funcs"]
    layer_configs = self.config["layer_config"]

    all_ops = []
    for lfunc, lconfig in zip(layer_funcs, layer_configs):
      layer = lfunc() if lconfig is None else lfunc(lconfig)
      all_ops.append(layer)
    self._module_op = nn.Sequential(*all_ops)

  def forward(self, tx):
    return self._module_op(tx)


class LatentClassifier(GANModule):
  # Module C as per the paper
  def __init__(self, config):
    super(LatentClassifier, self).__init__(config)

  def forward(self, tx):
    tx_out = self._module_op(tx)
    return torch.softmax(tx_out, dim=-1)


class EncoderDecoder(GANModule):
  # Module A, B or D as per the paper
  def __init__(self, config):
    super(EncoderDecoder, self).__init__(config)


class ViewDiscriminator(GANModule):
  # Partial module E as per the paper (only discriminator)
  def __init__(self, config):
    super(ViewDiscriminator, self).__init__(config)

  def forward(self, tx):
    tx_out = self._module_op(tx)
    return torch.softmax(tx_out, dim=-1)


class SSGTrainingScheduler(object):
  # This class is for properly alternating the training between the different
  # modules. It is meant to provide the order of modules for taking batch
  # gradient steps, as well as, for each "epoch", the number of times a step is
  # taken consecutively for each module in this order.
  # You can provide multiple such step-splits.
  # "Epoch" is in quotes because it doesn't correspond to a data epoch, but
  # rather to one round of training all available modules.
  # Some of this code is silly, I know.
  # TODO: Also provide a way to have variability in order?
  def __init__(
      self, ae_itrs, cla_itrs, gen_itrs, dis_itrs, num_epochs, order, use_cla,
      use_gen_dis):
      # *_itrs: List with the ith element being the number of times that module
      # is gradient-stepped in the ith "epoch".
      # When the epoch exceeds the length of any *itr list, the last value is
      # used thereafter.
      # num_epochs: List with number of "epochs" each set of itrs is run for

    self.ae_itrs = ae_itrs

    self.use_cla = use_cla
    self.cla_itrs = cla_itrs

    self.use_gen_dis = use_gen_dis
    self.gen_itrs = gen_itrs
    self.dis_itrs = gen_itrs

    self.num_epochs = num_epochs

    self.epoch_idx = 0
    self._num_epoch_idx = 0
    self._inner_idx = 0

    self._epoch_order = None
    self._check_and_save_order(order)
    self._initialize_new_epoch()

  def _check_and_save_order(self, order):
    for module in order:
      if module not in _MODULES:
        raise ValueError("%s not a valid module." % module)

    self.order = []
    self._max_len = len(self.ae_itrs)
    for module in order:
      if self.use_cla and module == "cla":
        self._max_len = max(len(self.cla_itrs), self._max_len)
        self.order.append(module)
      if self.use_gen_dis and module in ["gen", "dis"]:
        self._max_len = max(
            len(self.gen_itrs), len(self.dis_itrs), self._max_len)
        self.order.append(module)
      if module == "ae":
        self.order.append(module)

  def _initialize_new_epoch(self):
    self._inner_idx = 0
    if self.epoch_idx >= self._max_len:
      return

    if self._epoch_order is not None:
      curr_num_epoch = (
          self.num_epochs[-1] if self.epoch_idx >= len(self.num_epochs) else
          self.num_epochs[self.epoch_idx])
      if self._num_epoch_idx < curr_num_epoch:
        self._num_epoch_idx += 1
        return

    self._epoch_order = []
    for module in self.order:
      if module == "ae":
        ae_idx = min(self.epoch_idx, len(self.ae_itrs) - 1)
        self._epoch_order.extend(["ae"] * self.ae_itrs[ae_idx])
      elif self.use_cla and module == "cla":
        cla_idx = min(self.epoch_idx, len(self.cla_itrs) - 1)
        self._epoch_order.extend(["cla"] * self.cla_itrs[cla_idx])
      elif self.use_gen_dis:
        if module == "gen":
          gen_idx = min(self.epoch_idx, len(self.gen_itrs) - 1)
          self._epoch_order.extend(["gen"] * self.gen_itrs[gen_idx])
        elif module == "dis":
          dis_idx = min(self.epoch_idx, len(self.dis_itrs) - 1)
          self._epoch_order.extend(["dis"] * self.dis_itrs[dis_idx])

    self._num_epoch_idx = 0
    self.epoch_idx += 1

  def get_next_train_module(self):
    if self._inner_idx >= len(self._epoch_order):
      self._initialize_new_epoch()

    next_module = self._epoch_order[self._inner_idx]
    self._inner_idx += 1
    return next_module


class SSGConfig(object):
  def __init__(
      self, n_views, encoder_params, decoder_params, classifier_params,
      generator_params, discriminator_params, use_cla, ae_dis_alpha,
      use_gen_dis, t_length, t_scheduler_config, enable_cuda, lr, batch_size,
      max_iters, verbose, *args, **kwargs):

    self.n_views = n_views

    self.encoder_params = encoder_params
    self.decoder_params = decoder_params

    self.use_cla = use_cla
    self.classifier_params = classifier_params
    self.ae_dis_alpha = ae_dis_alpha

    self.use_gen_dis = use_gen_dis
    self.generator_params = generator_params
    self.discriminator_params = discriminator_params

    self.t_length = t_length

    t_scheduler_config["use_cla"] = use_cla
    t_scheduler_config["use_gen_dis"] = use_gen_dis
    self.t_scheduler_config = t_scheduler_config

    self.enable_cuda = enable_cuda
    self.lr = lr
    self.batch_size = batch_size
    self.max_iters = max_iters

    self.verbose = verbose


class SeqStarGAN(nn.Module):
  # Auto encoder with multi-views
  def __init__(self, config):
    super(SeqStarGAN, self).__init__()
    self.config = config
    self.t_scheduler = SSGTrainingScheduler(**config.t_scheduler_config)

    self._n_views = config.n_views

    self._initialize_layers()
    self._setup_loss_funcs()
    self._setup_optimizers()

    self.trained = False

  def _enable_cuda(self):
    if not self.config.enable_cuda or not torch.cuda.is_available():
      return
    for module in self.modules():
      module.cuda()

  def _initialize_layers(self):
    # We need to initialize each module.
    # A: Encoder for each view
    # B: Decoder for each view
    # C: Classifier of encoding into views
    # D: Generator of reconstruction residual for each view
    # E: Discriminator of real data for each view

    self._encoders = nn.ModuleDict()  # module A
    self._decoders = nn.ModuleDict()  # module B
    if self.config.use_gen_dis:
      self._generators = nn.ModuleDict()  # module D
      self._discriminators = nn.ModuleDict()  # module E

    # Module C is common across all views
    self._classifier = LatentClassifier(self.config.classifier_params)

    for vi in range(self._n_views):
      self._encoders[vi] = EncoderDecoder(self.config.encoder_params[vi])
      self._decoders[vi] = EncoderDecoder(self.config.decoder_params[vi])

      if self.config.use_gen_dis:
        self._generators[vi] = EncoderDecoder(self.config.generator_params[vi])
        self._discriminators[vi] = ViewDiscriminator(
            self.config.discriminator_params[vi])

  def get_parameters(self, psets="all"):
    if not isinstance(psets, list):
      psets = [psets]

    if "all" in psets:
      params = list(self.parameters())
      return params

    params = []
    if "encoder" in psets:
      for vi in range(self._n_views):
        params = params + list(self._encoders[vi].parameters())
    if "decoder" in psets:
      for vi in range(self._n_views):
        params = params + list(self._decoders[vi].parameters())
    if "classifier" in psets:
      params = params + list(self._classifier.parameters())

    if self.config.use_gen_dis:
      if "generator" in psets:
        for vi in range(self._n_views):
          params = params + list(self._generators[vi].parameters())
      if "discriminator" in psets:
        for vi in range(self._n_views):
          params = params + list(self._discriminators[vi].parameters())
    return params

  def _setup_loss_funcs(self):
    self._mse_loss = nn.MSELoss(reduction="elementwise_mean")
    self._l1_loss = nn.L1Loss(reduction="elementwise_mean")
    self._ce_loss = nn.CrossEntropyLoss(reduction="elementwise_mean")

  def _setup_optimizers(self):
    self._enable_cuda()

    # TODO: betas?
    # TODO: different learning rates?
    # Set up optimizers
    ae_params = self.get_parameters(psets=["encoder", "decoder"])
    self._ae_opt = optim.Adam(ae_params, self.config.lr)
    if self.config.use_cla:
      cla_params = self.get_parameters(psets="classifier")
      self._cla_opt = optim.Adam(cla_params, self.config.lr)
    if self.config.use_gen_dis:
      gen_params = self.get_parameters(psets="generator")
      self._gen_opt = optim.Adam(gen_params, self.config.lr)
      dis_params = self.get_parameters(psets="discriminator")
      self._dis_opt = optim.Adam(dis_params, self.config.lr)

  def encode(self, tx, vi):
    encoder = self._encoders[vi]
    if len(tx.shape) < 3:
      tx = tx.reshape(tx.shape[0], 1, -1)
    # if isinstance(tx, list):
    #   # Improve this by concatenating to encode in one function call
    #   return [encoder(txi) for txi in tx]
    return encoder(tx)

  def decode(self, code, vi):
    decoder = self._decoders[vi]
    # if isinstance(code, list):
    #   # Improve this by concatenating to decode in one function call
    #   return [decoder(codei) for codei in code]
    return decoder(code)

  def _encode_views(self, v_tx):
    return {vi: self.encode(v_tx[vi], vi) for vi in v_tx}

  def _decode_views(self, codes):
    return {vi: self.decode(codes[vi], vi) for vi in codes}

  def classify_latent(self, code):
    return self._classifier(code)

  def generate_residual(self, code, vi):
    if self.config.use_gen_dis:
      generator = self._generators[vi]
      return generator(code)

  def discriminate_view(self, tx, vi):
    if self.config.use_gen_dis:
      discriminator = self._discriminators[vi]
    return discriminator(tx)

  def forward(self, tx, vi, vo):
    code = self.encode(tx, vi)
    recon = self.decode(code, vo)
    if self.config.use_gen_dis:
      recon += self.generate_residual(code, vo)
    return code, recon

  def _cla_loss(self, code, vi):
    # adv currently uses last-step hidden state for t-series
    logits = self.classify_latent(code).reshape(-1, self._n_views)
    # Create target tensor
    targets = torch.ones((logits.shape[0]), dtype=torch.long) * int(vi)
    return self._ce_loss(logits, targets)

  def _ae_loss(self, tx, code, recon, vi):
    # loss_rec += self._mse_loss(recon - tx)
    loss = self._l1_loss(recon, tx)
    if self.config.use_cla:
      loss -= self.config.ae_dis_alpha * self._cla_loss(code, vi)
    return loss

  def _dis_loss(self, true_txs, fake_txs=None):
    loss = 0.
    if true_txs is not None:
      for vo, tx in true_txs.items():
        logits = self.discriminate_view(tx, vo).reshape(-1, 2)
        targets = torch.ones((logits.shape[0]), dtype=torch.long)
        loss += self._ce_loss(logits, targets)
    if fake_txs is not None:
      for vo, tx in fake_txs.items():
        logits = self.discriminate_view(tx, vo).reshape(-1, 2)
        targets = torch.zeros((logits.shape[0]), dtype=torch.long)
        loss += self._ce_loss(logits, targets)
    return loss

  def _gen_loss(self, fake_outputs):
    return -self._dis_loss(None, fake_outputs)

  def _train_ae(self, tx_batch):
    # txs -- batch of time series
    # vis -- corresponding view indices
    # codes = {}
    # recons = {}
    loss = 0.
    n_batch = 0
    for vi, tx in tx_batch.items():
      n_batch += tx.shape[0]
      # tx = torch.from_numpy(tx.astype(np.float32))
      tx = tx.permute(1, 0, 2)
      code = self.encode(tx, vi)
      recon = self.decode(code, vi)
      # code = code.squeeze()
      # recon = recon.squeeze()
      # codes[vi] = code
      # recons[vi] = recon
      loss += self._ae_loss(tx, code, recon, vi)
    # for tx, vi in zip(txs, vis):
    #   tx = torch.from_numpy(tx.astype(np.float32))
    #   tx = tx.reshape(tx.shape[0], 1, -1)
    #   n_batch += 1
    #   code = self.encode(tx, vi)
    #   recon = self.decode(code, vi)
    #   # code = code.squeeze()
    #   # recon = recon.squeeze()
    #   codes.append(code)
    #   recons.append(recon)
    #   loss += self._ae_loss(tx, code, recon, vi)
    # loss /= n_batch

    self._ae_opt.zero_grad()
    loss.backward(retain_graph=True)
    self._ae_opt.step()

    # return codes, recons, loss
    return loss

  def _evaluate_classifier_on_codes(self, codes):

    n_pts = 0
    n_correct = 0
    for vi, code in codes.items():
      preds = torch.argmax(self.classify_latent(code), dim=2)
      n_correct += float(torch.sum(preds == float(vi)))
      n_pts += float(preds.numel())

    return n_correct, n_pts

  def _evaluate_classifier(self, txs):
    codes = self._encode_views(txs)
    return self._evaluate_classifier_on_codes(codes)

  def _train_cla(self, tx_batch):
    loss = 0.
    n_batch = 0
    codes = self._encode_views(tx_batch)
    for vi, code in codes.items():
      n_batch += code.shape[0]
      loss += self._cla_loss(code, vi)
    # loss /= n_batch

    self._cla_opt.zero_grad()
    loss.backward(retain_graph=True)
    self._cla_opt.step()

    return loss

  # def _train_cla(self, codes, vis):
  #   loss = 0.
  #   n_batch = 0
  #   for code, vi in zip(codes, vis):
  #     n_batch += 1
  #     loss += self._cla_loss(code, vi)
  #   loss /= n_batch

  #   self._cla_opt.zero_grad()
  #   loss.backward(retain_graph=self.config.use_gen_dis)
  #   self._cla_opt.step()

  #   return loss

  def _sample_random_views(self, nv, exclude):
    if not isinstance(exclude, list):
      exclude = [exclude]
    allowed_views = [i for i in range(self._n_views) if i not in exclude]
    n_views = len(allowed_views)
    sampled_views = np.random.randint(0, n_views, nv)
    sampled_views = np.array(allowed_views)[sampled_views]
    return sampled_views[0] if nv == 1 else sampled_views

  def _generate_fake_outputs(self, codes, vos):
    code_remap = {}
    for idx in range(vos.shape[0]):
      code = codes[idx]
      vo = vos[idx]
      if vo not in code_remap:
        code_remap[vo] = []
      code_remap[vo].append(code)

    code_remap = {vo: torch.stack(code_remap[vo], dim=0) for vo in code_remap}
    fake_outputs = {
        vo: (self.decode(code, vo) + self.generate_residual(code, vo))
        for vo, code in code_remap.items()
    }
    return fake_outputs

  def _train_gen(self, tx_batch):
    loss = 0.
    n_batch = 0
    codes = self._encode_views(tx_batch)
    fake_outputs = {}
    for vi, code in codes.items():
      n_batch += code.shape[0]
      vos = self._sample_random_views(code.shape[0], exclude=[vi])
      fake_outputs_vi = self._generate_fake_outputs(code, vos)
      for vo in fake_outputs_vi:
        if vo not in fake_outputs:
          fake_outputs[vo] = []
        fake_outputs[vo].append(fake_outputs_vi[vo])

    fake_outputs = {vo: torch.cat(fake_outputs[vo], 0) for vo in fake_outputs}
    # for code, vi in zip(codes, vis):
    #   n_batch += 1
    #   vo = self._sample_random_views(1, exclude=[vi])

    loss += self._gen_loss(fake_outputs)
    # loss /= n_batch

    self._gen_opt.zero_grad()
    loss.backward()
    self._gen_opt.step()

    return loss

  def _evaluate_discriminator(self, txs):
    n_pts = 0
    n_correct = 0
    for vi, tx in txs.items():
      preds = torch.argmax(self.discriminate_view(tx, vi), dim=2)
      n_correct += float(torch.sum(preds))
      n_pts += float(preds.numel())

    return n_correct, n_pts

  def _train_dis(self, tx_batch):
    n_batch = 0
    codes = self._encode_views(tx_batch)
    fake_outputs = {}
    for vi, code in codes.items():
      n_batch += code.shape[0]
      vos = self._sample_random_views(code.shape[0], exclude=[vi])
      fake_outputs_vi = self._generate_fake_outputs(code, vos)
      for vo in fake_outputs_vi:
        if vo not in fake_outputs:
          fake_outputs[vo] = []
        fake_outputs[vo].append(fake_outputs_vi[vo])
    fake_outputs = {vo: torch.cat(fake_outputs[vo], 0) for vo in fake_outputs}
    loss = self._dis_loss(tx_batch, fake_outputs)
    # loss /= n_batch
    self._dis_opt.zero_grad()
    loss.backward()
    self._dis_opt.step()

    return loss

  def _shuffle(self, txs, vis):
    r_inds = np.random.permutation(self._n_ts)
    shuffled_txs = txs[r_inds]
    shuffled_vis = vis[r_inds]
    return shuffled_txs, shuffled_vis

  def _train_loop(self, dset):
    # 1. Train the encoder + decoder to:
    #    (i) reconstruct image well
    #    (ii) produce speaker-independent latent representation
    self.ae_loss, self.ae_itr = 0., 0
    self.cla_loss, self.cla_itr = 0., 0
    self.gen_loss, self.gen_itr = 0., 0
    self.dis_loss, self.dis_itr = 0., 0
    for tx_batch in dset.get_ts_batches(
        self.config.batch_size, permutation=(1, 0, 2)):
      # Rearranging dimensions -- no need to do this now
      # tx_batch = [tx.permute((1, 0, 2)) for tx in tx_batch]

      # Assuming scheduler adheres to use_cla and use_gen_dis.
      next_module = self.t_scheduler.get_next_train_module()
      if next_module == "ae":
        loss_ae = self._train_ae(tx_batch)
        self.ae_loss += loss_ae
        self.ae_itr += 1
      elif next_module == "cla": 
        loss_cla = self._train_cla(tx_batch)
        self.cla_loss += loss_cla
        self.cla_itr += 1
      elif next_module == "gen":
        loss_gen = self._train_gen(tx_batch)
        self.gen_loss += loss_gen
        self.gen_itr += 1
      else:
        loss_dis = self._train_dis(tx_batch)
        self.dis_loss += loss_dis
        self.dis_itr += 1

    n_correct, n_pts = self._evaluate_classifier(dset.v_txs)
    self.cla_acc = n_correct / n_pts
    n_correct, n_pts = self._evaluate_discriminator(dset.v_txs)
    self.dis_acc = n_correct / n_pts

    self.ae_loss, self.cla_loss, self.dis_loss, self.gen_loss = (
        utils.convert_torch_to_float(
            self.ae_loss, self.cla_loss, self.dis_loss, self.gen_loss))
    self.cla_acc, self.dis_acc = utils.convert_torch_to_float(
        self.cla_acc, self.dis_acc)

  def fit(self, dset): 
    # dset: dataset.MultimodalTimeSeriesDataset
    self._n_ts = dset.n_ts
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    dset.convert_to_torch()
    self._n_batches = int(np.ceil(self._n_ts / self.config.batch_size))
    try:
      for self.itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("Epoch %i out of %i." % (self.itr + 1, self.config.max_iters))

          self._train_loop(dset)

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          if self.ae_itr > 0:
            print("AE Loss in %i iters: %.5f" % (self.ae_itr, self.ae_loss))
          if self.config.use_cla and self.cla_itr > 0:
            print("CLA Loss in %i iters: %.5f" % (self.cla_itr, self.cla_loss))
            print("CLA Accuracy: %.5f" % self.cla_acc)
          if self.config.use_gen_dis and self.gen_itr > 0:
            print("GEN Loss in %i iters: %.5f" % (self.gen_itr, self.gen_loss))
          if self.config.use_gen_dis and self.dis_itr > 0:
            print("DIS Loss in %i iters: %.5f" % (self.dis_itr, self.dis_loss))
            print("DIS Accuracy: %.5f" % self.dis_acc)
          print("Epoch %i took %0.2fs.\n" % (self.itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")

    self.trained = True
    print("Training finished in %0.2f s." % (time.time() - all_start_time))

  def predict(self, txs, vi_outs, rtn_torch=False):
    # TODO: Seems best to have simple dictionary txs as input
    # txs -- Dictionary from view to n_ts x n_steps x n_features time-series.
    # vi_outs -- Dictionary from view to list of output views
    # if vi_out is None:
    #   vi_out = np.arange(self._n_views).tolist()

    # TODO: Fix this to work with multi-view input.
    # if isinstance(vi_in, list):
    #   raise Exception("Not yet ready for multiview input.")

    v_check = list(txs.keys())[0]
    if not isinstance(txs[v_check], torch.Tensor):
      txs = {
          vi:torch.from_numpy(np.asarray(txs[vi])).type(_DTYPE)
          for vi in txs
    }
    txs = {vi:txs[vi].permute(1, 0, 2) for vi in txs}


    preds = {}
    for vi, vtx in txs.items():
      vos = vi_outs[vi]
      codes = self.encode(vtx, vi)
      vos_tx = []
      # TODO: maybe do this more efficiently by grouping based on common vo.
      for idx, vo in enumerate(vos):
        code = codes[:, [idx]]  # Add "batch" dimension
        output = (
            self.decode(code, vo) + self.generate_residual(code, vo)
            if self.config.use_gen_dis else self.decode(code, vo)
        )
        vos_tx.append(output)
      preds[vi] = torch.cat(vos_tx, 1).permute(1, 0, 2)

    if not rtn_torch:
      preds = {vi:preds[vi].detach().numpy() for vi in preds}

    return preds
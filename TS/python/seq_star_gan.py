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
import time

import numpy as np
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


class SSGConfig(object):
  def __init__(
      self, n_views, encoder_params, decoder_params, classifier_params,
      generator_params, discriminator_params, use_cla, ae_dis_alpha,
      use_gen_dis, t_length, enable_cuda, lr, batch_size, max_iters, verbose):

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

    self._n_views = config.n_views

    self._initialize_layers()
    self._setup_loss_funcs()
    self._setup_optimizers()

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

    self._encoders = {}  # module A
    self._decoders = {}  # module B
    if self.config.use_gen_dis:
      self._generators = {}  # module D
      self._discriminators = {}  # module E

    # Module C is common across all views
    self._classifier = LatentClassifier(self.config.classifier_params)

    # Need to call "add_module" so the parameters are found.
    def set_value(vdict, key, name, value):
      self.add_module(name, value)
      vdict[key] = value

    for vi in range(self._n_views):
      set_value(self._encoders, vi, "_encoder_%i" % vi,
                EncoderDecoder(self.config.encoder_params[vi]))
      set_value(self._decoders, vi, "_decoder_%i" % vi,
                EncoderDecoder(self.config.decoder_params[vi]))

      if self.config.use_gen_dis:
        set_value(self._generators, vi, "_generator_%i" % vi,
                  EncoderDecoder(self.config.generator_params[vi]))
        set_value(self._discriminators, vi, "discriminator_%i" % vi,
                  ViewDiscriminator(self.config.discriminator_params[vi]))

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

  def _dis_loss(self, tx, vi, org=True):
    logits = self.discriminate_view(tx, vi)
    org = int(org)
    return self._ce_loss(logits, org)

  # TODO
  def _gen_loss(self, code, vi):
    tx_out = self.decode(code, vi) + self.generate_residual(code, vi)
    loss = -self._dis_loss(tx_out, vi, False)

  def _train_ae(self, tx_batch):
    # txs -- batch of time series
    # vis -- corresponding view indices
    codes = {}
    recons = {}
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
      codes[vi] = code
      recons[vi] = recon
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

    return codes, recons, loss

  def _train_cla(self, codes):
    loss = 0.
    n_batch = 0
    for vi, code in codes.items():
      n_batch += code.shape[0]
      loss += self._cla_loss(code, vi)
    # loss /= n_batch

    self._cla_opt.zero_grad()
    loss.backward(retain_graph=self.config.use_gen_dis)
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

  def _train_gen(self, codes):
    loss = 0.
    n_batch = 0
    for vi, code in codes.items():
      n_batch += code.shape[0]
      vo = self._sample_random_views(code.shape[0], exclude=[vi])
    # for code, vi in zip(codes, vis):
    #   n_batch += 1
    #   vo = self._sample_random_views(1, exclude=[vi])
    #   loss += self._gen_loss(code, vo)
    # loss /= n_batch

    self._gen_opt.zero_grad()
    loss.backward()
    self._gen_opt.step()

    return loss

  def _train_dis(self, txs, codes, vis):
    loss = 0.
    n_batch = 0
    for tx, code, vi in zip(txs, codes, vis):
      n_batch += 1
      vo = self._sample_random_views(1, exclude=[vi])
      tx_fake = self.decode(code, vo) + self.generate_residual(code, vo)
      loss += self._dis_loss(tx_fake, vo, False) + self._dis_loss(tx, vi, True)
    loss /= n_batch

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
    self.ae_loss = 0.
    self.cla_loss = 0.
    self.gen_loss = 0.
    self.dis_loss = 0.
    for tx_batch in dset.get_ts_batches(
        self.config.batch_size, permutation=(1, 0, 2)):
      # Rearranging dimensions -- no need to do this now
      # tx_batch = [tx.permute((1, 0, 2)) for tx in tx_batch]
      codes, _, loss_ae = self._train_ae(tx_batch)
      self.ae_loss += loss_ae
      if self.config.use_cla:
        loss_cla = self._train_cla(codes)
        self.cla_loss += loss_cla
      if self.config.use_gen_dis:
        loss_gen = self._train_gen_dis(codes, vis_batch)
        loss_dis = self._train_dis(txs_batch, codes, vis_batch)
        self.gen_loss += loss_gen
        self.dis_loss += loss_dis

    return self.ae_loss, self.cla_loss, self.gen_loss, self.dis_loss

  def fit(self, dset): 
    # dset: dataset.MultimodalAsyncTimeSeriesDataset
    # Assuming this is coming in as non-tensor
    self._n_ts = dset.n_ts
    if self.config.verbose:
      all_start_time = time.time()
      print("Starting training loop.")

    dset.convert_to_torch()
    self._n_batches = int(np.ceil(self._n_ts / self.config.batch_size))
    try:
      for itr in range(self.config.max_iters):
        if self.config.verbose:
          itr_start_time = time.time()
          print("Epoch %i out of %i." % (itr + 1, self.config.max_iters))

          self._train_loop(dset)

        if self.config.verbose:
          itr_duration = time.time() - itr_start_time
          print("AE Loss: %.5f" % float(self.ae_loss.detach()))
          if self.config.use_cla:
            print("CLA Loss: %.5f" % float(self.cla_loss.detach()))
          if self.config.use_gen_dis:
            print("GEN Loss: %.5f" % float(self.gen_loss.detach()))
            print("DIS Loss: %.5f" % float(self.dis_loss.detach()))
          print("Epoch %i took %0.2fs.\n" % (itr + 1, itr_duration))
    except KeyboardInterrupt:
      print("Training interrupted. Quitting now.")
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
        vos_tx.append(self.decode(code, vo))
      preds[vi] = torch.cat(vos_tx, 1).permute(1, 0, 2)

    if not rtn_torch:
      preds = {vi:preds[vi].detach().numpy() for vi in preds}

    return preds
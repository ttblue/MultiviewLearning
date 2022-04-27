# Util function for MNIST + other things.
import numpy as np
import onnx
import onnx_tf
import torch
from torch import nn

from dataprocessing import split_single_view_dsets as ssvd
from utils import torch_utils, utils


torch.set_default_dtype(torch.float64)


def evaluate_mnist_onnx_model(tf_rep, data, fname=None):
  if tf_rep is None:
    print("THIS IS HAPPENING")
    onnx_model = onnx.load(fname)
    tf_rep = onnx_tf.backend.prepare(onnx_model)

  n_pts = data.shape[0]
  logits = []
  for j in range(n_pts):
    print("Evaluating digit %i/%i." % (j+1, n_pts), end="\r")
    jdat = data[j].reshape(1, 1, 28, 28)
    j_logit = tf_rep.run(jdat).Plus214_Output_0
    logits.append(j_logit)

  return np.array(logits).squeeze()


# def convert_truncSVD_inverse_torch(self, svd_models):
#   svd_components = {vi:smdl.components_ for vi, smdl in svd_models.items()}

_mnist_w = _mnist_h = 28
class MNIST8(nn.Module):
  # Class to recreate ONNX model "mnist8" from ONNX zoo.
  def __init__(self, fname, n_views):
    super(MNIST8, self).__init__()
    self._n_views = n_views
    self.initialize_layers()
    self.load_parameter_values(fname)

  def initialize_layers(self):
    self._conv1 = nn.Conv2d(
        in_channels=1, out_channels=8, bias=True, kernel_size=5,
        padding="same")
    self._maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self._conv2 = nn.Conv2d(
        in_channels=8, out_channels=16, bias=True, kernel_size=5,
        padding="same")
    self._maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)

    self._lin = nn.Linear(in_features=256, out_features=10, bias=True)
    self._components_vs = None
    self._pre_op = None

  def load_parameter_values(self, fname):
    # Hardcoded for this particular model.
    onnx_model = onnx.load(fname)
    initializers = onnx_model.graph.initializer
    model_data = {
        i.name: i.float_data
        for i in initializers
    }

    conv1_W = torch.as_tensor(
        model_data["Parameter5"], dtype=torch.float64).view(8, 1, 5, 5)
    conv1_b = torch.as_tensor(
        model_data["Parameter6"], dtype=torch.float64)
    conv2_W = torch.as_tensor(
        model_data["Parameter87"], dtype=torch.float64).view(16, 8, 5, 5)
    conv2_b = torch.as_tensor(
        model_data["Parameter88"], dtype=torch.float64)
    lin_W = torch.as_tensor(
        model_data["Parameter193"], dtype=torch.float64).view(256, 10)
    lin_b = torch.as_tensor(
        model_data["Parameter194"], dtype=torch.float64)

    # with torch.no_grad():
    self._conv1.weight.requires_grad = False
    self._conv1.weight.copy_(conv1_W)
    self._conv1.bias.requires_grad = False
    self._conv1.bias.copy_(conv1_b)

    self._conv2.weight.requires_grad = False
    self._conv2.weight.copy_(conv2_W)
    self._conv2.bias.requires_grad = False
    self._conv2.bias.copy_(conv2_b)

    self._lin.weight.requires_grad = False
    self._lin.weight.copy_(lin_W.transpose(0, 1))
    self._lin.bias.requires_grad = False
    self._lin.bias.copy_(lin_b)

    self._mnist_img_inds = ssvd.get_mnist_split_inds(
        n_views=self._n_views, shape="grid")

    self.cross_ent_loss = nn.CrossEntropyLoss()
    self.eval()

  def save_svd_models(self, svd_models):
    self._components_vs = {
        vi:torch_utils.numpy_to_torch(smdl.components_)
        for vi, smdl in svd_models.items()
    }

  def convert_to_imgs(self, x_vs):
    n_pts = x_vs[utils.get_any_key(x_vs)].shape[0]
    if self._components_vs is not None:
      x_vs = {
          vi: x_vs[vi].matmul(comp_vi)
          for vi, comp_vi in self._components_vs.items()
      }
    imgs = torch.zeros((n_pts, _mnist_h * _mnist_w))
    for vi, x_vi in x_vs.items():
      vi_inds = self._mnist_img_inds[vi]
      imgs[:, vi_inds] = x_vi

    imgs = imgs.view((-1, 1, _mnist_h, _mnist_w))
    return imgs
    # n_pts = base_vs[utils.get_any_key(base_vs)].shape[0]
    # img_vs = {
    #     vi: (base_vi * b_vs[vi] + (1 - b_vs[vi]) * sample_vs[vi])
    #     for vi, base_vi in base_vs.items()
    # }
    # if self._components_vs is not None:
    #   img_vs = {
    #       vi: img_vi[vi].matmul(comp_vi)
    #       for vi, comp_vi in self._components_vs.items()
    #   }
    # imgs = torch.zeros((n_pts, _mnist_h * _mnist_w))
    # for vi, img_vi in imgs_vs.items():
    #   vi_inds = self._mnist_img_inds[vi]
    #   imgs[: vi_inds] = img_vi

    # imgs = imgs.view((-1, 1, _mnist_h, _mnist_w))
    # return imgs

  def get_pre_logits(self, x):

    # if len(x.shape) < 4:
    #   x = torch.unsqueeze(x, 1)
    npts = x.shape[0]

    if self._pre_op:
      x = self._pre_op(x)

    x_conv1 = torch.relu(self._conv1(x))
    x_mp1 = self._maxpool1(x_conv1)

    x_conv2 = torch.relu(self._conv2(x_mp1))
    x_mp2 = self._maxpool2(x_conv2)

    x_output = x_mp2.view(npts, -1)
    x_pre_logit = self._lin(x_output)

    return x_pre_logit

  def forward(self, x_vs, y, *args, **kwargs):
    x = self.convert_to_imgs(x_vs)
    x_pre_logit = self.get_pre_logits(x)
    # x_logit = x_pre_logit.softmax(dim=1)
    loss_val = self.cross_ent_loss(x_pre_logit, y.long())
    return loss_val

  def predict(self, x_vs):
    x_vs = torch_utils.dict_numpy_to_torch(x_vs)
    x = self.convert_to_imgs(x_vs)
    x_pre_logit = self.get_pre_logits(x)
    preds = torch.argmax(x_pre_logit, dim=1)
    return torch_utils.torch_to_numpy(preds)

  def predict_imgs(self, imgs):
    imgs = torch_utils.numpy_to_torch(imgs)
    imgs = imgs.view((-1, 1, _mnist_h, _mnist_w))
    x_pre_logit = self.get_pre_logits(imgs)
    preds = torch.argmax(x_pre_logit, dim=1)
    return torch_utils.torch_to_numpy(preds).astype(int)  


if __name__=="__main__":
  from onnx_tf.backend import prepare
  import os
  import time

  from dataprocessing.split_single_view_dsets import load_original_mnist

  mnist_model_file = os.path.join(
      os.getenv("HOME"), "Research/MultiviewLearning/Code/pretrained_models",
      "mnist/mnist-8.onnx")
  mnist_model = MNIST8(mnist_model_file)
  onnx_model = onnx.load(mnist_model_file)
  tf_rep = onnx_tf.backend.prepare(onnx_model)

  tr_inds_file = os.path.join(
      os.getenv("HOME"), "Research/MultiviewLearning/Code/python/tests",
      "data/mnist/mv/mnist_mv_tr_inds.npy")
  train_set, valid_set, test_set = load_original_mnist()
  tr_inds = np.load(tr_inds_file)
  tr_x, tr_y = train_set[0][tr_inds], train_set[1][tr_inds]
  torch_tr_x = torch.from_numpy(tr_x.astype("float64").reshape(-1, 1, 28, 28))

  npts = 2000
  xn, yn = train_set[0][:npts], train_set[1][:npts]
  torch_xn = torch.from_numpy(xn.astype("float64").reshape(-1, 1, 28, 28))
  t1 = time.time()
  onnx_plogits = evaluate_mnist_onnx_model(tf_rep, xn, mnist_model_file)
  t2 = time.time()
  torch_plogits = mnist_model(torch_xn)
  t3 = time.time()

  print("ONNX time: %.2fs" % (t2 - t1))
  print("Torch time: %.2fs" % (t3 - t2))
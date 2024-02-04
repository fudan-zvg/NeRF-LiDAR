# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch
import yaml

class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes, bn_d=0.1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                           stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
    self.relu1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
    self.relu2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out += residual
    return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params):
    super(Backbone, self).__init__()
    self.use_range = params["input_depth"]["range"]
    self.use_xyz = params["input_depth"]["xyz"]
    self.use_remission = params["input_depth"]["remission"]
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]
    self.OS = params["OS"]
    self.layers = params["extra"]["layers"]
    print("Using DarknetNet" + str(self.layers) + " Backbone")

    # input depth calc
    self.input_depth = 0
    self.input_idxs = []
    if self.use_range:
      self.input_depth += 1
      self.input_idxs.append(0)
    if self.use_xyz:
      self.input_depth += 3
      self.input_idxs.extend([1, 2, 3])
    if self.use_remission:
      self.input_depth += 1
      self.input_idxs.append(4)
    print("Depth of backbone input = ", self.input_depth)

    # stride play
    self.strides = [2, 2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    # make the new stride
    if self.OS > current_os:
      print("Can't do OS, ", self.OS,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == self.OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)

    # check that darknet exists
    assert self.layers in model_blocks.keys()

    # generate layers depending on darknet type
    self.blocks = model_blocks[self.layers]

    # input layer
    self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
    self.relu1 = nn.LeakyReLU(0.1)

    # encoder
    self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                     stride=self.strides[0], bn_d=self.bn_d)
    self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                     stride=self.strides[1], bn_d=self.bn_d)
    self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                     stride=self.strides[2], bn_d=self.bn_d)
    self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                     stride=self.strides[3], bn_d=self.bn_d)
    self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                     stride=self.strides[4], bn_d=self.bn_d)

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 1024

  # make layer useful function
  def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
    layers = []

    #  downsample
    layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                     kernel_size=3,
                                     stride=[1, stride], dilation=1,
                                     padding=1, bias=False)))
    layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
    layers.append(("relu", nn.LeakyReLU(0.1)))

    #  blocks
    inplanes = planes[1]
    for i in range(0, blocks):
      layers.append(("residual_{}".format(i),
                     block(inplanes, planes, bn_d)))

    return nn.Sequential(OrderedDict(layers))

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x,return_features = False):
    # filter input
    # import pdb;pdb.set_trace()
    features = []
    x = x[:, self.input_idxs]
    # features.append(x)

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # first layer
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.relu1, skips, os)
    features.append(x)
    # all encoder blocks with intermediate dropouts
    x, skips, os = self.run_layer(x, self.enc1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    features.append(x)
    x, skips, os = self.run_layer(x, self.enc2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    features.append(x)
    x, skips, os = self.run_layer(x, self.enc3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    features.append(x)
    x, skips, os = self.run_layer(x, self.enc4, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    features.append(x)
    x, skips, os = self.run_layer(x, self.enc5, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    features.append(x)
    if not return_features:
      return x, skips
    else:
      return x, skips, features

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth

class FeatureLoss(nn.Module):
  def __init__(self,device):
    super(FeatureLoss, self).__init__()
    ARCH = yaml.safe_load(open('./model/darknet53-1024px_noremission.yaml', 'r'))
    self.backbone = Backbone(params=ARCH["backbone"]).to(device)

    path = '/SSD_DISK/users/zhangjunge/lidar-bonnetal/train/tasks/logs/2022-11-04-14:37'
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone",
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
    weights = [1/16,1/16,1/8,1/4,1/2,1]
    self.weights = [k/2 for k in weights]
    sensor = ARCH["dataset"]["sensor"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                      dtype=torch.float)[:4].to(device)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)[:4].to(device)
  def forward(self,x,x_points,mask,y,y_points):
    # x =torch.zeros(1,4,32,1024)
    # import pdb;pdb.set_trace()
    x = torch.cat([x.unsqueeze(1),x_points],axis=1)
    x = x* mask.unsqueeze(1)
    y = torch.cat([y.unsqueeze(1),y_points],axis=1)
    x = (x - self.sensor_img_means[None,:, None, None]
        ) / self.sensor_img_stds[None,:, None, None]
    y = (y - self.sensor_img_means[None,:, None, None]
        ) / self.sensor_img_stds[None,:, None, None]

    _,_,x_features = self.backbone(x,return_features=True)
    _,_,y_features = self.backbone(y,return_features=True)
    loss = 0.
    for i in range(len(x_features)):

      loss+= self.weights[i] * ((x_features[i]-y_features[i])**2).mean()
    # print('feature_loss:',loss)
    return loss

if __name__ == "__main__":
  # ARCH = yaml.safe_load(open('/SSD_DISK/users/zhangjunge/lidar-bonnetal/train/tasks/semantic/config/arch/darknet53-1024px_noremission.yaml', 'r'))
  ARCH = yaml.safe_load(open('/SSD_DISK/users/zhangjunge/lidar-bonnetal/train/tasks/semantic/config/arch/darknet53-1024px.yaml', 'r'))
  backbone = Backbone(params=ARCH["backbone"])
  path = '/SSD_DISK/users/zhangjunge/lidar-bonnetal/train/tasks/logs/2022-11-03-20:38'
  if path is not None:
    # try backbone
    try:
      w_dict = torch.load(path + "/backbone",
                          map_location=lambda storage, loc: storage)
      backbone.load_state_dict(w_dict, strict=True)
      print("Successfully loaded model backbone weights")
    except Exception as e:
      print()
      print("Couldn't load backbone, using random weights. Error: ", e)
  x =torch.zeros(1,5,32,1024)
  y,skips,features = backbone(x,return_features=True)
  import pdb;pdb.set_trace()
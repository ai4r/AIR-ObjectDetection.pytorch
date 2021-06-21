# This file is copied and modified from https://github.com/jindongwang/transferlearning/blob/master/code/deep/DAAN/model/backbone.py
import pdb

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F


# convnet without the last layer
class ResNet50Fc(nn.Module):
    def __init__(self, use_all_layers=False):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        
        self.use_all_layers = use_all_layers

    def forward(self, x):
        if self.use_all_layers:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x0 = self.maxpool(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x4 = self.avgpool(x4)
            x4_avgpool = x4.view(x4.size(0), -1)

            x0_avgpool = F.avg_pool2d(x0, kernel_size=56)       # x0: b, c, h, w, 64, 64, 56, 56
            x1_avgpool = F.avg_pool2d(x1, kernel_size=56)
            x2_avgpool = F.avg_pool2d(x2, kernel_size=28)
            x3_avgpool = F.avg_pool2d(x3, kernel_size=14)

            x0_avgpool = x0_avgpool.view(x0_avgpool.size(0), -1)
            x1_avgpool = x1_avgpool.view(x1_avgpool.size(0), -1)
            x2_avgpool = x2_avgpool.view(x2_avgpool.size(0), -1)
            x3_avgpool = x3_avgpool.view(x3_avgpool.size(0), -1)

            x = torch.cat((x0_avgpool, x1_avgpool, x2_avgpool, x3_avgpool, x4_avgpool), dim=1)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        return x

class ResNet101Fc(nn.Module):
    def __init__(self):
        super(ResNet101Fc, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResNet152Fc(nn.Module):
    def __init__(self):
        super(ResNet152Fc, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


network_dict = {"ResNet50": ResNet50Fc,
                "ResNet101": ResNet101Fc,
                "ResNet152": ResNet152Fc}

# -*- coding: utf-8 -*-
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.loss import _Loss

import models.classifiers.evolve as evolve


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        res = self.fc2(x)
        return [x, res]


class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return [feature, res]

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return out


class VGG16_vib(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_vib, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.k = self.feat_dim // 2
        self.n_classes = n_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)

    def forward(self, x, mode="train"):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)

        return [feature, out, mu, std]

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)

        return out


class CrossEntropyLoss(_Loss):
    def forward(self, out, gt, mode="reg"):
        bs = out.size(0)
        loss = - torch.mul(gt.float(), torch.log(out.float() + 1e-7))
        if mode == "dp":
            loss = torch.sum(loss, dim=1).view(-1)
        else:
            loss = torch.sum(loss) / bs
        return loss


class BinaryLoss(_Loss):
    def forward(self, out, gt):
        bs = out.size(0)
        loss = - (gt * torch.log(out.float() + 1e-7) + (1 - gt) * torch.log(1 - out.float() + 1e-7))
        loss = torch.mean(loss)
        return loss


class FaceNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet, self).__init__()
        self.feature = evolve.IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def predict(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out

    def forward(self, x):
        # print("input shape:", x.shape)
        # import pdb; pdb.set_trace()

        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return [feat, out]


class FaceNet64(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet64, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out


class IR152(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat, out


class IR152_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152_vib, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.k = self.feat_dim // 2
        self.n_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * 4 * 4, 512),
                                          nn.BatchNorm1d(512))

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]

        std = F.softplus(std - 5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return [feature, mu, std, out]

class IR18(nn.Module):
    def __init__(self, num_classes=5):
        super(IR18, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.feature = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,

            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,

            model.avgpool,
            Flatten()
        )
        self.fc_layer = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        feat = self.feature(x)
        out = self.fc_layer(feat)
        return feat, out

############################# BiDO #############################

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    blocks = []
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            blocks.append(nn.Sequential(*layers))
            layers = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return blocks

class VGG16_BiDO(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_BiDO, self).__init__()

        blocks = make_layers(cfgs['D'], batch_norm=True)
        self.layer1 = blocks[0]
        self.layer2 = blocks[1]
        self.layer3 = blocks[2]
        self.layer4 = blocks[3]
        self.layer5 = blocks[4]

        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x):
        hiddens = []

        out = self.layer1(x)
        hiddens.append(out)

        out = self.layer2(out)
        hiddens.append(out)

        out = self.layer3(out)
        hiddens.append(out)

        out = self.layer4(out)
        hiddens.append(out)

        feature = self.layer5(out)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)

        hiddens.append(feature)

        res = self.fc_layer(feature)

        return hiddens, res

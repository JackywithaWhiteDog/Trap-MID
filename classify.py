import torch.nn as nn
import torchvision

import evolve

############################# Normal #############################

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

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
        return feature, res


class FaceNet64(nn.Module):
    def __init__(self, num_classes = 1000):
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

class IR18_BiDO(nn.Module):
    def __init__(self, num_classes=5):
        super(IR18_BiDO, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.input_layer = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.output_layer = nn.Sequential(
            model.avgpool,
            Flatten()
        )
        self.fc_layer = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        hiddens = []
        feat = self.input_layer(x)
        feat = self.layer1(feat)
        hiddens.append(feat)
        feat = self.layer2(feat)
        hiddens.append(feat)
        feat = self.layer3(feat)
        hiddens.append(feat)
        feat = self.layer4(feat)
        hiddens.append(feat)
        feat = self.output_layer(feat)
        out = self.fc_layer(feat)
        return hiddens, out

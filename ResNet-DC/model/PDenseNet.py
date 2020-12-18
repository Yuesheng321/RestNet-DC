import torch.nn as nn
import torch
from torchvision import models
from tool.config import opt
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print(m)


class PDenseNet(nn.Module):
    def __init__(self, feature, inplanes):
        super(PDenseNet, self).__init__()
        self.feature = feature
        self.unsample = nn.Sequential(nn.Conv2d(inplanes, 128, 3, 1, 1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),

                                      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                                                         output_padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),

                                      nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1,
                                                         output_padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),

                                      nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1,
                                                         output_padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True),
                                      )
        self.peak = nn.Sequential(
                                  # nn.Conv2d(16, 16, 3, 1, 1),
                                  # nn.BatchNorm2d(16),
                                  # nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  )
        # self.mask = nn.Sequential(nn.Conv2d(16, 2, 1, 1, 0))     # 前景和后景
        initialize_weights(self.unsample)
        initialize_weights(self.peak)
        # initialize_weights(self.mask)

    def forward(self, x):
        x = self.feature(x)
        x = self.unsample(x)
        peak = self.peak(x)
        # mask = self.mask(x)
        return peak

def make_res_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


def freeze(feature, layer_name='6.0.conv1.weight'):
    for name, param in feature.named_parameters():
        if name == layer_name:
            break
        param.requires_grad = False


def PDenseNet18():
    res = models.resnet18(pretrained=False)
    pre_wts = torch.load(opt.resnet18)
    res.load_state_dict(pre_wts)
    layer_3 = make_res_layer(BasicBlock, 128, 256, 2, stride=1)
    layer_3.load_state_dict(res.layer3.state_dict())
    feature = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, layer_3)
    # freeze some feature
    # freeze(feature)
    model = PDenseNet(feature, inplanes=256)
    return model


def PDenseNet34():
    res = models.resnet34(pretrained=False)
    pre_wts = torch.load(opt.resnet34)
    res.load_state_dict(pre_wts)
    layer_3 = make_res_layer(BasicBlock, 128, 256, 6, stride=1)
    layer_3.load_state_dict(res.layer3.state_dict())
    feature = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, layer_3)
    # freeze some feature
    # freeze(feature)
    model = PDenseNet(feature, inplanes=256)
    return model


def PDenseNet50():
    res = models.resnet50(pretrained=False)
    pre_wts = torch.load(opt.resnet50)
    res.load_state_dict(pre_wts)
    layer_3 = make_res_layer(Bottleneck, 512, 256, 6, stride=1)
    layer_3.load_state_dict(res.layer3.state_dict())
    feature = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, layer_3)
    # freeze some feature
    # freeze(feature)
    model = PDenseNet(feature, inplanes=1024)
    return model


def PDenseNet101():
    res = models.resnet101(pretrained=False)
    pre_wts = torch.load(opt.resnet101)
    res.load_state_dict(pre_wts)
    layer_3 = make_res_layer(Bottleneck, 512, 256, 23, stride=1)
    layer_3.load_state_dict(res.layer3.state_dict())
    feature = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, layer_3)
    # freeze some feature
    # freeze(feature)
    model = PDenseNet(feature, inplanes=1024)
    return model


def vgg16():
    # Load the caffe model
    model = models.vgg16(pretrained=False)
    model.load_state_dict(torch.load(opt.vgg16))
    features = list(model.features)[:23]
    return nn.Sequential(*features)


def PVGG():
    feature = vgg16()
    model = PDenseNet(feature, inplanes=512)
    return model


if __name__ == '__main__':
    x = torch.rand((1, 3, 640, 360))
    # print(x.size())
    model = PVGG()
    print(model(x).size())
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         print(m.weight.data)
    #         if m.bias is not None:
    #             print(m.bias.data)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         print(m.weight.data)
    #         print(m.bias.data)
    # x = torch.randn(1, 3, 640, 360)
    # print(model(x).size())

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torchvision.models.resnet import Bottleneck


class RPN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_anchors):
        super().__init__()
        self.standardise = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.classifier = nn.Conv2d(out_channels, num_anchors, 1)
        self.regressor = nn.Conv2d(out_channels, num_anchors * 4, 1)

    def forward(self, x):
        x = self.standardise(x)
        foreground = self.classifier(x)
        boxes = self.regressor(x)
        return foreground, boxes


class ROIPooling(nn.Module):
    def __init__(self, output_shape, pooling=functional.max_pool2d):
        super().__init__()
        self.pooling = pooling
        self.output_shape = output_shape

    def forward(self, x):
        shape = np.array(x.shape[2:])
        scale = shape / self.output_shape
        stride = tuple(map(int, np.floor(scale)))
        kernel_size = tuple(map(int, np.ceil(scale)))

        x = self.pooling(x, kernel_size=kernel_size, stride=stride)
        return x


# params = Bottleneck, [3, 4, 6, 3]


class ResNetC4(nn.Module):
    def __init__(self, block, layers, strides):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

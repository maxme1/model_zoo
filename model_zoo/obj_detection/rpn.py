import numpy as np
import torch.nn as nn
from torch.nn import functional


class RPN(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size, num_anchors):
        super().__init__()
        self.standardise = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=kernel_size // 2)
        self.classifier = nn.Conv2d(mid_channels, num_anchors, 1)
        self.regressor = nn.Conv2d(mid_channels, num_anchors * 4, 1)

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


class ResNetC4(nn.Module):
    def __init__(self, model):
        super().__init__()
        del model.avgpool, model.fc
        self.model = model

    def forward(self, x):
        model = self.model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x

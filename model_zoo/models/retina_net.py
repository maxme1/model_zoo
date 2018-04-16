from torch import nn
from torch.nn import functional
from torchvision.models import resnet


class Backbone(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        del base_model.avgpool, base_model.fc
        self.model = base_model

    def forward(self, x):
        model = self.model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)

        x = model.layer2(x)
        y = model.layer3(x)
        z = model.layer4(y)
        return x, y, z


def subnet(in_channels, out_channels):
    mid_channels = in_channels
    return nn.Sequential(
#         nn.Conv2d(in_channels, mid_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 1),
    )


class RetinaNet(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        inp_chans = [
            4 * 512, 4 * 256, 4 * 128
        ]
        mid_chans = 256
        self.normalize = nn.ModuleList([nn.Conv2d(channel, mid_chans, 1) for channel in inp_chans])

        self.backbone = Backbone(resnet.resnet50(pretrained=True))
        self.classification = subnet(mid_chans, num_anchors * num_classes)
        self.regression = subnet(mid_chans, 4 * num_anchors)

    def head(self, x):
        return self.classification(x), self.regression(x)

    def forward(self, x):
        pyramid = self.backbone(x)
        pyramid = [layer(level) for layer, level in zip(self.normalize, reversed(pyramid))]
        current = pyramid[0]
        result = [current]
        for level in pyramid[1:]:
            current = functional.upsample(current, size=level.shape[2:], mode='bilinear') + level
            result.append(current)
        return [self.head(x) for x in result]

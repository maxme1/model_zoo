import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    apply_conv = nn.Conv2d
    apply_pool = nn.MaxPool2d
    apply_bn = nn.BatchNorm2d

    def __init__(self, nchannels, kernel_size=3):
        super().__init__()

        self.convolution = self.apply_conv(nchannels, 16 - nchannels, kernel_size,
                                           stride=2, padding=1, bias=True)
        self.max_pool = self.apply_pool(2, stride=2)
        self.batch_norm = self.apply_bn(16, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.convolution(input), self.max_pool(input)], 1)
        output = self.batch_norm(output)
        return F.relu(output)


def get_conv_block(conv_layer, bn_layer, input_channels, output_channels, kernel_size, **kwargs):
    return nn.Sequential(
        conv_layer(input_channels, output_channels, kernel_size, **kwargs, bias=False),
        bn_layer(output_channels, eps=1e-03),
        nn.PReLU(),
    )


class Bottleneck(nn.Module):
    apply_conv = nn.Conv2d
    apply_conv_transpose = nn.ConvTranspose2d
    apply_pool = nn.MaxPool2d
    apply_bn = nn.BatchNorm2d
    apply_dropout = nn.Dropout2d

    def __init__(self, input_channels, output_channels, kernel_size=3, downsample=False, upsample=False,
                 dropout_prob=.1, internal_scale=4):
        # it can be either upsampling or downsampling:
        assert not (upsample and downsample)
        super().__init__()

        internal_channels = output_channels // internal_scale
        input_stride = 2 if downsample else 1

        self.downsample = downsample
        self.upsample = upsample
        self.output_channels = output_channels
        self.input_channels = input_channels

        enter = get_conv_block(
            self.apply_conv, self.apply_bn, input_channels, internal_channels, input_stride, stride=input_stride
        )

        if upsample:
            middle = get_conv_block(self.apply_conv_transpose, self.apply_bn, internal_channels, internal_channels,
                                    kernel_size, stride=2, padding=1, output_padding=1)
        else:
            # TODO: use dilated and asymmetric convolutions
            middle = get_conv_block(self.apply_conv, self.apply_bn, internal_channels, internal_channels,
                                    kernel_size, padding=1)

        end = get_conv_block(self.apply_conv, self.apply_bn, internal_channels, output_channels, 1)

        self.conv_block = nn.Sequential(
            enter, middle, end,
            self.apply_dropout(dropout_prob)
        )

        # main path
        if downsample:
            self.max_pool = self.apply_pool(2, stride=2)
        if upsample:
            # TODO: implement unpooling
            self.unpool = self.apply_conv_transpose(
                output_channels, output_channels, kernel_size, stride=2, padding=1, output_padding=1, bias=True
            )

        if output_channels != input_channels:
            self.adjust = nn.Sequential(
                self.apply_conv(input_channels, output_channels, 1,
                                stride=1, padding=0, bias=False),
                self.apply_bn(output_channels, eps=1e-03),
            )

    def forward(self, input):
        conv_path = self.conv_block(input)
        main_path = input

        if self.downsample:
            main_path = self.max_pool(main_path)
        if self.output_channels != self.input_channels:
            main_path = self.adjust(main_path)
        if self.upsample:
            main_path = self.unpool(main_path)

        return F.relu(conv_path + main_path)


class Stage(nn.Module):
    res_block = Bottleneck

    def __init__(self, input_channels, output_channels, num_blocks, downsample=False, upsample=False):
        super().__init__()

        blocks = [self.res_block(input_channels, output_channels, downsample=downsample, upsample=upsample)]
        blocks.extend([self.res_block(output_channels, output_channels) for _ in range(num_blocks - 1)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class ENet2D(nn.Module):
    conv_transpose = nn.ConvTranspose2d
    stage = Stage
    initial = InitialBlock

    def __init__(self, nclasses, nchannels):
        super().__init__()

        self.layers = nn.Sequential(
            self.initial(nchannels),
            self.stage(16, 64, 5, downsample=True),
            self.stage(64, 128, 9, downsample=True),
            self.stage(128, 128, 8),
            self.stage(128, 64, 3, upsample=True),
            self.stage(64, 16, 2, upsample=False),
            self.conv_transpose(16, nclasses, 2, stride=2, padding=0, output_padding=0, bias=True),
        )

    def forward(self, input):
        return self.layers(input)


class InitialBlock3D(InitialBlock):
    apply_conv = nn.Conv3d
    apply_pool = nn.MaxPool3d
    apply_bn = nn.BatchNorm3d


class Bottleneck3D(Bottleneck):
    apply_conv = nn.Conv3d
    apply_conv_transpose = nn.ConvTranspose3d
    apply_pool = nn.MaxPool3d
    apply_bn = nn.BatchNorm3d
    apply_dropout = nn.Dropout3d


class Stage3D(Stage):
    res_block = Bottleneck3D


class ENet3D(ENet2D):
    conv_transpose = nn.ConvTranspose3d
    stage = Stage3D
    initial = InitialBlock3D

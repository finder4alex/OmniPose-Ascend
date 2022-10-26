# ------------------------------------------------------------------------------
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mindspore import nn
from mindspore import ops
# from mindspore.nn import BatchNorm2d
# from mindspore.nn import SyncBatchNorm as BatchNorm2d
from mindspore.ops import AdaptiveAvgPool2D

from src.config import config
if config["TRAIN"]["DEVICE_TARGET"] == "Ascend" and config["TRAIN"]["DEVICE_NUM"] > 1:
    from mindspore.nn import SyncBatchNorm as BatchNorm2d
else:
    from mindspore.nn import BatchNorm2d

# GPU version
# class AdaptiveAvgPool2d(nn.Cell):
#     """AdaptiveAvgPool2d"""
#
#     def __init__(self, output_size):
#         super(AdaptiveAvgPool2d, self).__init__()
#         self.adaptive_avg_pool2d = AdaptiveAvgPool2D(output_size=output_size)
#         nn.AvgPool2d
#
#     def construct(self, x):
#         x = self.adaptive_avg_pool2d(x)
#         return x


class AdaptiveAvgPool2d(nn.Cell):
    """AdaptiveAvgPool2d"""
    def __init__(self):
        super(AdaptiveAvgPool2d, self).__init__()
        self.mean = ops.ReduceMean(True)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class SepConv2d(nn.Cell):
    """ SepConv2d """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=0,
                 dilation=1, has_bias=True, depth_multiplier=1):
        super(SepConv2d, self).__init__()
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size, stride,
            pad_mode=pad_mode, padding=padding,
            dilation=dilation, group=in_channels, has_bias=has_bias)
        self.pointConv = nn.Conv2d(
            intermediate_channels, out_channels, kernel_size=1, stride=1,
            pad_mode=pad_mode, padding=0, dilation=1, has_bias=has_bias)

        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.spatialConv(x)
        x = self.relu(x)
        x = self.pointConv(x)

        return x


conv_dict = {
    "CONV2D": nn.Conv2d,
    "SEPARABLE": SepConv2d
}


class _AtrousModule(nn.Cell):
    """_AtrousModule"""

    def __init__(self, conv_type, inplanes, planes, kernel_size, padding, dilation, batch_norm):
        super(_AtrousModule, self).__init__()
        self.conv = conv_dict[conv_type]
        self.atrous_conv = self.conv(
            inplanes, planes, kernel_size=kernel_size, stride=1,
            pad_mode="pad", padding=padding, dilation=dilation, has_bias=False)

        self.bn = batch_norm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def _init_weight(self):
        pass

    def construct(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class WASP(nn.Cell):
    """WASP"""

    def __init__(self, inplanes, planes, upDilations=[4,8]):
        super(WASP, self).__init__()
        dilations = [6, 12, 18, 24]
        batch_norm = BatchNorm2d

        self.aspp1 = _AtrousModule(
            'CONV2D', inplanes, planes, 1, padding=0, dilation=dilations[0], batch_norm=batch_norm)
        self.aspp2 = _AtrousModule(
            'CONV2D', planes, planes, 3, padding=dilations[1], dilation=dilations[1], batch_norm=batch_norm)
        self.aspp3 = _AtrousModule(
            'CONV2D', planes, planes, 3, padding=dilations[2], dilation=dilations[2], batch_norm=batch_norm)
        self.aspp4 = _AtrousModule(
            'CONV2D', planes, planes, 3, padding=dilations[3], dilation=dilations[3], batch_norm=batch_norm)

        self.global_avg_pool = nn.SequentialCell(
            [
                # AdaptiveAvgPool2d((1, 1)),
                AdaptiveAvgPool2d(),
                nn.Conv2d(inplanes, planes, 1, stride=1, pad_mode="valid", has_bias=False),
                batch_norm(planes),
                nn.ReLU()
            ])

        self.conv1 = nn.Conv2d(5*planes, planes, 1, pad_mode="valid", has_bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 1, pad_mode="valid", has_bias=False)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=1-0.5)
        self.cat = ops.Concat(axis=1)
        self._init_weight()

    def _init_weight(self):
        pass

    def construct(self, x):
        N, C, H, W = x.shape
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        # F.interpolate ResizeBilinear
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x5 = ops.ResizeBilinear(size=(H, W), align_corners=True)(x5)

        x = self.cat([x1, x2, x3, x4, x5])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


def build_wasp(inplanes, planes, upDilations):
    return WASP(inplanes, planes, upDilations)


class WASPv2(nn.Cell):
    """WASPv2"""

    def __init__(self, conv_type, inplanes, planes, n_classes=17):
        super(WASPv2, self).__init__()
        # WASP
        dilations = [1, 6, 12, 18]
        # dilations = [1, 12, 24, 36]

        # convs = conv_dict[conv_type]

        reduction = planes // 8

        batch_norm = BatchNorm2d

        self.aspp1 = _AtrousModule(
            conv_type, inplanes, planes, 1, padding=0, dilation=dilations[0], batch_norm=batch_norm)
        self.aspp2 = _AtrousModule(
            conv_type, planes, planes, 3, padding=dilations[1], dilation=dilations[1], batch_norm=batch_norm)
        self.aspp3 = _AtrousModule(
            conv_type, planes, planes, 3, padding=dilations[2], dilation=dilations[2], batch_norm=batch_norm)
        self.aspp4 = _AtrousModule(
            conv_type, planes, planes, 3, padding=dilations[3], dilation=dilations[3], batch_norm=batch_norm)

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.SequentialCell(
            [
                # AdaptiveAvgPool2d((1, 1)),
                AdaptiveAvgPool2d(),
                nn.Conv2d(planes, planes, 1, stride=1, pad_mode="valid", has_bias=False),
                batch_norm(planes),
                nn.ReLU()
            ])

        self.conv1 = nn.Conv2d(5*planes, planes, 1, pad_mode="valid", has_bias=False)
        self.bn1 = BatchNorm2d(planes)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, reduction, 1, pad_mode="valid", has_bias=False)
        self.bn2 = batch_norm(reduction)

        self.last_conv = nn.SequentialCell(
            [nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=False),
             batch_norm(planes),
             nn.ReLU(),
             nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=False),
             batch_norm(planes),
             nn.ReLU(),
             nn.Conv2d(planes, n_classes, kernel_size=1, stride=1, pad_mode="valid", has_bias=True)])

        self.cat = ops.Concat(axis=1)

    def construct(self, x, low_level_features):
        N, C, H, W = x.shape
        # input = x
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)
        x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x5 = ops.ResizeBilinear(size=(H, W), align_corners=True)(x5)

        x = self.cat([x1, x2, x3, x4, x5])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = self.cat([x, low_level_features])
        x = self.last_conv(x)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

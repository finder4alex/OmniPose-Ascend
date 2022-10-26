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

import math
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common import Parameter, Tensor
from mindspore.common import initializer as weight_init

from src.config import config
from src.models.wasp import build_wasp
from src.models.wasp import WASPv2

if config["TRAIN"]["DEVICE_TARGET"] == "Ascend" and config["TRAIN"]["DEVICE_NUM"] > 1:
    from mindspore.nn import SyncBatchNorm as BatchNorm2d
else:
    from mindspore.nn import BatchNorm2d

BN_MOMENTUM = 0.1
convs = nn.Conv2d


class NoneCell(nn.Cell):
    """NoneCell"""

    def __init__(self):
        super(NoneCell, self).__init__()

    def construct(self, x):
        return x


class ExpandDimsCell(nn.Cell):
    """ExpandDimsCell"""

    def __init__(self, axis=0):
        super(ExpandDimsCell, self).__init__()
        self.axis = axis
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        return self.expand_dims(x, self.axis)


class SqueezeCell(nn.Cell):
    """SqueezeCell"""

    def __init__(self, axis=0):
        super(SqueezeCell, self).__init__()
        self.squeeze = ops.Squeeze(axis)

    def construct(self, x):
        return self.squeeze(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_op = nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, pad_mode="pad", padding=1, has_bias=False)

    return conv_op


class Decoder(nn.Cell):
    """Decoder"""

    def __init__(self, low_level_inplanes, planes, num_classes):
        super(Decoder, self).__init__()
        reduction = planes // 8

        self.conv1 = nn.Conv2d(low_level_inplanes, reduction, 1, pad_mode="valid", has_bias=False)
        self.bn1 = BatchNorm2d(reduction)
        self.relu = nn.ReLU()

        self.last_conv = nn.SequentialCell(
            [nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1,
                       pad_mode="pad", padding=1, has_bias=False),
             BatchNorm2d(planes),
             nn.ReLU(),
             nn.Dropout(keep_prob=1.0-0.5),
             nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=False),
             BatchNorm2d(planes),
             nn.ReLU(),
             nn.Dropout(keep_prob=1.0-0.1),
             nn.Conv2d(planes, num_classes, kernel_size=1, stride=1, padding="valid", has_bias=True)])

        self.cat = ops.Concat(axis=1)

    def _init_weight(self):
        pass

    def construct(self, x, low_level_feat):
        N, C, H, W = low_level_feat.shape
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = ops.ResizeBilinear(size=(H, W), align_corners=True)(x)
        x = self.cat([x, low_level_feat])
        x = self.last_conv(x)

        return x


class BasicBlock(nn.Cell):
    """BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=1.0 - BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=1.0 - BN_MOMENTUM)

        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
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


class Bottleneck(nn.Cell):
    """Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = convs(inplanes, planes, kernel_size=1, pad_mode="valid", has_bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=1.0 - BN_MOMENTUM)
        self.conv2 = convs(planes, planes, kernel_size=3, stride=stride, pad_mode="pad", padding=1, has_bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=1.0 - BN_MOMENTUM)
        self.conv3 = convs(planes, planes * self.expansion, kernel_size=1, pad_mode="valid", has_bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=1.0 - BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
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


class GaussianFilter(nn.Cell):
    """GaussianFilter"""

    def __init__(self, channels, kernel_size, sigma):
        super(GaussianFilter, self).__init__()
        x_cord = np.arange(kernel_size)
        x_grid = np.tile(x_cord, kernel_size).reshape(kernel_size, kernel_size)
        y_grid = x_grid.transpose()
        xy_grid = np.stack([x_grid, y_grid], axis=-1).astype(np.float64)
        mean = (kernel_size - 1) / 2

        gaussian_kernel = (1. / (2.*math.pi*sigma**2)) * np.exp(-np.sum((xy_grid - mean)**2., axis=-1) / (2*sigma**2))
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_size, kernel_size)
        gaussian_kernel = np.tile(gaussian_kernel, (channels, 1, 1, 1))

        self.weight = Parameter(Tensor(gaussian_kernel, dtype=mstype.float32), requires_grad=False)
        self.conv = ops.Conv2D(
            out_channel=channels, kernel_size=kernel_size,
            pad_mode="pad", pad=int(kernel_size//2), group=channels)

    def construct(self, x):
        return self.conv(x, self.weight)


class HighResolutionModule(nn.Cell):
    """HighResolutionModule"""

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialCell(
                [convs(
                    self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, pad_mode="valid", has_bias=False),
                    BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=1.0 - BN_MOMENTUM)])

        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]

        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.SequentialCell(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.CellList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i and (j - i) == 1:
                    fuse_layer.append(
                        nn.SequentialCell([
                            convs(num_inchannels[j], num_inchannels[i], 1, 1,
                                  pad_mode="pad", padding=0, has_bias=False),
                            BatchNorm2d(num_inchannels[i]),
                            # nn.Conv2dTranspose(in_channels=num_inchannels[i],
                            #                    out_channels=num_inchannels[i],
                            #                    kernel_size=3, stride=2,
                            #                    pad_mode="pad", padding=1, has_bias=False),
                            ExpandDimsCell(axis=2),
                            nn.Conv3dTranspose(in_channels=num_inchannels[i],
                                               out_channels=num_inchannels[i],
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               pad_mode="pad",
                                               padding=(0, 0, 1, 1, 1, 1),
                                               output_padding=(0, 1, 1),
                                               has_bias=False),
                            SqueezeCell(axis=2),
                            BatchNorm2d(num_inchannels[i], momentum=1.0-0.1),
                            nn.ReLU(),
                            # self.gaussian_filter(num_inchannels[i], 3, 3)]))
                            GaussianFilter(num_inchannels[i], 3, 3)]))
                elif j > i and (j - i) == 2:
                    fuse_layer.append(
                        nn.SequentialCell([
                            convs(num_inchannels[j], num_inchannels[i], 1, 1,
                                  pad_mode="pad", padding=0, has_bias=False),
                            BatchNorm2d(num_inchannels[i]),
                            # nn.Conv2dTranspose(in_channels=num_inchannels[i],
                            #                    out_channels=num_inchannels[i],
                            #                    kernel_size=3, stride=2,
                            #                    pad_mode="pad", padding=1, has_bias=False),
                            ExpandDimsCell(axis=2),
                            nn.Conv3dTranspose(in_channels=num_inchannels[i],
                                               out_channels=num_inchannels[i],
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               pad_mode="pad",
                                               padding=(0, 0, 1, 1, 1, 1),
                                               output_padding=(0, 1, 1),
                                               has_bias=False),
                            SqueezeCell(axis=2),
                            BatchNorm2d(num_inchannels[i], momentum=1.0 - 0.1),
                            nn.ReLU(),
                            # nn.Conv2dTranspose(in_channels=num_inchannels[i],
                            #                    out_channels=num_inchannels[i],
                            #                    kernel_size=3, stride=2,
                            #                    pad_mode="pad", padding=1, has_bias=False),
                            ExpandDimsCell(axis=2),
                            nn.Conv3dTranspose(in_channels=num_inchannels[i],
                                               out_channels=num_inchannels[i],
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               pad_mode="pad",
                                               padding=(0, 0, 1, 1, 1, 1),
                                               output_padding=(0, 1, 1),
                                               has_bias=False),
                            SqueezeCell(axis=2),
                            BatchNorm2d(num_inchannels[i], momentum=1.0 - 0.1),
                            nn.ReLU(),
                            # self.gaussian_filter(num_inchannels[i], 3, 3)]))
                            GaussianFilter(num_inchannels[i], 3, 3)]))
                elif j > i and (j - i) == 3:
                    fuse_layer.append(
                        nn.SequentialCell([
                            convs(num_inchannels[j], num_inchannels[i], 1, 1,
                                  pad_mode="pad", padding=0, has_bias=False),
                            BatchNorm2d(num_inchannels[i]),
                            # nn.Conv2dTranspose(in_channels=num_inchannels[i],
                            #                    out_channels=num_inchannels[i],
                            #                    kernel_size=3, stride=2,
                            #                    pad_mode="pad", padding=1, has_bias=False),
                            ExpandDimsCell(axis=2),
                            nn.Conv3dTranspose(in_channels=num_inchannels[i],
                                               out_channels=num_inchannels[i],
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               pad_mode="pad",
                                               padding=(0, 0, 1, 1, 1, 1),
                                               output_padding=(0, 1, 1),
                                               has_bias=False),
                            SqueezeCell(axis=2),
                            BatchNorm2d(num_inchannels[i], momentum=1.0 - 0.1),
                            nn.ReLU(),
                            # nn.Conv2dTranspose(in_channels=num_inchannels[i],
                            #                    out_channels=num_inchannels[i],
                            #                    kernel_size=3, stride=2,
                            #                    pad_mode="pad", padding=1, has_bias=False),
                            ExpandDimsCell(axis=2),
                            nn.Conv3dTranspose(in_channels=num_inchannels[i],
                                               out_channels=num_inchannels[i],
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               pad_mode="pad",
                                               padding=(0, 0, 1, 1, 1, 1),
                                               output_padding=(0, 1, 1),
                                               has_bias=False),
                            SqueezeCell(axis=2),
                            BatchNorm2d(num_inchannels[i], momentum=1.0 - 0.1),
                            nn.ReLU(),
                            # nn.Conv2dTranspose(in_channels=num_inchannels[i],
                            #                    out_channels=num_inchannels[i],
                            #                    kernel_size=3, stride=2,
                            #                    pad_mode="pad", padding=1, has_bias=False),
                            ExpandDimsCell(axis=2),
                            nn.Conv3dTranspose(in_channels=num_inchannels[i],
                                               out_channels=num_inchannels[i],
                                               kernel_size=(1, 3, 3),
                                               stride=(1, 2, 2),
                                               pad_mode="pad",
                                               padding=(0, 0, 1, 1, 1, 1),
                                               output_padding=(0, 1, 1),
                                               has_bias=False),
                            SqueezeCell(axis=2),
                            BatchNorm2d(num_inchannels[i], momentum=1.0 - 0.1),
                            nn.ReLU(),
                            # self.gaussian_filter(num_inchannels[i], 3, 3)]))
                            GaussianFilter(num_inchannels[i], 3, 3)]))
                elif j == i:
                    # fuse_layer.append(None)
                    # print("HighResolutionModule NoneCell", flush=True)
                    fuse_layer.append(NoneCell())
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.SequentialCell([
                                    convs(num_inchannels[j], num_outchannels_conv3x3, 3, 2,
                                          pad_mode="pad", padding=1, has_bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3)
                                ]))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.SequentialCell([
                                    convs(num_inchannels[j], num_outchannels_conv3x3, 3, 2,
                                          pad_mode="pad", padding=1, has_bias=False),
                                    BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU()]))
                    fuse_layer.append(nn.SequentialCell(conv3x3s))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def construct(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class OmniPose(nn.Cell):
    """OmniPose"""

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        # extra = cfg.MODEL.EXTRA
        super(OmniPose, self).__init__()

        # stem net
        self.conv1 = convs(3, 64, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False)
        self.bn1 = BatchNorm2d(64, momentum=1.0-BN_MOMENTUM)
        self.conv2 = convs(64, 64, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False)
        self.bn2 = BatchNorm2d(64, momentum=1.0-BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1, self.transition1_tag = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2, self.transition2_tag = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3, self.transition3_tag = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)

        # If using WASPv1
        # self.wasp = build_wasp(48, 48,[0,0])
        # self.decoder = Decoder(256, 48, cfg.MODEL.NUM_JOINTS)

        # If using WASPv2
        # self.waspv2 = WASPv2('SEPARABLE', 48, 48, cfg.MODEL.NUM_JOINTS)
        self.waspv2 = WASPv2('SEPARABLE', 48, 48, cfg["MODEL"]["NUM_JOINTS"])

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self._init_weight()

    def _init_weight(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    weight_init.initializer(weight_init.Normal(sigma=0.001), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(
                    weight_init.initializer(weight_init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(
                    weight_init.initializer(weight_init.Zero(), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv3dTranspose):
                cell.weight.set_data(
                    weight_init.initializer(weight_init.Normal(sigma=0.001), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        transition_layers_tag = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialCell([
                            convs(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                  3, 1, pad_mode="pad", padding=1, has_bias=False),
                            BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU()]))
                    transition_layers_tag.append(1)
                else:
                    # transition_layers.append(None)
                    # print("OmniPose NoneCell", flush=True)
                    transition_layers.append(NoneCell())
                    transition_layers_tag.append(0)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.SequentialCell([
                            convs(inchannels, outchannels, 3, 2, pad_mode="pad", padding=1, has_bias=False),
                            BatchNorm2d(outchannels),
                            nn.ReLU()]))
                transition_layers.append(nn.SequentialCell(conv3x3s))
                transition_layers_tag.append(1)

        return nn.CellList(transition_layers), transition_layers_tag

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                convs(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, pad_mode="valid", has_bias=False),
                BatchNorm2d(planes * block.expansion, momentum=1.0-BN_MOMENTUM)])

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.SequentialCell(modules), num_inchannels

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        low_level_feat = x

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            # if self.transition1[i] is not None:
            # if not isinstance(self.transition1[i], NoneCell):
            if self.transition1_tag[i] != 0:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        level_2 = y_list[0]

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            # if self.transition2[i] is not None:
            # if not isinstance(self.transition2[i], NoneCell):
            if self.transition2_tag[i] != 0:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        level_3 = y_list[0]

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            # if self.transition3[i] is not None:
            # if not isinstance(self.transition3[i], NoneCell):
            if self.transition3_tag[i] != 0:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # If using WASPv1
        #         x = self.wasp(y_list[0])
        #         x = self.decoder(x,low_level_feat)

        # If using WASPv2
        x = self.waspv2(y_list[0], low_level_feat)

        return x


def get_omnipose(cfg, is_train, **kwargs):
    model = OmniPose(cfg, **kwargs)

    return model


def main():
    import yaml
    from mindspore import context
    yaml_file = "/Users/kaierlong/Codes/OpenI/kaierlong/OmniPose/src/configs/omnipose.yaml"
    yaml_txt = open(yaml_file).read()
    cfg = yaml.load(yaml_txt, Loader=yaml.FullLoader)

    print(cfg, flush=True)
    print(cfg['MODEL']['EXTRA']['STAGE2'], flush=True)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    data = np.random.rand(2, 3, 288, 384).astype(np.float32)
    x = Tensor(data, dtype=mstype.float32)

    model = get_omnipose(cfg, True)

    y = model(x)

    print(y.shape, flush=True)


if __name__ == "__main__":
    main()

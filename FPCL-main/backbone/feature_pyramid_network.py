from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict
from .layers import (Conv2dStaticSamePadding, MaxPool2dStaticSamePadding,
                         MemoryEfficientSwish, Swish)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            #print("name,module",name,module)
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        #print("features")
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                #print(x.shape)
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
        # 将resnet layer4的channel调整到指定的out_channels
        # last_inner = self.inner_blocks[-1](x[-1])
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names


# ----------------------------------#
#   Xception中深度可分离卷积
#   先3x3的深度可分离卷积
#   再1x1的普通卷积
# ----------------------------------#
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1,
                                                      groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            # 获取到了efficientnet的最后三层，对其进行通道的下压缩
            self.p6_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # 对输入进来的p5进行宽高的下采样
            self.p6_to_p7 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            '''self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )'''

            # BIFPN第一轮的时候，跳线那里并不是同一个in
            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p6_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # 简易注意力机制的weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """ bifpn模块结构示意图
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            out = self._forward_fast_attention(inputs)
        else:
            out = self._forward(inputs)

        return out

    def _forward_fast_attention(self, inputs):
        # ------------------------------------------------#
        #   当phi=0、1、2、3、4、5的时候使用fast_attention
        #   获得三个shape的有效特征层
        #   分别是C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        # ------------------------------------------------#
        if self.first_time:
            #print("1")
            #for i in list(inputs.values()):
                #print(i.shape)
            # ------------------------------------------------------------------------#
            #   第一次BIFPN需要 下采样 与 调整通道 获得 p3_in p4_in p5_in p6_in p7_in
            # ------------------------------------------------------------------------#
            names =["0","1","2","3","4"]
            #print(names)
            p3, p4, p5, p6 = list(inputs.values())
            #print(p3.shape,p4.shape,p5.shape,p6.shape)
            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C3 64, 64, 40 -> 64, 64, 64
            # -------------------------------------------#
            p3_in = self.p3_down_channel(p3)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C4 32, 32, 112 -> 32, 32, 64
            #                  -> 32, 32, 64
            # -------------------------------------------#
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            # -------------------------------------------#
            #   首先对通道数进行调整
            #   C5 16, 16, 320 -> 16, 16, 64
            #                  -> 16, 16, 64
            # -------------------------------------------#
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)

            # -------------------------------------------#
            #   对C5进行下采样，调整通道数与宽高
            #   C5 16, 16, 320 -> 8, 8, 64
            # -------------------------------------------#
            p6_in_1 = self.p6_down_channel(p6)
            p6_in_2 = self.p6_down_channel_2(p6)
            # -------------------------------------------#
            #   对P6_in进行下采样，调整宽高
            #   P6_in 8, 8, 64 -> 4, 4, 64
            # -------------------------------------------#
            p7_in = self.p6_to_p7(p6)

            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            #print(self.p6_upsample(p7_in).shape,p6_in_1.shape)
            if (self.p6_upsample(p7_in).shape[2] != p6_in_1.shape[2] or self.p6_upsample(p7_in).shape[3] != p6_in_1.shape[3]):
                x=self.p6_upsample(p7_in)
                x = x[:, :, :p6_in_1.shape[2], :p6_in_1.shape[3]]
                #print(x.shape)
                p6_td = self.conv6_up(self.swish(weight[0] * p6_in_1 + weight[1] * x))
            else:
                p6_td = self.conv6_up(self.swish(weight[0] * p6_in_1 + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in_2还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in_2还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in_2 + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        else:
            #print("2")
            names = list(inputs.keys())
            #for i in list(inputs.values()):
                #print(i.shape)
            p3_in, p4_in, p5_in, p6_in, p7_in = list(inputs.values())

            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            #p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))
            if (self.p6_upsample(p7_in).shape[2] != p6_in.shape[2] or self.p6_upsample(p7_in).shape[3] != p6_in.shape[3]):
                x=self.p6_upsample(p7_in)
                x = x[:, :, :p6_in.shape[2], :p6_in.shape[3]]
                p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * x))
            else:
                p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        results=[p3_out, p4_out, p5_out, p6_out, p7_out]
        #for i in results:
            #print(i.shape)
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out

    def _forward(self, inputs):
        # 当phi=6、7的时候使用_forward
        if self.first_time:
            # 第一次BIFPN需要下采样与降通道获得
            # p3_in p4_in p5_in p6_in p7_in
            names = list(inputs.keys())
            p3, p4, p5, p6 = list(inputs.values())
            p3_in = self.p3_down_channel(p3)
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            p6_in_1 = self.p6_down_channel(p6)
            p6_in_2 = self.p6_down_channel_2(p6)
            p7_in = self.p6_to_p7(p6)

            p6_td = self.conv6_up(self.swish(p6_in_1 + self.p6_upsample(p7_in)))

            p5_td = self.conv5_up(self.swish(p5_in_1 + self.p5_upsample(p6_td)))

            p4_td = self.conv4_up(self.swish(p4_in_1 + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in_2 + p4_td + self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in_2 + p5_td + self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in_2 + p6_td + self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        else:
            names = list(inputs.keys())
            p3_in, p4_in, p5_in, p6_in, p7_in = list(inputs.values())

            p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_td)))

            p4_td = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td + self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in + p5_td + self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))
        results = [p3_out, p4_out, p5_out, p6_out, p7_out]
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out

class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks: ExtraFPNBlock
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if re_getter is True:
            #print("1")
            assert return_layers is not None
            #print("return_layers",return_layers){'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            #print("2")
            self.body = backbone
        #print("in_channels_list",in_channels_list)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        phi=0
        self.fpn_cell_repeats = [3, 4, 5, 6, 2, 1, 8, 8]
        self.bifpn = nn.Sequential(
            *[BiFPN(out_channels,#输出通道数
                    in_channels_list,#输入通道数
                    True if _ == 0 else False,
                    attention=True if phi < 6 else False)
              for _ in range(self.fpn_cell_repeats[phi])])#BiFPN重复次数

        #print("out_channels",out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        #x1=list(x.keys())
        #print(x1)
        x =self.bifpn(x)
        #x = self.fpn(x)
        #print(x1)
        return x

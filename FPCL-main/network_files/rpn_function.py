import os
from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
import numpy as np
from torchvision.ops.roi_align import roi_align

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList
from .poolers import LevelMapper
from .boxes import box_area


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    anchors生成器
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        compute anchor sizes
        Arguments:
            scales: sqrt(anchor_area)
            aspect_ratios: h/w ratios
            dtype: float32
            device: cpu/gpu
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        # [r1, r2, r3]' * [s1, s2, s3]
        # number of elements is len(ratios)*len(scales)
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # left-top, right-bottom coordinate relative to anchor center(0, 0)
        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round 四舍五入

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        anchors position in grid coordinate axis map into origin image
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的x坐标(列)
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的y坐标(行)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # 获取每个预测特征层的尺寸(height, width)
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]
        #print("-----------------------image_size",image_size)[704,1344]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # one step in feature map equate n pixel stride in origin image
        # 计算特征层上的一步等于原始图像上的步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios生成anchors模板
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，list元素对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors


class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]x主干网络输入特征层
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            #print("-----------------------feature.shape",feature.shape)遍历特征层
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C,  H, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)#[batchsize,anchor数，类别数或者4个参数]reshape的作用便于后续将预测值与anchor进行结合，对proposal过滤也方便
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    对box_cla和box_regression两个list中的每个预测特征层的预测信息
    的tensor排列顺序以及shape进行调整 -> [N, -1, C]
    Args:
        box_cls: 每个预测特征层上的预测目标概率
        box_regression: 每个预测特征层上的预测目标bboxes regression参数

    Returns:

    """
    box_cls_flattened = []
    box_regression_flattened = []

    # 遍历每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width]
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        N, AxC, H, W = box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        # [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim将一个batchsize的anchor放在一起[一个batchsize的anchor数，预测类别参数1]
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)#[一个batchsize的anchor数，位置参数4]
    return box_cls, box_regression

class PixelShuffleUpscale(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffleUpscale, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(
            in_channels=256,
            out_channels=(upscale_factor ** 2)*64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ).to("cuda")
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        #print("x",x.shape)
        x = self.conv(x)
        #print("conv",x.shape)
        x = self.pixel_shuffle(x)
        return x

class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,#batch_size_per_image RPN在计算损失时采用正负样本的总个数 positive_fraction正样本占所有用于计算损失样本的比例
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):#NMS处理前后目标个数，NMS阈值
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        Args：
            anchors: (List[Tensor])
            targets: (List[Dict[Tensor])
        Returns:
            labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
            matched_gt_boxes：与anchors匹配的gt
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            #print("gt_boxes",gt_boxes.numel())gt_boxes四个顶点坐标
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                #print("match_quality_matrix",match_quality_matrix.shape)[gt_boxes个数，anchors数]
                #每一个anchor和每一个gtboxes的iou值
                # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                #print("matched_idxs",matched_idxs)给每个anchor分配最匹配(iou最大)的gt_boxes索引，舍弃的anchor值为-2负样本为-1
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                # 这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息
                # 负样本和舍弃的样本都是负值，所以为了防止越界直接置为0
                # 因为后面是通过labels_per_image变量来记录正样本位置的，
                # 所以负样本和舍弃的样本对应的gt_boxes信息并没有什么意义，
                # 反正计算目标边界框回归损失时只会用到正样本。
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                #print("matched_gt_boxes_per_image",len(matched_gt_boxes_per_image))

                # 记录所有anchors匹配后的标签(正样本处标记为1，负样本处标记为0，丢弃样本处标记为-2)
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                #print("labels_per_image",labels_per_image)

                # background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                #print("bg_indices",len(bg_indices))
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                #print("inds_to_discard",len(inds_to_discard))
                labels_per_image[inds_to_discard] = -1.0

            
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            #print("---------------labels,matched_gt_boxes",len(labels),len(matched_gt_boxes))
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
                #print("1num_anchors,pre_nms_top_n",num_anchors,pre_nms_top_n)
            else:
                num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
                #print("2num_anchors,pre_nms_top_n",num_anchors,pre_nms_top_n)每层的anchor数，每层选取的anchor数

            # Returns the k largest elements of the given input tensor along a given dimension根据objectness进行排序
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)#真正索引=当前层索引+偏移量
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        Args:
            proposals: 预测的bbox坐标
            objectness: 预测的目标概率
            image_shapes: batch中每张图片的size信息
            num_anchors_per_level: 每个预测特征层上预测anchors的数目

        Returns:

        """
        num_images = proposals.shape[0]#4
        #print("num_images",num_images)
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        #print("objectness",objectness.shape)[一个batchsize的anchor数，1]
        objectness = objectness.reshape(num_images, -1)
        #print("objectness",objectness.shape)[一个图片的anchor数，4]

        # Returns a tensor of size size filled with fill_value
        # levels负责记录分隔不同预测特征层上的anchors索引信息
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)#[0,0,0,...1,1,1,......4,4,4]一个batchsize的anchor索引（一维）
        #print("levels",levels.shape)

        # Expand this tensor to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)
        #print("levels1",levels)
        #print("levels2",levels.shape)[一个batchsize的anchor索引，4]索引信息复制4份

        # select top_n boxes independently per level before applying nms
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        #print("top_n_idx",top_n_idx.shape)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        objectness = objectness[batch_idx, top_n_idx]
        #print("1objectness",objectness)
        levels = levels[batch_idx, top_n_idx]
        #print("levels",levels)
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)
        #print("objectness",objectness_prob)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 返回boxes满足宽，高都大于min_size的索引
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            #print("boxes",boxes)

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        Arguments:
            objectness (Tensor)：预测的前景概率
            pred_bbox_deltas (Tensor)：预测的bbox regression
            labels (List[Tensor])：真实的标签 1, 0, -1（batch中每一张图片的labels对应List的一个元素中）
            regression_targets (List[Tensor])：真实的bbox regression

        Returns:
            objectness_loss (Tensor) : 类别损失
            box_loss (Tensor)：边界框回归损失
        """
        num=objectness.shape[0]

        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        #print("sampled_pos_inds",len(sampled_pos_inds))
        #for i in sampled_pos_inds:
              #print("i",i.shape)
        
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        #print("sampled_pos_inds",sampled_pos_inds.shape)
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        #print("sampled_neg_inds",sampled_neg_inds.shape)

        #------------------------------------------------------------------------------------------------------------------------------------
        pos_id=[]
        for i in sampled_pos_inds:
              if i>=0 and i<=num/4-1:
                  pos_id.append([0])
              if i>=num/4 and i<=num/2-1:
                  pos_id.append([1])
              if i>=num/2 and i<=num*3/4-1:
                  pos_id.append([2])
              if i>=num*3/4 and i<=num-1:
                  pos_id.append([3])
        pos_ids=torch.tensor(pos_id).cuda()
        #print("pos_ids", pos_ids.shape)
        neg_id=[]
        for i in sampled_neg_inds:
              if i>=0 and i<=num/4-1:
                  neg_id.append([0])
              if i>=num/4 and i<=num/2-1:
                  neg_id.append([1])
              if i>=num/2 and i<=num*3/4-1:
                  neg_id.append([2])
              if i>=num*3/4 and i<=num-1:
                  neg_id.append([3])
        neg_ids=torch.tensor(neg_id).cuda()
        #print("ids",ids1.shape,ids1)
        # ------------------------------------------------------------------------------------------------------------------------------------

        # 将所有正负样本索引拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        #print("pred_bbox_deltas",pred_bbox_deltas.shape)
        # 计算边界框回归损失
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失
        objectness_loss = F.binary_cross_entropy_with_logits( 
            objectness[sampled_inds], labels[sampled_inds]
        )
        #print("sampled_pos_inds", sampled_pos_inds.shape)

        return objectness_loss, box_loss, sampled_pos_inds, sampled_neg_inds, pos_ids, neg_ids

    #----------------------------------------------------------------------------------------------------------------------------------------
    def level_mapper(self,boxes):
        #print("boxes",boxes.shape,boxes)
        #for i in range(boxes.shape[0]):
              #print("i",i)
              #print("boxes[i]",boxes[i])
        s = torch.sqrt(box_area(boxes))
        #print("sf",box_area(boxes))

        # Eqn.(1) in FPN paper
        #print(self.lvl0,s,self.s0,self.eps,self.k_min,self.k_max)
        target_lvls = torch.floor(4 + torch.log2(s / 224) + torch.tensor(1e-6, dtype=s.dtype))
        #print("target_lvls1",target_lvls.shape)[2048]
        target_lvls = torch.clamp(target_lvls, min=2, max=5)
        #print("target_lvls2",target_lvls.shape)[2048]
        #print((target_lvls.to(torch.int64) - 2).to(torch.int64).shape)
        return (target_lvls.to(torch.int64) - 2).to(torch.int64)

    def _filter_input(self,x: Dict[str, Tensor], featmap_names: List[str]) -> List[Tensor]:
        x_filtered = []
        for k, v in x.items():
            if k in featmap_names:
                x_filtered.append(v)
        return x_filtered

    def _multiscale_roi_align(self,x_filtered,output_size,scales,levels,rois,sampling_ratio):
        num_rois=len(rois)
        #print("num_rois1",num_rois)
        num_channels = x_filtered[0].shape[1]
        #print("num_channels1", num_channels)
        dtype, device = x_filtered[0].dtype, x_filtered[0].device

        result = torch.zeros(
            (
                num_rois,
                num_channels,
            )
            + output_size,
            dtype=dtype,
            device=device,
        )

        #print("result", result)
        tracing_results = []

        #print("rois", rois.shape)
        #print("levels1", levels.shape)
        #print("x_filtered1", x_filtered)
        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
            if(level==0):
                idx_in_level = torch.where(levels == level)[0]
                # print("idx_in_level",idx_in_level.shape,idx_in_level)
                rois_per_level = rois[idx_in_level]
                # print("rois_per_level",rois_per_level.shape,rois_per_level)
                # print("per_level_feature",per_level_feature.shape)
                result_idx_in_level = roi_align(
                    per_level_feature,
                    rois_per_level,
                    output_size=output_size,
                    spatial_scale=scale,
                    sampling_ratio=sampling_ratio,
                )
                if torchvision._is_tracing():
                    print("1")
                    tracing_results.append(result_idx_in_level.to(dtype))
                else:
                    print("2")
                    # result and result_idx_in_level's dtypes are based on dtypes of different
                    # elements in x_filtered.  x_filtered contains tensors output by different
                    # layers.  When autocast is active, it may choose different dtypes for
                    # different layers' outputs.  Therefore, we defensively match result's dtype
                    # before copying elements from result_idx_in_level in the following op.
                    # We need to cast manually (can't rely on autocast to cast for us) because
                    # the op acts on result in-place, and autocast only affects out-of-place ops.
                    result[idx_in_level] = result_idx_in_level.to(result.dtype)
            else:
                break

        if torchvision._is_tracing():
            print("11")
            result = _onnx_merge_levels(levels, tracing_results)

        #print("result", result.shape, result)
        return result


    def up(self,levels,boxes,ids,x_filtered,scales,original_image_sizes,image_sizes,target_size):
        features = torch.rand(0).cuda()
        new_levels= torch.rand(0).cuda()
        # print(images.tensors[0].shape)
        for level, box, id in zip(levels, boxes, ids):
            # print(pos_box.shape)
            #feat_map_size = torch.tensor([x_filtered[level][id].shape[2],
                                         # x_filtered[level][id].shape[3]]).cuda()  # tensor([w,h])
            if(box[0]<0 or box[1]<0 or box[2]<0 or box[3]<0):
                continue
            else:
                #print("scales[level]", scales[level])
                #print("pos_box", box[0], box[1], box[2], box[3])
                #print("new_levels",new_levels.shape,new_levels)
                level = level.view(1)
                s = round(image_sizes[id][1] / original_image_sizes[id][1], 2)
                #print("image_sizes", image_sizes[id][1], original_image_sizes[id][1])
                #print("s", s)
                fm_x1, fm_y1 = int(box[0] * scales[level] * s), int(box[1] * scales[level] * s)
                fm_x2, fm_y2 = int(box[2] * scales[level] * s), int(box[3] * scales[level] * s)
                #print("fm", fm_x1, fm_y1, fm_x2, fm_y2)
                #print("x_filtered",x_filtered.shape)
                #print("level",level,id)
                feat_map = x_filtered[level][id]
                #print("feat_map", feat_map.shape)
                cropped_feat = feat_map[:, :, fm_y1:fm_y2 + 1, fm_x1:fm_x2 + 1]
                #output_tensor = F.interpolate(cropped_feat, size=(10, 10), mode='bilinear', align_corners=True)
                #features = torch.cat([features, output_tensor])
                #print("cropped_feat", cropped_feat.shape)
                # 定义目标尺寸
                if(cropped_feat.size()[2]>=12 or cropped_feat.size()[3]>=12):
                    continue
                else:
                    new_levels = torch.cat([new_levels, level])
                    upscale_factor = target_size[1] // cropped_feat.size()[2]
                    # print(upscale_factor)
                    with torch.no_grad():
                        pixel_shuffle = PixelShuffleUpscale(upscale_factor)
                        output_tensor = pixel_shuffle(cropped_feat)
                        # print(output_tensor.shape)
                        # 将所有特征向量放大到指定尺寸
                        output_tensor = F.interpolate(output_tensor, size=(12, 12), mode='bilinear', align_corners=True)
                    # print(output_tensor.shape)  # should be torch.Size([1, 64, 30, 30])
                    features = torch.cat([features, output_tensor])
                    del cropped_feat, output_tensor
                    torch.cuda.empty_cache()
        #print("new_levels",features.shape,new_levels.shape)'''
        return features,new_levels
    # ----------------------------------------------------------------------------------------------------------------------------------------

    def forward(self,
                images,        # type: ImageList
                features,      # type: Dict[str, Tensor]
                original_image_sizes,
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        # features是所有预测特征层组成的OrderedDict
        feat=features
        #print(list(features.keys()))
        features = list(features.values())
        #for i in features:
            #print(i.shape)


        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        # objectness和pred_bbox_deltas都是list
        objectness, pred_bbox_deltas = self.head(features)
        #print("objectness",len(objectness))5层特征层
        #for i in objectness:
              #print("objectness",i.shape)[batchsize(4),参数个数channel(3),W,H]
        #for i in pred_bbox_deltas:
              #print("pred_bbox_deltas",i.shape)[batchsize(4),参数个数channel(12),W,H]

        # 生成一个batch图像的所有anchors信息,list(tensor)元素个数等于batch_size
        anchors = self.anchor_generator(images, features)
        #for i in anchors:
              #print("anchor",i)

        # batch_size4
        num_images = len(anchors)

        # numel() Returns the total number of elements in the input tensor.
        # 计算每个预测特征层上的对应的anchors数量
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        #print("num_anchors_per_level_shape_tensors",num_anchors_per_level_shape_tensors)
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        #for s in num_anchors_per_level_shape_tensors:
              #print("s",s[0] ,s[1] , s[2])5层featuremap，一层一个位置产生3个anchor，一共W*H个位置，总共产生每层anchor数*W*H之和个anchor数

        # 调整内部tensor格式以及shape
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,#[一个batchsize的anchor数，预测类别参数1]
                                                                    pred_bbox_deltas)#[一个batchsize的anchor数，预测位置参数4]

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        # 将预测的bbox regression参数应用到anchors上得到最终预测bbox坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)#[一个batchsize的anchor数,1,proposal坐标信息]
        #print("proposals1",proposals.shape,proposals)
        proposals = proposals.view(num_images, -1, 4)
        #print("proposals2",proposals.shape,proposals)

        res=np.reshape(proposals,(-1,4))
        res=res.cuda()
        #print("res.shape",res.shape)

        # 筛除小boxes框，nms处理，根据预测概率得分在每个预测特征层获取前post_nms_top_n（2000）个目标，并在初筛选的框进行调整剔除
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        #print("boxes1",boxes)
        #for i in boxes:
              #print("boxes",i.shape)
        #for (i,j) in zip(boxes,features):
              #print("boxes",i.shape)
              #print("feature",j.shape)
        #for i in scores:
              #print("scores",i.shape)[2000,1]*4

        losses = {}
        pos_align_feature=torch.rand(0).cuda()
        pos_level=torch.rand(0).cuda()
        #global sampled_pos_inds, sampled_neg_inds, pos_ids, neg_ids, pos_boxes, neg_boxes, pos_levels, neg_levels, pos_rois, neg_rois, pos_align_feature, pos_align_feature

        if self.training:
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            #for i in labels:
                  #print("labels",i.shape)每张图片的样本索引
            #for i in matched_gt_boxes:
                  #print("matched_gt_boxes",i)gt_boxes坐标信息
            # 结合anchors以及对应的gt，计算regression参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg, sampled_pos_inds, sampled_neg_inds, pos_ids, neg_ids= self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }

            #------------------------------------------------------------------------------------------------------------------------------------
            featmap_names=['0', '1', '2', '3']
            output_size = [15, 15]
            scales = [0.25, 0.125, 0.0625, 0.03125]
            target_size = [12,12]
            scales_martix=torch.tensor(scales).cuda()
            sampling_ratio =2

            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            output_size = tuple(output_size)

            x_filtered =self. _filter_input(feat, featmap_names)
            #for i in x_filtered:
             #   print(i.shape)

            #正负样本坐标信息
            pos_boxes=res[sampled_pos_inds].cuda()
            #neg_boxes=res[sampled_neg_inds].cuda()
            #print("pos_boxes", pos_boxes.shape)

            #正负样本坐标对应特征层
            pos_levels=self.level_mapper(pos_boxes)
            pos_align_feature,pos_level=self.up(pos_levels,pos_boxes,pos_ids,x_filtered,scales_martix,original_image_sizes,images.image_sizes,target_size)

        return boxes, losses, pos_align_feature, pos_level

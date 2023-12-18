from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from .boxes import box_area

from torchvision.utils import _log_api_usage_once
from torchvision.ops.roi_align import roi_align


# copying result_idx_in_level to a specific index in result[]
# is not supported by ONNX tracing yet.
# _onnx_merge_levels() is an implementation supported by ONNX
# that merges the levels to the right indices
@torch.jit.unused
def _onnx_merge_levels(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros(
        (levels.size(0), first_result.size(1), first_result.size(2), first_result.size(3)), dtype=dtype, device=device
    )
    for level in range(len(unmerged_results)):
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        index = index.expand(
            index.size(0),
            unmerged_results[level].size(1),
            unmerged_results[level].size(2),
            unmerged_results[level].size(3),
        )
        res = res.scatter(0, index, unmerged_results[level])
    return res


# TODO: (eellison) T54974082 https://github.com/pytorch/pytorch/issues/26744/pytorch/issues/26744
def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level idsproposal面积开根号
        #for boxlist in boxlists:
              #print("-------------------boxlists",boxlist)
        
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))
        #print("-------------------sp",s.shape)

        # Eqn.(1) in FPN paper
        #print(self.lvl0,s,self.s0,self.eps,self.k_min,self.k_max)
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        #print("-------------------target_lvls1",target_lvls.shape)[2048]
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        #print("-------------------target_lvls2",target_lvls.shape)[2048]
        #print((target_lvls.to(torch.int64) - self.k_min).to(torch.int64).shape)[2048]
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


def _convert_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = torch.cat(boxes, dim=0)
    #print("-------------------concat_boxes",concat_boxes.shape)[2048,4]
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)],
        dim=0,
    )
    #print("-------------------ids",ids.shape,ids)
    rois = torch.cat([ids, concat_boxes], dim=1)
    #print("-------------------rois",rois.shape,rois)
    return rois


def _infer_scale(feature: Tensor, original_size: List[int]) -> float:
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    return possible_scales[0]


@torch.fx.wrap
def _setup_scales(
    features: List[Tensor], image_shapes: List[Tuple[int, int]], canonical_scale: int, canonical_level: int
) -> Tuple[List[float], LevelMapper]:
    if not image_shapes:
        raise ValueError("images list should not be empty")
    max_x = 0
    max_y = 0
    #print("----------------image_shapes",image_shapes)[699,1333]
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
        #print("----------------max_x,max_y",max_x,max_y)
    original_input_shape = (max_x, max_y)
    #print("----------------original_input_shape",original_input_shape)

    scales = [_infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
    #print("----------------lvl_min,lvl_max",lvl_min,lvl_max)2.0,5.0(最小和最大采样率次数)
    
    map_levels = initLevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    #print("----------------scales",scales)
    #print("----------------map_levels",map_levels)
    return scales, map_levels


@torch.fx.wrap
def _filter_input(x: Dict[str, Tensor], featmap_names: List[str]) -> List[Tensor]:
    x_filtered = []
    for k, v in x.items():
        if k in featmap_names:
            x_filtered.append(v)
    return x_filtered


@torch.fx.wrap
def _multiscale_roi_align(
    x_filtered: List[Tensor],
    boxes: List[Tensor],
    output_size: List[int],
    sampling_ratio: int,
    scales: Optional[List[float]],
    mapper: Optional[LevelMapper],
) -> Tensor:
    """
    Args:
        x_filtered (List[Tensor]): List of input tensors.
        boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
            (x1, y1, x2, y2) format and in the image reference size, not the feature map
            reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        output_size (Union[List[Tuple[int, int]], List[int]]): size of the output
        sampling_ratio (int): sampling ratio for ROIAlign
        scales (Optional[List[float]]): If None, scales will be automatically inferred. Default value is None.
        mapper (Optional[LevelMapper]): If none, mapper will be automatically inferred. Default value is None.
    Returns:
        result (Tensor)
    """
    if scales is None or mapper is None:
        raise ValueError("scales and mapper should not be None")

    num_levels = len(x_filtered)#4
    rois = _convert_to_roi_format(boxes)#图片索引与坐标信息
    #print(rois)

    if num_levels == 1:#without fpn
        return roi_align(
            x_filtered[0],
            rois,
            output_size=output_size,
            spatial_scale=scales[0],
            sampling_ratio=sampling_ratio,
        )
    #print("---------------boxes",boxes)
    levels = mapper(boxes)
    #print("---------------levelsp",levels)

    num_rois = len(rois)#2048
    #print("---------------num_rois2",num_rois)
    num_channels = x_filtered[0].shape[1]
    #print("---------------num_channels2",num_channels)

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
    #print("---------------result",result.shape,result)[2048,256,7,7]
    
    tracing_results = []
    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = torch.where(levels == level)[0]
        #print("---------------idx_in_level",idx_in_level.shape,idx_in_level)
        rois_per_level = rois[idx_in_level]
        #print("---------------rois_per_level",rois_per_level.shape,rois_per_level)循环4次，将同一feature_map不同图片的box的坐标信息存入rois_per_level
        result_idx_in_level = roi_align(
            per_level_feature,
            rois_per_level,
            output_size=output_size,
            spatial_scale=scale,
            sampling_ratio=sampling_ratio,
        )

        if torchvision._is_tracing():
            tracing_results.append(result_idx_in_level.to(dtype))
        else:
            # result and result_idx_in_level's dtypes are based on dtypes of different
            # elements in x_filtered.  x_filtered contains tensors output by different
            # layers.  When autocast is active, it may choose different dtypes for
            # different layers' outputs.  Therefore, we defensively match result's dtype
            # before copying elements from result_idx_in_level in the following op.
            # We need to cast manually (can't rely on autocast to cast for us) because
            # the op acts on result in-place, and autocast only affects out-of-place ops.
            result[idx_in_level] = result_idx_in_level.to(result.dtype)

    if torchvision._is_tracing():
        result = _onnx_merge_levels(levels, tracing_results)

    return result


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.
    It infers the scale of the pooling via the heuristics specified in eq. 1
    of the `Feature Pyramid Network paper <https://arxiv.org/abs/1612.03144>`_.
    They keyword-only parameters ``canonical_scale`` and ``canonical_level``
    correspond respectively to ``224`` and ``k0=4`` in eq. 1, and
    have the following meaning: ``canonical_level`` is the target level of the pyramid from
    which to pool a region of interest with ``w x h = canonical_scale x canonical_scale``.
    Args:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
        canonical_scale (int, optional): canonical_scale for LevelMapper
        canonical_level (int, optional): canonical_level for LevelMapper
    Examples::
        >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random bounding boxes
        >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])
    """

    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        _log_api_usage_once(self)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        #print("--------------x",x)features字典
        x_filtered = _filter_input(x, self.featmap_names)
        #print("--------------x_filtered2",x_filtered.shape)
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )
        #print("--------------self.scales",self.scales)
        #print("--------------self.map_levels",self.map_levels)
        return _multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )
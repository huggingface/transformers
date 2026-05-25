# Copyright 2026 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Sapiens2 License. You may obtain a copy of the License at
#
#     https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size_for_max_height_width,
    make_list_of_images,
)
from ...modeling_outputs import ModelOutput, SemanticSegmenterOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, is_cv2_available, logging, requires_backends
from ...utils.generic import can_return_tuple, maybe_autocast
from ..beit.modeling_beit import BeitConvLayer
from ..dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTAttention,
    DINOv3ViTBackbone,
    DINOv3ViTEmbeddings,
    DINOv3ViTEncoder,
    DINOv3ViTLayer,
    DINOv3ViTLayerScale,
    DINOv3ViTModel,
    DINOv3ViTPreTrainedModel,
    augment_patches_center_coordinates,
    get_patches_center_coordinates,
    rotate_half,
)
from ..gemma2.modeling_gemma2 import eager_attention_forward
from ..vitpose.image_processing_vitpose import get_keypoint_predictions, get_warp_matrix
from ..vitpose.modeling_vitpose import flip_back


if is_cv2_available():
    import cv2


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Class for outputs of pose estimation models.
    """
)
@dataclass
class Sapiens2PoseEstimatorOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Keypoint loss computed between the predicted heatmaps and the ground-truth heatmaps.
    heatmaps (`torch.FloatTensor` of shape `(batch_size, num_keypoints, height, width)`):
        Heatmaps as predicted by the model.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage)
        of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
        each layer plus the initial embedding outputs.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one per layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights after the attention softmax.
    """

    loss: torch.FloatTensor | None = None
    heatmaps: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Class for outputs of normal estimation models.
    """
)
@dataclass
class Sapiens2NormalEstimatorOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Normal estimation loss.
    normals (`torch.FloatTensor` of shape `(batch_size, num_labels, height, width)`):
        Raw normal map predictions as output by the model (unnormalized).
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage)
        of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
        each layer plus the initial embedding outputs.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one per layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights after the attention softmax.
    """

    loss: torch.FloatTensor | None = None
    normals: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Class for outputs of pointmap estimation models.
    """
)
@dataclass
class Sapiens2PointmapEstimatorOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Pointmap estimation loss.
    pointmaps (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
        Per-pixel 3D XYZ coordinate predictions in canonical camera space.
    scales (`torch.FloatTensor` of shape `(batch_size, 1)`, *optional*):
        Canonical focal length / actual focal length ratio. `None` when no scale branch is configured.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage)
        of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
        each layer plus the initial embedding outputs.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one per layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights after the attention softmax.
    """

    loss: torch.FloatTensor | None = None
    pointmaps: torch.FloatTensor | None = None
    scales: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Class for outputs of matting models.
    """
)
@dataclass
class Sapiens2MattingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Matting loss.
    alphas (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`):
        Alpha matte predictions in `[0, 1]` (sigmoid-activated).
    foregrounds (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
        Pre-multiplied RGB foreground predictions in `[0, 1]` (sigmoid-activated).
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage)
        of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
        each layer plus the initial embedding outputs.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one per layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights after the attention softmax.
    """

    loss: torch.FloatTensor | None = None
    alphas: torch.FloatTensor | None = None
    foregrounds: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


def box_to_center_and_scale(bbox: np.ndarray, padding: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    """Convert COCO bbox (x, y, w, h) to center and scale for affine warp."""
    x, y, w, h = bbox[:4].astype(np.float32)
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    scale = np.array([w * padding, h * padding], dtype=np.float32)
    return center, scale


def cv2_warp_affine(image_np: np.ndarray, warp_mat: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Warp-affine crop a HWC uint8 numpy array. output_size = (width, height)."""
    scale_factor = min(np.linalg.norm(warp_mat[0, :2]), np.linalg.norm(warp_mat[1, :2]))
    flags = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    return cv2.warpAffine(image_np, warp_mat, output_size, flags=flags)


def cv2_gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Gaussian blur per-keypoint heatmap, preserving the original max value."""
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        if np.max(heatmaps[k]) > 0:
            heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def fix_aspect_ratio(scale: np.ndarray, width: int, height: int) -> np.ndarray:
    """Adjust scale so the crop aspect ratio matches the given width/height."""
    aspect_ratio = width / height
    sw, sh = scale
    if sw > sh * aspect_ratio:
        return np.array([sw, sw / aspect_ratio], dtype=scale.dtype)
    else:
        return np.array([sh * aspect_ratio, sh], dtype=scale.dtype)


def post_dark_unbiased_data_processing(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int = 11
) -> np.ndarray:
    """Sub-pixel refinement via Hessian on log-heatmaps (UDP Dark Pose). In-place, returns keypoints."""
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    heatmaps = cv2_gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1).reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze()

    return keypoints


class Sapiens2ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """

    do_reduce_labels: bool


class Sapiens2ImageProcessor(TorchvisionBackend):
    # Note: original Sapiens2 uses cv2.INTER_AREA for downsampling and cv2.INTER_CUBIC for upsampling
    valid_kwargs = Sapiens2ImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 1024, "width": 768}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_reduce_labels = False
    do_pad = False  # Set to True for normal, albedo, and pointmap estimation

    def __init__(self, **kwargs: Unpack[Sapiens2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        boxes: list[list[list[float]]] | None = None,
        **kwargs: Unpack[Sapiens2ImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        boxes (`list[list[list[float]]]` or `np.ndarray`, *optional*):
            List or array of bounding boxes for each image. Each box should be a list of 4 floats
            representing the bounding box coordinates in COCO format
            (top_left_x, top_left_y, width, height). When provided, each person crop is
            affine-warped to the model input size instead of resizing the full image.
            Requires the `cv2` package.
        """
        if boxes is not None:
            requires_backends(self, ["cv2"])
        return super().preprocess(images, segmentation_maps, boxes, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        boxes: list[list[list[float]]] | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        device: Union[str, "torch.device"] | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Handle extra inputs beyond images."""
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        images_kwargs = kwargs.copy()
        images_kwargs["do_reduce_labels"] = False
        data = {}
        data["pixel_values"] = self._preprocess(images, boxes=boxes, **images_kwargs)

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update({"do_normalize": False, "do_rescale": False})
            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            )

            processed_segmentation_maps = [
                processed_segmentation_map.squeeze(0).to(torch.int64)
                for processed_segmentation_map in processed_segmentation_maps
            ]
            data["labels"] = processed_segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    def reduce_label(self, labels: list["torch.Tensor"]) -> list["torch.Tensor"]:
        """Reduce label values by 1, replacing 0 with 255."""
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype, device=label.device), label)
            label = label - 1
            label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype, device=label.device), label)
            labels[idx] = label
        return labels

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        do_reduce_labels: bool = False,
        do_pad: bool = False,
        boxes: list[list[list[float]]] | None = None,
        **kwargs,
    ) -> list["torch.Tensor"]:
        if boxes is not None:
            output_size = (size["width"], size["height"])  # (W, H) for cv2
            crops = []
            for image, image_boxes in zip(images, boxes):
                image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
                for bbox in image_boxes:
                    bbox_np = np.array(bbox, dtype=np.float32)
                    center, scale = box_to_center_and_scale(bbox_np)
                    scale = fix_aspect_ratio(scale, size["width"], size["height"])
                    warp_mat = get_warp_matrix(0, center * 2, np.array(output_size, dtype=np.float32) - 1, scale)
                    crop_np = cv2_warp_affine(image_np, warp_mat, output_size)
                    crops.append(torch.from_numpy(crop_np).permute(2, 0, 1).to(image.device))
            images = crops
            do_resize = False  # affine crop already produces the target size

        if do_reduce_labels:
            images = self.reduce_label(images)

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                if do_pad:
                    # Resize while preserving aspect ratio. Then add symmetric padding on all sides to reach the target size.
                    aspect_ratio_size = SizeDict(max_height=size["height"], max_width=size["width"])
                    stacked_images = self.resize(stacked_images, aspect_ratio_size, resample)
                    stacked_images = self.center_crop(stacked_images, size)
                else:
                    stacked_images = self.resize(stacked_images, size, resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        return reorder_images(processed_images_grouped, grouped_images_index)

    def post_process_pose_estimation(
        self,
        outputs: Sapiens2PoseEstimatorOutput,
        boxes: list[list[list[float]]] | np.ndarray,
        blur_kernel_size: int = 11,
        threshold: float | None = None,
        target_sizes: TensorType | list[tuple] | None = None,
    ) -> list[list[dict[str, torch.Tensor]]]:
        """
        Converts the output of [`Sapiens2ForPoseEstimation`] into keypoint predictions in image space.

        Args:
            outputs (`Sapiens2PoseEstimatorOutput`):
                Raw outputs of the model. `outputs.heatmaps` must have shape
                `(N_total, num_keypoints, heatmap_height, heatmap_width)` where
                `N_total = sum(len(b) for b in boxes)`.
            boxes (`list[list[list[float]]]` or `np.ndarray`):
                List or array of bounding boxes for each image. Each box should be a list of 4 floats
                representing the bounding box coordinates in COCO format
                (top_left_x, top_left_y, width, height). Must match the `boxes` argument passed to
                `preprocess`.
            blur_kernel_size (`int`, *optional*, defaults to 11):
                Kernel size for the Gaussian blur used in UDP Dark Pose refinement.
            threshold (`float`, *optional*):
                Score threshold. Keypoints with scores at or below this value are
                filtered out from the result dictionaries.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(N_total, 2)` or list of `(width, height)` tuples, one per person (i.e. one
                per flattened box). When provided, each box is multiplied by
                `[width, height, width, height]` before processing, which is useful when boxes are given in
                normalized [0, 1] coordinates.

        Returns:
            `list[list[dict]]`: Outer list is over images, inner list is over persons.
            Each dict contains:
            - `keypoints` (`torch.FloatTensor` of shape `(num_keypoints, 2)`): x/y in image coords.
            - `scores` (`torch.FloatTensor` of shape `(num_keypoints,)`): per-keypoint confidence.
            - `labels` (`torch.LongTensor` of shape `(num_keypoints,)`): keypoint indices.
            - `bbox` (`torch.FloatTensor` of shape `(4,)`): the padded, aspect-ratio-corrected input bounding box
              in xyxy format (x_min, y_min, x_max, y_max), i.e. the actual image region used for preprocessing.
        """
        requires_backends(self, ["cv2"])

        heatmaps_np = outputs.heatmaps.cpu().numpy()  # (N_total, K, H_hm, W_hm)
        flattened_boxes = [np.array(bbox, dtype=np.float32) for image_boxes in boxes for bbox in image_boxes]

        if target_sizes is not None and len(flattened_boxes) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the total number of persons "
                "(i.e. the sum of boxes across all images)."
            )

        _, K, H_hm, W_hm = heatmaps_np.shape
        heatmap_size = np.array([W_hm - 1, H_hm - 1], dtype=np.float32)

        person_results = []
        for i, bbox in enumerate(flattened_boxes):
            if target_sizes is not None:
                image_width, image_height = target_sizes[i][0], target_sizes[i][1]
                bbox = bbox * np.array([image_width, image_height, image_width, image_height], dtype=np.float32)
            center, scale = box_to_center_and_scale(bbox)
            scale = fix_aspect_ratio(scale, self.size["width"], self.size["height"])
            heatmap_i = heatmaps_np[i].copy()  # copy to avoid mutating caller's tensor

            locs, scores_i = get_keypoint_predictions(heatmap_i[None])  # (1, K, 2), (1, K, 1)
            scores_i = scores_i[0, :, 0]  # (K,)
            locs = post_dark_unbiased_data_processing(locs, heatmap_i, blur_kernel_size)
            locs = locs[0]  # (K, 2)

            locs = locs / heatmap_size * scale + center - 0.5 * scale

            keypoints = torch.tensor(locs, dtype=torch.float32)
            scores = torch.tensor(scores_i, dtype=torch.float32)
            labels = torch.arange(0, K)
            bbox_tensor = torch.tensor(
                [
                    center[0] - scale[0] / 2,
                    center[1] - scale[1] / 2,
                    center[0] + scale[0] / 2,
                    center[1] + scale[1] / 2,
                ],
                dtype=torch.float32,
            )

            if threshold is not None:
                keep = scores > threshold
                keypoints = keypoints[keep]
                scores = scores[keep]
                labels = labels[keep]

            person_results.append({"keypoints": keypoints, "scores": scores, "labels": labels, "bbox": bbox_tensor})

        # Reassemble into list[list[dict]] grouped by image
        result = []
        idx = 0
        for image_boxes in boxes:
            n = len(image_boxes)
            result.append(person_results[idx : idx + n])
            idx += n
        return result

    def post_process_semantic_segmentation(
        self, outputs: SemanticSegmenterOutput, target_sizes: list[tuple] | None = None
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`Sapiens2ForSemanticSegmentation`] into semantic segmentation maps.

        Args:
            outputs (`SemanticSegmenterOutput`):
                Raw outputs of the model.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size `(height, width)` of each prediction.
                If unset, predictions will not be resized.

        Returns:
            `list[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of
            shape `(height, width)` corresponding to the target size (if `target_sizes` is specified).
            Each entry corresponds to a semantic class id.
        """
        logits = outputs.logits

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(len(logits)):
                resized_logits = F.interpolate(
                    logits[idx].unsqueeze(0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                semantic_segmentation.append(resized_logits[0].argmax(dim=0))
        else:
            semantic_segmentation = list(logits.argmax(dim=1))

        return semantic_segmentation

    def post_process_normal_estimation(
        self,
        outputs: Sapiens2NormalEstimatorOutput,
        source_sizes: list[tuple] | None = None,
        target_sizes: list[tuple] | None = None,
        do_remove_padding: bool | None = None,
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`Sapiens2ForNormalEstimation`] into L2-normalized surface normal maps.

        Args:
            outputs (`Sapiens2NormalEstimatorOutput`):
                Raw outputs of the model.
            source_sizes (`list[tuple]` of length `batch_size`, *optional*):
                Original `(height, width)` of each image before preprocessing. When provided,
                the padding added during preprocessing is removed and predictions are resized back
                to the original image size (unless `target_sizes` overrides the final size).
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                Requested final `(height, width)` for each prediction. When provided, used as the
                resize target instead of `source_sizes`. Resized with bilinear interpolation after
                L2 normalization.
            do_remove_padding (`bool`, *optional*):
                Whether to crop away the zero-padding added during preprocessing before resizing.
                Defaults to `True` when `source_sizes` is provided, `False` otherwise.

        Returns:
            `list[torch.Tensor]` of length `batch_size`, each of shape `(3, height, width)`.
            Values are L2-normalized unit vectors in `[-1, 1]` per channel (XYZ surface normals).
        """
        if do_remove_padding is None:
            do_remove_padding = source_sizes is not None

        if do_remove_padding and source_sizes is None:
            raise ValueError("`source_sizes` must be provided when `do_remove_padding=True`.")

        normals = outputs.normals

        if source_sizes is not None and len(normals) != len(source_sizes):
            raise ValueError("Make sure that you pass in as many source sizes as the batch dimension of the normals")
        if target_sizes is not None and len(normals) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the normals")

        result = []
        model_h = self.size["height"]
        model_w = self.size["width"]

        normals = F.normalize(normals, p=2, dim=1, eps=1e-8)

        for idx in range(len(normals)):
            normal = normals[idx]

            if do_remove_padding:
                orig_h, orig_w = source_sizes[idx]
                new_h, new_w = get_image_size_for_max_height_width((orig_h, orig_w), model_h, model_w)
                pad_top = (model_h - new_h) // 2 if new_h < model_h else 0
                pad_left = (model_w - new_w) // 2 if new_w < model_w else 0
                normal = normal[:, pad_top : pad_top + min(new_h, model_h), pad_left : pad_left + min(new_w, model_w)]

            final_size = (
                target_sizes[idx]
                if target_sizes is not None
                else (source_sizes[idx] if source_sizes is not None else None)
            )

            if final_size is not None:
                normal = F.interpolate(
                    normal.unsqueeze(0),
                    size=final_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]

            result.append(normal)

        return result

    def post_process_pointmap(
        self,
        outputs: Sapiens2PointmapEstimatorOutput,
        source_sizes: list[tuple] | None = None,
        target_sizes: list[tuple] | None = None,
        do_remove_padding: bool | None = None,
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`Sapiens2ForPointmapEstimation`] into pointmap tensors in image space.

        Args:
            outputs (`Sapiens2PointmapEstimatorOutput`):
                Raw outputs of the model.
            source_sizes (`list[tuple]` of length `batch_size`, *optional*):
                Original `(height, width)` of each image before preprocessing. When provided,
                the padding added during preprocessing is removed and predictions are resized back
                to the original image size (unless `target_sizes` overrides the final size).
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                Requested final `(height, width)` for each prediction. Overrides `source_sizes`
                as the resize target.
            do_remove_padding (`bool`, *optional*):
                Whether to crop away the zero-padding added during preprocessing before resizing.
                Defaults to `True` when `source_sizes` is provided, `False` otherwise.

        Returns:
            `list[torch.Tensor]` of length `batch_size`, each of shape `(3, height, width)`.
            Values are per-pixel 3D XYZ coordinates in canonical camera space, optionally divided
            by `outputs.scales` to convert to metric coordinates.
        """
        if do_remove_padding is None:
            do_remove_padding = source_sizes is not None

        if do_remove_padding and source_sizes is None:
            raise ValueError("`source_sizes` must be provided when `do_remove_padding=True`.")

        pointmaps = outputs.pointmaps

        if source_sizes is not None and len(pointmaps) != len(source_sizes):
            raise ValueError("Make sure that you pass in as many source sizes as the batch dimension of the pointmap")
        if target_sizes is not None and len(pointmaps) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the pointmap")

        result = []
        model_h = self.size["height"]
        model_w = self.size["width"]

        for idx in range(len(pointmaps)):
            pm = pointmaps[idx]

            if outputs.scales is not None:
                pm = pm / outputs.scales[idx]

            if do_remove_padding:
                orig_h, orig_w = source_sizes[idx]
                new_h, new_w = get_image_size_for_max_height_width((orig_h, orig_w), model_h, model_w)
                pad_top = (model_h - new_h) // 2 if new_h < model_h else 0
                pad_left = (model_w - new_w) // 2 if new_w < model_w else 0
                pm = pm[:, pad_top : pad_top + min(new_h, model_h), pad_left : pad_left + min(new_w, model_w)]

            final_size = (
                target_sizes[idx]
                if target_sizes is not None
                else (source_sizes[idx] if source_sizes is not None else None)
            )

            if final_size is not None:
                pm = F.interpolate(
                    pm.unsqueeze(0),
                    size=final_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]

            result.append(pm)

        return result

    def post_process_matting(
        self,
        outputs: Sapiens2MattingOutput,
        target_sizes: list[tuple] | None = None,
        backgrounds: ImageInput | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the output of [`Sapiens2ForMatting`] into alpha mattes and foreground maps.

        Args:
            outputs (`Sapiens2MattingOutput`):
                Raw outputs of the model.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                Requested final `(height, width)` for each prediction. Resized with bilinear
                interpolation. If unset, predictions are returned at the model output resolution.
            backgrounds (`ImageInput`, *optional*):
                Background image(s) to composite over. Can be a single image (applied to every item
                in the batch) or a list of images, one per batch item. Accepts PIL images, numpy
                arrays, or torch tensors of any dtype; integer types (e.g. uint8) are scaled to
                `[0, 1]` automatically. When provided, each result dict gains a `"composite"` key
                with the composited image as a uint8 tensor in `[0, 255]`.

        Returns:
            `list[dict]` of length `batch_size`. Each dict has:
            - `"alpha"` (`torch.Tensor` of shape `(1, height, width)`): alpha values in `[0, 1]`.
            - `"foreground"` (`torch.Tensor` of shape `(3, height, width)`): pre-multiplied RGB in `[0, 1]`.
            - `"composite"` (`torch.Tensor` of shape `(3, height, width)` or `None`): foreground composited
              over `backgrounds` as a uint8 tensor in `[0, 255]`; `None` when `backgrounds` is not provided.
        """
        matting = torch.cat([outputs.foregrounds, outputs.alphas], dim=1)  # (B, 4, H, W)

        if target_sizes is not None:
            if len(matting) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the matting output"
                )

        background_tensors = None
        if backgrounds is not None:
            background_list = make_list_of_images(backgrounds)
            if len(background_list) != 1 and len(background_list) != len(matting):
                raise ValueError(
                    "Make sure that you pass in as many backgrounds as the batch dimension of the matting output"
                )
            device = matting.device
            dtype = matting.dtype
            background_tensors = [
                tvF.to_dtype_image(tvF.to_image(bg), dtype=dtype, scale=True).to(device) for bg in background_list
            ]

        result = []
        for idx in range(len(matting)):
            mat = matting[idx]

            if target_sizes is not None:
                mat = F.interpolate(
                    mat.unsqueeze(0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]

            mat = mat.clamp(0.0, 1.0)
            foreground = mat[:3]
            alpha = mat[3:]
            composite = None

            if background_tensors is not None:
                background = background_tensors[0] if len(background_tensors) == 1 else background_tensors[idx]
                if background.shape[-2:] != mat.shape[-2:]:
                    background = F.interpolate(
                        background.unsqueeze(0),
                        size=mat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                        antialias=False,
                    )[0]
                composite = tvF.to_dtype_image(
                    (foreground + (1 - alpha) * background).clamp(0.0, 1.0), dtype=torch.uint8, scale=True
                )

            result.append({"foreground": foreground, "alpha": alpha, "composite": composite})

        return result


@auto_docstring(checkpoint="facebook/sapiens2-pretrain-0.4b")
@strict
class Sapiens2Config(DINOv3ViTConfig):
    r"""
    rope_theta (`float`, *optional*, defaults to 100.0):
        The base period of the RoPE embeddings.
    query_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the query projection.
    key_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the key projection.
    value_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the value projection.
    proj_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the output projection.
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value to use for layer scale.
    use_gated_mlp (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feedforward neural network.
    num_register_tokens (`int`, *optional*, defaults to 0):
        The number of register tokens.
    pos_embed_shift (`float`, *optional*):
        Amount to randomly shift position embedding coordinates in [-shift, shift],
        applied only in training mode if not `None`.
    pos_embed_jitter (`float`, *optional*):
        Amount to randomly jitter position embedding coordinates in log-uniform value in [1/jitter, jitter],
        applied only in training mode if not `None`.
    pos_embed_rescale (`float`, *optional*, defaults to 2.0):
        Amount to randomly rescale position embedding coordinates in log-uniform value in [1/rescale, rescale],
        applied only in training mode if not `None`.
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization to the feature maps when used as backbone.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the hidden states to spatial dimensions when used as backbone.
    use_mask_token (`bool`, *optional*, defaults to `False`):
        Whether to use a mask token in the embeddings (needed for masked image modeling pretraining).
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply RMSNorm to queries and keys before RoPE in attention layers.
    num_key_value_heads_per_layer (`list[int]`, *optional*):
        Number of key/value heads for each transformer layer. Setting a layer's value equal to
        `num_attention_heads` gives full multi-head attention; a smaller value gives grouped-query
        attention. Defaults to `num_attention_heads` for the first 8 and last 8 layers and
        `num_attention_heads // 2` for all other layers.
    pos_embed_dtype (`str`, *optional*, defaults to `"bfloat16"`):
        Dtype used for positional embedding computations (RoPE angles, cos/sin).
    semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
        Label index ignored when computing the segmentation loss.
    head_upsample_out_channels (`list[int]`, *optional*):
        Output channel counts for each upsample block in the decode head.
        The first block takes `hidden_size` channels as input; subsequent blocks use the previous output.
    head_upsample_kernel_sizes (`list[int]`, *optional*):
        Kernel size for each upsample block. Auto-filled with `[4, ...]` when
        `head_upsample_out_channels` is set but this is `None`.
        Must have the same length as `head_upsample_out_channels`.
    head_conv_out_channels (`list[int]`, *optional*):
        Output channel counts for the refinement conv layers that follow the upsample blocks.
    head_conv_kernel_sizes (`list[int]`, *optional*):
        Kernel size for each refinement conv layer. Auto-filled with `[1, ...]` when
        `head_conv_out_channels` is set but this is `None`.
        Must have the same length as `head_conv_out_channels`.
    head_scale_conv_out_channels (`list[int]`, *optional*):
        Output channel counts for the stride-2 conv layers used to predict the focal-length scale.
        When `None` (default), no scale branch is built.
    head_scale_conv_kernel_sizes (`list[int]`, *optional*):
        Kernel size for each scale conv layer. Auto-filled with `[1, ...]` when
        `head_scale_conv_out_channels` is set but this is `None`.
    head_scale_final_hidden_sizes (`list[int]`, *optional*):
        Hidden-layer sizes for the MLP that maps flattened scale features to the scalar scale output.
        When `None` (default), no scale branch is built.
    flip_pairs (`list[list[int]]`, *optional*):
        Pairs of keypoint indices that are left/right mirrors of each other (e.g. `[[1, 2], [3, 4]]`).
        Used when running the model on horizontally flipped images and averaging heatmaps for test-time
        augmentation.
    """

    model_type = "sapiens2"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    use_mask_token: bool = False
    use_gated_mlp: bool = True
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-6
    num_register_tokens: int = 8
    key_bias: bool = True
    use_qk_norm: bool = True
    num_key_value_heads_per_layer: list[int] | None = None
    pos_embed_dtype: str = "bfloat16"
    semantic_loss_ignore_index: int = 255
    head_upsample_out_channels: list[int] | None = None
    head_upsample_kernel_sizes: list[int] | None = None
    head_conv_out_channels: list[int] | None = None
    head_conv_kernel_sizes: list[int] | None = None
    head_scale_conv_out_channels: list[int] | None = None
    head_scale_conv_kernel_sizes: list[int] | None = None
    head_scale_final_hidden_sizes: list[int] | None = None
    flip_pairs: list[list[int]] | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads_per_layer is None:
            self.num_key_value_heads_per_layer = [
                self.num_attention_heads
                if (i < 8 or i >= self.num_hidden_layers - 8)
                else self.num_attention_heads // 2
                for i in range(self.num_hidden_layers)
            ]
        if self.head_upsample_out_channels is not None and self.head_upsample_kernel_sizes is None:
            self.head_upsample_kernel_sizes = [4] * len(self.head_upsample_out_channels)
        if self.head_conv_out_channels is not None and self.head_conv_kernel_sizes is None:
            self.head_conv_kernel_sizes = [1] * len(self.head_conv_out_channels)
        if self.head_scale_conv_out_channels is not None and self.head_scale_conv_kernel_sizes is None:
            self.head_scale_conv_kernel_sizes = [1] * len(self.head_scale_conv_out_channels)
        super().__post_init__(**kwargs)


class Sapiens2Embeddings(DINOv3ViTEmbeddings):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        if not config.use_mask_token:
            del self.mask_token

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        if bool_masked_pos is not None and not self.config.use_mask_token:
            raise ValueError("bool_masked_pos requires use_mask_token=True in the config")
        return super().forward(pixel_values, bool_masked_pos)


class Sapiens2RopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Sapiens2Config):
        super().__init__()

        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / self.base ** (
            2 * torch.arange(self.head_dim // 4, dtype=getattr(torch, config.pos_embed_dtype)) / (self.head_dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_size
        num_patches_w = width // self.config.patch_size

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with maybe_autocast(device_type=device_type, enabled=False):
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=self.inv_freq.dtype, device=device
            )
            if self.training:
                patch_coords = augment_patches_center_coordinates(
                    patch_coords,
                    shift=self.config.pos_embed_shift,
                    jitter=self.config.pos_embed_jitter,
                    rescale=self.config.pos_embed_rescale,
                )

            # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors, but only to the patch tokens,
    ignoring the prefix tokens (cls token and register tokens).

    Casts all q/k tokens to the rope dtype before applying the rotation and casts back afterwards.
    This matches the original model behavior where all tokens are cast to rope_dtype before the
    prefix/patch split, even though RoPE is only applied to patch tokens.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    rope_dtype = cos.dtype
    q = q.to(rope_dtype)
    k = k.to(rope_dtype)

    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_patches = q[..., num_prefix_tokens:, :]
    k_patches = k[..., num_prefix_tokens:, :]
    q[..., num_prefix_tokens:, :] = q_patches * cos + rotate_half(q_patches) * sin
    k[..., num_prefix_tokens:, :] = k_patches * cos + rotate_half(k_patches) * sin

    return q.to(q_dtype), k.to(k_dtype)


class Sapiens2Attention(DINOv3ViTAttention):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__(config)
        del self.k_proj
        del self.v_proj
        self.num_key_value_heads = config.num_key_value_heads_per_layer[layer_idx]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        kv_dim = self.num_key_value_heads * self.head_dim

        self.k_proj = nn.Linear(self.embed_dim, kv_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, kv_dim, bias=config.value_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps) if config.use_qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Sapiens2LayerScale(DINOv3ViTLayerScale):
    pass


class Sapiens2Layer(DINOv3ViTLayer):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__(config)
        self.attention = Sapiens2Attention(config, layer_idx=layer_idx)
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_scale2 = nn.Identity()


class Sapiens2ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(hidden_states)))


class Sapiens2PixelShuffleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale_factor: int = 2,
        padding: int = 1,
        bias: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels * scale_factor**2, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.pixel_shuffle(self.conv(hidden_states))))


class Sapiens2ConvLayer(BeitConvLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        stride: int = 1,
        padding: int | tuple[int, int] | str = 0,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        activation: str = "silu",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation,
            groups=groups,
            activation=activation,
        )
        self.normalization = nn.InstanceNorm2d(out_channels)


class Sapiens2SegmentationHead(nn.Module):
    def __init__(self, config: Sapiens2Config):
        super().__init__()
        upsample_in_channels = [config.hidden_size] + config.head_upsample_out_channels[:-1]
        self.deconv_layers = nn.ModuleList(
            Sapiens2ConvTransposeLayer(in_ch, out_ch, kernel_size=ks)
            for in_ch, out_ch, ks in zip(
                upsample_in_channels, config.head_upsample_out_channels, config.head_upsample_kernel_sizes
            )
        )
        conv_in_channels = [config.head_upsample_out_channels[-1]] + config.head_conv_out_channels[:-1]
        self.conv_layers = nn.ModuleList(
            Sapiens2ConvLayer(in_ch, out_ch, kernel_size=ks)
            for in_ch, out_ch, ks in zip(
                conv_in_channels, config.head_conv_out_channels, config.head_conv_kernel_sizes
            )
        )
        predictor_in = (
            config.head_conv_out_channels[-1]
            if config.head_conv_out_channels
            else config.head_upsample_out_channels[-1]
        )
        self.predictor = nn.Conv2d(predictor_in, config.num_labels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.deconv_layers:
            hidden_states = layer(hidden_states)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return self.predictor(hidden_states)


class Sapiens2PointmapHead(nn.Module):
    def __init__(self, config: Sapiens2Config):
        super().__init__()
        self.input_conv = Sapiens2ConvLayer(config.hidden_size, config.hidden_size, kernel_size=3, padding=1)
        upsample_in_channels = [config.hidden_size] + config.head_upsample_out_channels[:-1]
        self.upsample_layers = nn.ModuleList(
            Sapiens2PixelShuffleLayer(in_ch, out_ch, kernel_size=ks, padding=(ks - 1) // 2)
            for in_ch, out_ch, ks in zip(
                upsample_in_channels, config.head_upsample_out_channels, config.head_upsample_kernel_sizes
            )
        )
        conv_in_channels = [config.head_upsample_out_channels[-1]] + config.head_conv_out_channels[:-1]
        self.conv_layers = nn.ModuleList(
            Sapiens2ConvLayer(in_ch, out_ch, kernel_size=ks, padding=(ks - 1) // 2)
            for in_ch, out_ch, ks in zip(
                conv_in_channels, config.head_conv_out_channels, config.head_conv_kernel_sizes
            )
        )
        predictor_in = (
            config.head_conv_out_channels[-1]
            if config.head_conv_out_channels
            else config.head_upsample_out_channels[-1]
        )
        self.predictor = nn.Conv2d(predictor_in, config.num_labels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_conv(hidden_states)
        for block in self.upsample_layers:
            hidden_states = block(hidden_states)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return self.predictor(hidden_states)


class Sapiens2PointmapFinalLayer(nn.Module):
    def __init__(self, in_size: int, hidden_sizes: list[int]):
        super().__init__()
        layers = [nn.Flatten()]
        in_channels = [in_size] + hidden_sizes[:-1]
        for in_ch, out_ch in zip(in_channels, hidden_sizes):
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class Sapiens2PointmapScaleHead(nn.Module):
    def __init__(self, config: Sapiens2Config):
        super().__init__()
        image_size = config.image_size
        image_h, image_w = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        patch_size = config.patch_size if isinstance(config.patch_size, int) else config.patch_size[0]
        h = image_h // patch_size
        w = image_w // patch_size
        for ks in config.head_scale_conv_kernel_sizes:
            padding = (ks - 1) // 2
            h = (h + 2 * padding - ks) // 2 + 1
            w = (w + 2 * padding - ks) // 2 + 1
        flat_size = h * w * config.head_scale_conv_out_channels[-1]

        self.conv_layers = nn.ModuleList()
        scale_in_channels = [config.hidden_size] + config.head_scale_conv_out_channels[:-1]
        for in_ch, out_ch, ks in zip(
            scale_in_channels,
            config.head_scale_conv_out_channels,
            config.head_scale_conv_kernel_sizes,
        ):
            self.conv_layers.append(Sapiens2ConvLayer(in_ch, out_ch, kernel_size=ks, stride=2, padding=(ks - 1) // 2))
        self.predictor = Sapiens2PointmapFinalLayer(flat_size, config.head_scale_final_hidden_sizes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return self.predictor(hidden_states)


class Sapiens2PreTrainedModel(DINOv3ViTPreTrainedModel):
    base_model_prefix = "sapiens2"
    _keys_to_ignore_on_load_unexpected = [r"periods"]

    @torch.no_grad()
    def _init_weights(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Backbone layers
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, Sapiens2Embeddings):
            init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
            if module.config.num_register_tokens > 0:
                init.trunc_normal_(module.register_tokens, mean=0.0, std=self.config.initializer_range)
            if module.config.use_mask_token:
                init.zeros_(module.mask_token)
        elif isinstance(module, Sapiens2LayerScale):
            init.constant_(module.lambda1, self.config.layerscale_value)
        elif isinstance(module, Sapiens2RopePositionEmbedding):
            inv_freq = 1 / module.base ** (
                2 * torch.arange(module.head_dim // 4, dtype=module.inv_freq.dtype) / (module.head_dim // 2)
            )
            init.copy_(module.inv_freq, inv_freq)
        elif isinstance(
            module,
            (Sapiens2SegmentationHead, Sapiens2PointmapHead, Sapiens2PointmapScaleHead),
        ):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                    if m.bias is not None:
                        init.zeros_(m.bias)


class Sapiens2Encoder(DINOv3ViTEncoder):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([Sapiens2Layer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.post_init()


class Sapiens2Model(DINOv3ViTModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()


class Sapiens2Backbone(DINOv3ViTBackbone):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()


@auto_docstring(checkpoint="facebook/sapiens2-seg-0.4b")
class Sapiens2ForSemanticSegmentation(Sapiens2PreTrainedModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.sapiens2 = Sapiens2Model(config)
        self.decode_head = Sapiens2SegmentationHead(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SemanticSegmenterOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
        """
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.sapiens2(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size

        patch_tokens = outputs.last_hidden_state[:, 1 + self.config.num_register_tokens :]
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, -1, patch_height, patch_width)

        logits = self.decode_head(feature_map)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, ignore_index=self.config.semantic_loss_ignore_index)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    checkpoint="facebook/sapiens2-pose-0.4b",
    custom_intro="""
    The Sapiens2 model with a pose estimation head on top (a set of heatmap predictors on top of the hidden states output).
    """,
)
class Sapiens2ForPoseEstimation(Sapiens2PreTrainedModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.sapiens2 = Sapiens2Model(config)
        self.decode_head = Sapiens2SegmentationHead(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        flip_pairs: torch.Tensor | None = None,
        labels: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sapiens2PoseEstimatorOutput:
        r"""
        flip_pairs (`torch.Tensor` of shape `(num_pairs, 2)`, *optional*):
            Pairs of keypoints which are mirrored (for example, left ear -- right ear), used for
            test-time flip augmentation. When provided, the model assumes `pixel_values` contains
            horizontally-flipped images and calls `flip_back` on the output heatmaps to restore the
            original orientation.

            Typical usage: run a second forward pass on `pixel_values.flip(-1)` with this argument,
            then average the two heatmap outputs:

            ```python
            outputs = model(pixel_values)
            outputs_flipped = model(pixel_values.flip(-1), flip_pairs=flip_pairs)
            heatmaps = (outputs.heatmaps + outputs_flipped.heatmaps) / 2
            ```
        labels (`torch.FloatTensor` of shape `(batch_size, num_keypoints, height, width)`, *optional*):
            Heatmap ground truth for computing the loss.
        """
        outputs = self.sapiens2(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size

        patch_tokens = outputs.last_hidden_state[:, 1 + self.config.num_register_tokens :]
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, -1, patch_height, patch_width)

        heatmaps = self.decode_head(feature_map)
        if flip_pairs is not None:
            heatmaps = flip_back(heatmaps, flip_pairs)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return Sapiens2PoseEstimatorOutput(
            loss=loss,
            heatmaps=heatmaps,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    checkpoint="facebook/sapiens2-normal-0.4b",
    custom_intro="""
    The Sapiens2 model with a normal estimation head on top (a PixelShuffle-based decoder that predicts surface normal maps).
    """,
)
class Sapiens2ForNormalEstimation(Sapiens2PreTrainedModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.sapiens2 = Sapiens2Model(config)
        self.decode_head = Sapiens2PointmapHead(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sapiens2NormalEstimatorOutput:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, num_labels, height, width)`, *optional*):
            Ground-truth surface normal maps for computing the loss.
        """
        outputs = self.sapiens2(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size

        patch_tokens = outputs.last_hidden_state[:, 1 + self.config.num_register_tokens :]
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, -1, patch_height, patch_width)

        normals = self.decode_head(feature_map)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return Sapiens2NormalEstimatorOutput(
            loss=loss,
            normals=normals,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    checkpoint="facebook/sapiens2-pointmap-0.4b",
    custom_intro="""
    The Sapiens2 model with a pointmap head on top (a PixelShuffle-based decoder that predicts per-pixel 3D XYZ
    coordinates, plus an optional scale branch for focal-length normalization).
    """,
)
class Sapiens2ForPointmapEstimation(Sapiens2PreTrainedModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.sapiens2 = Sapiens2Model(config)
        self.decode_head = Sapiens2PointmapHead(config)
        self.scale_head = (
            Sapiens2PointmapScaleHead(config) if config.head_scale_conv_out_channels is not None else nn.Identity()
        )
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sapiens2PointmapEstimatorOutput:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`, *optional*):
            Ground-truth pointmap for computing the loss.
        """
        outputs = self.sapiens2(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size

        patch_tokens = outputs.last_hidden_state[:, 1 + self.config.num_register_tokens :]
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, -1, patch_height, patch_width)

        pointmaps = self.decode_head(feature_map)
        scales = None if isinstance(self.scale_head, nn.Identity) else self.scale_head(feature_map)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return Sapiens2PointmapEstimatorOutput(
            loss=loss,
            pointmaps=pointmaps,
            scales=scales,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    checkpoint="facebook/sapiens2-matting-1b",
    custom_intro="""
    The Sapiens2 model with a matting head on top (a PixelShuffle-based decoder that predicts a
    pre-multiplied RGB foreground and an alpha matte).
    """,
)
class Sapiens2ForMatting(Sapiens2PreTrainedModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.sapiens2 = Sapiens2Model(config)
        self.decode_head = Sapiens2PointmapHead(config)  # config.num_labels = 4
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sapiens2MattingOutput:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, 4, height, width)`, *optional*):
            Ground-truth matting targets for computing the loss.
        """
        outputs = self.sapiens2(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size

        patch_tokens = outputs.last_hidden_state[:, 1 + self.config.num_register_tokens :]
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, -1, patch_height, patch_width)

        matting = self.decode_head(feature_map).sigmoid()  # (B, 4, H, W)
        foregrounds = matting[:, :3]  # (B, 3, H, W)
        alphas = matting[:, 3:]  # (B, 1, H, W)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return Sapiens2MattingOutput(
            loss=loss,
            alphas=alphas,
            foregrounds=foregrounds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Sapiens2Config",
    "Sapiens2ForSemanticSegmentation",
    "Sapiens2ForPoseEstimation",
    "Sapiens2ForNormalEstimation",
    "Sapiens2ForPointmapEstimation",
    "Sapiens2ForMatting",
    "Sapiens2Model",
    "Sapiens2PreTrainedModel",
    "Sapiens2Backbone",
    "Sapiens2ImageProcessor",
]

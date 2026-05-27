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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from transformers.image_processing_backends import TorchvisionBackend

from ... import initialization as init
from ...activations import ACT2FN
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
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..beit.image_processing_beit import BeitImageProcessor, BeitImageProcessorKwargs
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
    DINOv3ViTRopePositionEmbedding,
    apply_rotary_pos_emb,
)
from ..gemma2.modeling_gemma2 import eager_attention_forward
from ..sam3.processing_sam3 import box_xywh_to_cxcywh, box_xywh_to_xyxy
from ..vitmatte.modeling_vitmatte import ImageMattingOutput
from ..vitpose.modeling_vitpose import VitPoseEstimatorOutput, flip_back


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Class for outputs of pose estimation models.
    """
)
@dataclass
class Sapiens2PoseEstimatorOutput(VitPoseEstimatorOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Pose estimation loss.
    heatmaps (`torch.FloatTensor` of shape `(batch_size, num_keypoints, height, width)`):
        Heatmaps as predicted by the model.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
        (also called feature maps) of the model at the output of each stage.
    """


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
    Class for outputs of image matting models.
    """
)
@dataclass
class Sapiens2ImageMattingOutput(ImageMattingOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Loss.
    alphas (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`):
        Estimated alpha values.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
        (also called feature maps) of the model at the output of each stage.
    foregrounds (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
        Pre-multiplied RGB foreground predictions in `[0, 1]` (sigmoid-activated).
    """

    foregrounds: torch.FloatTensor | None = None


def boxes_to_crop_params(
    boxes: torch.Tensor,
    output_size: tuple[int, int],
    padding: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute crop center and scale from bboxes, applying padding and aspect ratio correction.

    Accepts either a single box `(4,)` or multiple boxes `(num_boxes, 4)` and returns center/scale with a matching
    leading dimension.

    Args:
        boxes (`torch.Tensor` of shape `(4,)` or `(num_boxes, 4)`): Bounding box in
            (center-x, center-y, width, height) format, with values in absolute pixel coordinates.
        output_size (`tuple[int, int]`): Target output size as `(height, width)`, used to compute
            the aspect ratio for scale correction.
        padding (`float`, *optional*, defaults to `1.25`): Multiplicative factor applied to the
            bounding box dimensions, adding context around the region of interest.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: A pair `(center, scale)` where `center` has shape
        `(..., 2)` with (x, y) in input-image pixel coordinates, and `scale` has shape `(..., 2)`
        with (width, height) in input-image pixels representing the dimensions of the padded,
        aspect-ratio-corrected crop window.
    """
    center_x, center_y, width, height = boxes.unbind(-1)
    center = torch.stack([center_x, center_y], dim=-1)
    scaled_width = width * padding
    scaled_height = height * padding
    output_height, output_width = output_size
    aspect_ratio = output_width / output_height
    scale = torch.where(
        (scaled_width > scaled_height * aspect_ratio)[..., None],
        torch.stack([scaled_width, scaled_width / aspect_ratio], dim=-1),
        torch.stack([scaled_height * aspect_ratio, scaled_height], dim=-1),
    )
    return center, scale


def crop_and_resize(
    image: torch.Tensor,
    boxes: torch.Tensor,
    output_size: tuple[int, int],
    padding: float = 1.25,
) -> torch.Tensor:
    """Crops and resizes bbox regions from the input image to the target output size.

    Applies padding and aspect ratio correction to each crop before resizing.
    Uses bilinear interpolation for downscaling and bicubic for upscaling.

    This implementation is equivalent to the cv2 affine warp with rotation=0 used in the original
    Sapiens2 codebase. Rotation is always zero because we don't support rotated bounding boxes.

    Args:
        image (`torch.Tensor`): Input image tensor of shape `(C, H, W)` in float32.
        boxes (`torch.Tensor`): Bounding boxes in (center-x, center-y, width, height) format,
            shape `(num_boxes, 4)`, with values in absolute pixel coordinates.
        output_size (`tuple[int, int]`): Target output size as `(height, width)`.
        padding (`float`, *optional*, defaults to `1.25`): Multiplicative factor applied to the
            bounding box dimensions before cropping, adding context around the region of interest.

    Returns:
        `torch.Tensor`: Cropped and resized images of shape `(num_boxes, C, output_height, output_width)`.
    """
    output_height, output_width = output_size
    num_channels, input_height, input_width = image.shape
    center, scale = boxes_to_crop_params(boxes, output_size=output_size, padding=padding)
    center_x, center_y = center.unbind(-1)
    bbox_w, bbox_h = scale.unbind(-1)

    scale_x = (output_width - 1) / bbox_w  # (num_boxes,)
    scale_y = (output_height - 1) / bbox_h  # (num_boxes,)
    is_bilinear = torch.minimum(scale_x, scale_y) < 1.0  # (num_boxes,)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(output_height, dtype=torch.float32, device=image.device),
        torch.arange(output_width, dtype=torch.float32, device=image.device),
        indexing="ij",
    )
    in_x = grid_x / scale_x[:, None, None] + center_x[:, None, None] - 0.5 * bbox_w[:, None, None]
    in_y = grid_y / scale_y[:, None, None] + center_y[:, None, None] - 0.5 * bbox_h[:, None, None]
    # (num_boxes, output_height, output_width, 2)
    grids = torch.stack([2.0 * in_x / (input_width - 1) - 1.0, 2.0 * in_y / (input_height - 1) - 1.0], dim=-1)

    num_boxes = boxes.shape[0]
    output = torch.empty(num_boxes, num_channels, output_height, output_width, device=image.device, dtype=image.dtype)

    # Apply grid sampling separately for upscaling and downscaling to use the appropriate interpolation mode
    image_4d = image.unsqueeze(0)
    for mask, mode in [(is_bilinear, "bilinear"), (~is_bilinear, "bicubic")]:
        if mask.any():
            output[mask] = F.grid_sample(
                image_4d.expand(mask.sum(), -1, -1, -1),
                grids[mask],
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )

    return output


def gaussian_blur_preserve_max(heatmaps: torch.Tensor, kernel: int = 11) -> torch.Tensor:
    """Gaussian blur per-keypoint heatmap, preserving the original max value.

    Matches cv2.GaussianBlur with sigma=0 which means that the sigma is automatically
    computed from the kernel size.

    Args:
        heatmaps: Shape `(K, height, width)`.
        kernel: Odd integer kernel size for the Gaussian blur. Must be greater than 1.

    Returns:
        `torch.Tensor`: Blurred heatmaps of the same shape as the input.
    """
    if kernel % 2 == 0 or kernel <= 1:
        raise ValueError("Kernel size must be an odd integer greater than 1.")
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    border = (kernel - 1) // 2
    origin_maxes = heatmaps.amax(dim=(1, 2))  # (K,)

    # Padding required to prevent border effect from gaussian blur. Torchvision uses reflect padding internally.
    padded = F.pad(heatmaps, (border, border, border, border), mode="constant", value=0.0)
    blurred = tvF.gaussian_blur(padded, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    result = blurred[:, border:-border, border:-border]

    result_maxes = result.amax(dim=(1, 2))  # (K,)
    safe_maxes = torch.where(result_maxes > 0, result_maxes, torch.ones_like(result_maxes))
    scale = torch.where(result_maxes > 0, origin_maxes / safe_maxes, torch.ones_like(origin_maxes))
    return result * scale[:, None, None]


def get_keypoint_predictions(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict keypoint locations and confidence scores from heatmaps.

    Args:
        heatmaps: Shape `(num_persons, num_keypoints, height, width)`.

    Returns:
        locations: `(num_persons, num_keypoints, 2)` x/y in heatmap pixel coordinates.
        scores: `(num_persons, num_keypoints)` per-keypoint confidence.
    """
    num_persons, num_keypoints, _, heatmap_width = heatmaps.shape
    device = heatmaps.device
    heatmap_flat = heatmaps.reshape(num_persons, num_keypoints, -1)
    scores = heatmap_flat.amax(dim=-1)
    flat_index = heatmap_flat.argmax(dim=-1)
    locations_x = (flat_index % heatmap_width).float()
    locations_y = (flat_index // heatmap_width).float()
    locations = torch.where(
        scores.unsqueeze(-1) > 0.0,
        torch.stack([locations_x, locations_y], dim=-1),
        torch.full((num_persons, num_keypoints, 2), -1.0, device=device),
    )
    return locations, scores


def post_dark_unbiased_data_processing(
    keypoints: torch.Tensor, heatmaps: torch.Tensor, blur_kernel_size: int = 11
) -> torch.Tensor:
    """Sub-pixel refinement via Hessian on log-heatmaps (UDP Dark Pose).

    Args:
        keypoints: Shape `(num_persons, num_keypoints, 2)` x/y in heatmap pixel coordinates.
        heatmaps: Shape `(num_persons, num_keypoints, height, width)`.

    Returns:
        `(num_persons, num_keypoints, 2)` refined keypoint locations.
    """
    num_persons, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape
    device = heatmaps.device

    heatmaps = gaussian_blur_preserve_max(
        heatmaps.reshape(num_persons * num_keypoints, heatmap_height, heatmap_width), blur_kernel_size
    ).reshape(num_persons, num_keypoints, heatmap_height, heatmap_width)
    heatmaps = heatmaps.clamp(1e-3, 50.0).log()  # Clamp values based on original Sapiens2 implementation

    heatmaps_padded = F.pad(heatmaps, (1, 1, 1, 1), mode="replicate")
    heatmaps_flattened = heatmaps_padded.flatten()

    padded_height = heatmap_height + 2
    padded_width = heatmap_width + 2
    keypoint_stride = padded_height * padded_width
    person_stride = num_keypoints * keypoint_stride

    index = keypoints[:, :, 0].long() + 1 + (keypoints[:, :, 1].long() + 1) * padded_width
    index = index + keypoint_stride * torch.arange(num_keypoints, device=device, dtype=torch.long)[None, :]
    index = index + person_stride * torch.arange(num_persons, device=device, dtype=torch.long)[:, None]
    index = index.unsqueeze(-1)  # (num_persons, num_keypoints, 1)

    position_to_index_offset = {
        (0, 0): 0,
        (0, 1): 1,
        (0, -1): -1,
        (1, 0): padded_width,
        (-1, 0): -padded_width,
        (1, 1): padded_width + 1,
        (-1, -1): -(padded_width + 1),
    }
    # Dict mapping from (dx, dy) offsets to the corresponding values in the heatmap
    h = {(dx, dy): heatmaps_flattened[index + offset] for (dx, dy), offset in position_to_index_offset.items()}

    gradient_x = 0.5 * (h[0, 1] - h[0, -1])
    gradient_y = 0.5 * (h[1, 0] - h[-1, 0])
    derivative = torch.cat([gradient_x, gradient_y], dim=-1).reshape(num_persons, num_keypoints, 2, 1)

    hessian_xx = h[0, 1] - 2 * h[0, 0] + h[0, -1]
    hessian_yy = h[1, 0] - 2 * h[0, 0] + h[-1, 0]
    hessian_xy = 0.5 * (h[1, 1] - h[0, 1] - h[1, 0] + h[0, 0] + h[0, 0] - h[0, -1] - h[-1, 0] + h[-1, -1])
    hessian = torch.cat([hessian_xx, hessian_xy, hessian_xy, hessian_yy], dim=-1).reshape(
        num_persons, num_keypoints, 2, 2
    )
    hessian = torch.linalg.inv(hessian + torch.finfo(hessian.dtype).eps * torch.eye(2, device=device))
    return keypoints - (hessian @ derivative).squeeze(-1)


class Sapiens2ImageProcessorKwargs(BeitImageProcessorKwargs, total=False):
    pass


class Sapiens2ImageProcessor(BeitImageProcessor):
    valid_kwargs = Sapiens2ImageProcessorKwargs
    # Note: original Sapiens2 uses cv2.INTER_AREA for downsampling and cv2.INTER_CUBIC for upsampling
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 1024, "width": 768}
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
        """
        return TorchvisionBackend.preprocess(images, segmentation_maps, boxes, **kwargs)

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
        kwargs["boxes"] = boxes
        return super()._preprocess_image_like_inputs(
            self,
            images=images,
            segmentation_maps=segmentation_maps,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            return_tensors=return_tensors,
            device=device,
            **kwargs,
        )

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
            output_size = (size["height"], size["width"])
            crops = []
            for image, image_boxes in zip(images, boxes):
                image = tvF.to_dtype_image(image, dtype=torch.float32, scale=False)
                boxes_cxcywh = box_xywh_to_cxcywh(torch.tensor(image_boxes, dtype=torch.float32, device=image.device))
                crops.extend(crop_and_resize(image, boxes=boxes_cxcywh, output_size=output_size))
            images = crops
            do_resize = False  # crop_and_resize already produces the target size

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
        boxes: list[list[list[float]]],
        outputs_flipped: Sapiens2PoseEstimatorOutput | None = None,
        kernel_size: int = 11,
        threshold: float | None = None,
        source_sizes: TensorType | list[tuple[int, int]] | None = None,
        target_sizes: TensorType | list[tuple[int, int]] | None = None,
    ) -> list[list[dict[str, torch.Tensor]]]:
        """
        Converts the output of [`Sapiens2ForPoseEstimation`] into keypoint predictions in image space.

        Args:
            outputs (`Sapiens2PoseEstimatorOutput`):
                Raw outputs of the model. `outputs.heatmaps` must have shape
                `(N_total, num_keypoints, heatmap_height, heatmap_width)` where
                `N_total = sum(len(b) for b in boxes)`.
            boxes (`list[list[list[float]]]` or `np.ndarray`):
                List or array of bounding boxes for each image in absolute pixel coordinates. Each box
                should be a list of 4 floats representing the bounding box coordinates in COCO format
                (top_left_x, top_left_y, width, height). Must match the `boxes` argument passed to
                `preprocess`.
            outputs_flipped (`Sapiens2PoseEstimatorOutput`, *optional*):
                Outputs from running the model on horizontally flipped inputs. When provided, heatmaps
                are averaged with `outputs` before keypoint extraction to improve accuracy:
                `avg_heatmaps = (outputs.heatmaps + outputs_flipped.heatmaps) / 2`.
            kernel_size (`int`, *optional*, defaults to 11):
                Kernel size for the Gaussian blur used in UDP Dark Pose refinement.
            threshold (`float`, *optional*):
                Score threshold. Keypoints with scores at or below this value are
                filtered out from the result dictionaries.
            source_sizes (`torch.Tensor` or `list[tuple[int, int]]` of length `batch_size`, *optional*):
                Original `(height, width)` of each image in pixels. Required when `target_sizes` is
                provided, as the source coordinate space for scaling keypoints and bounding boxes.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]` of length `batch_size`, *optional*):
                Desired output `(height, width)` coordinate space for each image. When provided
                alongside `source_sizes`, keypoint coordinates and bounding boxes are scaled from
                source to target space.

        Returns:
            `list[list[dict]]`: Outer list is over images, inner list is over persons.
            Each dict contains:
            - `keypoints` (`torch.FloatTensor` of shape `(num_keypoints, 2)`): absolut x/y coordinates in
              the source image space, or in target space if `target_sizes` is provided.
            - `scores` (`torch.FloatTensor` of shape `(num_keypoints,)`): per-keypoint confidence.
            - `labels` (`torch.LongTensor` of shape `(num_keypoints,)`): keypoint indices.
            - `bbox` (`torch.FloatTensor` of shape `(4,)`): bounding box in absolute (x_min, y_min, x_max, y_max)
               format, in the same coordinate space as `keypoints`.
        """
        if isinstance(source_sizes, torch.Tensor):
            source_sizes = source_sizes.tolist()
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = target_sizes.tolist()

        num_images = len(boxes)

        if target_sizes is not None and source_sizes is None:
            raise ValueError("`source_sizes` must be provided when `target_sizes` is specified.")
        if source_sizes is not None and num_images != len(source_sizes):
            raise ValueError("Make sure that you pass in as many source sizes as the number of images.")
        if target_sizes is not None and num_images != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the number of images.")

        heatmaps = outputs.heatmaps  # (num_total_persons, num_keypoints, heatmap_height, heatmap_width)
        if outputs_flipped is not None:
            heatmaps = (heatmaps + outputs_flipped.heatmaps) / 2

        device = heatmaps.device
        num_total_persons, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape

        if num_total_persons == 0:
            return [[] for _ in boxes]

        # (num_total_persons, 4)
        boxes_xywh = torch.tensor(
            [bbox for image_boxes in boxes for bbox in image_boxes], dtype=torch.float32, device=device
        )

        heatmaps = heatmaps.float()  # For consistency with original numpy/cv2 implementation which uses float32.

        # (num_total_persons, num_keypoints, 2), (num_total_persons, num_keypoints)
        all_keypoints, all_scores = get_keypoint_predictions(heatmaps)
        all_keypoints = post_dark_unbiased_data_processing(
            keypoints=all_keypoints, heatmaps=heatmaps, blur_kernel_size=kernel_size
        )

        # Remap coordinates from heatmap space to original image space
        boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
        centers, scales = boxes_to_crop_params(boxes_cxcywh, output_size=(self.size["height"], self.size["width"]))
        heatmap_size = torch.tensor([heatmap_width - 1, heatmap_height - 1], dtype=torch.float32, device=device)
        all_keypoints = (
            all_keypoints / heatmap_size * scales[:, None, :] + centers[:, None, :] - 0.5 * scales[:, None, :]
        )
        all_boxes = box_xywh_to_xyxy(boxes_xywh)  # (num_total_persons, 4)

        if source_sizes is not None and target_sizes is not None:
            # (num_images, 2)
            per_image_scale = torch.tensor(
                [
                    [target_width / source_width, target_height / source_height]
                    for (source_height, source_width), (target_height, target_width) in zip(source_sizes, target_sizes)
                ],
                dtype=torch.float32,
                device=device,
            )
            # (num_total_persons, 2)
            per_person_scale = torch.cat(
                [per_image_scale[i].unsqueeze(0).expand(len(boxes[i]), 2) for i in range(num_images)]
            )
            all_keypoints = all_keypoints * per_person_scale[:, None, :]
            all_boxes = all_boxes * per_person_scale[:, [0, 1, 0, 1]]

        person_results = []
        for person_index in range(num_total_persons):
            keypoints = all_keypoints[person_index]
            scores = all_scores[person_index]
            labels = torch.arange(num_keypoints, device=device)

            if threshold is not None:
                keep = scores > threshold
                keypoints = keypoints[keep]
                scores = scores[keep]
                labels = labels[keep]

            person_results.append(
                {"keypoints": keypoints, "scores": scores, "labels": labels, "bbox": all_boxes[person_index]}
            )

        # Reassemble into list[list[dict]] grouped by image
        result = []
        person_offset = 0
        for image_boxes in boxes:
            num_persons_in_image = len(image_boxes)
            result.append(person_results[person_offset : person_offset + num_persons_in_image])
            person_offset += num_persons_in_image
        return result

    def post_process_normal_estimation(
        self,
        outputs: Sapiens2NormalEstimatorOutput,
        source_sizes: TensorType | list[tuple[int, int]] | None = None,
        target_sizes: TensorType | list[tuple[int, int]] | None = None,
        do_remove_padding: bool | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the output of [`Sapiens2ForNormalEstimation`] into L2-normalized surface normal maps.

        Args:
            outputs (`Sapiens2NormalEstimatorOutput`):
                Raw outputs of the model.
            source_sizes (`torch.Tensor` or `list[tuple[int, int]]` of length `batch_size`, *optional*):
                Original `(height, width)` of each image before preprocessing. When provided,
                the padding added during preprocessing is removed and predictions are resized back
                to the original image size (unless `target_sizes` overrides the final size).
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]` of length `batch_size`, *optional*):
                Requested final `(height, width)` for each prediction. When provided, used as the
                resize target instead of `source_sizes`. Resized with bilinear interpolation after
                L2 normalization.
            do_remove_padding (`bool`, *optional*):
                Whether to crop away the zero-padding added during preprocessing before resizing.
                Defaults to `True` when `source_sizes` is provided, `False` otherwise.

        Returns:
            `list[dict[str, torch.Tensor]]` of length `batch_size`. Each dict has a `"normals"` key
            mapping to a tensor of shape `(3, height, width)` with L2-normalized unit vectors in
            `[-1, 1]` per channel (XYZ surface normals).
        """
        if isinstance(source_sizes, torch.Tensor):
            source_sizes = source_sizes.tolist()
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = target_sizes.tolist()

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
        model_height = self.size["height"]
        model_width = self.size["width"]

        normals = F.normalize(normals, p=2, dim=1, eps=1e-8)

        for index in range(len(normals)):
            normal = normals[index]

            if do_remove_padding:
                original_height, original_width = source_sizes[index]
                new_height, new_width = get_image_size_for_max_height_width(
                    (original_height, original_width), model_height, model_width
                )
                pad_top = (model_height - new_height) // 2 if new_height < model_height else 0
                pad_left = (model_width - new_width) // 2 if new_width < model_width else 0
                normal = normal[
                    :,
                    pad_top : pad_top + min(new_height, model_height),
                    pad_left : pad_left + min(new_width, model_width),
                ]

            final_size = (
                target_sizes[index]
                if target_sizes is not None
                else (source_sizes[index] if source_sizes is not None else None)
            )

            if final_size is not None:
                normal = F.interpolate(
                    normal.unsqueeze(0),
                    size=final_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]

            result.append({"normals": normal})

        return result

    def post_process_pointmap(
        self,
        outputs: Sapiens2PointmapEstimatorOutput,
        source_sizes: TensorType | list[tuple[int, int]] | None = None,
        target_sizes: TensorType | list[tuple[int, int]] | None = None,
        do_remove_padding: bool | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the output of [`Sapiens2ForPointmapEstimation`] into pointmap tensors in image space.

        Args:
            outputs (`Sapiens2PointmapEstimatorOutput`):
                Raw outputs of the model.
            source_sizes (`torch.Tensor` or `list[tuple[int, int]]` of length `batch_size`, *optional*):
                Original `(height, width)` of each image before preprocessing. When provided,
                the padding added during preprocessing is removed and predictions are resized back
                to the original image size (unless `target_sizes` overrides the final size).
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]` of length `batch_size`, *optional*):
                Requested final `(height, width)` for each prediction. Overrides `source_sizes`
                as the resize target.
            do_remove_padding (`bool`, *optional*):
                Whether to crop away the zero-padding added during preprocessing before resizing.
                Defaults to `True` when `source_sizes` is provided, `False` otherwise.

        Returns:
            `list[dict[str, torch.Tensor]]` of length `batch_size`. Each dict has a `"pointmap"` key
            mapping to a tensor of shape `(3, height, width)` with per-pixel 3D XYZ coordinates in
            canonical camera space, optionally divided by `outputs.scales` to convert to metric coordinates.
        """
        if isinstance(source_sizes, torch.Tensor):
            source_sizes = source_sizes.tolist()
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = target_sizes.tolist()

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
        model_height = self.size["height"]
        model_width = self.size["width"]

        for index in range(len(pointmaps)):
            pointmap = pointmaps[index]

            if outputs.scales is not None:
                pointmap = pointmap / outputs.scales[index]

            if do_remove_padding:
                original_height, original_width = source_sizes[index]
                new_height, new_width = get_image_size_for_max_height_width(
                    (original_height, original_width), model_height, model_width
                )
                pad_top = (model_height - new_height) // 2 if new_height < model_height else 0
                pad_left = (model_width - new_width) // 2 if new_width < model_width else 0
                pointmap = pointmap[
                    :,
                    pad_top : pad_top + min(new_height, model_height),
                    pad_left : pad_left + min(new_width, model_width),
                ]

            final_size = (
                target_sizes[index]
                if target_sizes is not None
                else (source_sizes[index] if source_sizes is not None else None)
            )

            if final_size is not None:
                pointmap = F.interpolate(
                    pointmap.unsqueeze(0),
                    size=final_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]

            result.append({"pointmap": pointmap})

        return result

    def post_process_image_matting(
        self,
        outputs: Sapiens2ImageMattingOutput,
        target_sizes: TensorType | list[tuple[int, int]] | None = None,
        backgrounds: ImageInput | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the output of [`Sapiens2ForImageMatting`] into alpha mattes and foreground maps.

        Args:
            outputs (`Sapiens2ImageMattingOutput`):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]` of length `batch_size`, *optional*):
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
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = target_sizes.tolist()

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
                tvF.to_dtype_image(tvF.to_image(background_image), dtype=dtype, scale=True).to(device)
                for background_image in background_list
            ]

        result = []
        for index in range(len(matting)):
            matting_item = matting[index]

            if target_sizes is not None:
                matting_item = F.interpolate(
                    matting_item.unsqueeze(0),
                    size=target_sizes[index],
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]

            matting_item = matting_item.clamp(0.0, 1.0)
            foreground = matting_item[:3]
            alpha = matting_item[3:]
            composite = None

            if background_tensors is not None:
                background = background_tensors[0] if len(background_tensors) == 1 else background_tensors[index]
                if background.shape[-2:] != matting_item.shape[-2:]:
                    background = F.interpolate(
                        background.unsqueeze(0),
                        size=matting_item.shape[-2:],
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
    semantic_loss_ignore_index: int = 255
    head_upsample_out_channels: list[int] | None = None
    head_upsample_kernel_sizes: list[int] | None = None
    head_conv_out_channels: list[int] | None = None
    head_conv_kernel_sizes: list[int] | None = None
    head_scale_conv_out_channels: list[int] | None = None
    head_scale_conv_kernel_sizes: list[int] | None = None
    head_scale_final_hidden_sizes: list[int] | None = None

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


class Sapiens2RopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    def __init__(self, config: Sapiens2Config):
        super().__init__(self)

        del self.num_patches_h
        del self.num_patches_w
        image_size = config.image_size
        image_h, image_w = image_size if isinstance(image_size, Iterable) else (image_size, image_size)
        patch_size = config.patch_size if isinstance(config.patch_size, int) else config.patch_size[0]
        self.num_patches_h = image_h // patch_size
        self.num_patches_w = image_w // patch_size


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
            inv_freq = 1 / module.base ** torch.arange(0, 1, 4 / module.head_dim, dtype=torch.float32)
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
class Sapiens2ForImageMatting(Sapiens2PreTrainedModel):
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
    ) -> Sapiens2ImageMattingOutput:
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

        return Sapiens2ImageMattingOutput(
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
    "Sapiens2ForImageMatting",
    "Sapiens2Model",
    "Sapiens2PreTrainedModel",
    "Sapiens2Backbone",
    "Sapiens2ImageProcessor",
]

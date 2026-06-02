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
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTBackboneOutput

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
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
from ...modeling_outputs import BaseModelOutputWithPooling, ModelOutput, SemanticSegmenterOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..beit.image_processing_beit import BeitImageProcessor, BeitImageProcessorKwargs
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
from ..llama.modeling_llama import LlamaRMSNorm
from ..mask2former.modeling_mask2former import Mask2FormerPredictionBlock
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import PPOCRV5ServerDetConvBatchnormLayer
from ..sam3.processing_sam3 import box_xywh_to_cxcywh, box_xywh_to_xyxy
from ..vitmatte.modeling_vitmatte import ImageMattingOutput
from ..vitpose.modeling_vitpose import VitPoseEstimatorOutput, flip_back


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Output type of [`Sapiens2Backbone`], extending [`BackboneOutput`] with optional CLS tokens from
    each selected feature stage (used when `config.return_class_token=True`).
    """
)
@dataclass
class Sapiens2BackboneOutput(DINOv3ViTBackboneOutput):
    pass


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
    """Compute crop center and scale from bounding boxes, applying padding and aspect ratio correction.

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
    """Crops and resizes bounding box regions from the input image to the target output size.

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
    boxes_width, boxes_height = scale.unbind(-1)

    scale_x = (output_width - 1) / boxes_width  # (num_boxes,)
    scale_y = (output_height - 1) / boxes_height  # (num_boxes,)
    is_bilinear = torch.minimum(scale_x, scale_y) < 1.0  # (num_boxes,)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(output_height, dtype=torch.float32, device=image.device),
        torch.arange(output_width, dtype=torch.float32, device=image.device),
        indexing="ij",
    )
    in_x = grid_x / scale_x[:, None, None] + center_x[:, None, None] - 0.5 * boxes_width[:, None, None]
    in_y = grid_y / scale_y[:, None, None] + center_y[:, None, None] - 0.5 * boxes_height[:, None, None]
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
    heatmap_values = {
        (dx, dy): heatmaps_flattened[index + offset] for (dx, dy), offset in position_to_index_offset.items()
    }

    gradient_x = 0.5 * (heatmap_values[0, 1] - heatmap_values[0, -1])
    gradient_y = 0.5 * (heatmap_values[1, 0] - heatmap_values[-1, 0])

    hessian_xx = heatmap_values[0, 1] - 2 * heatmap_values[0, 0] + heatmap_values[0, -1]
    hessian_yy = heatmap_values[1, 0] - 2 * heatmap_values[0, 0] + heatmap_values[-1, 0]
    hessian_xy = 0.5 * (
        heatmap_values[1, 1]
        - heatmap_values[0, 1]
        - heatmap_values[1, 0]
        + heatmap_values[0, 0]
        + heatmap_values[0, 0]
        - heatmap_values[0, -1]
        - heatmap_values[-1, 0]
        + heatmap_values[-1, -1]
    )

    eps = torch.finfo(hessian_xx.dtype).eps
    hessian_xx = hessian_xx + eps
    hessian_yy = hessian_yy + eps
    determinant = hessian_xx * hessian_yy - hessian_xy * hessian_xy
    offset_x = (hessian_yy * gradient_x - hessian_xy * gradient_y) / determinant
    offset_y = (-hessian_xy * gradient_x + hessian_xx * gradient_y) / determinant
    return keypoints - torch.cat([offset_x, offset_y], dim=-1)


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
        kwargs["boxes"] = boxes  # modular trick
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
        images: list[torch.Tensor],
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
    ) -> list[torch.Tensor]:
        if boxes is not None:
            output_size = (size["height"], size["width"])
            crops = []
            for image, image_boxes in zip(images, boxes):
                image = tvF.to_dtype_image(image, dtype=torch.float32, scale=False)
                boxes_tensor = box_xywh_to_cxcywh(torch.tensor(image_boxes, dtype=torch.float32, device=image.device))
                crops.extend(crop_and_resize(image, boxes=boxes_tensor, output_size=output_size))
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
        boxes_tensor = torch.tensor(
            [box for image_boxes in boxes for box in image_boxes], dtype=torch.float32, device=device
        )

        heatmaps = heatmaps.float()  # For consistency with original numpy/cv2 implementation which uses float32.

        # (num_total_persons, num_keypoints, 2), (num_total_persons, num_keypoints)
        all_keypoints, all_scores = get_keypoint_predictions(heatmaps)
        all_keypoints = post_dark_unbiased_data_processing(
            keypoints=all_keypoints, heatmaps=heatmaps, blur_kernel_size=kernel_size
        )

        # Remap coordinates from heatmap space to original image space
        centers, scales = boxes_to_crop_params(
            box_xywh_to_cxcywh(boxes_tensor), output_size=(self.size["height"], self.size["width"])
        )
        heatmap_size = torch.tensor([heatmap_width - 1, heatmap_height - 1], dtype=torch.float32, device=device)
        all_keypoints = (
            all_keypoints / heatmap_size * scales[:, None, :] + centers[:, None, :] - 0.5 * scales[:, None, :]
        )
        all_boxes = box_xywh_to_xyxy(boxes_tensor)  # (num_total_persons, 4)

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
                [
                    per_image_scale[image_index].unsqueeze(0).expand(len(boxes[image_index]), 2)
                    for image_index in range(num_images)
                ]
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
        normals = F.normalize(outputs.normals, p=2, dim=1, eps=1e-8)
        results = self._post_process_maps(
            maps=normals, source_sizes=source_sizes, target_sizes=target_sizes, do_remove_padding=do_remove_padding
        )
        return [{"normals": result} for result in results]

    def post_process_pointmap_estimation(
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
        pointmaps = outputs.pointmaps
        if outputs.scales is not None:
            pointmaps = pointmaps / outputs.scales[:, :, None, None]
        results = self._post_process_maps(
            maps=pointmaps, source_sizes=source_sizes, target_sizes=target_sizes, do_remove_padding=do_remove_padding
        )
        return [{"pointmap": result} for result in results]

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

        batch_size = outputs.foregrounds.shape[0]
        device = outputs.foregrounds.device
        dtype = outputs.foregrounds.dtype

        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the matting output"
                )
        all_target_sizes_equal = target_sizes is None or all(
            tuple(size) == tuple(target_sizes[0]) for size in target_sizes
        )

        background_tensors = []
        if backgrounds is not None:
            background_list = make_list_of_images(backgrounds)
            if len(background_list) != 1 and len(background_list) != batch_size:
                raise ValueError(
                    "Make sure that you pass in as many backgrounds as the batch dimension of the matting output"
                )
            background_tensors = [
                tvF.to_dtype_image(tvF.to_image(background_image), dtype=dtype, scale=True).to(device)
                for background_image in background_list
            ]
        all_background_sizes_equal = not background_tensors or all(
            background.shape[-2:] == background_tensors[0].shape[-2:] for background in background_tensors
        )

        matting = torch.cat([outputs.foregrounds, outputs.alphas], dim=1)  # (batch_size, 4, height, width)

        if target_sizes is not None and all_target_sizes_equal:
            target_size = tuple(target_sizes[0])
            matting = F.interpolate(
                matting,
                size=target_size,
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            matting = matting.clamp(0.0, 1.0)

        result = []
        if all_target_sizes_equal and all_background_sizes_equal:
            # Fast path
            foregrounds = matting[:, :3]
            alphas = matting[:, 3:]
            composites = [None] * batch_size
            if background_tensors:
                background = torch.stack(background_tensors)
                if background.shape[-2:] != matting.shape[-2:]:
                    background = F.interpolate(
                        background,
                        size=matting.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                        antialias=False,
                    )
                composites = (foregrounds + (1 - alphas) * background).clamp(0.0, 1.0)
                composites = tvF.to_dtype_image(composites, dtype=torch.uint8, scale=True)

            for foreground, alpha, composite in zip(foregrounds, alphas, composites):
                result.append(
                    {
                        "foreground": foreground,
                        "alpha": alpha,
                        "composite": composite,
                    }
                )

        else:
            # Slow path
            for index in range(len(matting)):
                matting_item = matting[index]

                if target_sizes and not all_target_sizes_equal:
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

                if background_tensors:
                    background = background_tensors[0] if len(background_tensors) == 1 else background_tensors[index]
                    if background.shape[-2:] != matting_item.shape[-2:]:
                        background = F.interpolate(
                            background.unsqueeze(0),
                            size=matting_item.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                            antialias=False,
                        )[0]
                    composite = (foreground + (1 - alpha) * background).clamp(0.0, 1.0)
                    composite = tvF.to_dtype_image(composite, dtype=torch.uint8, scale=True)

                result.append({"foreground": foreground, "alpha": alpha, "composite": composite})

        return result

    def _post_process_maps(
        self,
        maps: torch.Tensor,
        source_sizes: TensorType | list[tuple[int, int]] | None,
        target_sizes: TensorType | list[tuple[int, int]] | None,
        do_remove_padding: bool | None,
    ) -> list[torch.Tensor]:
        if isinstance(source_sizes, torch.Tensor):
            source_sizes = source_sizes.tolist()
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = target_sizes.tolist()
        if do_remove_padding is None:
            do_remove_padding = source_sizes is not None
        if do_remove_padding and source_sizes is None:
            raise ValueError("`source_sizes` must be provided when `do_remove_padding=True`.")

        if source_sizes is not None and len(maps) != len(source_sizes):
            raise ValueError("Make sure that you pass in as many source sizes as the batch dimension of the outputs")
        if target_sizes is not None and len(maps) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the outputs")

        model_height = self.size["height"]
        model_width = self.size["width"]

        crops = []
        if do_remove_padding:
            for original_height, original_width in source_sizes:
                new_height, new_width = get_image_size_for_max_height_width(
                    (original_height, original_width), model_height, model_width
                )
                pad_top = (model_height - new_height) // 2 if new_height < model_height else 0
                pad_left = (model_width - new_width) // 2 if new_width < model_width else 0
                crops.append(
                    (
                        pad_top,
                        pad_left,
                        pad_top + min(new_height, model_height),
                        pad_left + min(new_width, model_width),
                    )
                )
        all_crops_equal = not crops or all(crop == crops[0] for crop in crops)

        final_sizes = []
        if target_sizes is not None:
            final_sizes = [tuple(size) for size in target_sizes]
        elif source_sizes is not None:
            final_sizes = [tuple(size) for size in source_sizes]
        all_final_sizes_equal = not final_sizes or all(size == final_sizes[0] for size in final_sizes)

        result = []
        if all_crops_equal and all_final_sizes_equal:
            # Fast path
            if do_remove_padding:
                top, left, bottom, right = crops[0]
                maps = maps[:, :, top:bottom, left:right]

            if final_sizes:
                maps = F.interpolate(
                    maps,
                    size=final_sizes[0],
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )

            result = list(maps)
        else:
            # Slow path
            for index in range(len(maps)):
                map_item = maps[index]

                if do_remove_padding:
                    top, left, bottom, right = crops[index]
                    map_item = map_item[:, top:bottom, left:right]

                if final_sizes:
                    map_item = F.interpolate(
                        map_item.unsqueeze(0),
                        size=final_sizes[index],
                        mode="bilinear",
                        align_corners=False,
                        antialias=False,
                    )[0]

                result.append(map_item)

        return result


@auto_docstring(checkpoint="facebook/sapiens2-seg-0.4b")
@strict
class Sapiens2HeadConfig(PreTrainedConfig):
    r"""
    upsample_out_channels (`list[int]`, *optional*):
        Output channel counts for each upsample block.
        The first block takes `hidden_size` channels as input; subsequent blocks use the previous output.
    upsample_kernel_sizes (`list[int]`, *optional*):
        Kernel size for each upsample block. Auto-filled with `[4, ...]` when
        `upsample_out_channels` is set but this is `None`.
        Must have the same length as `upsample_out_channels`.
    upsample_kernel_size (`int`, defaults to 4):
        Default kernel size for upsample blocks when `upsample_kernel_sizes` is not set.
    use_pixel_shuffle (`bool`, *optional*):
        Whether the upsample head uses pixel-shuffle upsampling instead of transposed convolutions.
        When `None` (default), the head uses transposed convolutions.
    conv_out_channels (`list[int]`, *optional*):
        Output channel counts for the refinement conv layers that follow the upsample blocks.
    conv_kernel_sizes (`list[int]`, *optional*):
        Kernel size for each refinement conv layer. Auto-filled with `[1, ...]` when
        `conv_out_channels` is set but this is `None`.
        Must have the same length as `conv_out_channels`.
    conv_kernel_size (`int`, defaults to 1):
        Default kernel size for conv layers when `conv_kernel_sizes` is not set.
    scale_conv_out_channels (`list[int]`, *optional*):
        Output channel counts for the stride-2 conv layers used to predict the focal-length scale.
        When `None` (default), no scale branch is built.
    scale_conv_kernel_sizes (`list[int]`, *optional*):
        Kernel size for each scale conv layer. Auto-filled with `[1, ...]` when
        `scale_conv_out_channels` is set but this is `None`.
        Must have the same length as `scale_conv_out_channels`.
    scale_conv_kernel_size (`int`, defaults to 1):
        Default kernel size for scale conv layers when `scale_conv_kernel_sizes` is not set.
    scale_final_input_size (`int`, *optional*):
        Flattened feature size passed into the scale MLP.
        When `None` (default), it is automatically inferred from `image_size` and `patch_size`
        in the parent [`Sapiens2Config`].
    scale_final_hidden_sizes (`list[int]`, *optional*):
        Hidden-layer sizes for the MLP that maps flattened scale features to the scalar scale output.
        When `None` (default), no scale branch is built.
    """

    model_type = "sapiens2_head"
    base_config_key = "head_config"

    upsample_out_channels: list[int] | None = None
    upsample_kernel_sizes: list[int] | None = None
    upsample_kernel_size: int = 4
    use_pixel_shuffle: bool | None = None
    conv_out_channels: list[int] | None = None
    conv_kernel_sizes: list[int] | None = None
    conv_kernel_size: int = 1
    scale_conv_out_channels: list[int] | None = None
    scale_conv_kernel_sizes: list[int] | None = None
    scale_conv_kernel_size: int = 1
    scale_final_input_size: int | None = None
    scale_final_hidden_sizes: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.upsample_out_channels is not None and self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [self.upsample_kernel_size] * len(self.upsample_out_channels)
        if self.conv_out_channels is not None and self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [self.conv_kernel_size] * len(self.conv_out_channels)
        if self.scale_conv_out_channels is not None and self.scale_conv_kernel_sizes is None:
            self.scale_conv_kernel_sizes = [self.scale_conv_kernel_size] * len(self.scale_conv_out_channels)
        super().__post_init__(**kwargs)

    def _init_scale_final_input_size(
        self, image_size: int | list[int] | tuple[int, int], patch_size: int | list[int] | tuple[int, int]
    ) -> None:
        if (
            self.scale_final_input_size is not None
            or self.scale_conv_out_channels is None
            or self.scale_conv_kernel_sizes is None
        ):
            return
        image_height, image_width = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        patch_height = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_width = patch_size if isinstance(patch_size, int) else patch_size[1]
        features_height = image_height // patch_height
        features_width = image_width // patch_width
        for kernel_size in self.scale_conv_kernel_sizes:
            padding = (kernel_size - 1) // 2
            features_height = (features_height + 2 * padding - kernel_size) // 2 + 1
            features_width = (features_width + 2 * padding - kernel_size) // 2 + 1
        self.scale_final_input_size = features_height * features_width * self.scale_conv_out_channels[-1]


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
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the hidden states to spatial dimensions when used as backbone.
    use_mask_token (`bool`, *optional*, defaults to `False`):
        Whether to use a mask token in the embeddings (needed for masked image modeling pretraining).
    rms_norm_eps (`float`, *optional*, defaults to 1e-6):
        Epsilon for the RMS normalization layers.
    normalize_backbone_outputs (`bool`, *optional*, defaults to `True`):
        Whether to apply RMSNorm to the backbone `feature_maps` and `cls_tokens` outputs before
        returning them from the forward pass. Only applies when the model is used as a backbone.
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply RMSNorm to queries and keys before RoPE in attention layers.
    num_key_value_heads_per_layer (`list[int]`, *optional*):
        Number of key/value heads for each transformer layer. Setting a layer's value equal to
        `num_attention_heads` gives full multi-head attention; a smaller value gives grouped-query
        attention. Defaults to `num_attention_heads` for the first `num_first_full_attention_layers`
        and last `num_last_full_attention_layers` layers and `num_key_valueattention_heads` for all other
        layers.
    num_key_value_attention_heads (`int`):
        Number of key/value heads for layers that use grouped-query attention when `num_key_value_heads_per_layer`
        is not set. Ignored when `num_key_value_heads_per_layer` is set.
    num_first_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of leading transformer layers that use full multi-head attention.
        Only used when `num_key_value_heads_per_layer` is `None`.
    num_last_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of trailing transformer layers that use full multi-head attention.
        Only used when `num_key_value_heads_per_layer` is `None`.
    semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
        Label index ignored when computing the segmentation loss.
    flip_pairs (`list[list[int]]`, *optional*):
        Pairs of keypoint indices that are mirrored horizontally (e.g., left ear ↔ right ear).
        Each pair is a two-element list `[left_index, right_index]`. Used for test-time
        horizontal flip augmentation in pose estimation: pass these pairs to the second
        forward call so the model flips heatmaps back before returning them.
    head_config (`Sapiens2HeadConfig`, *optional*):
        Configuration for the decode head. See [`Sapiens2HeadConfig`] for the available options.
    """

    model_type = "sapiens2"
    sub_configs = {"head_config": Sapiens2HeadConfig}

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    use_mask_token: bool = False
    use_gated_mlp: bool = True
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    normalize_backbone_outputs: bool = True
    num_register_tokens: int = 8
    key_bias: bool = True
    use_qk_norm: bool = True
    num_key_value_heads_per_layer: list[int] | None = None
    num_key_value_attention_heads: int = 8
    num_first_full_attention_layers: int = 8
    num_last_full_attention_layers: int = 8
    semantic_loss_ignore_index: int = 255
    flip_pairs: list[list[int]] | None = None
    head_config: Sapiens2HeadConfig | dict | None = None

    layer_norm_eps = AttributeError()  # inherited from DINOv3 but not used
    apply_layernorm = AttributeError()  # inherited from DINOv3 but not used

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads_per_layer is None:
            self.num_key_value_heads_per_layer = [
                self.num_attention_heads
                if (
                    layer_index < self.num_first_full_attention_layers
                    or layer_index >= self.num_hidden_layers - self.num_last_full_attention_layers
                )
                else self.num_key_value_attention_heads
                for layer_index in range(self.num_hidden_layers)
            ]
        if isinstance(self.head_config, dict):
            self.head_config = Sapiens2HeadConfig(**self.head_config)
        if self.head_config is not None:
            self.head_config._init_scale_final_input_size(image_size=self.image_size, patch_size=self.patch_size)
        super().__post_init__(**kwargs)


class Sapiens2Embeddings(DINOv3ViTEmbeddings):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_mask_token else None

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        if bool_masked_pos is not None and self.mask_token is None:
            raise ValueError("bool_masked_pos requires use_mask_token=True in the config")
        return super().forward(pixel_values, bool_masked_pos)


class Sapiens2RopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    def __init__(self, config: Sapiens2Config):
        super().__init__(self)

        del self.num_patches_h
        del self.num_patches_w
        image_size = config.image_size
        image_h, image_w = image_size if isinstance(image_size, Iterable) else (image_size, image_size)
        patch_size = config.patch_size
        patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        self.num_patches_h = image_h // patch_size_h
        self.num_patches_w = image_w // patch_size_w


class Sapiens2RMSNorm(LlamaRMSNorm):
    pass


class Sapiens2Attention(DINOv3ViTAttention):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__(config)
        del self.k_proj
        del self.v_proj
        self.num_key_value_heads = config.num_key_value_heads_per_layer[layer_idx]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.num_key_value_heads * self.head_dim, bias=config.value_bias)
        self.q_norm = Sapiens2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = Sapiens2RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Sapiens2LayerScale(DINOv3ViTLayerScale):
    pass


class Sapiens2Layer(DINOv3ViTLayer):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__(config)
        self.attention = Sapiens2Attention(config, layer_idx=layer_idx)
        self.norm1 = Sapiens2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Sapiens2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scale2 = nn.Identity()


class Sapiens2ConvLayer(PPOCRV5ServerDetConvBatchnormLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        stride: int = 1,
        padding: int | tuple[int, int] | str = 0,
        groups: int = 1,
        activation: str = "silu",
        bias: bool = True,
        convolution_transpose: bool = False,
        pixel_shuffle: bool = False,
        scale_factor: int = 2,
    ):
        super().__init__()
        if convolution_transpose:
            self.convolution = nn.ConvTranspose2d(
                in_channels,
                out_channels * scale_factor**2 if pixel_shuffle else out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )
        else:
            self.convolution = nn.Conv2d(
                in_channels,
                out_channels * scale_factor**2 if pixel_shuffle else out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor) if pixel_shuffle else nn.Identity()
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act_fn = ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class Sapiens2Head(nn.Module):
    def __init__(self, config: Sapiens2Config):
        super().__init__()
        self.input_conv = (
            Sapiens2ConvLayer(config.hidden_size, config.hidden_size, kernel_size=3, padding=1)
            if config.head_config.use_pixel_shuffle
            else nn.Identity()
        )
        upsample_in_channels = [config.hidden_size] + config.head_config.upsample_out_channels[:-1]
        self.upsample_layers = nn.ModuleList(
            Sapiens2ConvLayer(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=1 if config.head_config.use_pixel_shuffle else 2,
                padding=(kernel_size - 1) // 2 if config.head_config.use_pixel_shuffle else 1,
                bias=bool(config.head_config.use_pixel_shuffle),
                pixel_shuffle=bool(config.head_config.use_pixel_shuffle),
                convolution_transpose=not config.head_config.use_pixel_shuffle,
            )
            for in_ch, out_ch, kernel_size in zip(
                upsample_in_channels,
                config.head_config.upsample_out_channels,
                config.head_config.upsample_kernel_sizes,
            )
        )
        conv_in_channels = [config.head_config.upsample_out_channels[-1]] + config.head_config.conv_out_channels[:-1]
        self.conv_layers = nn.ModuleList(
            Sapiens2ConvLayer(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2 if config.head_config.use_pixel_shuffle else 0,
            )
            for in_ch, out_ch, kernel_size in zip(
                conv_in_channels, config.head_config.conv_out_channels, config.head_config.conv_kernel_sizes
            )
        )
        predictor_in = (
            config.head_config.conv_out_channels[-1]
            if config.head_config.conv_out_channels
            else config.head_config.upsample_out_channels[-1]
            if config.head_config.upsample_out_channels
            else config.hidden_size
        )
        self.predictor = nn.Conv2d(predictor_in, config.num_labels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_conv(hidden_states)
        for layer in self.upsample_layers:
            hidden_states = layer(hidden_states)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return self.predictor(hidden_states)


class Sapiens2PointmapFinalLayerBlock(Mask2FormerPredictionBlock):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        nn.Module.__init__(self)
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim), activation])


class Sapiens2PointmapFinalLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: tuple[int, int], out_dim: int = 1, activation: str = "silu"):
        super().__init__()
        self.flatten = nn.Flatten()
        self.block1 = Sapiens2PointmapFinalLayerBlock(
            in_dim=in_dim, out_dim=hidden_sizes[0], activation=ACT2FN[activation]
        )
        self.block2 = Sapiens2PointmapFinalLayerBlock(
            in_dim=hidden_sizes[0], out_dim=hidden_sizes[1], activation=ACT2FN[activation]
        )
        self.proj = nn.Linear(hidden_sizes[1], out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.flatten(hidden_states)
        hidden_states = self.block1(hidden_states)
        hidden_states = self.block2(hidden_states)
        return self.proj(hidden_states)


class Sapiens2PointmapScaleHead(nn.Module):
    def __init__(self, config: Sapiens2Config):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        scale_in_channels = [config.hidden_size] + config.head_config.scale_conv_out_channels[:-1]
        for in_ch, out_ch, kernel_size in zip(
            scale_in_channels,
            config.head_config.scale_conv_out_channels,
            config.head_config.scale_conv_kernel_sizes,
        ):
            self.conv_layers.append(
                Sapiens2ConvLayer(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2)
            )
        self.predictor = Sapiens2PointmapFinalLayer(
            config.head_config.scale_final_input_size,
            config.head_config.scale_final_hidden_sizes,
            activation=config.hidden_act,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return self.predictor(hidden_states)


class Sapiens2PreTrainedModel(DINOv3ViTPreTrainedModel):
    base_model_prefix = "model"

    # Ignore periods as we use inv_freq instead which is automatically calculated from the config.
    _keys_to_ignore_on_load_unexpected = [r"periods"]
    # mask_token is only used for masked image modeling pretraining and is absent in most checkpoints.
    _keys_to_ignore_on_load_missing = [r"mask_token"]

    @torch.no_grad()
    def _init_weights(self, module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.ConvTranspose2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
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
        elif isinstance(module, (Sapiens2Head, Sapiens2PointmapScaleHead)):
            for head_module in module.modules():
                if isinstance(head_module, nn.Conv2d):
                    init.kaiming_normal_(head_module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(head_module, nn.Linear):
                    init.kaiming_normal_(head_module.weight, mode="fan_in", nonlinearity="linear")


class Sapiens2Encoder(DINOv3ViTEncoder):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [Sapiens2Layer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class Sapiens2Model(DINOv3ViTModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.norm = Sapiens2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from transformers.image_utils import load_image
        >>> import torch

        >>> image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pretrain-0.4b")
        >>> model = AutoModel.from_pretrained("facebook/sapiens2-pretrain-0.4b")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)

        >>> cls_token = outputs.pooler_output
        >>> cls_token.shape
        torch.Size([1, 1024])
        ```
        """
        return super().forward(pixel_values, bool_masked_pos=bool_masked_pos, **kwargs)


class Sapiens2Backbone(DINOv3ViTBackbone):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.norm = Sapiens2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sapiens2BackboneOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoBackbone, AutoImageProcessor
        >>> from transformers.image_utils import load_image
        >>> import torch

        >>> image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pretrain-0.4b")
        >>> model = AutoBackbone.from_pretrained("facebook/sapiens2-pretrain-0.4b")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs, return_class_token=True)

        >>> outputs.feature_maps[0].shape
        torch.Size([1, 1024, 64, 48])
        >>> outputs.cls_tokens[0].shape
        torch.Size([1, 1024])
        ```
        """
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)

        kwargs["output_hidden_states"] = True  # required to extract layers for the stages
        output = self.model(hidden_states, position_embeddings, **kwargs)
        stage_hidden_states = output.hidden_states

        batch_size, _, image_height, image_width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        num_patches_height = image_height // patch_size_h
        num_patches_width = image_width // patch_size_w

        num_prefix = 1 + getattr(self.config, "num_register_tokens", 0)
        return_class_token = getattr(self.config, "return_class_token", False)

        feature_maps, cls_tokens = [], []
        for idx, (stage_name, hidden_state) in enumerate(zip(self.stage_names, stage_hidden_states)):
            if self.config.normalize_backbone_outputs:
                hidden_state = self.norm(hidden_state)

            if stage_name in self.out_features:
                if return_class_token:
                    cls_tokens.append(hidden_state[:, 0, :])
                patch_tokens = hidden_state[:, num_prefix:, :]
                if self.config.reshape_hidden_states:
                    feature_map = (
                        patch_tokens.reshape(batch_size, num_patches_height, num_patches_width, patch_tokens.shape[-1])
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                else:
                    feature_map = patch_tokens

                feature_maps.append(feature_map)

        return Sapiens2BackboneOutput(
            feature_maps=tuple(feature_maps),
            cls_tokens=tuple(cls_tokens) if return_class_token else None,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


@auto_docstring(checkpoint="facebook/sapiens2-seg-0.4b")
class Sapiens2ForSemanticSegmentation(Sapiens2PreTrainedModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Sapiens2Model(config)
        self.decode_head = Sapiens2Head(config)
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

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from transformers.image_utils import load_image
        >>> import torch

        >>> image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-seg-0.4b")
        >>> model = AutoModel.from_pretrained("facebook/sapiens2-seg-0.4b")

        >>> inputs = image_processor(image, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)

        >>> outputs.logits.shape
        torch.Size([1, 29, 1024, 768])
        ```
        """
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.model(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_h
        patch_width = width // patch_size_w

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
        self.model = Sapiens2Model(config)
        self.decode_head = Sapiens2Head(config)
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
        labels (`torch.FloatTensor` of shape `(batch_size, num_keypoints, height, width)`, *optional*):
            Heatmap ground truth for computing the loss.

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from transformers.image_utils import load_image
        >>> import torch

        >>> image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pose-0.4b")
        >>> model = AutoModel.from_pretrained("facebook/sapiens2-pose-0.4b")

        >>> boxes = [[[270.8, 0.6, 294.1, 379.5]]]
        >>> inputs = image_processor(image, boxes=boxes, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)

        >>> outputs.heatmaps.shape
        torch.Size([1, 308, 256, 192])
        ```
        """
        outputs = self.model(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_h
        patch_width = width // patch_size_w

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
        self.model = Sapiens2Model(config)
        self.decode_head = Sapiens2Head(config)
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

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from transformers.image_utils import load_image
        >>> import torch

        >>> image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-normal-0.4b")
        >>> model = AutoModel.from_pretrained("facebook/sapiens2-normal-0.4b")

        >>> inputs = image_processor(image, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)

        >>> outputs.normals.shape
        torch.Size([1, 3, 1024, 768])
        ```
        """
        outputs = self.model(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_h
        patch_width = width // patch_size_w

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
        self.model = Sapiens2Model(config)
        self.decode_head = Sapiens2Head(config)
        self.scale_head = (
            Sapiens2PointmapScaleHead(config)
            if config.head_config is not None and config.head_config.scale_conv_out_channels is not None
            else nn.Identity()
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

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from transformers.image_utils import load_image
        >>> import torch

        >>> image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pointmap-0.4b")
        >>> model = AutoModel.from_pretrained("facebook/sapiens2-pointmap-0.4b")

        >>> inputs = image_processor(image, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)

        >>> outputs.pointmaps.shape
        torch.Size([1, 3, 1024, 768])
        ```
        """
        outputs = self.model(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_h
        patch_width = width // patch_size_w

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
        self.model = Sapiens2Model(config)
        self.decode_head = Sapiens2Head(config)  # config.num_labels = 4
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

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from transformers.image_utils import load_image
        >>> import torch

        >>> image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-matting-1b")
        >>> model = AutoModel.from_pretrained("facebook/sapiens2-matting-1b")

        >>> inputs = image_processor(image, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)

        >>> outputs.alphas.shape
        torch.Size([1, 1, 1024, 768])
        >>> outputs.foregrounds.shape
        torch.Size([1, 3, 1024, 768])
        ```
        """
        outputs = self.model(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_size_h = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_w = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_h
        patch_width = width // patch_size_w

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
    "Sapiens2HeadConfig",
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

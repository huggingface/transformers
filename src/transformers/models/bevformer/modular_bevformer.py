# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modular implementation of BEVFormer, a multi-view camera-based 3D object detector."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import load_backbone
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling, SizeDict
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...models.auto import AutoConfig
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    ModelOutput,
    TensorType,
    TransformersKwargs,
    auto_docstring,
    logging,
    torch_compilable_check,
)
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoderLayer,
    DeformableDetrMLP,
    DeformableDetrMLPPredictionHead,
    DeformableDetrMultiscaleDeformableAttention,
    DeformableDetrSelfAttention,
    inverse_sigmoid,
    replace_batch_norm,
)


logger = logging.get_logger(__name__)


NUSCENES_CLASS_NAMES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

# Normalization statistics of the BEVFormer pipeline. The nuScenes dataset loader rescales images to the
# RGB convention and normalizes with *un-rescaled* mean/std, i.e. directly in the 0-255 pixel value space.
DEFAULT_IMAGE_MEAN = [123.675, 116.28, 103.53]
DEFAULT_IMAGE_STD = [58.395, 57.12, 57.375]
DEFAULT_POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


@auto_docstring(checkpoint="hasanfaesal/bevformer-tiny")
@strict(accept_kwargs=True)
class BevformerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BevformerModel`]. It is used to instantiate a
    BEVFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BEVFormer-tiny architecture.
    BEVFormer-tiny consumes multi-view images from 6 cameras surrounding the ego vehicle and predicts 3D bounding
    boxes in a bird's-eye-view (BEV) representation.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        point_cloud_range (`list[float]`, *optional*, defaults to `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`):
            The lidar coordinate system range `(x_min, y_min, z_min, x_max, y_max, z_max)` covered by the BEV plane.
        bev_height (`int`, *optional*, defaults to 50):
            The height (number of cells along the y axis) of the BEV query plane.
        bev_width (`int`, *optional*, defaults to 50):
            The width (number of cells along the x axis) of the BEV query plane.
        num_cameras (`int`, *optional*, defaults to 6):
            The number of surrounding cameras expected by the model.
        num_feature_levels (`int`, *optional*, defaults to 1):
            The number of image feature levels fed to the encoder.
        d_model (`int`, *optional*, defaults to 256):
            Dimensionality of the BEV queries, image features and attention layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each BEVFormer encoder layer.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each object detection decoder layer.
        encoder_layers (`int`, *optional*, defaults to 3):
            Number of BEVFormer encoder layers.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of object detection decoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the "intermediate" (feed-forward) layer inside the encoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the "intermediate" (feed-forward) layer inside the decoder layers.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the feed-forward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability of attention weights and feed-forward layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities in the decoder layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the feed-forward layers of the decoder.
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries.
        num_points_in_pillar (`int`, *optional*, defaults to 4):
            Number of sampled depth positions (pillar points) along the z axis for each BEV query.
        num_points_in_bev (`int`, *optional*, defaults to 8):
            Number of sampling points on the image plane for each pillar point in the spatial cross attention.
        encoder_n_points (`int`, *optional*, defaults to 4):
            Number of sampling points per attention head in the temporal self-attention of the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            Number of sampling points per attention head in the deformable attention of the decoder.
        can_bus_dim (`int`, *optional*, defaults to 18):
            Dimensionality of the CAN bus ego-motion features.
        can_bus_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply a layer normalization on the encoded CAN bus signals.
        rotate_center (`list[float]`, *optional*, defaults to `[100.0, 100.0]`):
            The rotation center `[x, y]` (in BEV cell coordinates) used when rotating previous BEV embeddings.
    """

    model_type = "bevformer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    sub_configs_defaults = {
        "backbone_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="resnet",
            init_kwargs={"out_features": ["stage4"]},
        ),
    }

    backbone_config: dict | PreTrainedConfig | None = None
    point_cloud_range: list[float] | None = None
    bev_height: int = 50
    bev_width: int = 50
    num_cameras: int = 6
    num_feature_levels: int = 1
    d_model: int = 256
    encoder_layers: int = 3
    encoder_ffn_dim: int = 512
    encoder_attention_heads: int = 8
    encoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_ffn_dim: int = 512
    decoder_attention_heads: int = 8
    decoder_n_points: int = 4
    activation_function: str = "relu"
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    num_queries: int = 900
    num_points_in_pillar: int = 4
    num_points_in_bev: int = 8
    can_bus_dim: int = 18
    can_bus_norm: bool = True
    rotate_center: list[float] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.point_cloud_range is None:
            self.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if self.rotate_center is None:
            self.rotate_center = [100.0, 100.0]
        if self.id2label is None:
            self.id2label = dict(enumerate(NUSCENES_CLASS_NAMES))
        super().__post_init__(**kwargs)


class BevformerImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    pad_size_divisor (`int`, *optional*, defaults to `self.pad_size_divisor`):
        Spatial dimensions of the padded images are set to multiples of this value.
    point_cloud_range (`list[float]`, *optional*, defaults to `self.point_cloud_range`):
        3D LiDAR point cloud bounds `[x_min, y_min, z_min, x_max, y_max, z_max]` used to build 3D pillar reference
        points, ego-motion shifts, and metric box denormalization.
    bev_height (`int`, *optional*, defaults to `self.bev_height`):
        Height of the Bird's-Eye-View (BEV) query grid.
    bev_width (`int`, *optional*, defaults to `self.bev_width`):
        Width of the Bird's-Eye-View (BEV) query grid.
    num_points_in_pillar (`int`, *optional*, defaults to `self.num_points_in_pillar`):
        Number of sampled vertical anchor points along the z-axis for each BEV pillar.
    """

    pad_size_divisor: int
    point_cloud_range: list[float]
    bev_height: int
    bev_width: int
    num_points_in_pillar: int


def get_bev_reference_points(
    height: int,
    width: int,
    z_height: float = 8.0,
    num_points_in_pillar: int = 4,
    dim: str = "3d",
    batch_size: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate normalized 2D or 3D reference points on the BEV grid."""
    if dim == "3d":
        zs = (
            torch.linspace(0.5, z_height - 0.5, num_points_in_pillar, dtype=dtype, device=device)
            .view(-1, 1, 1)
            .expand(num_points_in_pillar, height, width)
            / z_height
        )
        xs = (
            torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device)
            .view(1, 1, width)
            .expand(num_points_in_pillar, height, width)
            / width
        )
        ys = (
            torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device)
            .view(1, height, 1)
            .expand(num_points_in_pillar, height, width)
            / height
        )
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        return ref_3d[None].repeat(batch_size, 1, 1, 1)

    if dim == "2d":
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        ref_2d = torch.stack((ref_x, ref_y), -1)
        return ref_2d.repeat(batch_size, 1, 1).unsqueeze(2)

    raise ValueError(f"dim must be '2d' or '3d', got {dim}")


def point_sampling(
    reference_points: torch.Tensor,
    point_cloud_range: list[float] | tuple[float, ...],
    lidar2img: torch.Tensor,
    img_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D pillar reference points onto multi-camera 2D image coordinates.

    Returns batch-first tensors:
        - `reference_points_cam` of shape `(batch_size, num_cameras, num_queries, num_points_in_pillar, 2)`
        - `bev_mask` of shape `(batch_size, num_cameras, num_queries, num_points_in_pillar)`
    """
    ref_pts = reference_points.clone()
    ref_pts[..., 0:1] = ref_pts[..., 0:1] * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
    ref_pts[..., 1:2] = ref_pts[..., 1:2] * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
    ref_pts[..., 2:3] = ref_pts[..., 2:3] * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]

    # Homogeneous coordinates: (B, D, num_queries, 4) -> (B, 1, num_queries, D, 4, 1)
    ref_pts = torch.cat([ref_pts, torch.ones_like(ref_pts[..., :1])], dim=-1)
    ref_pts = ref_pts.transpose(1, 2).unsqueeze(1).unsqueeze(-1)

    reference_points_cam = torch.matmul(
        lidar2img[:, :, None, None, :, :].to(torch.float32), ref_pts.to(torch.float32)
    ).squeeze(-1)

    eps = 1e-5
    bev_mask = reference_points_cam[..., 2:3] > eps
    depth_clamped = torch.maximum(reference_points_cam[..., 2:3], torch.full_like(reference_points_cam[..., 2:3], eps))
    reference_points_cam = reference_points_cam[..., 0:2] / depth_clamped

    height, width = img_shape
    reference_points_cam[..., 0] /= width
    reference_points_cam[..., 1] /= height

    bev_mask = (
        bev_mask
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0)
        & (reference_points_cam[..., 0:1] < 1.0)
    )
    bev_mask = torch.nan_to_num(bev_mask).squeeze(-1)
    return reference_points_cam, bev_mask


def compute_ego_motion_shift(
    can_bus: torch.Tensor,
    point_cloud_range: list[float] | tuple[float, ...],
    bev_height: int,
    bev_width: int,
) -> torch.Tensor:
    """Compute normalized BEV translation shifts `(shift_x, shift_y)` from CAN bus ego-motion signals."""
    delta_x, delta_y = can_bus[:, 0], can_bus[:, 1]
    ego_angle = can_bus[:, -2] / math.pi * 180.0

    grid_length_y = (point_cloud_range[4] - point_cloud_range[1]) / bev_height
    grid_length_x = (point_cloud_range[3] - point_cloud_range[0]) / bev_width
    translation_length = torch.sqrt(delta_x**2 + delta_y**2)
    translation_angle = torch.atan2(delta_y, delta_x) / math.pi * 180.0
    bev_angle = ego_angle - translation_angle

    shift_y = translation_length * torch.cos(bev_angle / 180.0 * math.pi) / grid_length_y / bev_height
    shift_x = translation_length * torch.sin(bev_angle / 180.0 * math.pi) / grid_length_x / bev_width
    return torch.stack([shift_x, shift_y], dim=-1)


@auto_docstring
class BevformerImageProcessor(TorchvisionBackend):
    valid_kwargs = BevformerImageProcessorKwargs
    do_resize = True
    size = {"height": 450, "width": 800}
    resample = PILImageResampling.BILINEAR
    do_rescale = False
    do_normalize = True
    image_mean = DEFAULT_IMAGE_MEAN
    image_std = DEFAULT_IMAGE_STD
    do_pad = True
    pad_size_divisor = 32
    point_cloud_range = DEFAULT_POINT_CLOUD_RANGE
    bev_height = 50
    bev_width = 50
    num_points_in_pillar = 4
    model_input_names = ["pixel_values", "reference_points_cam", "bev_mask", "reference_points_2d", "can_bus"]

    def __init__(self, **kwargs: Unpack[BevformerImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

    def _prepare_multiview_samples(
        self,
        images: ImageInput,
        do_convert_rgb: bool | None,
        input_data_format: str | ChannelDimension | None,
        device: str | torch.device | None,
    ) -> list[torch.Tensor]:
        """Convert multi-view inputs into a list of `(num_cameras, 3, height, width)` float32 tensors."""
        if isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("`images` cannot be empty.")
            first = images[0]
            if isinstance(first, (list, tuple)):
                raw_samples = [list(sample) for sample in images]
            elif hasattr(first, "ndim") and first.ndim == 4:
                raw_samples = [[sample[c] for c in range(sample.shape[0])] for sample in images]
            else:
                raw_samples = [list(images)]
        elif hasattr(images, "ndim"):
            if images.ndim == 5:
                raw_samples = [[images[b, c] for c in range(images.shape[1])] for b in range(images.shape[0])]
            elif images.ndim == 4:
                raw_samples = [[images[c] for c in range(images.shape[0])]]
            else:
                raise ValueError(
                    "Multi-view images must be a 4D `(num_cameras, ...)` or 5D `(batch_size, num_cameras, ...)` "
                    f"array/tensor, got {images.ndim}D."
                )
        else:
            raise ValueError(f"Unsupported multi-view `images` type: {type(images)}")

        samples = []
        for sample_index, sample in enumerate(raw_samples):
            cam_tensors = [
                self.process_image(
                    cam_image,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    device=device,
                ).to(torch.float32)
                for cam_image in sample
            ]
            if not all(cam.shape == cam_tensors[0].shape for cam in cam_tensors):
                raise ValueError(f"All cameras of sample {sample_index} must share the same (height, width) size.")
            samples.append(torch.stack(cam_tensors, dim=0))
        return samples

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        lidar2img: ImageInput | None = None,
        can_bus: ImageInput | None = None,
        **kwargs: Unpack[BevformerImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        lidar2img (`ImageInput`, *optional*):
            Projection matrices from LiDAR coordinates to each camera image plane of shape
            `(batch_size, num_cameras, 4, 4)` or `(num_cameras, 4, 4)`. When provided, they are rescaled by the
            image resize factor and used to project 3D pillar reference points into `reference_points_cam` and
            `bev_mask`.
        can_bus (`ImageInput`, *optional*):
            CAN bus ego-motion vectors of shape `(batch_size, 18)` or `(18,)`. Used to compute the ego-motion
            translation shift for `reference_points_2d` and forwarded to the model.
        """
        return super().preprocess(images, lidar2img, can_bus, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        lidar2img: ImageInput | None,
        can_bus: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> BatchFeature:
        samples = self._prepare_multiview_samples(
            images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        return self._preprocess(
            samples,
            lidar2img=lidar2img,
            can_bus=can_bus,
            return_tensors=return_tensors,
            device=device,
            **kwargs,
        )

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size_divisor: int,
        point_cloud_range: list[float],
        bev_height: int,
        bev_width: int,
        num_points_in_pillar: int,
        return_tensors: str | TensorType | None,
        lidar2img: ImageInput | None = None,
        can_bus: ImageInput | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> BatchFeature:
        batch_size = len(images)
        num_cameras = images[0].shape[0]
        tensor_device = images[0].device

        lidar2img_tensor = None
        if lidar2img is not None:
            lidar2img_tensor = torch.as_tensor(lidar2img, dtype=torch.float32, device=tensor_device)
            if lidar2img_tensor.ndim == 3:
                lidar2img_tensor = lidar2img_tensor.unsqueeze(0)
            if lidar2img_tensor.shape[:2] != (batch_size, num_cameras) or lidar2img_tensor.shape[-2:] != (4, 4):
                raise ValueError(
                    f"Expected `lidar2img` to have shape ({batch_size}, {num_cameras}, 4, 4), got "
                    f"{tuple(lidar2img_tensor.shape)}."
                )

        processed_images = []
        scaled_lidar2img = []
        for sample_index, sample in enumerate(images):
            orig_height, orig_width = sample.shape[-2:]
            scale = 1.0
            if do_resize:
                height_scale = size.height / orig_height
                width_scale = size.width / orig_width
                if not math.isclose(height_scale, width_scale, rel_tol=0.0, abs_tol=1e-4):
                    raise ValueError(
                        f"The ratio applied to height ({height_scale}) and width ({width_scale}) must be equal, as "
                        f"a single scalar rescales both image dimensions and the `lidar2img` matrices."
                    )
                scale = width_scale
                # antialias=False matches upstream cv2.resize(INTER_LINEAR)
                sample = self.resize(image=sample, size=size, resample=resample, antialias=False)

            sample = self.rescale_and_normalize(
                sample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )

            scaled_height, scaled_width = sample.shape[-2:]
            pad_height = -scaled_height % pad_size_divisor if do_pad else 0
            pad_width = -scaled_width % pad_size_divisor if do_pad else 0
            if pad_height > 0 or pad_width > 0:
                sample = F.pad(sample, (0, pad_width, 0, pad_height), value=0.0)
            processed_images.append(sample)

            if lidar2img_tensor is not None:
                scale_matrix = torch.eye(4, dtype=torch.float32, device=tensor_device)
                scale_matrix[0, 0] = scale
                scale_matrix[1, 1] = scale
                scaled_lidar2img.append(torch.matmul(scale_matrix, lidar2img_tensor[sample_index]))

        pixel_values = torch.stack(processed_images, dim=0)
        padded_height, padded_width = pixel_values.shape[-2:]

        # Build 2D reference points with ego-motion shift: [shift_ref_2d, ref_2d] -> (B, 2, bev_h * bev_w, 1, 2)
        ref_2d = get_bev_reference_points(
            bev_height, bev_width, dim="2d", batch_size=batch_size, device=tensor_device, dtype=torch.float32
        )
        can_bus_tensor = None
        if can_bus is not None:
            can_bus_tensor = torch.as_tensor(can_bus, dtype=torch.float32, device=tensor_device)
            if can_bus_tensor.ndim == 1:
                can_bus_tensor = can_bus_tensor.unsqueeze(0)
            shift = compute_ego_motion_shift(can_bus_tensor, point_cloud_range, bev_height, bev_width)
            shift_ref_2d = ref_2d + shift[:, None, None, :]
        else:
            shift_ref_2d = ref_2d
        reference_points_2d = torch.stack([shift_ref_2d, ref_2d], dim=1)

        data: dict[str, torch.Tensor] = {
            "pixel_values": pixel_values,
            "reference_points_2d": reference_points_2d,
        }

        if lidar2img_tensor is not None:
            lidar2img_scaled = torch.stack(scaled_lidar2img, dim=0)
            z_height = point_cloud_range[5] - point_cloud_range[2]
            ref_3d = get_bev_reference_points(
                bev_height,
                bev_width,
                z_height=z_height,
                num_points_in_pillar=num_points_in_pillar,
                dim="3d",
                batch_size=batch_size,
                device=tensor_device,
                dtype=torch.float32,
            )
            reference_points_cam, bev_mask = point_sampling(
                ref_3d, point_cloud_range, lidar2img_scaled, img_shape=(padded_height, padded_width)
            )
            data["reference_points_cam"] = reference_points_cam
            data["bev_mask"] = bev_mask

        if can_bus_tensor is not None:
            data["can_bus"] = can_bus_tensor

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.1,
        top_k: int = 300,
        point_cloud_range: list[float] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Converts the raw outputs of [`BevformerFor3DObjectDetection`] into 3D bounding boxes in the LiDAR coordinate
        frame, with confidence scores and class labels.

        Args:
            outputs ([`BevformerObjectDetectionOutput`]):
                Raw outputs from [`BevformerFor3DObjectDetection`], with `pred_boxes` centers `(cx, cy, cz)` in
                normalized `[0, 1]` space.
            threshold (`float`, *optional*, defaults to 0.1):
                Score threshold to filter detections.
            top_k (`int`, *optional*, defaults to 300):
                Maximum number of boxes to retain per sample.
            point_cloud_range (`list[float]`, *optional*, defaults to `self.point_cloud_range`):
                3D point cloud bounds `[x_min, y_min, z_min, x_max, y_max, z_max]` used to denormalize box centers.

        Returns:
            `list[dict[str, torch.Tensor]]`:
                A list of dictionaries containing:
                - `"scores"`: Confidence scores for each detected box.
                - `"labels"`: Class label indices for each detected box.
                - `"boxes_3d"`: 9-parameter 3D bounding boxes `[cx, cy, cz_bottom, w, l, h, yaw, vx, vy]` in LiDAR frame.
        """
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        pc_range = point_cloud_range if point_cloud_range is not None else self.point_cloud_range
        pc_min = pred_boxes.new_tensor(pc_range[:3])
        pc_diff = pred_boxes.new_tensor(pc_range[3:]) - pc_min

        batch_size = logits.shape[0]
        results = []

        for b in range(batch_size):
            scores = logits[b].sigmoid()
            boxes = pred_boxes[b].clone()

            # Denormalize (cx, cy, cz) from [0, 1] into metric LiDAR coordinates
            boxes[:, 0] = boxes[:, 0] * pc_diff[0] + pc_min[0]
            boxes[:, 1] = boxes[:, 1] * pc_diff[1] + pc_min[1]
            boxes[:, 4] = boxes[:, 4] * pc_diff[2] + pc_min[2]

            # Upstream 10-vector: cx, cy, w, l, cz, h, sin_yaw, cos_yaw, vx, vy
            # Adjust cz to bottom-center z: z_bottom = cz - 0.5 * h
            boxes[:, 4] = boxes[:, 4] - 0.5 * boxes[:, 5]

            # Compute continuous yaw angle in radians
            yaw = torch.atan2(boxes[:, 6], boxes[:, 7]).unsqueeze(-1)
            # Reconstruct 9-dim box [cx, cy, cz_bottom, w, l, h, yaw, vx, vy]
            boxes_3d = torch.cat(
                [boxes[:, :2], boxes[:, 4:5], boxes[:, 2:4], boxes[:, 5:6], yaw, boxes[:, 8:10]], dim=-1
            )

            max_scores, labels = scores.max(dim=-1)
            keep = max_scores >= threshold
            scores_filtered = max_scores[keep]
            labels_filtered = labels[keep]
            boxes_filtered = boxes_3d[keep]

            if len(scores_filtered) > top_k:
                topk_idx = scores_filtered.topk(top_k).indices
                scores_filtered = scores_filtered[topk_idx]
                labels_filtered = labels_filtered[topk_idx]
                boxes_filtered = boxes_filtered[topk_idx]

            results.append(
                {
                    "scores": scores_filtered,
                    "labels": labels_filtered,
                    "boxes_3d": boxes_filtered,
                }
            )

        return results


class BevformerConvEncoder(nn.Module):
    """
    Multi-view convolutional backbone encoder for BEVFormer.

    Wraps a convolutional backbone (e.g. ResNet-50) using `load_backbone`.
    Multi-view camera inputs of shape `(batch_size, num_cameras, channels, height, width)`
    are reshaped to `(batch_size * num_cameras, channels, height, width)` before being passed
    through the backbone, and the multi-camera dimension is restored on the output feature map.

    `nn.BatchNorm2d` layers are replaced by `BevformerFrozenBatchNorm2d` via `replace_batch_norm`.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config

        backbone = load_backbone(config)
        self.intermediate_channel_sizes = backbone.channels

        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, num_cameras, num_channels, height, width)`):
                Multi-view camera input images.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_cameras, out_channels, out_height, out_width)`:
                The extracted stage4 feature map for each camera view.
        """
        batch_size, num_cameras, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_cameras, channels, height, width)

        feature_map = self.model(pixel_values).feature_maps[-1]

        _, out_channels, out_height, out_width = feature_map.shape
        feature_map = feature_map.view(batch_size, num_cameras, out_channels, out_height, out_width)
        return feature_map


class BevformerFPN(nn.Module):
    """
    Feature Pyramid Network (FPN) neck for BEVFormer.

    For BEVFormer-tiny, projects single-level stage4 (C5) backbone features into `d_model`
    channels via a 1x1 lateral convolution followed by a 3x3 output convolution.
    Submodule containers `lateral_convs` and `fpn_convs` align with upstream MMDetection FPN
    naming conventions (`img_neck.lateral_convs.0` and `img_neck.fpn_convs.0`).
    """

    def __init__(self, config: BevformerConfig, in_channels: int | list[int] | None = None):
        super().__init__()
        self.config = config

        if in_channels is None:
            if hasattr(config, "backbone_config") and config.backbone_config is not None:
                hidden_sizes = getattr(config.backbone_config, "hidden_sizes", None)
                in_channels = hidden_sizes[-1] if hidden_sizes else 2048
            else:
                in_channels = 2048
        elif isinstance(in_channels, (list, tuple)):
            in_channels = in_channels[-1]

        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels, config.d_model, kernel_size=1)])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(config.d_model, config.d_model, kernel_size=3, padding=1)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (`torch.Tensor` of shape `(batch_size, num_cameras, in_channels, height, width)` or `(batch_size * num_cameras, in_channels, height, width)`):
                Multi-view feature maps from the backbone.

        Returns:
            `torch.Tensor`: The projected and smoothed feature map, matching the input batch/camera dimensioning
            with channel dimension `config.d_model`.
        """
        restore_shape = False
        if features.ndim == 5:
            batch_size, num_cameras, channels, height, width = features.shape
            features = features.view(batch_size * num_cameras, channels, height, width)
            restore_shape = True

        lateral = self.lateral_convs[0](features)
        out = self.fpn_convs[0](lateral)

        if restore_shape:
            _, out_channels, out_height, out_width = out.shape
            out = out.view(batch_size, num_cameras, out_channels, out_height, out_width)

        return out


class BevformerLearnedPositionalEncoding(nn.Module):
    """
    Learned 2D Bird's-Eye-View (BEV) positional encoding for BEVFormer.

    Decomposes the 2D grid positional encoding into independent row and column embeddings
    that are concatenated along the feature dimension to form a `d_model`-dimensional embedding
    at each `(y, x)` grid position. Submodule names `row_embed` and `col_embed` exactly match
    upstream MMCV `LearnedPositionalEncoding` (`pts_bbox_head.positional_encoding.row_embed`
    and `pts_bbox_head.positional_encoding.col_embed`).

    Args:
        config ([`BevformerConfig`]):
            Configuration of the model.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        num_feats = config.d_model // 2
        self.row_embed = nn.Embedding(config.bev_height, num_feats)
        self.col_embed = nn.Embedding(config.bev_width, num_feats)

    def forward(self, mask: torch.Tensor | None = None, batch_size: int = 1) -> torch.Tensor:
        """
        Args:
            mask (`torch.Tensor`, *optional*):
                Optional attention mask of shape `(batch_size, bev_height, bev_width)`. If provided,
                the batch size is inferred from the mask.
            batch_size (`int`, *optional*, defaults to 1):
                Batch size to expand the positional embeddings over if `mask` is not provided.

        Returns:
            `torch.Tensor` of shape `(batch_size, bev_height * bev_width, d_model)`:
                Flattened 2D learned BEV positional embeddings.
        """
        if mask is not None:
            batch_size = mask.shape[0]

        height, width = self.config.bev_height, self.config.bev_width
        y = torch.arange(height, device=self.row_embed.weight.device)
        x = torch.arange(width, device=self.col_embed.weight.device)

        x_embed = self.col_embed(x).unsqueeze(0).repeat(height, 1, 1)  # (H, W, num_feats)
        y_embed = self.row_embed(y).unsqueeze(1).repeat(1, width, 1)  # (H, W, num_feats)

        pos = torch.cat([x_embed, y_embed], dim=-1)  # (H, W, d_model)
        pos = pos.flatten(0, 1).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, H * W, d_model)
        return pos


class BevformerQueryEmbedding(nn.Module):
    """
    Embedding holder and generator for BEV and object detection queries in BEVFormer.

    Encompasses:
    - `bev_embedding`: Learned BEV query grid (`pts_bbox_head.transformer.encoder.bev_embedding.weight`).
    - `query_embedding`: Concatenated object queries and position embeddings
      (`pts_bbox_head.transformer.decoder.query_embedding.weight`).
    - `reference_points`: Linear projection mapping object query position embeddings to initial
      normalized 3D coordinates (`pts_bbox_head.reference_points.weight` and `.bias`).

    Args:
        config ([`BevformerConfig`]):
            Configuration of the model.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.bev_embedding = nn.Embedding(config.bev_height * config.bev_width, config.d_model)
        self.query_embedding = nn.Embedding(config.num_queries, config.d_model * 2)
        self.reference_points = nn.Linear(config.d_model, 3)

    def get_bev_queries(self, batch_size: int = 1) -> torch.Tensor:
        """
        Returns:
            `torch.Tensor` of shape `(batch_size, bev_height * bev_width, d_model)`:
                The initial BEV query embeddings expanded across the batch.
        """
        return self.bev_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

    def get_object_queries(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:
                - `query` of shape `(batch_size, num_queries, d_model)`: Content query embeddings.
                - `query_pos` of shape `(batch_size, num_queries, d_model)`: Positional query embeddings.
                - `reference_points` of shape `(batch_size, num_queries, 3)`: Normalized initial 3D coordinates in `[0, 1]`.
        """
        weights = self.query_embedding.weight
        query = weights[:, : self.config.d_model].unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = weights[:, self.config.d_model :].unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = torch.sigmoid(self.reference_points(query_pos))
        return query, query_pos, reference_points


class BevformerCanBusMLP(nn.Module):
    """
    MLP encoder for ego-motion CAN bus signals in BEVFormer.

    Encodes vehicle translation, rotation, velocity and acceleration signals into an additive
    conditioning embedding for the BEV query plane.

    Args:
        config ([`BevformerConfig`]):
            Configuration of the model.
        can_bus_dim (`int`, *optional*):
            Dimensionality of input CAN bus vector. Defaults to `config.can_bus_dim` (18).
    """

    def __init__(self, config: BevformerConfig, can_bus_dim: int | None = None):
        super().__init__()
        self.config = config
        can_bus_dim = can_bus_dim if can_bus_dim is not None else config.can_bus_dim
        self.fc1 = nn.Linear(can_bus_dim, config.d_model // 2)
        self.fc2 = nn.Linear(config.d_model // 2, config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.can_bus_norm = nn.LayerNorm(config.d_model) if config.can_bus_norm else None

    def forward(self, can_bus: torch.Tensor) -> torch.Tensor:
        """
        Args:
            can_bus (`torch.Tensor` of shape `(batch_size, can_bus_dim)`):
                Raw ego CAN bus motion features.

        Returns:
            `torch.Tensor` of shape `(batch_size, 1, d_model)`:
                Encoded CAN bus conditioning embedding.
        """
        can_bus_embed = self.activation_fn(self.fc1(can_bus))
        can_bus_embed = self.activation_fn(self.fc2(can_bus_embed))
        if self.can_bus_norm is not None:
            can_bus_embed = self.can_bus_norm(can_bus_embed)
        return can_bus_embed.unsqueeze(1)


class BevformerClsHead(nn.Module):
    """
    Classification head for BEVFormer object detection.

    Projects transformer decoder hidden states into raw class logits (pre-sigmoid).

    Args:
        config ([`BevformerConfig`]):
            Configuration of the model.
        num_classes (`int`, *optional*):
            Number of object detection classes. Defaults to `len(config.id2label)`.
    """

    def __init__(self, config: BevformerConfig, num_classes: int | None = None):
        super().__init__()
        self.config = config
        num_classes = num_classes if num_classes is not None else len(config.id2label)
        self.fc1 = nn.Linear(config.d_model, config.d_model)
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.fc2 = nn.Linear(config.d_model, config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.fc_out = nn.Linear(config.d_model, num_classes)
        self.activation_fn = ACT2FN[config.activation_function]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_queries, d_model)`):
                Decoder hidden states.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, num_classes)`:
                Raw classification scores.
        """
        hidden_states = self.activation_fn(self.layer_norm1(self.fc1(hidden_states)))
        hidden_states = self.activation_fn(self.layer_norm2(self.fc2(hidden_states)))
        return self.fc_out(hidden_states)


class BevformerRegHead(DeformableDetrMLPPredictionHead):
    pass


class BevformerMultiscaleDeformableAttention(DeformableDetrMultiscaleDeformableAttention):
    def __init__(self, config: BevformerConfig, num_heads: int, n_points: int):
        super().__init__(config, num_heads=num_heads, n_points=n_points)
        self.n_levels = 1
        del self.disable_custom_kernels


class BevformerSpatialDeformableAttention(DeformableDetrMultiscaleDeformableAttention):
    """Each BEV query samples `num_points_in_pillar` z-anchors, the sampling offsets are split across them."""

    def __init__(self, config: BevformerConfig, num_heads: int, n_points: int):
        super().__init__(config, num_heads=num_heads, n_points=n_points)
        del self.output_proj
        del self.disable_custom_kernels

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        _, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
        torch_compilable_check(
            total_elements == sequence_length,
            "Make sure to align the spatial shapes with the sequence length of the encoder hidden states",
        )

        value = self.value_proj(encoder_hidden_states)
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )

        num_z_anchors = reference_points.shape[2]
        if self.n_points % num_z_anchors != 0:
            raise ValueError(f"n_points ({self.n_points}) must be divisible by num_z_anchors ({num_z_anchors})")
        num_points_per_anchor = self.n_points // num_z_anchors

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_offsets = sampling_offsets.view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            num_points_per_anchor,
            num_z_anchors,
            2,
        )
        sampling_locations = (
            reference_points[:, :, None, None, None, :, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]
        ).view(batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2)

        return self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )


class BevformerTemporalSelfAttention(DeformableDetrMultiscaleDeformableAttention):
    """Temporal self-attention over a 2-frame BEV queue (`[prev_bev, current_bev]`)."""

    def __init__(self, config: BevformerConfig, num_heads: int, n_points: int, num_bev_queue: int = 2):
        super().__init__(config, num_heads=num_heads, n_points=n_points)
        del self.disable_custom_kernels
        self.n_levels = 1
        self.num_bev_queue = num_bev_queue
        # Offsets and weights are conditioned on [prev_bev, current_bev] across the temporal queue
        self.sampling_offsets = nn.Linear(2 * self.d_model, num_bev_queue * num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(2 * self.d_model, num_bev_queue * num_heads * self.n_levels * n_points)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        total_batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
        torch_compilable_check(
            total_elements == sequence_length,
            "Make sure to align the spatial shapes with the sequence length of the encoder hidden states",
        )

        # Delta 1: condition offsets and attention weights on [prev_bev, current_bev]
        prev_bev = encoder_hidden_states.view(batch_size, self.num_bev_queue, sequence_length, self.d_model)[:, 0]
        query_concat = torch.cat([prev_bev, hidden_states], dim=-1)

        value = self.value_proj(encoder_hidden_states).view(
            total_batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads
        )
        # Delta 2: fold num_bev_queue into the batch dimension while preserving upstream (n_heads, num_bev_queue) layout
        sampling_offsets = (
            self.sampling_offsets(query_concat)
            .view(batch_size, num_queries, self.n_heads, self.num_bev_queue, self.n_levels, self.n_points, 2)
            .movedim(3, 1)
            .reshape(total_batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2)
        )
        attention_weights = (
            F.softmax(
                self.attention_weights(query_concat).view(
                    batch_size, num_queries, self.n_heads, self.num_bev_queue, self.n_levels * self.n_points
                ),
                dim=-1,
            )
            .view(batch_size, num_queries, self.n_heads, self.num_bev_queue, self.n_levels, self.n_points)
            .movedim(3, 1)
            .reshape(total_batch_size, num_queries, self.n_heads, self.n_levels, self.n_points)
        )

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        # Delta 3: fuse over the temporal queue via mean before output projection
        output = output.view(batch_size, self.num_bev_queue, num_queries, self.d_model).mean(dim=1)
        output = self.output_proj(output)
        return output, attention_weights


class BevformerSpatialCrossAttention(nn.Module):
    """Spatial Cross Attention (SCA) module for BEVFormer.

    Extracts multi-camera image features for each BEV query using pre-projected 3D pillar reference points,
    selecting valid frustum hits per camera and sampling multi-scale deformable attention offsets.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.embed_dims = config.d_model
        self.num_cams = config.num_cameras

        self.deformable_attention = BevformerSpatialDeformableAttention(
            config,
            num_heads=config.encoder_attention_heads,
            n_points=config.num_points_in_bev,
        )
        self.output_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        reference_points_cam: torch.Tensor | None = None,
        bev_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slots = torch.zeros_like(hidden_states)

        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        reference_points_cam = reference_points_cam.to(hidden_states.dtype)
        batch_size, _, _ = hidden_states.shape
        num_z_anchors = reference_points_cam.shape[3]

        indexes = [
            [bev_mask[b, cam_idx].any(dim=-1).nonzero().squeeze(-1) for cam_idx in range(self.num_cams)]
            for b in range(batch_size)
        ]
        max_len = max((len(idx) for sample_indexes in indexes for idx in sample_indexes), default=0)

        if max_len > 0:
            queries_rebatch = hidden_states.new_zeros([batch_size, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [batch_size, self.num_cams, max_len, num_z_anchors, 2]
            )

            for b in range(batch_size):
                for cam_idx, idx in enumerate(indexes[b]):
                    if len(idx) > 0:
                        queries_rebatch[b, cam_idx, : len(idx)] = hidden_states[b, idx]
                        reference_points_rebatch[b, cam_idx, : len(idx)] = reference_points_cam[b, cam_idx, idx]

            num_cams, num_keys, bs, embed_dims = encoder_hidden_states.shape
            value_reshaped = encoder_hidden_states.permute(2, 0, 1, 3).reshape(
                bs * self.num_cams, num_keys, self.embed_dims
            )

            queries_in = queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims)
            ref_pts_in = reference_points_rebatch.view(bs * self.num_cams, max_len, num_z_anchors, 2)

            queries_out = self.deformable_attention(
                hidden_states=queries_in,
                encoder_hidden_states=value_reshaped,
                reference_points=ref_pts_in,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
            ).view(bs, self.num_cams, max_len, self.embed_dims)

            for b in range(batch_size):
                for cam_idx, idx in enumerate(indexes[b]):
                    if len(idx) > 0:
                        slots[b, idx] += queries_out[b, cam_idx, : len(idx)]

        # Average over visible views observing each query pillar
        count = bev_mask.any(dim=-1).sum(dim=1).clamp(min=1.0).to(slots.dtype)
        slots = slots / count.unsqueeze(-1)
        return self.output_proj(slots)


class BevformerMLP(DeformableDetrMLP):
    pass


class BevformerEncoderLayer(DeformableDetrDecoderLayer):
    def __init__(self, config: BevformerConfig):
        super().__init__(config)
        self.self_attn = BevformerTemporalSelfAttention(
            config,
            num_heads=config.encoder_attention_heads,
            n_points=config.encoder_n_points,
        )
        self.encoder_attn = BevformerSpatialCrossAttention(config)
        self.mlp = BevformerMLP(config, self.hidden_size, config.encoder_ffn_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        ref_2d: torch.Tensor | None = None,
        bev_spatial_shapes: torch.Tensor | None = None,
        bev_spatial_shapes_list: list[tuple[int, int]] | None = None,
        bev_level_start_index: torch.Tensor | None = None,
        reference_points_cam: torch.Tensor | None = None,
        bev_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # 1. Temporal self-attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=prev_bev,
            position_embeddings=position_embeddings,
            reference_points=ref_2d,
            spatial_shapes=bev_spatial_shapes,
            spatial_shapes_list=bev_spatial_shapes_list,
            level_start_index=bev_level_start_index,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        # 2. Spatial cross-attention
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # 3. Feed-forward network
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BevformerEncoder(nn.Module):
    """BEVFormer BEV Encoder.

    Encapsulates multi-layer BEV refinement, CAN bus MLP injection, previous-BEV rotation,
    and camera/level feature embedding.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.encoder_layers
        self.bev_h = config.bev_height
        self.bev_w = config.bev_width
        self.embed_dims = config.d_model
        self.rotate_center = config.rotate_center

        self.layers = nn.ModuleList([BevformerEncoderLayer(config) for _ in range(self.num_layers)])
        self.can_bus_mlp = BevformerCanBusMLP(config)

        # Camera & Level embeddings
        self.level_embeds = nn.Parameter(torch.empty(config.num_feature_levels, config.d_model))
        self.cams_embeds = nn.Parameter(torch.empty(config.num_cameras, config.d_model))

    def apply_prev_bev_rotation(self, prev_bev: torch.Tensor, can_bus: torch.Tensor) -> torch.Tensor:
        """Apply yaw rotation to previous BEV embeddings in a single batched, on-device op.

        Matches upstream `torchvision.transforms.functional.rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)`
        bitwise via `F.affine_grid` + `F.grid_sample` (`mode="nearest"`, `padding_mode="zeros"`, `align_corners=False`):
        - `can_bus[:, -1]` holds the ego yaw (or frame-to-frame delta yaw) in degrees
          (see https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L161-L165
          and https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py#L146-L156).
        - Upstream quirk: `PerceptionTransformer` defaults `rotate_center=[100, 100]`
          (https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py#L50),
          which `bevformer_tiny.py` does not override despite using a `50x50` BEV grid.

        Args:
            prev_bev (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Previous BEV embeddings.
            can_bus (`torch.Tensor` of shape `(batch_size, 18)`):
                CAN bus status, with yaw / delta-yaw angle in degrees at index `-1`.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`:
                Rotated BEV embeddings.
        """
        batch_size = prev_bev.shape[0]
        rad = can_bus[:, -1].to(torch.float32) * (math.pi / 180.0)
        cos_a = torch.cos(rad)
        sin_a = torch.sin(rad)

        # Upstream quirk: rotate_center defaults to [100.0, 100.0] even on a 50x50 grid
        cx = self.rotate_center[0] - 0.5 * self.bev_w
        cy = self.rotate_center[1] - 0.5 * self.bev_h
        tx = (cx * (1.0 - cos_a) + cy * sin_a) / (0.5 * self.bev_w)
        ty = (-cx * sin_a + cy * (1.0 - cos_a)) / (0.5 * self.bev_h)

        theta = torch.stack(
            [
                torch.stack([cos_a, -sin_a * (self.bev_h / self.bev_w), tx], dim=-1),
                torch.stack([sin_a * (self.bev_w / self.bev_h), cos_a, ty], dim=-1),
            ],
            dim=1,
        ).to(prev_bev.dtype)

        grid = F.affine_grid(theta, size=(batch_size, self.embed_dims, self.bev_h, self.bev_w), align_corners=False)
        bev_map = prev_bev.view(batch_size, self.bev_h, self.bev_w, self.embed_dims).permute(0, 3, 1, 2)
        rotated = F.grid_sample(bev_map, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        return rotated.permute(0, 2, 3, 1).reshape(batch_size, self.bev_h * self.bev_w, self.embed_dims)

    def forward(
        self,
        bev_queries: torch.Tensor,
        multi_level_feats: list[torch.Tensor] | tuple[torch.Tensor, ...],
        bev_pos: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        reference_points_2d: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = bev_queries.shape
        device = bev_queries.device
        dtype = bev_queries.dtype

        if can_bus is None:
            can_bus = bev_queries.new_zeros((batch_size, 18))

        # 1. Inject CAN bus signals into BEV queries
        bev_queries = bev_queries + self.can_bus_mlp(can_bus)

        # 2. Build hybrid 2D reference points and temporal queue
        reference_points_2d = reference_points_2d.to(dtype)
        if prev_bev is not None:
            prev_bev_rotated = self.apply_prev_bev_rotation(prev_bev, can_bus)
            temporal_bev_queue = torch.stack([prev_bev_rotated, bev_queries], dim=1).reshape(
                batch_size * 2, num_queries, -1
            )
            hybrid_ref_2d = reference_points_2d.reshape(batch_size * 2, num_queries, 1, 2)
        else:
            temporal_bev_queue = None
            ref_2d = reference_points_2d[:, 1]
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], dim=1).reshape(batch_size * 2, num_queries, 1, 2)

        # 6. Flatten multi-level features and add camera + level embeddings
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(multi_level_feats):
            # feat: (batch_size, num_cams, C, H, W)
            bs, num_cams, c, h, w = feat.shape
            spatial_shape = (h, w)
            # Permute to (num_cams, batch_size, H*W, C)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        # Concatenate along the spatial sequence dimension: (num_cams, batch_size, total_keys, C)
        feat_flatten = torch.cat(feat_flatten, dim=2)
        # Permute to (num_cams, total_keys, batch_size, C) matching upstream SCA
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat(
            (spatial_shapes_tensor.new_zeros((1,)), spatial_shapes_tensor.prod(1).cumsum(0)[:-1])
        )

        bev_spatial_shapes_list = [(self.bev_h, self.bev_w)]
        bev_spatial_shapes = torch.as_tensor(bev_spatial_shapes_list, dtype=torch.long, device=device)
        bev_level_start_index = bev_spatial_shapes.new_zeros((1,))

        # 7. Execute encoder layers iteratively
        output = bev_queries
        for layer in self.layers:
            layer_prev_bev = (
                temporal_bev_queue
                if temporal_bev_queue is not None
                else torch.stack([output, output], dim=1).reshape(batch_size * 2, num_queries, -1)
            )
            output = layer(
                hidden_states=output,
                encoder_hidden_states=feat_flatten,
                position_embeddings=bev_pos,
                prev_bev=layer_prev_bev,
                ref_2d=hybrid_ref_2d,
                bev_spatial_shapes=bev_spatial_shapes,
                bev_spatial_shapes_list=bev_spatial_shapes_list,
                bev_level_start_index=bev_level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                spatial_shapes=spatial_shapes_tensor,
                spatial_shapes_list=spatial_shapes,
                level_start_index=level_start_index,
            )

        return output


class BevformerSelfAttention(DeformableDetrSelfAttention):
    pass


class BevformerDecoderLayer(DeformableDetrDecoderLayer):
    pass


class BevformerReferencePointRefiner(nn.Module):
    """Applies one stage of iterative 3D reference-point refinement and returns `(reference_points, pred_boxes)`."""

    def forward(self, bbox_deltas: torch.Tensor, reference_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_ref = inverse_sigmoid(reference_points)
        pred_xy = (bbox_deltas[..., 0:2] + inv_ref[..., 0:2]).sigmoid()
        pred_z = (bbox_deltas[..., 4:5] + inv_ref[..., 2:3]).sigmoid()
        pred_boxes = torch.cat([pred_xy, bbox_deltas[..., 2:4], pred_z, bbox_deltas[..., 5:]], dim=-1)
        next_reference_points = torch.cat([pred_xy, pred_z], dim=-1).detach()
        return next_reference_points, pred_boxes


@dataclass
class BevformerDecoderOutput(BaseModelOutputWithCrossAttentions):
    """Output type of [`BevformerDecoder`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, d_model)`):
            Sequence of hidden states at the output of the last decoder layer.
        reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 3)`):
            Normalized 3D reference points `(cx, cy, cz)` after the final decoder layer.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 10)`, *optional*):
            Normalized 3D box predictions from the final decoder layer when iterative box refinement (`bbox_embed`)
            is attached.
        intermediate_reference_points (`tuple[torch.FloatTensor]`, *optional*):
            Per-layer refined 3D reference points recorded when `output_intermediate_reference_points=True`.
    """

    reference_points: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    intermediate_reference_points: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BevformerModelOutput(ModelOutput):
    """Output type of [`BevformerModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, d_model)`):
            Sequence of hidden states at the output of the last decoder layer.
        bev_embed (`torch.FloatTensor` of shape `(batch_size, bev_height * bev_width, d_model)`):
            Bird's-Eye-View (BEV) embedding generated by the BEV encoder. Can be passed as `prev_bev` to
            subsequent frames.
        init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 3)`):
            Initial normalized 3D reference points passed to the decoder.
        reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 3)`):
            Final normalized 3D reference points returned by the decoder.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 10)`, *optional*):
            Normalized 3D box predictions from the final decoder layer when `decoder.bbox_embed` is attached.
        decoder_hidden_states (`tuple[torch.FloatTensor]`, *optional*):
            Decoder hidden states at the output of each layer (plus initial embeddings) when
            `output_hidden_states=True`.
        decoder_attentions (`tuple[torch.FloatTensor]`, *optional*):
            Decoder self-attention weights when `output_attentions=True`.
        cross_attentions (`tuple[torch.FloatTensor]`, *optional*):
            Decoder deformable cross-attention weights when `output_attentions=True`.
        intermediate_reference_points (`tuple[torch.FloatTensor]`, *optional*):
            Per-layer refined reference points when `output_intermediate_reference_points=True`.
    """

    last_hidden_state: torch.FloatTensor | None = None
    bev_embed: torch.FloatTensor | None = None
    init_reference_points: torch.FloatTensor | None = None
    reference_points: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    intermediate_reference_points: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BevformerObjectDetectionOutput(ModelOutput):
    """Output type of [`BevformerFor3DObjectDetection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`):
            Classification logits (pre-sigmoid) for each query from the final decoder layer.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 10)`):
            Predicted 3D bounding box coordinates `[cx, cy, w, l, cz, h, sin(yaw), cos(yaw), vx, vy]` from the final
            decoder layer, with `(cx, cy, cz)` in normalized `[0, 1]` space relative to `config.point_cloud_range`.
        bev_embed (`torch.FloatTensor` of shape `(batch_size, bev_height * bev_width, d_model)`):
            Bird's-Eye-View (BEV) embedding generated by the BEV encoder. Can be passed as `prev_bev` to
            subsequent frames.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, d_model)`):
            Sequence of hidden states at the output of the last decoder layer.
        init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 3)`):
            Initial normalized 3D reference points passed to the decoder.
        reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 3)`):
            Final normalized 3D reference points returned by the decoder.
        decoder_hidden_states (`tuple[torch.FloatTensor]`, *optional*):
            Decoder hidden states when `output_hidden_states=True`.
        decoder_attentions (`tuple[torch.FloatTensor]`, *optional*):
            Decoder self-attention weights when `output_attentions=True`.
        cross_attentions (`tuple[torch.FloatTensor]`, *optional*):
            Decoder deformable cross-attention weights when `output_attentions=True`.
        intermediate_reference_points (`tuple[torch.FloatTensor]`, *optional*):
            Per-layer refined reference points when `output_intermediate_reference_points=True`.
    """

    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    bev_embed: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    init_reference_points: torch.FloatTensor | None = None
    reference_points: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    intermediate_reference_points: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
class BevformerPreTrainedModel(PreTrainedModel):
    config: BevformerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _keys_to_ignore_on_load_unexpected = [
        r".*\.num_batches_tracked",
        r"pts_bbox_head\.code_weights",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, BevformerEncoder):
            init.normal_(module.level_embeds)
            init.normal_(module.cams_embeds)
        elif isinstance(
            module,
            (
                BevformerMultiscaleDeformableAttention,
                BevformerSpatialDeformableAttention,
                BevformerTemporalSelfAttention,
            ),
        ):
            init.constant_(module.sampling_offsets.weight, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            if isinstance(module, BevformerTemporalSelfAttention):
                grid_init = grid_init.unsqueeze(1).repeat(1, module.num_bev_queue, 1, 1, 1)

            init.copy_(module.sampling_offsets.bias, grid_init.reshape(-1))
            init.constant_(module.attention_weights.weight, 0.0)
            init.constant_(module.attention_weights.bias, 0.0)
            init.xavier_uniform_(module.value_proj.weight)
            init.constant_(module.value_proj.bias, 0.0)
            if not isinstance(module, BevformerSpatialDeformableAttention):
                init.xavier_uniform_(module.output_proj.weight)
                init.constant_(module.output_proj.bias, 0.0)


class BevformerDecoder(BevformerPreTrainedModel):
    """Detection transformer decoder for BEVFormer with iterative 3D reference point refinement."""

    _can_record_outputs = {
        "hidden_states": BevformerDecoderLayer,
        "attentions": OutputRecorder(BevformerSelfAttention, layer_name="self_attn", index=1),
        "cross_attentions": OutputRecorder(BevformerMultiscaleDeformableAttention, layer_name="encoder_attn", index=1),
        "intermediate_reference_points": OutputRecorder(BevformerReferencePointRefiner, index=0),
    }

    def __init__(self, config: BevformerConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([BevformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_point_refiner = BevformerReferencePointRefiner()
        self.bbox_embed = None
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        object_queries_position_embeddings: torch.Tensor,
        reference_points: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BevformerDecoderOutput:
        spatial_shapes_list = [(self.config.bev_height, self.config.bev_width)]
        spatial_shapes = inputs_embeds.new_tensor(spatial_shapes_list, dtype=torch.long)
        level_start_index = inputs_embeds.new_tensor([0], dtype=torch.long)

        hidden_states = inputs_embeds
        pred_boxes = None

        for lid, layer in enumerate(self.layers):
            ref_input = reference_points[..., :2].unsqueeze(2)  # (batch_size, num_queries, 1, 2)
            hidden_states = layer(
                hidden_states,
                object_queries_position_embeddings=object_queries_position_embeddings,
                reference_points=ref_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )

            if self.bbox_embed is not None:
                bbox_deltas = self.bbox_embed[lid](hidden_states)
                reference_points, pred_boxes = self.reference_point_refiner(bbox_deltas, reference_points)

        return BevformerDecoderOutput(
            last_hidden_state=hidden_states,
            reference_points=reference_points,
            pred_boxes=pred_boxes,
        )


@auto_docstring
class BevformerModel(BevformerPreTrainedModel):
    def __init__(self, config: BevformerConfig):
        super().__init__(config)
        self.config = config
        self.conv_encoder = BevformerConvEncoder(config)
        self.fpn = BevformerFPN(config)
        self.positional_encoding = BevformerLearnedPositionalEncoding(config)
        self.query_embedding = BevformerQueryEmbedding(config)
        self.encoder = BevformerEncoder(config)
        self.decoder = BevformerDecoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.conv_encoder

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        reference_points_2d: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BevformerModelOutput:
        r"""
        reference_points_cam (`torch.Tensor` of shape `(batch_size, num_cameras, bev_height * bev_width, num_points_in_pillar, 2)`):
            Pre-projected normalized 2D image coordinates of 3D BEV pillar anchor points for each camera, as returned
            by [`BevformerImageProcessor`].
        bev_mask (`torch.Tensor` of shape `(batch_size, num_cameras, bev_height * bev_width, num_points_in_pillar)`):
            Boolean visibility mask indicating which projected pillar points fall within each camera's frustum.
        reference_points_2d (`torch.Tensor` of shape `(batch_size, 2, bev_height * bev_width, 1, 2)`):
            2D BEV grid reference points `[shift_ref_2d, ref_2d]` for temporal self-attention.
        can_bus (`torch.Tensor` of shape `(batch_size, 18)`, *optional*):
            CAN bus ego-motion signals (translation, rotation, velocity, acceleration). Defaults to zeros if omitted.
        prev_bev (`torch.Tensor` of shape `(batch_size, bev_height * bev_width, d_model)`, *optional*):
            Bird's-eye-view embedding from the previous frame for temporal fusion.
        """
        batch_size = pixel_values.shape[0]

        # 1. Multi-camera feature extraction (backbone + FPN)
        features = self.conv_encoder(pixel_values)
        fpn_features = self.fpn(features)

        # 2. Positional encoding & initial BEV queries
        bev_pos = self.positional_encoding(batch_size=batch_size)
        bev_queries = self.query_embedding.get_bev_queries(batch_size=batch_size)

        # 3. BEV Encoder
        bev_embed = self.encoder(
            bev_queries=bev_queries,
            multi_level_feats=[fpn_features],
            bev_pos=bev_pos,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            reference_points_2d=reference_points_2d,
            can_bus=can_bus,
            prev_bev=prev_bev,
        )

        # 4. Object queries and detection transformer decoder
        query, query_pos, init_reference_points = self.query_embedding.get_object_queries(batch_size=batch_size)
        decoder_outputs = self.decoder(
            inputs_embeds=query,
            encoder_hidden_states=bev_embed,
            object_queries_position_embeddings=query_pos,
            reference_points=init_reference_points,
            **kwargs,
        )

        return BevformerModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            bev_embed=bev_embed,
            init_reference_points=init_reference_points,
            reference_points=decoder_outputs.reference_points,
            pred_boxes=decoder_outputs.pred_boxes,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
        )


@auto_docstring(
    custom_intro="""
    BEVFormer model with a 3D object detection head on top.
    """
)
class BevformerFor3DObjectDetection(BevformerPreTrainedModel):
    _tied_weights_keys = {"model.decoder.bbox_embed": "reg_branches"}

    def __init__(self, config: BevformerConfig):
        super().__init__(config)
        self.config = config
        self.model = BevformerModel(config)

        self.cls_branches = nn.ModuleList([BevformerClsHead(config) for _ in range(config.decoder_layers)])
        self.reg_branches = nn.ModuleList(
            [BevformerRegHead(config.d_model, config.d_model, 10, 3) for _ in range(config.decoder_layers)]
        )
        self.model.decoder.bbox_embed = self.reg_branches
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        reference_points_2d: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BevformerObjectDetectionOutput:
        r"""
        reference_points_cam (`torch.Tensor` of shape `(batch_size, num_cameras, bev_height * bev_width, num_points_in_pillar, 2)`):
            Pre-projected normalized 2D image coordinates of 3D BEV pillar anchor points for each camera, as returned
            by [`BevformerImageProcessor`].
        bev_mask (`torch.Tensor` of shape `(batch_size, num_cameras, bev_height * bev_width, num_points_in_pillar)`):
            Boolean visibility mask indicating which projected pillar points fall within each camera's frustum.
        reference_points_2d (`torch.Tensor` of shape `(batch_size, 2, bev_height * bev_width, 1, 2)`):
            2D BEV grid reference points `[shift_ref_2d, ref_2d]` for temporal self-attention.
        can_bus (`torch.Tensor` of shape `(batch_size, 18)`, *optional*):
            CAN bus ego-motion signals (translation, rotation, velocity, acceleration). Defaults to zeros if omitted.
        prev_bev (`torch.Tensor` of shape `(batch_size, bev_height * bev_width, d_model)`, *optional*):
            Bird's-eye-view embedding from the previous frame for temporal fusion.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            reference_points_2d=reference_points_2d,
            can_bus=can_bus,
            prev_bev=prev_bev,
            **kwargs,
        )

        logits = self.cls_branches[-1](outputs.last_hidden_state)

        return BevformerObjectDetectionOutput(
            logits=logits,
            pred_boxes=outputs.pred_boxes,
            bev_embed=outputs.bev_embed,
            last_hidden_state=outputs.last_hidden_state,
            init_reference_points=outputs.init_reference_points,
            reference_points=outputs.reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            intermediate_reference_points=outputs.intermediate_reference_points,
        )


__all__ = [
    "BevformerConfig",
    "BevformerFor3DObjectDetection",
    "BevformerImageProcessor",
    "BevformerModel",
    "BevformerPreTrainedModel",
]

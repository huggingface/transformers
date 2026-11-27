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

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import load_backbone
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import normalize
from ...image_utils import ImageInput, PILImageResampling
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...models.auto import AutoConfig
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TensorType,
    TransformersKwargs,
    auto_docstring,
    is_torch_tensor,
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


class BevformerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a BEVFormer image processor which rescales, normalizes and pads a stack of multi-view images, and
    rescales the matching `lidar2img` calibration matrices. This mirrors the inference pipeline of BEVFormer: the
    images are rescaled to `size` (by default `(450, 800)`, i.e. a 0.5 ratio of the original `(900, 1600)` inputs),
    normalized in the 0-255 pixel value space and padded at the bottom/right so that the spatial dimensions are
    multiples of `pad_size_divisor`. The same ratio is applied to the top two rows of the projection matrices so
    that projected pixel coordinates stay aligned with the rescaled images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image's (height, width) dimensions to the specified `size`. Can be overridden by
            the `do_resize` parameter in the `preprocess` method.
        size (`dict[str, int]`, *optional*, defaults to `{"height": 450, "width": 800}`):
            Size of the output image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Note that, unlike most image processors, the rescale step is not applied
            first and the statistics operate directly on the 0-255 pixel value space. Can be overridden by the
            `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[123.675, 116.28, 103.53]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `[58.395, 57.12, 57.375]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
            method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image at the bottom and at the right so that both dimensions are divisible by
            `pad_size_divisor`. Can be overridden by the `do_pad` parameter in the `preprocess` method.
        pad_size_divisor (`int`, *optional*, defaults to 32):
            Spatial dimensions of the padded images are set to multiples of this value.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        resample: PILImageResampling | int = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_pad: bool = True,
        pad_size_divisor: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 450, "width": 800}
        size = get_size_dict(size, param_name="size")

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else DEFAULT_IMAGE_MEAN
        self.image_std = image_std if image_std is not None else DEFAULT_IMAGE_STD
        self.do_pad = do_pad
        self.pad_size_divisor = pad_size_divisor

    def _to_numpy(self, value) -> np.ndarray:
        """Convert the supported input formats (nested lists, tensors or arrays) to a numpy array."""
        if is_torch_tensor(value):
            return value.cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.array([self._to_numpy(element) for element in value])
        return np.asarray(value)

    def _prepare_sample_list(self, images: ImageInput) -> list[np.ndarray]:
        """Normalize the supported multi-view stacking formats to a list of `(num_cameras, height, width, 3)` arrays."""
        if isinstance(images, (list, tuple)):
            elements = [self._to_numpy(element) for element in images]
            if all(element.ndim == 3 for element in elements):
                # A list of single images: it is the camera stack of one sample.
                return [np.stack(elements)]
            if all(element.ndim == 4 for element in elements):
                # A batch of stacked samples.
                return elements
            raise ValueError(
                "Multi-view images given as a list must contain either images of shape (height, width, 3) (the camera "
                "stack of a single sample) or arrays of shape (num_cameras, height, width, 3) (a batch of samples)."
            )

        array = self._to_numpy(images)
        if array.ndim == 5:
            return [array[index] for index in range(array.shape[0])]
        if array.ndim == 4:
            return [array]
        raise ValueError(
            "Multi-view images must be a batched array or tensor of shape "
            "(batch_size, num_cameras, height, width, 3), or an array of shape (num_cameras, height, width, 3)."
        )

    def resize(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        resample: PILImageResampling | int = PILImageResampling.BILINEAR,
        input_data_format=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Rescale an image to `(height, width)`. Supports rescaling by a fractional ratio, e.g. the 0.5 factor of the
        BEVFormer pipeline, where `(1600, 900)` becomes `(800, 450)`.

        The interpolation is performed by bilinear sampling with half-pixel centers on the full-precision pixel
        values, replicating the `cv2.resize` call (`INTER_LINEAR`) used by the upstream pipeline.

        Args:
            image (`np.ndarray`):
                The image to rescale, in `(height, width, num_channels)` format.
            size (`tuple[int, int]`):
                Target size `(height, width)` of the rescaled image.
            resample (`PILImageResampling` or `int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image. Only bilinear sampling is supported.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The rescaled image, in `(height, width, num_channels)` format.
        """
        if resample is not None and resample != PILImageResampling.BILINEAR:
            raise ValueError("BEVFormer image processing uses bilinear interpolation, so `resample` must be bilinear.")
        height, width = size
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(torch.float32)
        resized = F.interpolate(image_tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0).numpy()

    def normalize(
        self,
        image: np.ndarray,
        mean: float | list[float],
        std: float | list[float],
        input_data_format=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image in the 0-255 pixel value space with the given mean and standard deviation, i.e.
        `(image - mean) / std` without any prior `(0, 255) -> (0, 1)` rescaling.

        Args:
            image (`np.ndarray`):
                The image to normalize.
            mean (`float` or `list[float]`):
                The mean to use for normalization.
            std (`float` or `list[float]`):
                The standard deviation to use for normalization.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The normalized image.
        """
        return normalize(image=image, mean=mean, std=std, input_data_format=input_data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        lidar2img: ImageInput | None = None,
        return_tensors: str | TensorType | None = None,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        resample: PILImageResampling | int | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_pad: bool | None = None,
        pad_size_divisor: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Rescale, normalize and pad a stack of multi-view images and rescale the matching `lidar2img` projection
        matrices. The rescaling factor applied to the images (the ratio between the target and original sizes) is
        also applied to the top two rows of each `lidar2img` matrix, matching the scaling of the image plane in
        BEVFormer's inference pipeline.

        Args:
            images (`ImageInput`):
                Multi-view images in one of the following formats:
                - a stacked array or tensor of shape `(batch_size, num_cameras, height, width, 3)`
                - an array or tensor of shape `(num_cameras, height, width, 3)` (a single sample)
                - a list of samples of either of the above
                Images are expected in the RGB channel convention, with the channel dimension last.
            lidar2img (`ImageInput`, *optional*):
                Projection matrices from the lidar coordinate system to each camera plane, of shape
                `(batch_size, num_cameras, 4, 4)`, `(num_cameras, 4, 4)` or a (possibly nested) list. If provided,
                they are rescaled with the same ratio that `images` are.
            return_tensors (`str` or [`TensorType`], *optional*):
                The type of tensors to return. Can be one of [`TensorType.PYTORCH`], [`TensorType.NUMPY`] or the
                corresponding string (`"pt"`, `"np"`).
            do_resize (`bool`, *optional*):
                Whether to rescale the images. Can be overridden by `preprocess`.
            size (`dict[str, int]`, *optional*):
                Target size `{"height": h, "width": w}` of the rescaled images. Can be overridden by `preprocess`.
            resample (`PILImageResampling`, *optional*):
                Resampling filter to use when resizing the images. Can be overridden by `preprocess`.
            do_normalize (`bool`, *optional*):
                Whether to normalize the images in the 0-255 pixel value space. Can be overridden by `preprocess`.
            image_mean (`float` or `list[float]`, *optional*):
                Mean to use if normalizing the images. Can be overridden by `preprocess`.
            image_std (`float` or `list[float]`, *optional*):
                Standard deviation to use if normalizing the images. Can be overridden by `preprocess`.
            do_pad (`bool`, *optional*):
                Whether to pad the images to dimensions divisible by `pad_size_divisor`. Can be overridden by
                `preprocess`.
            pad_size_divisor (`int`, *optional*):
                Spatial dimensions of the padded images are set to multiples of this value. Can be overridden by
                `preprocess`.

        Returns:
            [`BatchFeature`] with the following fields:
            - **pixel_values** — the processed multi-view images, of shape
              `(batch_size, num_cameras, 3, height, width)`.
            - **lidar2img** — the rescaled projection matrices, of shape `(batch_size, num_cameras, 4, 4)`,
              only returned if `lidar2img` was provided.
            - **image_shapes** — the `(height, width)` sizes of the images after rescaling but before padding,
              of shape `(batch_size, num_cameras, 2)`.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size_divisor = pad_size_divisor if pad_size_divisor is not None else self.pad_size_divisor

        size = get_size_dict(size, param_name="size")
        target_height, target_width = size["height"], size["width"]

        samples = self._prepare_sample_list(images)

        lidar2img_touched = False
        if lidar2img is not None:
            lidar2img = self._to_numpy(lidar2img)
            if lidar2img.ndim == 3:
                lidar2img = lidar2img[np.newaxis]
            if lidar2img.shape[:2] != (len(samples), len(samples[0])):
                raise ValueError(
                    f"Expected `lidar2img` to have shape ({len(samples)}, {len(samples[0])}, 4, 4), got "
                    f"{tuple(lidar2img.shape)}."
                )
            if lidar2img.shape[-2:] != (4, 4):
                raise ValueError(f"Calibration matrices must be 4x4, got shape {tuple(lidar2img.shape[-2:])}.")
            lidar2img_touched = True

        processed_images = []
        processed_lidar2img = []
        image_shapes = []
        for sample_index, sample in enumerate(samples):
            num_cameras = sample.shape[0]
            if sample.shape[-1] != 3:
                raise ValueError(
                    f"Expected images with the channel dimension last (shape (num_cameras, height, width, 3)), got "
                    f"the trailing dimensions {tuple(sample.shape[-2:])} for sample index {sample_index}."
                )
            if not all(sample[cam].shape == sample[0].shape for cam in range(num_cameras)):
                raise ValueError(f"All cameras of sample {sample_index} must share the same (height, width) size.")

            # The image scaling ratio is derived from the actual resize performed. Upstream BEVFormer rescales
            # images and `lidar2img` matrices with a single scalar factor: rows 0 and 1 of each projection
            # matrix are multiplied by the ratio that maps original size to target size.
            scale = 1.0
            if do_resize:
                height_scale = target_height / sample.shape[1]
                width_scale = target_width / sample.shape[2]
                if not np.isclose(height_scale, width_scale, rtol=0.0, atol=1e-4):
                    raise ValueError(
                        f"The ratio applied to height ({height_scale}) and width ({width_scale}) must be equal, as "
                        f"a single scalar rescales both image dimensions and the `lidar2img` matrices."
                    )
                scale = width_scale

            scaled_images = []
            scaled_shape = None
            for camera_index in range(num_cameras):
                image = sample[camera_index].astype(np.float32)
                if do_resize:
                    image = self.resize(image=image, size=(target_height, target_width), resample=resample)
                    scaled_shape = (image.shape[0], image.shape[1])
                if do_normalize:
                    image = self.normalize(image=image, mean=image_mean, std=image_std)
                if scaled_shape is None:
                    scaled_shape = (image.shape[0], image.shape[1])
                pad_height = -scaled_shape[0] % pad_size_divisor if do_pad else 0
                pad_width = -scaled_shape[1] % pad_size_divisor if do_pad else 0
                if pad_height > 0 or pad_width > 0:
                    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)))
                scaled_images.append(image.transpose(2, 0, 1))
            processed_images.append(np.stack(scaled_images))
            image_shapes.append(np.tile(np.array(scaled_shape, dtype=np.int64), (num_cameras, 1)))

            if lidar2img_touched:
                scale_matrix = np.eye(4, dtype=np.float32)
                scale_matrix[0, 0] = scale
                scale_matrix[1, 1] = scale
                processed_lidar2img.append(np.matmul(scale_matrix, lidar2img[sample_index].astype(np.float32)))

        data = {
            "pixel_values": np.stack(processed_images),
            "image_shapes": np.stack(image_shapes),
        }
        if lidar2img_touched:
            data["lidar2img"] = np.stack(processed_lidar2img)
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.1,
        top_k: int = 300,
    ) -> list[dict[str, torch.Tensor]]:
        """Converts the raw outputs of [`BevformerFor3DObjectDetection`] into 3D bounding boxes in the LiDAR coordinate
        frame, with confidence scores and class labels.

        Args:
            outputs ([`BevformerObjectDetectionOutput`]):
                Raw outputs from [`BevformerFor3DObjectDetection`].
            threshold (`float`, *optional*, defaults to 0.1):
                Score threshold to filter detections.
            top_k (`int`, *optional*, defaults to 300):
                Maximum number of boxes to retain per sample.

        Returns:
            `list[dict[str, torch.Tensor]]`:
                A list of dictionaries containing:
                - `"scores"`: Confidence scores for each detected box.
                - `"labels"`: Class label indices for each detected box.
                - `"boxes_3d"`: 9-parameter 3D bounding boxes `[cx, cy, cz_bottom, w, l, h, yaw, vx, vy]` in LiDAR frame.
        """
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        batch_size = logits.shape[0]
        results = []

        for b in range(batch_size):
            scores = logits[b].sigmoid()
            boxes = pred_boxes[b].clone()

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


def point_sampling(
    reference_points: torch.Tensor,
    pc_range: list[float] | tuple[float, ...],
    lidar2img: torch.Tensor,
    img_shape: tuple[int, int] | list[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D pillar reference points onto multi-camera 2D image coordinates.

    Args:
        reference_points (`torch.Tensor` of shape `(batch_size, num_queries, num_points_in_pillar, 3)`):
            Normalized 3D reference points in `[0, 1]`.
        pc_range (`list[float]` or `tuple[float, ...]`):
            Point cloud range `[x_min, y_min, z_min, x_max, y_max, z_max]`.
        lidar2img (`torch.Tensor` of shape `(batch_size, num_cams, 4, 4)`):
            Projection matrices from LiDAR to image plane.
        img_shape (`tuple[int, int]`, `list[int]`, or `torch.Tensor`):
            Padded image shape `(height, width)`. Upstream BEVFormer normalizes by the padded image shape.

    Returns:
        tuple[`torch.Tensor`, `torch.Tensor`]:
            - `reference_points_cam` (`torch.Tensor` of shape `(num_cams, batch_size, num_queries, num_points_in_pillar, 2)`):
                Normalized image coordinates `[u, v]` in `[0, 1]`.
            - `bev_mask` (`torch.Tensor` of shape `(num_cams, batch_size, num_queries, num_points_in_pillar)`):
                Boolean mask indicating whether each pillar point projects within camera frustum and bounds.
    """
    ref_pts = reference_points.clone()
    ref_pts[..., 0:1] = ref_pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    ref_pts[..., 1:2] = ref_pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    ref_pts[..., 2:3] = ref_pts[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

    # Convert to homogeneous coordinates (B, D, num_queries, 4)
    ref_pts = torch.cat([ref_pts, torch.ones_like(ref_pts[..., :1])], dim=-1)

    # Permute to (D, B, num_queries, 4)
    ref_pts = ref_pts.permute(1, 0, 2, 3)
    num_points_in_pillar, batch_size, num_queries, _ = ref_pts.shape
    num_cams = lidar2img.shape[1]

    # Expand for multi-camera batch matmul
    # ref_pts: (D, B, num_cams, num_queries, 4, 1)
    ref_pts = (
        ref_pts.view(num_points_in_pillar, batch_size, 1, num_queries, 4).repeat(1, 1, num_cams, 1, 1).unsqueeze(-1)
    )
    # lidar2img: (D, B, num_cams, num_queries, 4, 4)
    lidar2img_expanded = lidar2img.view(1, batch_size, num_cams, 1, 4, 4).repeat(
        num_points_in_pillar, 1, 1, num_queries, 1, 1
    )

    reference_points_cam = torch.matmul(lidar2img_expanded.to(torch.float32), ref_pts.to(torch.float32)).squeeze(-1)
    eps = 1e-5

    bev_mask = reference_points_cam[..., 2:3] > eps
    depth_clamped = torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
    )
    reference_points_cam = reference_points_cam[..., 0:2] / depth_clamped

    if isinstance(img_shape, torch.Tensor):
        h, w = img_shape[0].item(), img_shape[1].item()
    else:
        h, w = img_shape[0], img_shape[1]

    reference_points_cam[..., 0] /= w
    reference_points_cam[..., 1] /= h

    bev_mask = (
        bev_mask
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0)
    )
    bev_mask = torch.nan_to_num(bev_mask)

    # Upstream permutation: (2, 1, 3, 0, 4)
    # From (D, B, num_cams, num_queries, 2) -> (num_cams, B, num_queries, D, 2)
    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, bev_mask


class BevformerSpatialCrossAttention(nn.Module):
    """Spatial Cross Attention (SCA) module for BEVFormer.

    Extracts multi-camera image features for each BEV query by projecting 3D pillar reference points
    into the camera views, selecting valid frustum hits per camera, and sampling multi-scale deformable
    attention offsets.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.embed_dims = config.d_model
        self.num_cams = config.num_cameras
        self.pc_range = config.point_cloud_range

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

        batch_size, num_queries, _ = hidden_states.shape
        num_z_anchors = reference_points_cam.shape[3]

        indexes = []
        for mask_per_img in bev_mask:
            # mask_per_img[0]: (num_queries, D) -> sum(-1) > 0 indices
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)

        lens = [len(each) for each in indexes]
        max_len = max(lens) if len(lens) > 0 and max(lens) > 0 else 0

        if max_len > 0:
            queries_rebatch = hidden_states.new_zeros([batch_size, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [batch_size, self.num_cams, max_len, num_z_anchors, 2]
            )

            for b in range(batch_size):
                for cam_idx, reference_points_per_img in enumerate(reference_points_cam):
                    idx = indexes[cam_idx]
                    if len(idx) > 0:
                        queries_rebatch[b, cam_idx, : len(idx)] = hidden_states[b, idx]
                        reference_points_rebatch[b, cam_idx, : len(idx)] = reference_points_per_img[b, idx]

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
                for cam_idx, idx in enumerate(indexes):
                    if len(idx) > 0:
                        slots[b, idx] += queries_out[b, cam_idx, : len(idx)]

        # Average over visible views observing each query pillar
        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
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

    Encapsulates multi-layer BEV refinement, reference point generation (2D/3D),
    CAN bus ego-motion transformation (shift & rotation), and camera feature embedding.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.encoder_layers
        self.pc_range = config.point_cloud_range
        self.num_points_in_pillar = config.num_points_in_pillar
        self.bev_h = config.bev_height
        self.bev_w = config.bev_width
        self.embed_dims = config.d_model
        self.rotate_center = config.rotate_center

        self.layers = nn.ModuleList([BevformerEncoderLayer(config) for _ in range(self.num_layers)])
        self.can_bus_mlp = BevformerCanBusMLP(config)

        # Camera & Level embeddings
        self.level_embeds = nn.Parameter(torch.empty(config.num_feature_levels, config.d_model))
        self.cams_embeds = nn.Parameter(torch.empty(config.num_cameras, config.d_model))

    @staticmethod
    def get_reference_points(
        height: int,
        width: int,
        z_height: float = 8.0,
        num_points_in_pillar: int = 4,
        dim: str = "3d",
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate 2D or 3D reference points on the BEV grid.

        Args:
            height (`int`): BEV grid height.
            width (`int`): BEV grid width.
            z_height (`float`): Metric height span along Z. Defaults to 8.0.
            num_points_in_pillar (`int`): Number of vertical points per pillar. Defaults to 4.
            dim (`str`): '2d' or '3d'.
            batch_size (`int`): Batch size. Defaults to 1.
            device (`torch.device`, *optional*): Target device.
            dtype (`torch.dtype`): Target data type. Defaults to `torch.float32`.

        Returns:
            `torch.Tensor`:
                If dim='3d', shape `(batch_size, height * width, num_points_in_pillar, 3)`.
                If dim='2d', shape `(batch_size, height * width, 1, 2)`.
        """
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

        elif dim == "2d":
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

    def compute_ego_motion(
        self,
        can_bus: torch.Tensor,
        bev_queries: torch.Tensor,
        grid_length: tuple[float, float] | list[float],
    ) -> torch.Tensor:
        """Compute BEV translation shift from CAN bus ego-motion signals.

        Args:
            can_bus (`torch.Tensor` of shape `(batch_size, 18)`):
                CAN bus status containing velocity and yaw rate.
            bev_queries (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Reference tensor for device and dtype.
            grid_length (`tuple[float, float]`):
                Length per BEV cell in meters `(grid_length_y, grid_length_x)`.

        Returns:
            `torch.Tensor` of shape `(batch_size, 2)`:
                Translation shifts `(shift_x, shift_y)`.
        """
        can_bus_np = can_bus.detach().cpu().numpy()
        delta_x = can_bus_np[:, 0]
        delta_y = can_bus_np[:, 1]
        ego_angle = can_bus_np[:, -2] / np.pi * 180.0

        grid_length_y, grid_length_x = grid_length[0], grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180.0
        bev_angle = ego_angle - translation_angle

        shift_y = translation_length * np.cos(bev_angle / 180.0 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * np.sin(bev_angle / 180.0 * np.pi) / grid_length_x / self.bev_w

        return bev_queries.new_tensor(np.stack([shift_x, shift_y], axis=-1))

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
        lidar2img: torch.Tensor,
        bev_pos: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the BEVFormer Encoder.

        Args:
            bev_queries (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Initial learned BEV query embeddings.
            multi_level_feats (`list[torch.Tensor]`):
                List of multi-camera feature maps from FPN, each `(batch_size, num_cams, embed_dims, H, W)`.
            lidar2img (`torch.Tensor` of shape `(batch_size, num_cams, 4, 4)`):
                Projection matrices from LiDAR to image plane.
            bev_pos (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Learned 2D positional encodings for the BEV plane.
            can_bus (`torch.Tensor` of shape `(batch_size, 18)`, *optional*):
                CAN bus ego-motion signals. Defaults to zeros if not provided.
            prev_bev (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`, *optional*):
                BEV embedding from the previous frame.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`:
                Refined BEV feature representation.
        """
        batch_size, num_queries, _ = bev_queries.shape
        device = bev_queries.device
        dtype = bev_queries.dtype

        if can_bus is None:
            can_bus = bev_queries.new_zeros((batch_size, 18))

        # 1. Inject CAN bus signals into BEV queries
        bev_queries = bev_queries + self.can_bus_mlp(can_bus)

        # 2. Reference points in 3D and 2D
        z_height = self.pc_range[5] - self.pc_range[2]
        ref_3d = self.get_reference_points(
            self.bev_h,
            self.bev_w,
            z_height=z_height,
            num_points_in_pillar=self.num_points_in_pillar,
            dim="3d",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        ref_2d = self.get_reference_points(
            self.bev_h,
            self.bev_w,
            dim="2d",
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # 3. Project 3D reference points to camera views (point_sampling)
        # Upstream BEVFormer derives image height/width from padded feature stride
        # Feature level 0: shape is (batch_size, num_cams, embed_dims, H, W)
        feat_h, feat_w = multi_level_feats[0].shape[-2:]
        # In BEVFormer tiny, backbone stride is 32 -> img_shape = (feat_h * 32, feat_w * 32)
        padded_img_shape = (feat_h * 32, feat_w * 32)
        reference_points_cam, bev_mask = point_sampling(ref_3d, self.pc_range, lidar2img, img_shape=padded_img_shape)

        # 4. Translation shift & rotation for temporal fusion
        real_h = self.pc_range[4] - self.pc_range[1]
        real_w = self.pc_range[3] - self.pc_range[0]
        grid_length = (real_h / self.bev_h, real_w / self.bev_w)
        shift = self.compute_ego_motion(can_bus, bev_queries, grid_length)

        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # 5. Build hybrid 2D reference points and temporal queue
        if prev_bev is not None:
            prev_bev_rotated = self.apply_prev_bev_rotation(prev_bev, can_bus)
            # Stack [prev_bev, bev_queries] along queue dimension (2 * batch_size, num_queries, embed_dims)
            temporal_bev_queue = torch.stack([prev_bev_rotated, bev_queries], dim=1).reshape(
                batch_size * 2, num_queries, -1
            )
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], dim=1).reshape(batch_size * 2, num_queries, 1, 2)
        else:
            temporal_bev_queue = None
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
        lidar2img: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BevformerModelOutput:
        r"""
        lidar2img (`torch.Tensor` of shape `(batch_size, num_cameras, 4, 4)`):
            Projection matrices from LiDAR coordinates to each camera's image plane.
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
            lidar2img=lidar2img,
            bev_pos=bev_pos,
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
        lidar2img: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BevformerObjectDetectionOutput:
        r"""
        lidar2img (`torch.Tensor` of shape `(batch_size, num_cameras, 4, 4)`):
            Projection matrices from LiDAR coordinates to each camera's image plane.
        can_bus (`torch.Tensor` of shape `(batch_size, 18)`, *optional*):
            CAN bus ego-motion signals (translation, rotation, velocity, acceleration). Defaults to zeros if omitted.
        prev_bev (`torch.Tensor` of shape `(batch_size, bev_height * bev_width, d_model)`, *optional*):
            Bird's-eye-view embedding from the previous frame for temporal fusion.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            lidar2img=lidar2img,
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

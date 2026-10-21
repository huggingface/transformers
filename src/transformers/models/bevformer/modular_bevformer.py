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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...backbone_utils import load_backbone
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import normalize
from ...image_utils import ImageInput, PILImageResampling
from ...modeling_utils import PreTrainedModel
from ...models.auto import AutoConfig
from ...utils import ModelOutput, TensorType, auto_docstring, is_torch_tensor, is_torchvision_available, logging


if is_torchvision_available():
    from torchvision.transforms.functional import rotate


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
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        num_encoder_layers (`int`, *optional*, defaults to 3):
            Number of BEVFormer encoder layers.
        num_decoder_layers (`int`, *optional*, defaults to 6):
            Number of object detection decoder layers.
        ffn_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the "intermediate" (feed-forward) layer inside the transformer layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function in the feed-forward layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability of attention weights and feed-forward layers.
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries.
        num_points_in_pillar (`int`, *optional*, defaults to 4):
            Number of sampled depth positions (pillar points) along the z axis for each BEV query.
        num_points_in_bev (`int`, *optional*, defaults to 8):
            Number of sampling points on the image plane for each pillar point in the spatial cross attention.
        with_box_refine (`bool`, *optional*, defaults to `True`):
            Whether to refine 3D bounding box references across decoder layers.
        as_two_stage (`bool`, *optional*, defaults to `False`):
            Whether the model operates in a two-stage mode.
        rotate_prev_bev (`bool`, *optional*, defaults to `True`):
            Whether to rotate the previous BEV embeddings before temporal fusion.
        use_shift (`bool`, *optional*, defaults to `True`):
            Whether to shift the previous BEV embeddings before temporal fusion.
        use_can_bus (`bool`, *optional*, defaults to `True`):
            Whether to inject CAN bus ego motion signals into the BEV queries.
        can_bus_dim (`int`, *optional*, defaults to 18):
            Dimensionality of the CAN bus ego-motion features.
        can_bus_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply a layer normalization on the encoded CAN bus signals.
        rotate_center (`list[float]`, *optional*, defaults to `[100.0, 100.0]`):
            The rotation center `[x, y]` (in BEV cell coordinates) used when rotating previous BEV embeddings.
    """

    model_type = "bevformer"
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
    num_attention_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 6
    ffn_dim: int = 512
    hidden_act: str = "relu"
    dropout: float = 0.1
    num_queries: int = 900
    num_points_in_pillar: int = 4
    num_points_in_bev: int = 8
    with_box_refine: bool = True
    as_two_stage: bool = False
    rotate_prev_bev: bool = True
    use_shift: bool = True
    use_can_bus: bool = True
    can_bus_dim: int = 18
    can_bus_norm: bool = True
    rotate_center: list[float] | None = None

    def __post_init__(self, **kwargs):
        if self.point_cloud_range is None:
            self.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if self.rotate_center is None:
            self.rotate_center = [100.0, 100.0]
        if self.id2label is None:
            self.id2label = dict(enumerate(NUSCENES_CLASS_NAMES))
        super().__post_init__(**kwargs)

    @property
    def num_classes(self) -> int:
        return len(self.id2label)

    @num_classes.setter
    def num_classes(self, value: int):
        pass

    @property
    def hidden_size(self) -> int:
        return self.d_model

    @hidden_size.setter
    def hidden_size(self, value: int):
        self.d_model = value


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
        logits = outputs.logits[:, -1]
        pred_boxes = outputs.pred_boxes[:, -1]

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

    Batch normalization layers are forced to eval mode in `train()` to match the `norm_eval=True`
    parity behavior of upstream BEVFormer / MMDetection backbones.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config

        backbone = load_backbone(config)
        self.intermediate_channel_sizes = backbone.channels

        # Unwrap if loaded via timm directly
        if hasattr(backbone, "_backbone"):
            backbone = backbone._backbone
        self.model = backbone

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eval()
        return self

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

        features = self.model(pixel_values)
        if hasattr(features, "feature_maps"):
            feature_map = features.feature_maps[-1]
        elif isinstance(features, (list, tuple)):
            feature_map = features[-1]
        else:
            feature_map = features

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
    conditioning embedding for the BEV query plane. Submodule naming aligns with upstream
    `pts_bbox_head.transformer.encoder.can_bus_mlp.0` and `.2`.

    Args:
        config ([`BevformerConfig`]):
            Configuration of the model.
        can_bus_dim (`int`, *optional*):
            Dimensionality of input CAN bus vector. Defaults to `config.can_bus_dim` (18).
    """

    def __init__(self, config: BevformerConfig, can_bus_dim: int | None = None):
        super().__init__()
        self.config = config
        can_bus_dim = can_bus_dim if can_bus_dim is not None else getattr(config, "can_bus_dim", 18)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(can_bus_dim, config.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.ReLU(inplace=True),
        )
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
        can_bus_embed = self.can_bus_mlp(can_bus)
        if self.can_bus_norm is not None:
            can_bus_embed = self.can_bus_norm(can_bus_embed)
        return can_bus_embed.unsqueeze(1)


class BevformerClsHead(nn.Module):
    """
    Classification head for BEVFormer object detection.

    Projects transformer decoder hidden states into raw class logits (pre-sigmoid).
    Matches upstream `pts_bbox_head.cls_branches.{layer_idx}` with 2 intermediate Linear+LN+ReLU layers.

    Args:
        config ([`BevformerConfig`]):
            Configuration of the model.
        num_classes (`int`, *optional*):
            Number of object detection classes. Defaults to `config.num_classes`.
    """

    def __init__(self, config: BevformerConfig, num_classes: int | None = None):
        super().__init__()
        self.config = config
        num_classes = num_classes if num_classes is not None else config.num_classes
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(config.d_model, config.d_model))
            layers.append(nn.LayerNorm(config.d_model))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(config.d_model, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_queries, d_model)`):
                Decoder hidden states.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, num_classes)`:
                Raw classification scores.
        """
        return self.layers(hidden_states)


class BevformerRegHead(nn.Module):
    """
    3D Bounding Box Regression MLP head for BEVFormer.

    Predicts 3D bounding box coordinates or coordinate refinement offsets.
    Layer structure `(Linear -> ReLU -> Linear -> ReLU -> Linear)` preserves upstream
    `pts_bbox_head.reg_branches.{layer_idx}.0`, `.2`, and `.4` key numbering.

    Args:
        config ([`BevformerConfig`]):
            Configuration of the model.
        code_size (`int`, *optional*, defaults to 10):
            Number of output regression parameters `[cx, cy, w, l, cz, h, sin(yaw), cos(yaw), vx, vy]`.
        num_layers (`int`, *optional*, defaults to 3):
            Number of linear layers in the MLP.
    """

    def __init__(self, config: BevformerConfig, code_size: int = 10, num_layers: int = 3):
        super().__init__()
        self.config = config
        self.code_size = code_size

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(config.d_model, config.d_model))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(config.d_model, code_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_queries, d_model)`):
                Decoder hidden states.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, code_size)`:
                Predicted bounding box offsets or parameters.
        """
        return self.layers(hidden_states)


class MultiScaleDeformableAttention(nn.Module):
    """
    Pure PyTorch implementation of Multi-Scale Deformable Attention sampling kernel.

    Performs multi-scale bilinear sampling using `torch.nn.functional.grid_sample` with `align_corners=False`,
    achieving exact mathematical equivalence with MMCV's CUDA and C++ deformable attention kernels.
    """

    def forward(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor | list[tuple[int, int]],
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_heads, head_dim = value.shape
        _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

        if isinstance(value_spatial_shapes, torch.Tensor):
            value_spatial_shapes = value_spatial_shapes.tolist()

        value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1

        sampling_value_list = []
        for level_idx, (height, width) in enumerate(value_spatial_shapes):
            value_l = (
                value_list[level_idx]
                .flatten(2)
                .transpose(1, 2)
                .reshape(batch_size * num_heads, head_dim, height, width)
            )
            sampling_grid_l = sampling_grids[:, :, :, level_idx].transpose(1, 2).flatten(0, 1)

            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l)

        attention_weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads, 1, num_queries, num_levels * num_points
        )

        stacked_sampling_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)

        output = (
            (stacked_sampling_values * attention_weights)
            .sum(-1)
            .view(batch_size, num_heads * head_dim, num_queries)
            .transpose(1, 2)
            .contiguous()
        )
        return output


multi_scale_deformable_attention = MultiScaleDeformableAttention()


class BevformerMSDeformableAttention3D(nn.Module):
    """
    3D Multi-Scale Deformable Attention used in BEVFormer spatial cross-attention and detection decoder.

    Decomposes sampling offsets across 3D depth/pillar reference points, enabling continuous multi-view
    and multi-depth spatial feature aggregation.

    Args:
        embed_dims (`int`, *optional*, defaults to 256):
            Hidden embedding dimension.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_levels (`int`, *optional*, defaults to 1):
            Number of feature pyramid levels.
        num_points (`int`, *optional*, defaults to 8):
            Total sampling points per head per level across all depth anchors.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability on the output projection.
        has_output_proj (`bool`, *optional*, defaults to True):
            Whether to apply an output projection layer.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 8,
        dropout: float = 0.1,
        has_output_proj: bool = True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        if has_output_proj:
            self.output_proj = nn.Linear(embed_dims, embed_dims)
        else:
            self.output_proj = None

        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MultiScaleDeformableAttention()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Match MMCV deformable attention offset initialization
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * np.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        if self.output_proj is not None:
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Query embeddings.
            key (`torch.Tensor`, *optional*):
                Key features (not used directly; value is used).
            value (`torch.Tensor` of shape `(batch_size, num_keys, embed_dims)`, *optional*):
                Input feature representation. If `None`, defaults to `query`.
            identity (`torch.Tensor`, *optional*):
                Residual tensor. If `None`, defaults to `query`.
            query_pos (`torch.Tensor`, *optional*):
                Positional encoding for queries, added to `query` before computing offsets and weights.
            reference_points (`torch.Tensor` of shape `(batch_size, num_queries, 2)` or `(batch_size, num_queries, num_Z_anchors, 2)`):
                2D or 3D reference point coordinates normalized to `[0, 1]` as `(x, y)` = `(w, h)`.
            spatial_shapes (`torch.Tensor` or `list[tuple[int, int]]` of shape `(num_levels, 2)`):
                Spatial `(height, width)` per feature level.
            level_start_index (`torch.Tensor`, *optional*):
                Prefix sum index for multi-level flattening.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`:
                Attended output representation.
        """
        if value is None:
            value = query

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        batch_size, num_queries, _ = query.shape
        _, num_keys, _ = value.shape

        if reference_points is None:
            raise ValueError("`reference_points` must be provided to `BevformerMSDeformableAttention3D`")

        # Support both 2D reference points (bs, num_queries, 2) and 3D reference points (bs, num_queries, num_Z, 2)
        if reference_points.ndim == 3:
            reference_points = reference_points.unsqueeze(2)

        num_z_anchors = reference_points.shape[2]

        if self.num_points % num_z_anchors != 0:
            raise ValueError(f"num_points ({self.num_points}) must be divisible by num_z_anchors ({num_z_anchors})")
        num_points_per_anchor = self.num_points // num_z_anchors

        value = self.value_proj(value)
        value = value.view(batch_size, num_keys, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )

        attention_weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        if isinstance(spatial_shapes, list):
            spatial_shapes_tensor = query.new_tensor(spatial_shapes, dtype=torch.long)
        else:
            spatial_shapes_tensor = spatial_shapes

        # offset_normalizer is (width, height)
        offset_normalizer = torch.stack([spatial_shapes_tensor[..., 1], spatial_shapes_tensor[..., 0]], -1)

        # Decompose sampling offsets across Z anchors
        sampling_offsets = sampling_offsets.view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            num_points_per_anchor,
            num_z_anchors,
            2,
        )

        # Broadcast reference points across heads, levels, and points per anchor
        # reference_points: (batch_size, num_queries, 1, 1, 1, num_z_anchors, 2)
        ref_points_expanded = reference_points[:, :, None, None, None, :, :]
        normalizer_expanded = offset_normalizer[None, None, None, :, None, None, :]

        sampling_locations = ref_points_expanded + sampling_offsets / normalizer_expanded
        sampling_locations = sampling_locations.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )

        output = self.deformable_attention(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
        )

        if self.output_proj is not None:
            output = self.output_proj(output)

        return self.dropout(output)


class BevformerTemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention module for BEVFormer.

    Fuses the previous frame's BEV representation (`prev_bev`) into the current frame's BEV query
    representation using deformable multi-scale attention over a 2-frame history queue.

    Args:
        embed_dims (`int`, *optional*, defaults to 256):
            Hidden embedding dimension.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_levels (`int`, *optional*, defaults to 1):
            Number of feature pyramid levels.
        num_points (`int`, *optional*, defaults to 4):
            Number of sampling points per attention head per feature level per temporal queue item.
        num_bev_queue (`int`, *optional*, defaults to 2):
            Number of temporal frames in the queue (default 2: previous frame and current frame).
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability applied to the output projection.
    """

    def __init__(
        self,
        config: BevformerConfig | None = None,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if isinstance(config, BevformerConfig):
            embed_dims = config.d_model
            num_heads = config.num_attention_heads
            num_levels = config.num_feature_levels
            num_points = 4
            dropout = config.dropout
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.head_dim = embed_dims // num_heads

        # The query vector concatenated with prev_bev has dimension 2 * embed_dims
        self.sampling_offsets = nn.Linear(2 * embed_dims, num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(2 * embed_dims, num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MultiScaleDeformableAttention()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * np.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        grid_init = grid_init.view(1, self.num_heads, self.num_levels, self.num_points, 2).repeat(
            self.num_bev_queue, 1, 1, 1, 1
        )
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Current BEV queries.
            key (`torch.Tensor`, *optional*):
                Unused key placeholder.
            value (`torch.Tensor` of shape `(batch_size * 2, num_queries, embed_dims)`, *optional*):
                Stacked previous and current BEV representation. If `None`, constructed from `prev_bev` and `query`.
            identity (`torch.Tensor`, *optional*):
                Residual tensor. If `None`, defaults to `query`.
            query_pos (`torch.Tensor`, *optional*):
                BEV positional embeddings added to `query`.
            reference_points (`torch.Tensor` of shape `(batch_size, num_queries, 2)` or `(batch_size * 2, num_queries, 1, 2)`):
                Reference points on the BEV grid.
            spatial_shapes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Spatial `(bev_height, bev_width)` shape.
            level_start_index (`torch.Tensor`, *optional*):
                Prefix sum index.
            prev_bev (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`, *optional*):
                Ego-motion compensated previous BEV embedding. If `None`, falls back to duplicating `query`.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`:
                Fused BEV queries after temporal self-attention.
        """
        if identity is None:
            identity = query

        if query_pos is not None:
            query_with_pos = query + query_pos
        else:
            query_with_pos = query

        batch_size, num_queries, _ = query.shape

        # Upstream queue fallback: frame 0 without history duplicates current query
        if prev_bev is None:
            prev_bev = query

        if value is None:
            # Stack [prev_bev, query] along batch dimension -> (2 * batch_size, num_queries, embed_dims)
            value = torch.cat([prev_bev, query], dim=0)

        # Concatenate previous BEV and current query along features for offset & weight generation
        query_concat = torch.cat([prev_bev, query_with_pos], dim=-1)

        # Reference points handling
        if reference_points is None:
            raise ValueError("`reference_points` must be provided to `BevformerTemporalSelfAttention`")

        # If reference points are provided per-batch item (batch_size, num_queries, 2)
        # Repeat for the 2 queue items -> (2 * batch_size, num_queries, 1, 2)
        if reference_points.ndim == 3:
            reference_points = reference_points.unsqueeze(2).repeat(self.num_bev_queue, 1, 1, 1)
        elif reference_points.ndim == 4 and reference_points.shape[0] == batch_size:
            reference_points = reference_points.repeat(self.num_bev_queue, 1, 1, 1)

        total_batch_size = value.shape[0]
        num_keys = value.shape[1]

        value = self.value_proj(value)
        value = value.view(total_batch_size, num_keys, self.num_heads, self.head_dim)

        # Compute sampling offsets and attention weights from query_concat
        sampling_offsets = self.sampling_offsets(query_concat).view(
            batch_size, num_queries, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query_concat).view(
            batch_size, num_queries, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points
        )

        # Softmax normalized over (num_levels * num_points) per queue item
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            batch_size, num_queries, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points
        )

        # Permute queue dimension into batch dimension matching value (2 * batch_size)
        # (batch_size, num_queries, num_heads, num_bev_queue, num_levels, num_points, 2)
        # -> (num_bev_queue, batch_size, num_queries, num_heads, num_levels, num_points, 2)
        # -> (num_bev_queue * batch_size, num_queries, num_heads, num_levels, num_points, 2)
        sampling_offsets = (
            sampling_offsets.permute(3, 0, 1, 2, 4, 5, 6)
            .contiguous()
            .view(total_batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        )
        attention_weights = (
            attention_weights.permute(3, 0, 1, 2, 4, 5)
            .contiguous()
            .view(total_batch_size, num_queries, self.num_heads, self.num_levels, self.num_points)
        )

        if isinstance(spatial_shapes, list):
            spatial_shapes_tensor = query.new_tensor(spatial_shapes, dtype=torch.long)
        else:
            spatial_shapes_tensor = spatial_shapes

        offset_normalizer = torch.stack([spatial_shapes_tensor[..., 1], spatial_shapes_tensor[..., 0]], -1)

        # reference_points: (2 * batch_size, num_queries, 1, 2) -> (2 * batch_size, num_queries, 1, 1, 1, 2)
        ref_points_expanded = reference_points[:, :, None, :, None, :]
        normalizer_expanded = offset_normalizer[None, None, None, :, None, :]

        sampling_locations = ref_points_expanded + sampling_offsets / normalizer_expanded

        output = self.deformable_attention(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
        )

        # Fuse over the temporal queue dimension (num_bev_queue=2) via mean
        output = output.view(self.num_bev_queue, batch_size, num_queries, self.embed_dims).mean(dim=0)
        output = self.output_proj(output)
        return self.dropout(output) + identity


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
        self.dropout = nn.Dropout(config.dropout)
        self.pc_range = config.point_cloud_range

        self.deformable_attention = BevformerMSDeformableAttention3D(
            embed_dims=config.d_model,
            num_heads=config.num_attention_heads,
            num_levels=config.num_feature_levels,
            num_points=config.num_points_in_bev,
            dropout=config.dropout,
            has_output_proj=False,
        )
        self.output_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        reference_points_cam: torch.Tensor | None = None,
        bev_mask: torch.Tensor | None = None,
        spatial_shapes: list[list[int]] | torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for Spatial Cross Attention.

        Args:
            query (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Input BEV queries.
            key (`torch.Tensor` of shape `(num_cams, num_keys, batch_size, embed_dims)`):
                Flattened multi-camera feature maps.
            value (`torch.Tensor` of shape `(num_cams, num_keys, batch_size, embed_dims)`):
                Same as key.
            identity (`torch.Tensor`, *optional*):
                Residual tensor. Defaults to query.
            query_pos (`torch.Tensor`, *optional*):
                Positional encoding for BEV query.
            reference_points_cam (`torch.Tensor` of shape `(num_cams, batch_size, num_queries, D, 2)`):
                Projected 2D coordinates on camera views.
            bev_mask (`torch.Tensor` of shape `(num_cams, batch_size, num_queries, D)`):
                Boolean mask indicating valid projections.
            spatial_shapes (`torch.Tensor` or `list`):
                Spatial shapes `(H, W)` for each feature level.
            level_start_index (`torch.Tensor`, *optional*):
                Start indices for each feature level.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`:
                Aggregated and projected BEV features.
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if identity is None:
            inp_residual = query
        else:
            inp_residual = identity

        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        batch_size, num_queries, _ = query.shape
        num_z_anchors = reference_points_cam.shape[3]

        indexes = []
        for mask_per_img in bev_mask:
            # mask_per_img[0]: (num_queries, D) -> sum(-1) > 0 indices
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)

        lens = [len(each) for each in indexes]
        max_len = max(lens) if len(lens) > 0 and max(lens) > 0 else 0

        if max_len > 0:
            queries_rebatch = query.new_zeros([batch_size, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [batch_size, self.num_cams, max_len, num_z_anchors, 2]
            )

            for b in range(batch_size):
                for cam_idx, reference_points_per_img in enumerate(reference_points_cam):
                    idx = indexes[cam_idx]
                    if len(idx) > 0:
                        queries_rebatch[b, cam_idx, : len(idx)] = query[b, idx]
                        reference_points_rebatch[b, cam_idx, : len(idx)] = reference_points_per_img[b, idx]

            num_cams, num_keys, bs, embed_dims = key.shape
            key_reshaped = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, num_keys, self.embed_dims)
            value_reshaped = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, num_keys, self.embed_dims)

            queries_in = queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims)
            ref_pts_in = reference_points_rebatch.view(bs * self.num_cams, max_len, num_z_anchors, 2)

            queries_out = self.deformable_attention(
                query=queries_in,
                key=key_reshaped,
                value=value_reshaped,
                reference_points=ref_pts_in,
                spatial_shapes=spatial_shapes,
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
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


class BevformerEncoderLayer(nn.Module):
    """Single layer of the BEVFormer BEV Encoder.

    Order of operations strictly matches upstream BEVFormer:
    1. Temporal Self-Attention (`self_attn`) -> `norm1`
    2. Spatial Cross-Attention (`cross_attn`) -> `norm2`
    3. Feed-Forward Network (`ffn`) -> `norm3`
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.embed_dims = config.d_model

        self.self_attn = BevformerTemporalSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-5)

        self.cross_attn = BevformerSpatialCrossAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-5)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.norm3 = nn.LayerNorm(config.d_model, eps=1e-5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        bev_pos: torch.Tensor | None = None,
        ref_2d: torch.Tensor | None = None,
        ref_3d: torch.Tensor | None = None,
        bev_h: int | None = None,
        bev_w: int | None = None,
        spatial_shapes: list[list[int]] | torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        reference_points_cam: torch.Tensor | None = None,
        bev_mask: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for one encoder layer.

        Args:
            query (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                BEV queries.
            key (`torch.Tensor` of shape `(num_cams, num_keys, batch_size, embed_dims)`):
                Flattened multi-camera feature maps.
            value (`torch.Tensor` of shape `(num_cams, num_keys, batch_size, embed_dims)`):
                Flattened multi-camera feature maps.
            bev_pos (`torch.Tensor`, *optional*):
                Positional encoding for BEV query of shape `(batch_size, num_queries, embed_dims)`.
            ref_2d (`torch.Tensor` of shape `(2 * batch_size, num_queries, 1, 2)`):
                Hybrid 2D reference points for temporal self-attention.
            ref_3d (`torch.Tensor` of shape `(batch_size, num_queries, D, 3)`):
                3D pillar reference points.
            bev_h (`int`):
                Height of BEV feature map.
            bev_w (`int`):
                Width of BEV feature map.
            spatial_shapes (`torch.Tensor`):
                Spatial shapes of feature levels.
            level_start_index (`torch.Tensor`):
                Start indices for each feature level.
            reference_points_cam (`torch.Tensor`):
                Projected 2D coordinates on camera views.
            bev_mask (`torch.Tensor`):
                Visibility mask for projected points.
            prev_bev (`torch.Tensor`, *optional*):
                Previous BEV features queued with current BEV queries of shape `(2 * batch_size, num_queries, embed_dims)`.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`:
                Refined BEV queries.
        """
        # 1. Temporal self-attention
        query = self.self_attn(
            query=query,
            value=prev_bev,
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=[(bev_h, bev_w)],
            level_start_index=level_start_index,
        )
        query = self.norm1(query)

        # 2. Spatial cross-attention
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=bev_pos,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm2(query)

        # 3. Feed-forward network with residual connection
        query = self.norm3(query + self.ffn(query))

        return query


class BevformerEncoder(nn.Module):
    """BEVFormer BEV Encoder.

    Encapsulates multi-layer BEV refinement, reference point generation (2D/3D),
    CAN bus ego-motion transformation (shift & rotation), and camera feature embedding.
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_encoder_layers
        self.pc_range = config.point_cloud_range
        self.num_points_in_pillar = config.num_points_in_pillar
        self.bev_h = config.bev_height
        self.bev_w = config.bev_width
        self.embed_dims = config.d_model
        self.rotate_prev_bev = config.rotate_prev_bev
        self.use_shift = config.use_shift
        self.use_can_bus = config.use_can_bus
        self.rotate_center = config.rotate_center

        self.layers = nn.ModuleList([BevformerEncoderLayer(config) for _ in range(self.num_layers)])
        self.can_bus_mlp = BevformerCanBusMLP(config)

        # Camera & Level embeddings
        self.level_embeds = nn.Parameter(torch.empty(config.num_feature_levels, config.d_model))
        self.cams_embeds = nn.Parameter(torch.empty(config.num_cameras, config.d_model))
        self.init_weights()

    def init_weights(self):
        """Initialize level & camera embeddings matching upstream BEVFormer."""
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

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

        if self.use_shift:
            shift = bev_queries.new_tensor(np.stack([shift_x, shift_y], axis=-1))
        else:
            shift = bev_queries.new_zeros((can_bus.shape[0], 2))

        return shift

    def apply_prev_bev_rotation(self, prev_bev: torch.Tensor, can_bus: torch.Tensor) -> torch.Tensor:
        """Apply yaw rotation to previous BEV embeddings using the torchvision rotate quirk.

        Args:
            prev_bev (`torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`):
                Previous BEV embeddings.
            can_bus (`torch.Tensor` of shape `(batch_size, 18)`):
                CAN bus status.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_queries, embed_dims)`:
                Rotated BEV embeddings.
        """
        if not self.rotate_prev_bev or not is_torchvision_available():
            return prev_bev

        batch_size = prev_bev.shape[0]
        rotated_prev_bev = prev_bev.clone()

        for b in range(batch_size):
            # Upstream quirk: can_bus[-1] (yaw in radians) is passed directly as degrees
            rotation_angle = can_bus[b, -1].item()
            tmp = prev_bev[b].view(self.bev_h, self.bev_w, self.embed_dims).permute(2, 0, 1)
            # Upstream quirk: pivot center is at [100.0, 100.0] on a 50x50 grid
            tmp = rotate(tmp, rotation_angle, center=self.rotate_center)
            rotated_prev_bev[b] = tmp.permute(1, 2, 0).reshape(self.bev_h * self.bev_w, self.embed_dims)

        return rotated_prev_bev

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
        if self.use_can_bus:
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

        # 7. Execute encoder layers iteratively
        output = bev_queries
        for layer in self.layers:
            output = layer(
                query=output,
                key=feat_flatten,
                value=feat_flatten,
                bev_pos=bev_pos,
                ref_2d=hybrid_ref_2d,
                ref_3d=ref_3d,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                spatial_shapes=spatial_shapes_tensor,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=temporal_bev_queue,
            )

        return output


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse sigmoid function with clamping for numerical stability."""
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class BevformerDecoderLayer(nn.Module):
    """Detection decoder layer for BEVFormer.

    Order of operations:
    1. Multi-head self-attention on object queries (`self_attn`) -> `norm1`
    2. Deformable cross-attention over BEV embedding (`cross_attn`) -> `norm2`
    3. Feed-forward network (`ffn`) -> `norm3`
    """

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-5)

        self.cross_attn = BevformerMSDeformableAttention3D(
            embed_dims=config.d_model,
            num_heads=config.num_attention_heads,
            num_levels=1,
            num_points=4,
            dropout=config.dropout,
            has_output_proj=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-5)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.norm3 = nn.LayerNorm(config.d_model, eps=1e-5)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: list[list[int]] | torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Multi-head Self-Attention
        q = query + query_pos if query_pos is not None else query
        k = q
        attn_out, _ = self.self_attn(q, k, query)
        query = self.norm1(query + self.dropout1(attn_out))

        # 2. Cross-Attention over BEV feature map
        cross_out = self.cross_attn(
            query=query,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm2(query + cross_out)

        # 3. FFN
        query = self.norm3(query + self.ffn(query))
        return query


class BevformerDetrTransformerDecoder(nn.Module):
    """Detection transformer decoder for BEVFormer with iterative 2D reference point refinement."""

    def __init__(self, config: BevformerConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_decoder_layers
        self.layers = nn.ModuleList([BevformerDecoderLayer(config) for _ in range(self.num_layers)])

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        reg_branches: nn.ModuleList | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_shapes = query.new_tensor([[self.config.bev_height, self.config.bev_width]], dtype=torch.long)
        level_start_index = query.new_tensor([0], dtype=torch.long)

        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            ref_input = reference_points[..., :2].unsqueeze(2)  # (batch_size, num_queries, 1, 2)
            output = layer(
                query=output,
                value=value,
                query_pos=query_pos,
                reference_points=ref_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_ref = torch.zeros_like(reference_points)
                new_ref[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_ref[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                reference_points = new_ref.sigmoid().detach()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate, dim=1), torch.stack(intermediate_reference_points, dim=1)


@dataclass
class BevformerModelOutput(ModelOutput):
    """Output type of [`BevformerModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, bev_height * bev_width, d_model)`):
            Bird's-Eye-View (BEV) embedding generated by the BEV encoder. Can be passed as `prev_bev` to
            subsequent frames.
    """

    last_hidden_state: torch.FloatTensor = None


@dataclass
class BevformerObjectDetectionOutput(ModelOutput):
    """Output type of [`BevformerFor3DObjectDetection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_decoder_layers, num_queries, num_classes)`):
            Classification logits (pre-sigmoid) for each query at each decoder layer.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_decoder_layers, num_queries, 10)`):
            Predicted 3D bounding box coordinates `[cx, cy, w, l, cz, h, sin(yaw), cos(yaw), vx, vy]` in
            metric LiDAR frame.
        bev_embed (`torch.FloatTensor` of shape `(batch_size, bev_height * bev_width, d_model)`):
            Bird's-Eye-View (BEV) embedding generated by the BEV encoder. Can be passed as `prev_bev` to
            subsequent frames.
    """

    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    bev_embed: torch.FloatTensor = None


@auto_docstring
class BevformerPreTrainedModel(PreTrainedModel):
    config: BevformerConfig
    base_model_prefix = "bevformer"
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, BevformerEncoder):
            if hasattr(module, "level_embeds") and module.level_embeds is not None:
                torch.nn.init.normal_(module.level_embeds, mean=0.0, std=0.01)
            if hasattr(module, "cams_embeds") and module.cams_embeds is not None:
                torch.nn.init.normal_(module.cams_embeds, mean=0.0, std=0.01)


class BevformerModel(BevformerPreTrainedModel):
    """Base BEVFormer model generating BEV embeddings from multi-camera image streams."""

    def __init__(self, config: BevformerConfig):
        super().__init__(config)
        self.config = config
        self.conv_encoder = BevformerConvEncoder(config)
        self.fpn = BevformerFPN(config)
        self.positional_encoding = BevformerLearnedPositionalEncoding(config)
        self.query_embedding = BevformerQueryEmbedding(config)
        self.encoder = BevformerEncoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.conv_encoder

    def forward(
        self,
        pixel_values: torch.Tensor,
        lidar2img: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BevformerModelOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = pixel_values.shape[0]

        # 1. Multi-camera feature extraction (backbone + FPN)
        features = self.conv_encoder(pixel_values)
        fpn_features = self.fpn(features)

        # 2. Positional encoding & initial BEV queries
        bev_pos = self.positional_encoding(batch_size=batch_size).to(pixel_values.device)
        bev_queries = self.query_embedding.get_bev_queries(batch_size=batch_size).to(pixel_values.device)

        # 3. BEV Encoder
        bev_embed = self.encoder(
            bev_queries=bev_queries,
            multi_level_feats=[fpn_features],
            lidar2img=lidar2img,
            bev_pos=bev_pos,
            can_bus=can_bus,
            prev_bev=prev_bev,
        )

        if not return_dict:
            return (bev_embed,)

        return BevformerModelOutput(last_hidden_state=bev_embed)


class BevformerFor3DObjectDetection(BevformerPreTrainedModel):
    """BEVFormer Model for 3D object detection from multi-camera images."""

    def __init__(self, config: BevformerConfig):
        super().__init__(config)
        self.config = config
        self.bevformer = BevformerModel(config)
        self.decoder = BevformerDetrTransformerDecoder(config)

        self.cls_branches = nn.ModuleList([BevformerClsHead(config) for _ in range(config.num_decoder_layers)])
        self.reg_branches = nn.ModuleList([BevformerRegHead(config) for _ in range(config.num_decoder_layers)])
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        lidar2img: torch.Tensor,
        can_bus: torch.Tensor | None = None,
        prev_bev: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BevformerObjectDetectionOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = pixel_values.shape[0]

        # 1. Base BEVFormer forward to extract BEV feature embedding
        bev_outputs = self.bevformer(
            pixel_values=pixel_values,
            lidar2img=lidar2img,
            can_bus=can_bus,
            prev_bev=prev_bev,
            return_dict=True,
        )
        bev_embed = bev_outputs.last_hidden_state

        # 2. Object queries and initial reference points
        query, query_pos, reference_points = self.bevformer.query_embedding.get_object_queries(batch_size=batch_size)
        query = query.to(pixel_values.device)
        query_pos = query_pos.to(pixel_values.device)
        reference_points = reference_points.to(pixel_values.device)

        # 3. Detection transformer decoder
        inter_states, inter_references = self.decoder(
            query=query,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=self.reg_branches if self.config.with_box_refine else None,
        )

        # 4. Box parameterization matching upstream BEVFormer head
        num_layers = inter_states.shape[1]
        all_cls_scores = []
        all_bbox_preds = []

        pc_range = self.config.point_cloud_range
        pc_min = query.new_tensor(pc_range[:3])
        pc_max = query.new_tensor(pc_range[3:])
        pc_diff = pc_max - pc_min

        for lvl in range(num_layers):
            hs_lvl = inter_states[:, lvl]
            if lvl == 0:
                ref_lvl = reference_points
            else:
                ref_lvl = inter_references[:, lvl - 1]

            cls_score = self.cls_branches[lvl](hs_lvl)
            tmp = self.reg_branches[lvl](hs_lvl)

            inv_ref = inverse_sigmoid(ref_lvl)
            tmp[..., 0:2] = (tmp[..., 0:2] + inv_ref[..., 0:2]).sigmoid()
            tmp[..., 4:5] = (tmp[..., 4:5] + inv_ref[..., 2:3]).sigmoid()

            tmp[..., 0:1] = tmp[..., 0:1] * pc_diff[0] + pc_min[0]
            tmp[..., 1:2] = tmp[..., 1:2] * pc_diff[1] + pc_min[1]
            tmp[..., 4:5] = tmp[..., 4:5] * pc_diff[2] + pc_min[2]

            all_cls_scores.append(cls_score)
            all_bbox_preds.append(tmp)

        all_cls_scores = torch.stack(all_cls_scores, dim=1)
        all_bbox_preds = torch.stack(all_bbox_preds, dim=1)

        if not return_dict:
            return (all_cls_scores, all_bbox_preds, bev_embed)

        return BevformerObjectDetectionOutput(
            logits=all_cls_scores,
            pred_boxes=all_bbox_preds,
            bev_embed=bev_embed,
        )


__all__ = [
    "BevformerConfig",
    "BevformerFor3DObjectDetection",
    "BevformerImageProcessor",
    "BevformerModel",
    "BevformerPreTrainedModel",
]

# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torchvision.transforms.v2 import functional as tvF

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, TransformersKwargs, auto_docstring, logging
from ...utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from ...utils.generic import (
    maybe_autocast,
    merge_with_config_defaults,
)
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos
from ...vision_utils import (
    get_vision_cu_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)
from ..afmoe.modeling_afmoe import AfmoeAttention
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2MLP,
    Gemma2Model,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from ..gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast, Gemma3ModelOutputWithPast
from ..gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4VisionRotaryEmbedding
from ..glm4v.image_processing_glm4v import Glm4vImageProcessor, Glm4vImageProcessorKwargs
from ..kimi_k25.configuration_kimi_k25 import Kimi_K25VisionConfig
from ..kimi_k25.modeling_kimi_k25 import (
    Kimi_K25ForConditionalGeneration,
    Kimi_K25Model,
    Kimi_K25VisionAttention,
    Kimi_K25VisionEncoderLayer,
    Kimi_K25VisionMLP,
)
from ..llama.modeling_llama import eager_attention_forward
from ..paddleocr_vl.modeling_paddleocr_vl import PaddleOCRVisionEmbeddings


logger = logging.get_logger(__name__)


def smart_resize(
    height: int,
    width: int,
    patch_size: int,
    max_tokens: int,
) -> tuple[int, int]:
    """Pick the integer patch grid closest to the input aspect ratio under the token cap.

    Returns the resize target ``(target_height, target_width)`` in pixels.
    """
    ideal_patches_height = height / patch_size
    ideal_patches_width = width / patch_size
    ratio = ideal_patches_width / ideal_patches_height if ideal_patches_height > 0 else 1.0
    if ideal_patches_height * ideal_patches_width > max_tokens:
        ideal_patches_height = (max_tokens / ratio) ** 0.5
        ideal_patches_width = ideal_patches_height * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(ideal_patches_height), math.ceil(ideal_patches_height)],
                [math.floor(ideal_patches_width), math.ceil(ideal_patches_width)],
            )
        )
    )
    candidates = [
        (patches_height, patches_width)
        for patches_height, patches_width in candidates
        if patches_height >= 1 and patches_width >= 1 and patches_height * patches_width <= max_tokens
    ]
    if not candidates:
        candidates = [(max(1, round(ideal_patches_height)), max(1, round(ideal_patches_width)))]
    patches_height, patches_width = min(candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width))
    return patches_height * patch_size, patches_width * patch_size


class MuseGlimmerImageProcessorKwargs(Glm4vImageProcessorKwargs):
    """
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    max_image_tokens (`int`, *optional*, defaults to 4096):
        The maximum number of merged image tokens produced for one image.
    """

    max_image_tokens: int


class MuseGlimmerImageProcessor(Glm4vImageProcessor):
    resample = PILImageResampling.LANCZOS
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = None
    merge_size = 2
    max_image_tokens = 4096

    def _validate_preprocess_kwargs(self, **kwargs):
        # MuseGlimmer uses aspect_ratio_preserving_resize driven by patch_size,
        # not the standard `size` parameter. Temporarily disable do_resize so
        # the base validation doesn't raise an error
        kwargs["do_resize"] = False
        TorchvisionBackend._validate_preprocess_kwargs(**kwargs)

    def resize(
        self,
        images: torch.Tensor,
        patch_size: int,
        merge_size: int,
        max_tokens: int,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        **kwargs,
    ) -> torch.Tensor:
        """Resize dynamically based on input image aspect ratio."""
        height, width = images.shape[-2:]
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            patch_size=patch_size * merge_size,
            max_tokens=max_tokens,
        )
        return TorchvisionBackend.resize(
            image=images,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
            antialias=True,
        )

    def patchify(
        self,
        images: torch.Tensor,
        patch_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        """Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels."""
        batch_size, channel, resized_height, resized_width = images.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = images.view(
            batch_size,
            channel,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        # Unlike Glm4v, each flattened patch is laid out (temporal, channel), not (channel, temporal).
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        flatten_patches = (
            patches.unsqueeze(3)
            .expand(-1, -1, -1, temporal_patch_size, -1, -1, -1)
            .reshape(
                batch_size,
                grid_h * grid_w,
                temporal_patch_size * channel * patch_size * patch_size,
            )
        )
        return flatten_patches, grid_h, grid_w

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int,
        temporal_patch_size: int,
        max_image_tokens: int,
        merge_size: int,
        disable_grouping: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.
        """
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                # Unlike Glm4v's `smart_resize`, the target size keeps aspect ratio under a token cap.
                stacked_images = self.resize(
                    stacked_images,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    max_tokens=max_image_tokens,
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_h, grid_w = self.patchify(
                stacked_images,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
            )

            processed_images_grouped[shape] = patches
            processed_grids[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without an image input.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            patch_size=patch_size * merge_size,
            max_tokens=max_image_tokens,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class MuseGlimmerVideoProcessorInitKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        The spatial patch size of the vision encoder, in pixels.
    temporal_patch_size (`int`, *optional*):
        The temporal patch size of the vision encoder, in frames.
    max_video_frame_tokens (`int`, *optional*):
        Maximum number of vision tokens per video frame; frames are resized to stay under this cap.
    merge_size (`int`, *optional*):
        Factor by which the patch grid is downsampled by pixel shuffling after the vision encoder.
    """

    patch_size: int
    temporal_patch_size: int
    max_video_frame_tokens: int
    merge_size: int


@auto_docstring
class MuseGlimmerVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.LANCZOS
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_video_frame_tokens = 144
    num_frames = 96
    fps = 2.0
    do_sample_frames = True

    valid_kwargs = MuseGlimmerVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[MuseGlimmerVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # MuseGlimmer uses aspect_ratio_preserving_resize driven by patch_size,
        # not the standard `size` parameter. Temporarily disable do_resize so
        # the base validation doesn't raise an error
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def resize(
        self,
        videos: torch.Tensor,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        patch_size: int,
        merge_size: int,
        max_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Resize dynamically based on input video aspect ratio."""
        height, width = videos.shape[-2:]
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            patch_size=patch_size * merge_size,
            max_tokens=max_tokens,
        )

        return super().resize(
            videos,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
            antialias=True,
        )

    def patchify(
        self,
        videos: torch.Tensor,
        patch_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        "Patchifies each video into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels."
        batch_size, num_frames, channel, resized_height, resized_width = videos.shape

        # Check that videos have `num_frames` divisible by `temporal_patch_size`
        if pad := -num_frames % temporal_patch_size:
            repeats = videos[:, -1:].expand(-1, pad, -1, -1, -1)
            videos = torch.cat((videos, repeats), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = videos.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        # Unlike Glm4v, each flattened patch is laid out (temporal, channel), not (channel, temporal).
        patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            temporal_patch_size * channel * patch_size * patch_size,
        )

        return flatten_patches, grid_t, grid_h, grid_w

    def sample_frames(
        self,
        metadata: VideoMetadata,
        temporal_patch_size: int | None = None,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            temporal_patch_size (`int`, *optional*):
                The temporal patch size of the vision encoder. Number of sampled frames will be rounded to be divisible by frame factor.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if metadata.fps is None:
            logger.warning_once(
                "The `fps` of the input video could not be inferred. Defaulting to `fps=24`. "
                "Provide `video_metadata` for more accurate frame sampling."
            )
            metadata.fps = 24

        total_num_frames = metadata.total_num_frames
        num_frames = min(int(total_num_frames * fps / metadata.fps), num_frames, total_num_frames)
        num_frames = max(temporal_patch_size, (num_frames // temporal_patch_size) * temporal_patch_size)
        num_frames = min(num_frames, total_num_frames)
        indices = torch.linspace(0, total_num_frames - 1, num_frames).long()
        return indices

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        do_convert_rgb: bool,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int,
        temporal_patch_size: int,
        max_video_frame_tokens: int,
        merge_size: int,
        disable_grouping: bool = False,
        **kwargs,
    ) -> BatchFeature:
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(
                    stacked_videos,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    max_tokens=max_video_frame_tokens,
                    resample=resample,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
            )

            processed_videos_grouped[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )


class MuseGlimmerModelOutputWithPast(Gemma3ModelOutputWithPast):
    pass


class MuseGlimmerCausalLMOutputWithPast(Gemma3CausalLMOutputWithPast):
    pass


@auto_docstring(checkpoint="someorgtoo-hf/Muse-Glimmer-30B")
@strict
class MuseGlimmerVisionConfig(Kimi_K25VisionConfig):
    r"""
    pos_emb_height (`int`, *optional*):
        Initial position embedding height.
    pos_emb_width (`int`, *optional*):
        Initial position embedding width.
    patch_temporal (`int`, *optional*):
        The temporal patch size used to embed inputs.
    merge_size (`tuple[int] | list[int]`, *optional*):
        Kernel size for patch merging.
    """

    model_type = "muse_glimmer_vision"

    hidden_size: int = 1536
    num_hidden_layers: int = 50
    intermediate_size: int = 8960
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    hidden_act: str = "gelu"
    max_position_embeddings: int = 32 * 32  # == `pos_h * pos_w`
    layer_norm_eps: float = 1e-05
    layer_types: list[str] | None = None
    pos_emb_time = AttributeError()
    merge_kernel_size = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 or i == self.num_hidden_layers - 1 else "window_attention"
                for i in range(self.num_hidden_layers)
            ]
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="someorgtoo-hf/Muse-Glimmer-30B")
@strict
class MuseGlimmerTextConfig(Gemma2Config, PreTrainedConfig):
    r"""
    final_logit_softcapping (`float`, *optional*, defaults to 20.0):
        scaling factor when applying tanh softcapping on the logits.
    qk_scale_factor (`float`, *optional*, defaults to 3.87):
        Multiplier applied to Q after the scaleless QK-norm, on top of the standard `1/sqrt(head_dim)`
        attention scaling.
    output_multiplier (`float`, *optional*, defaults to 0.19611613513818404):
        Scale applied to logits before the final tanh softcap. Equal to `1/sqrt(hidden_size / 256)` for the
        released checkpoint.
    post_norm_eps (`float`, *optional*, defaults to 1e-8):
        Epsilon used for the post-attention and post-FFN norms (which sit between the sub-layer output and the residual).
    layer_rope_theta (`list[float]`, *optional*):
        Per-layer RoPE base theta; `0` disables rotary (NoPE) for that layer. Overrides the global
        `rope_parameters["rope_theta"]`. Defaults to the global theta everywhere except every 4th layer
        counted backward from the last, which is NoPE.
    """

    model_type = "muse_glimmer_text"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.gate_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 202_048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131_072
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 200_000
    eos_token_id: int | list[int] | None = 200_001
    pad_token_id: int | None = None
    sliding_window: int | None = 2048
    final_logit_softcapping: float = 20.0
    layer_types: list[str] | None = None
    query_pre_attn_scalar = AttributeError()
    attn_logit_softcapping = AttributeError()
    use_bidirectional_attention = AttributeError()

    # MuseGlimmer-specific fields
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    post_norm_eps: float = 1e-8
    layer_rope_theta: list[float | int] | None = None

    def __post_init__(self, **kwargs):
        # Full attention on NoPE layers (every 4th, counted backward from the last), sliding otherwise —
        # the reference config's sliding_window_pattern [w, w, w, 0].
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (self.num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        PreTrainedConfig.__post_init__(self, **kwargs)

        # Per-layer RoPE base theta (0 => NoPE). Needs `rope_parameters`, so runs after the super post-init. Not sure if it's the cleanest...
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if (self.num_hidden_layers - 1 - i) % 4 == 0 else self.rope_parameters["rope_theta"]
                for i in range(self.num_hidden_layers)
            ]


# TODO: point at the final repo name once it exists
@auto_docstring(checkpoint="someorgtoo-hf/Muse-Glimmer-30B")
@strict
class MuseGlimmerConfig(PreTrainedConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 6144):
        Output dimension of the vision encoder after patch merging (input width of the multimodal projection).
    projector_hidden_size (`int`, *optional*, defaults to 4096):
        Intermediate dimension of the multimodal projection.

    Example:

    ```python
    >>> from transformers import MuseGlimmerForConditionalGeneration, MuseGlimmerConfig

    >>> # Initializing an MuseGlimmer style configuration
    >>> configuration = MuseGlimmerConfig()

    >>> # Initializing a model from the configuration
    >>> model = MuseGlimmerForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "muse_glimmer"
    sub_configs = {"text_config": MuseGlimmerTextConfig, "vision_config": MuseGlimmerVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 200092
    video_token_id: int = 200091
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = MuseGlimmerTextConfig()
            logger.info("text_config is None, using default MuseGlimmerTextConfig text config.")
        elif isinstance(self.text_config, dict):
            self.text_config = MuseGlimmerTextConfig(**self.text_config)

        if isinstance(self.vision_config, dict):
            self.vision_config = MuseGlimmerVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = MuseGlimmerVisionConfig()
            logger.info("vision_config is None, using default MuseGlimmerVisionConfig vision config.")

        super().__post_init__(**kwargs)


class MuseGlimmerRMSNorm(Gemma4RMSNorm):
    def __init__(self, dim: int | None = None, eps: float = 1e-6, with_scale: bool = True):
        super().__init__(dim, eps, with_scale)


class MuseGlimmerTextCenteredRMSNorm(Gemma2RMSNorm):
    pass


class MuseGlimmerTextMLP(Gemma2MLP):
    pass


class MuseGlimmerTextRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class MuseGlimmerTextAttention(AfmoeAttention):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.q_norm
        del self.k_norm
        self.qk_norm = MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False)
        self.qk_scale_factor = config.qk_scale_factor

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.qk_norm(query_states) * self.qk_scale_factor
        key_states = self.qk_norm(key_states)

        # NoPE layers receive `position_embeddings=None` from the model.
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MuseGlimmerTextDecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = MuseGlimmerTextMLP(config)
        self.self_attn = MuseGlimmerTextAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = MuseGlimmerTextCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MuseGlimmerTextCenteredRMSNorm(config.hidden_size, eps=config.post_norm_eps)
        self.pre_feedforward_layernorm = MuseGlimmerTextCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = MuseGlimmerTextCenteredRMSNorm(config.hidden_size, eps=config.post_norm_eps)


class MuseGlimmerPreTrainedModel(Gemma2PreTrainedModel):
    _no_split_modules = ["MuseGlimmerTextDecoderLayer", "MuseGlimmerVisionEncoderLayer"]
    _can_record_outputs = None  # set on children directly as they are different for text and vision

    def _init_weights(self, module):  # trf-ignore: TRF018  @Tarek this ignore of the rule should not be needed!!
        raise NotImplementedError("No need to inherit, we can use the base one")


class MuseGlimmerTextNormedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, norm_eps: float = 1e-6):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        # Weight-less norm applied on top of the embeddings - cannot be merged to the embedding matrix, as Dflash implem needs
        # to embed without the norm
        self.embed_norm = MuseGlimmerRMSNorm(eps=norm_eps, with_scale=False)

    def forward(self, input_ids: torch.Tensor):
        return self.embed_norm(super().forward(input_ids))


class MuseGlimmerTextModel(Gemma2Model):
    config: MuseGlimmerTextConfig
    _can_record_outputs = {
        "hidden_states": MuseGlimmerTextDecoderLayer,
        "attentions": MuseGlimmerTextAttention,
    }

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__(config)
        self.embed_tokens = MuseGlimmerTextNormedEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, config.rms_norm_eps
        )
        self.layers = nn.ModuleList(
            [MuseGlimmerTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = MuseGlimmerTextRotaryEmbedding(config)
        self.norm = MuseGlimmerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                # NoPE layers (layer_rope_theta == 0) get no position embeddings.
                position_embeddings=position_embeddings if self.config.layer_rope_theta[i] else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MuseGlimmerVisionAttention(Kimi_K25VisionAttention):
    pass


class MuseGlimmerVisionMLP(Kimi_K25VisionMLP):
    pass


class MuseGlimmerVisionEncoderLayer(Kimi_K25VisionEncoderLayer):
    pass


# override the fn from `vision_utils.py` since muse_glimmer uses `F.grid_sample` in ref
# and `grid_sample` applies padding unlike `f.interpolate`. Custom interpolation
# export-friendly code used in muse_glimmer, thus kept in model file
def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fn is equivalent with F.grid_sample(inputs, align_coreners=False, padding="zeros")"""
    if kwargs is not None:
        bilinear_indices = kwargs.pop("bilinear_indices", None)
        bilinear_weights = kwargs.pop("bilinear_weights", None)
        if bilinear_indices is not None and bilinear_weights is not None:
            return bilinear_indices, bilinear_weights
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device

    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_grid = (torch.arange(h, device=device).float() + 0.5) * (side / h) - 0.5
        w_grid = (torch.arange(w, device=device).float() + 0.5) * (side / w) - 0.5

        # NOTE: use `floor()`, not `int()` to avoid truncation when align corner is False
        h_floor = torch.floor(h_grid).long()
        w_floor = torch.floor(w_grid).long()
        h_ceil = h_floor + 1
        w_ceil = w_floor + 1
        h_frac = h_grid - h_floor.float()
        w_frac = w_grid - w_floor.float()

        h_floor_valid = (h_floor >= 0) & (h_floor <= side - 1)
        h_ceil_valid = (h_ceil >= 0) & (h_ceil <= side - 1)
        w_floor_valid = (w_floor >= 0) & (w_floor <= side - 1)
        w_ceil_valid = (w_ceil >= 0) & (w_ceil <= side - 1)
        h_floor = h_floor.clamp(0, side - 1)
        h_ceil = h_ceil.clamp(0, side - 1)
        w_floor = w_floor.clamp(0, side - 1)
        w_ceil = w_ceil.clamp(0, side - 1)

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            (
                (1 - h_frac)[:, None] * (1 - w_frac)[None, :] * (h_floor_valid[:, None] & w_floor_valid[None, :])
            ).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :] * (h_floor_valid[:, None] & w_ceil_valid[None, :])).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :] * (h_ceil_valid[:, None] & w_floor_valid[None, :])).flatten(),
            (h_frac[:, None] * w_frac[None, :] * (h_ceil_valid[:, None] & w_ceil_valid[None, :])).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights


class MuseGlimmerVisionPatchEmbedder(PaddleOCRVisionEmbeddings):
    def __init__(self, config: MuseGlimmerVisionConfig):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_temporal * 3 * config.patch_size**2

        self.patch_embedding = nn.Linear(self.patch_size, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(config.pos_emb_height * config.pos_emb_width, self.hidden_size)
        # FIXME: only if square images - vision utils don't yet support non-square
        # For now assume pos_emb_height == pos_emb_width always, i.e. as in shared ckpt
        self.num_grid_per_side = config.pos_emb_height

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, image_channels, patch_size, patch_size)`):
                The tensors corresponding to the input images.
            grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        batch_sequence_len = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = embeddings.reshape(batch_sequence_len, -1)

        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=1,
            kwargs=kwargs,
        )
        # this doesn;t match ref since we compute manually in fp32. `F.grid_sample` has some numerical
        # error accumulated and for whatever reason, that might be important (see comment in ref code)
        pos_embeds = (self.position_embedding_table(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        embeddings = embeddings + pos_embeds.to(embeddings.dtype)

        return embeddings


class MuseGlimmerVisionRotaryEmbedding(Gemma4VisionRotaryEmbedding):
    def forward(self, x, position_ids):
        # We interleave as `[freq_w, freq_h, freq_w, freq_h]` in MuseGlimmer
        inv_freq_expanded = (
            self.inv_freq[None, :, None].to(device=x.device, dtype=torch.float32).expand(position_ids.shape[0], -1, 1)
        )
        w_ids = position_ids[:, :, 0][:, None, :].float()
        h_ids = position_ids[:, :, 1][:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freq_h = (inv_freq_expanded @ h_ids).transpose(1, 2)
            freq_w = (inv_freq_expanded @ w_ids).transpose(1, 2)
            freq = torch.cat([freq_w, freq_h, freq_w, freq_h], dim=-1)
            cos = freq.cos() * self.attention_scaling
            sin = freq.sin() * self.attention_scaling

        return cos.to(x.dtype), sin.to(x.dtype)


class MuseGlimmerVisionModel(MuseGlimmerPreTrainedModel):
    config: MuseGlimmerVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")
    _can_record_outputs = {
        "hidden_states": MuseGlimmerVisionEncoderLayer,
        "attentions": MuseGlimmerVisionAttention,
    }

    def __init__(self, config: MuseGlimmerVisionConfig):
        super().__init__(config)
        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(config)
        self.rotary_emb = MuseGlimmerVisionRotaryEmbedding(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([MuseGlimmerVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def pixel_shuffle(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        factor = self.config.merge_size
        dim = hidden_states.shape[-1]

        output = []
        offset = 0

        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            n_tokens = t * h * w

            hidden_states_chunk = hidden_states[offset : offset + n_tokens]

            # per-frame downsample (t frames share the same h,w perm)
            n_out_per_frame = (h // factor) * (w // factor)
            ds_perm = torch.arange(h * w, device=hidden_states.device)
            ds_perm = ds_perm.view(h // factor, factor, w // factor, factor).permute(0, 2, 1, 3).reshape(-1)

            if t > 1:
                # offset the perm per frame so it indexes correctly into the flattened (t*h*w) sequence
                frame_offsets = (torch.arange(t, device=hidden_states.device) * h * w).view(t, 1)
                ds_perm_all = (ds_perm.unsqueeze(0) + frame_offsets).reshape(-1)
            else:
                ds_perm_all = ds_perm

            hidden_states_downsampled = hidden_states_chunk[ds_perm_all]
            hidden_states_downsampled = hidden_states_downsampled.view(t * n_out_per_frame, factor * factor, dim)
            hidden_states_downsampled = (
                hidden_states_downsampled.permute(0, 2, 1)
                .contiguous()
                .view(t * n_out_per_frame, dim * factor * factor)
            )

            output.append(hidden_states_downsampled)
            offset += n_tokens

        return torch.cat(output, dim=0)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        cu_seqlens = get_vision_cu_seqlens(grid_thw, kwargs=kwargs)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=1,
            # assumes pos_emb_height==pos_emb_width, adapt to non-square if needed
            window_size=self.config.pos_emb_height * self.config.patch_size,
            patch_size=self.config.patch_size,
            kwargs=kwargs,
        )

        inputs_embeds = self.patch_embedder(pixel_values, grid_thw)
        hidden_states = self.ln_pre(inputs_embeds)
        hidden_states = hidden_states[window_index, :]

        # Add `1` because ref implementation's position offset is `1`!
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1)
        position_ids = position_ids.flip(-1) + 1
        position_ids = position_ids[None, window_index, :]  # unsqueeze single batch size
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens_mapping = {
            "full_attention": cu_seqlens,
            "window_attention": cu_window_seqlens,
        }
        for i, block in enumerate(self.layers):
            hidden_states = block(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens_mapping[self.config.layer_types[i]],
            )

        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states, grid_thw)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class MuseGlimmerVisionAdapter(nn.Module):
    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.out_hidden_size, config.projector_hidden_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.fc2 = nn.Linear(config.projector_hidden_size, config.projector_hidden_size, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.act(self.fc2(self.act(self.fc1(x))))


class MuseGlimmerModel(Kimi_K25Model):
    def __init__(self, config: MuseGlimmerConfig):
        super().__init__(config)
        del self.mm_projector
        self.vision_adapter = MuseGlimmerVisionAdapter(config)
        self.vision_projection = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=False)
        self.perception_emb_norm = MuseGlimmerRMSNorm(eps=config.text_config.rms_norm_eps, with_scale=False)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            **kwargs,
        )
        vision_features = self.vision_adapter(vision_outputs.last_hidden_state)
        vision_features = self.vision_projection(vision_features)
        vision_features = self.perception_emb_norm(vision_features)
        split_sizes = (image_grid_thw.prod(-1) // self.config.vision_config.merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_features, split_sizes)
        return vision_outputs


class MuseGlimmerForConditionalGeneration(Kimi_K25ForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # MuseGlimmer pre-scales logits by `output_multiplier` before the Gemma-style tanh softcap.
        # Together with `final_logit_softcapping = T`, this gives `T * tanh(logits * mult / T)`.
        logits = logits * self.config.text_config.output_multiplier
        logits = logits / self.config.text_config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

        return MuseGlimmerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "MuseGlimmerTextConfig",
    "MuseGlimmerVisionConfig",
    "MuseGlimmerConfig",
    "MuseGlimmerPreTrainedModel",
    "MuseGlimmerTextModel",
    "MuseGlimmerVisionModel",
    "MuseGlimmerModel",
    "MuseGlimmerForConditionalGeneration",
    "MuseGlimmerImageProcessor",
    "MuseGlimmerVideoProcessor",
]

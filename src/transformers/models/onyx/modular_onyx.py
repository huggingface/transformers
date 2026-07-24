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
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, logging
from ...utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from ...utils.generic import (
    maybe_autocast,
    merge_with_config_defaults,
)
from ...utils.output_capturing import capture_outputs
from ...vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2MLP,
    Gemma2Model,
    Gemma2PreTrainedModel,
    Gemma2RotaryEmbedding,
    eager_attention_forward,
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
from ..paddleocr_vl.modeling_paddleocr_vl import PaddleOCRVisionEmbeddings


logger = logging.get_logger(__name__)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_tokens: int,
) -> tuple[int, int]:
    """Pick the integer (H, W) grid closest to the aspect ratio under the token cap.

    Mirrors ``OnyxVisionEncoder._compute_grid_size`` so the processor needs no
    torch model import. Returns ``(target_h, target_w)``.
    """
    i_nph = height / patch_size
    i_npw = width / patch_size
    ratio = i_npw / i_nph if i_nph > 0 else 1.0
    if i_nph * i_npw > max_tokens:
        i_nph = (max_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(i_nph), math.ceil(i_nph)],
                [math.floor(i_npw), math.ceil(i_npw)],
            )
        )
    )
    candidates = [(nph, npw) for nph, npw in candidates if nph >= 1 and npw >= 1 and nph * npw <= max_tokens]
    if not candidates:
        candidates = [(max(1, round(i_nph)), max(1, round(i_npw)))]
    nph, npw = min(candidates, key=lambda c: abs(c[0] / c[1] - height / width))
    return nph * patch_size, npw * patch_size


class OnyxImageProcessorKwargs(Glm4vImageProcessorKwargs):
    max_image_tokens: int


class OnyxImageProcessor(Glm4vImageProcessor):
    resample = PILImageResampling.LANCZOS
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = None
    merge_size = 2
    max_image_tokens = 4096

    def __init__(self, **kwargs: Unpack[OnyxImageProcessorKwargs]):
        TorchvisionBackend.__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # Onyx uses aspect_ratio_preserving_resize driven by patch_size,
        # not the standard `size` parameter. Temporarily disable do_resize so
        # the base validation doesn't raise an error
        kwargs["do_resize"] = False
        TorchvisionBackend._validate_preprocess_kwargs(**kwargs)

    def _standardize_kwargs(self, **super_kwargs):
        raise NotImplementedError("Model doesn't need to override")

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
        # Different from Qwen-VL, we use new way to infer `resized_height/width` and we swap `channel` and `temporal_patch` dim before flattening
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                height, width = stacked_images.shape[-2:]
                resized_height, resized_width = get_aspect_ratio_preserving_size(
                    height=height,
                    width=width,
                    patch_size=patch_size * merge_size,
                    max_tokens=max_image_tokens,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                    antialias=True,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
            flatten_patches = patches.reshape(
                batch_size, grid_t * grid_h * grid_w, temporal_patch_size * channel * patch_size * patch_size
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

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

        resized_height, resized_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size * merge_size,
            max_tokens=max_image_tokens,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class OnyxModelOutputWithPast(Gemma3ModelOutputWithPast):
    pass


class OnyxCausalLMOutputWithPast(Gemma3CausalLMOutputWithPast):
    pass


@auto_docstring
@strict
class OnyxVisionConfig(Kimi_K25VisionConfig):
    r"""
    pos_emb_height (`int`, *optional*):
        Initial position embedding height.
    pos_emb_width (`int`, *optional*):
        Initial position embedding width.
    patch_temporal (`int`, *optional*):
        The temporal patch size used to embed inputs.
    output_dim (`int`, *optional*):
        Output dimension for encoded image last hidden states.
    adapter_dim (`int`, *optional*):
        Intermediate dimension used in multimodal projection.
    merge_size (`tuple[int] | list[int]`, *optional*):
        Kernel size for patch merging.
    """

    model_type = "onyx_vision"

    hidden_size: int = 1536
    output_dim: int = 6144
    num_hidden_layers: int = 50
    intermediate_size: int = 8960
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    adapter_dim: int = 4096
    hidden_act: str = "gelu"
    max_position_embeddings: int = 32 * 32  # == `pos_h * pos_w`
    layer_norm_eps: float = 1e-05
    layer_types: list[str] | None = None
    pos_emb_time = AttributeError()
    merge_kernel_size = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            stride = 4
            self.layer_types = [
                "full_attention" if (i + 1) % stride == 0 or i == self.num_hidden_layers - 1 else "window_attention"
                for i in range(self.num_hidden_layers)
            ]
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring
@strict
class OnyxTextConfig(Gemma2Config, PreTrainedConfig):
    r"""
    final_logit_softcapping (`float`, *optional*, defaults to 30.0):
        scaling factor when applying tanh softcapping on the logits.
    use_bidirectional_attention (`bool`, *optional*):
        If True, the model will attend to all text tokens instead of using a causal mask.
    qk_scale_factor (`float`, *optional*, defaults to 43.7840518911):
        Multiplier applied to Q after QK-norm, before the standard `1/sqrt(head_dim)` attention scaling.
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply a scaleless RMSNorm to Q and K before rotary.
    use_attn_output_gate (`bool`, *optional*, defaults to `True`):
        Whether to gate the per-head attention output with `sigmoid(output_gate_proj(hidden))`.
    output_multiplier (`float`, *optional*, defaults to 0.19611613513818404):
        Scale applied to logits before the final tanh softcap.
    post_norm_eps (`float`, *optional*, defaults to 1e-8):
        Epsilon used for the post-attention and post-FFN norms (which sit between the sub-layer output and the residual).
    no_rope_layers (`list[int]`, *optional*):
        Explicit per-layer rotary mask: 1 = apply rotary, 0 = NoPE. Defaults to an iRoPE pattern with NoPE
        every 4 layers, counting backward from the last layer.
    """

    model_type = "onyx_text"

    vocab_size: int = 202_048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 16_384
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 200_000
    eos_token_id: int | list[int] | None = 200_001
    pad_token_id: int | None = None
    sliding_window: int | None = 2048
    final_logit_softcapping: float | None = 20.0
    layer_types: list[str] | None = None
    query_pre_attn_scalar = AttributeError()
    attn_logit_softcapping = AttributeError()

    # Onyx-specific fields
    qk_scale_factor: float = 43.7840518911
    use_qk_norm: bool = True
    use_attn_output_gate: bool = True
    output_multiplier: float = 0.19611613513818404
    post_norm_eps: float = 1e-8
    no_rope_layers: list[int] | None = None

    def __post_init__(self, **kwargs):
        # iRoPE mask: default to NoPE every 4 layers, counted backward from the last layer.
        if self.no_rope_layers is None:
            stride = 4
            self.no_rope_layers = [
                0 if (self.num_hidden_layers - 1 - i) % stride == 0 else 1 for i in range(self.num_hidden_layers)
            ]

        # Full attention for NoPE layers, sliding otherwise (Onyx's default layout matches
        # the sliding_window_pattern [w, w, w, 0] used in the reference config).
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if self.no_rope_layers[i] == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring
@strict
class OnyxConfig(PreTrainedConfig):
    r"""
    TODO
    """

    model_type = "onyx"
    sub_configs = {
        "text_config": OnyxTextConfig,
        "vision_config": OnyxVisionConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 200092
    video_token_id: int = 200091

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = OnyxTextConfig()
            logger.info("text_config is None, using default OnyxTextConfig text config.")
        elif isinstance(self.text_config, dict):
            self.text_config = OnyxTextConfig(**self.text_config)

        if isinstance(self.vision_config, dict):
            self.vision_config = OnyxVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = OnyxVisionConfig()
            logger.info("vision_config is None, using default OnyxVisionConfig vision config.")

        super().__post_init__(**kwargs)


class OnyxRMSNorm(Gemma4RMSNorm):
    def __init__(self, dim: int | None = None, eps: float = 1e-6, with_scale: bool = True, weight_offset: int = 0):
        super().__init__(dim, eps, with_scale)
        # can we bake it in weights, i suppose this was needed for train-time stability?
        self.weight_offset = weight_offset

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * (self.weight.float() + self.weight_offset)
        return normed_output.type_as(hidden_states)


class OnyxNormalizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, eps: float = 1e-5):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.norm = OnyxRMSNorm(eps=eps, with_scale=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(super().forward(input_ids))


class OnyxMLP(Gemma2MLP):
    pass


class OnyxRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class OnyxAttention(Gemma2Attention):
    def __init__(self, config: OnyxTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.head_dim**-0.5
        self.attn_logit_softcapping = None

        self.use_rope = config.no_rope_layers[layer_idx] == 1

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_norm = OnyxRMSNorm(eps=config.rms_norm_eps, with_scale=False)
            self.scale_query_by = config.qk_scale_factor / (config.head_dim**0.5)

        self.use_output_gate = config.use_attn_output_gate
        if self.use_output_gate:
            self.output_gate_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
            )

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

        if self.use_qk_norm:
            query_states = self.qk_norm(query_states) * self.scale_query_by
            key_states = self.qk_norm(key_states)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_interleave(query_states, key_states, cos, sin)

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
        # attn_output shape here: (batch, seq, num_heads, head_dim)

        if self.use_output_gate:
            gate = torch.sigmoid(self.output_gate_proj(hidden_states).view(*attn_output.shape))
            attn_output = gate * attn_output

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OnyxDecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: OnyxTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Add an offset
        self.input_layernorm = OnyxRMSNorm(config.hidden_size, eps=config.rms_norm_eps, weight_offset=1)
        self.post_attention_layernorm = OnyxRMSNorm(config.hidden_size, eps=config.post_norm_eps, weight_offset=1)
        self.pre_feedforward_layernorm = OnyxRMSNorm(config.hidden_size, eps=config.rms_norm_eps, weight_offset=1)
        self.post_feedforward_layernorm = OnyxRMSNorm(config.hidden_size, eps=config.post_norm_eps, weight_offset=1)


class OnyxVisionRotaryEmbedding(Gemma4VisionRotaryEmbedding):
    def forward(self, x, position_ids):
        # We interleave as `[freq_w, freq_h, freq_w, freq_h]` in Onyx
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"

        w_ids = position_ids[:, :, 0][:, None, :].float()
        h_ids = position_ids[:, :, 1][:, None, :].float()
        with maybe_autocast(device_type=device_type, enabled=False):
            freq_h = (inv_freq_expanded @ h_ids).transpose(1, 2)
            freq_w = (inv_freq_expanded @ w_ids).transpose(1, 2)
            freq = torch.cat([freq_w, freq_h, freq_w, freq_h], dim=-1)
            cos = freq.cos() * self.attention_scaling
            sin = freq.sin() * self.attention_scaling

        return cos.to(x.dtype), sin.to(x.dtype)


class OnyxVisionAttention(Kimi_K25VisionAttention):
    pass


class OnyxVisionMLP(Kimi_K25VisionMLP):
    pass


class OnyxVisionEncoderLayer(Kimi_K25VisionEncoderLayer):
    pass


class OnyxVisionPatchEmbedder(PaddleOCRVisionEmbeddings):
    def __init__(self, config: OnyxVisionConfig):
        nn.Module.__init__()
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
            align_corners=False,
            kwargs=kwargs,
        )
        # this doesn;t match ref since we compute manually in fp32. `F.grid_sample` has some numerical
        # error accumulated and for whatever reason, that might be important (see comment in ref code)
        pos_embeds = (self.position_embedding_table(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        embeddings = embeddings + pos_embeds.to(embeddings.dtype)

        return embeddings


class OnyxPreTrainedModel(Gemma2PreTrainedModel):
    _no_split_modules = ["OnyxDecoderLayer", "OnyxVisionEncoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)


class OnyxVisionModel(OnyxPreTrainedModel):
    config: OnyxVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")
    _can_record_outputs = {
        "hidden_states": OnyxVisionEncoderLayer,
        "attentions": OnyxVisionAttention,
    }

    def __init__(self, config: OnyxVisionConfig):
        super().__init__(config)
        self.patch_embedder = OnyxVisionPatchEmbedder(config)
        self.rotary_emb = OnyxVisionRotaryEmbedding(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([OnyxVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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

        for i, block in enumerate(self.layers):
            is_global = self.config.layer_types[i] == "full_attention"
            hidden_states = block(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens if is_global else cu_window_seqlens,
            )

        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states, grid_thw)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class OnyxTextModel(Gemma2Model):
    config: OnyxTextConfig

    def __init__(self, config: OnyxTextConfig):
        super().__init__(config)
        # Replace Gemma2's sqrt(hidden_size)-scaled embedding — Onyx normalizes token embeddings instead.
        self.embed_tokens = OnyxNormalizedEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, eps=config.rms_norm_eps
        )
        # Final norm uses weight-as-scale (no offset), unlike the per-layer OnyxRMSNorm.
        self.norm = OnyxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()


class OnyxVisionAdapter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.output_dim, config.adapter_dim, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.adapter_dim, config.adapter_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.act(self.fc2(self.act(self.fc1(x))))


class OnyxModel(Kimi_K25Model):
    def __init__(self, config: OnyxConfig):
        super().__init__(config)
        del self.mm_projector
        self.vision_adapter = OnyxVisionAdapter(config.vision_config)
        self.vision_projection = nn.Linear(
            config.vision_config.adapter_dim, config.text_config.hidden_size, bias=False
        )
        self.perception_emb_norm = OnyxRMSNorm(eps=config.text_config.rms_norm_eps, with_scale=False)

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
        vision_outputs.pooler_output = self.perception_emb_norm(vision_features)
        return vision_outputs


class OnyxForConditionalGeneration(Kimi_K25ForConditionalGeneration):
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

        # Onyx pre-scales logits by `output_multiplier` before the Gemma-style tanh softcap.
        # Together with `final_logit_softcapping = T`, this gives `T * tanh(logits * mult / T)`.
        logits = logits * self.config.text_config.output_multiplier
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size, **kwargs)

        return OnyxCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "OnyxTextConfig",
    "OnyxVisionConfig",
    "OnyxConfig",
    "OnyxPreTrainedModel",
    "OnyxTextModel",
    "OnyxVisionModel",
    "OnyxModel",
    "OnyxForConditionalGeneration",
    "OnyxImageProcessor",
]

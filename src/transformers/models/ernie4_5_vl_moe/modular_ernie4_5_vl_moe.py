# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Ernie4.5-VL model."""

import itertools
from collections.abc import Callable
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_utils import BatchFeature
from ...image_transforms import (
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    PILImageResampling,
    SizeDict,
)
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchvision_available,
    logging,
)
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig
from ..ernie4_5_moe.modeling_ernie4_5_moe import (
    Ernie4_5_MoeAttention,
    Ernie4_5_MoeExperts,
    Ernie4_5_MoeMLP,
    Ernie4_5_MoeModel,
    Ernie4_5_MoeRMSNorm,
    Ernie4_5_MoeStatics,
    Ernie4_5_MoeTopKRouter,
)
from ..glm4v.image_processing_glm4v import Glm4vImageProcessor, Glm4vImageProcessorKwargs
from ..glm4v.image_processing_pil_glm4v import Glm4vImageProcessorPil
from ..glm4v.modeling_glm4v import Glm4vForConditionalGeneration
from ..mixtral.modeling_mixtral import load_balancing_loss_func
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from ..qwen2_vl.image_processing_qwen2_vl import smart_resize
from ..qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLModel, VisionMlp


if is_torchvision_available():
    import torchvision.transforms.v2.functional as tvF


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="baidu/ERNIE-4.5-VL-28B-A3B-PT")
@strict
class Ernie4_5_VLMoeVisionConfig(Qwen2VLVisionConfig):
    r"""
    temporal_merge_size (`int`, *optional*, defaults to 2):
        The size used for merge along the temporal dimension.
    """

    model_type = "ernie4_5_vl_moe_vision"

    base_model_tp_plan = {
        "blocks.*.attn.qkv": "colwise",
        "blocks.*.attn.proj": "rowwise",
        "blocks.*.mlp.fc1": "colwise",
        "blocks.*.mlp.fc2": "rowwise",
    }

    hidden_size: int = 1280
    intermediate_size: int = 4 * 1280
    temporal_merge_size: int = 2
    rms_norm_eps: float = 1e-6

    embed_dim = AttributeError()
    mlp_ratio = AttributeError()
    temporal_patch_size = AttributeError()


@auto_docstring(checkpoint="baidu/ERNIE-4.5-VL-28B-A3B-PT")
@strict
class Ernie4_5_VLMoeTextConfig(Ernie4_5_MoeConfig):
    r"""
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether to use a bias in any of the projections including mlp and attention for example
    moe_k (`int`, *optional*, defaults to 6):
        Number of selected experts.
    moe_num_experts (`int`, *optional*, defaults to 64):
        Number of routed experts.
    moe_num_shared_experts (`int`, *optional*, defaults to 2):
        The number of experts that are shared for all MoE forwards.
    moe_norm_min (`float`, *optional*, defaults to 1e-12):
        Minimum division value during routing normalization.
    mlp_layer_types (`list`, *optional*):
        MLP (Moe vs Dense) pattern for each layer.
    """

    model_type = "ernie4_5_vl_moe_text"
    base_config_key = "text_config"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    ignore_keys_at_rope_validation = {"mrope_section"}

    mlp_layer_types: list[str] | None = None
    moe_intermediate_size: list[int] | None = None
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    moe_layer_end_index = AttributeError()
    moe_layer_interval = AttributeError()
    moe_layer_start_index = AttributeError()

    def __post_init__(self, **kwargs):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)

        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = [1536, 512]

        PreTrainedConfig.__post_init__(**kwargs)


@auto_docstring(checkpoint="baidu/ERNIE-4.5-VL-28B-A3B-PT")
@strict
class Ernie4_5_VLMoeConfig(PreTrainedConfig):
    r"""
    image_start_token_id (`int`, *optional*, defaults to 101304):
        The image token index to encode the start of image.
    image_end_token_id (`int`, *optional*, defaults to 101305):
        The image token index to encode the end of image.
    image_token_id (`int`, *optional*, defaults to 100295):
        The image token index to encode the image prompt.
    video_start_token_id (`int`, *optional*, defaults to 101306):
        The video token index to encode the start of video.
    video_end_token_id (`int`, *optional*, defaults to 101307):
        The video token index to encode the end of video.
    video_token_id (`int`, *optional*, defaults to 103367):
        The video token index to encode the video prompt.

    Example:

    ```python
    >>> from transformers import Ernie4_5_VLMoeForConditionalGeneration, Ernie4_5_VLMoeConfig

    >>> # Initializing a Ernie4_5_VLMoe style configuration
    >>> configuration = Ernie4_5_VLMoeConfig()

    >>> # Initializing a model from the Ernie 4.5 VL 28B A3B configuration
    >>> model = Ernie4_5_VLMoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie4_5_vl_moe"
    sub_configs = {"vision_config": Ernie4_5_VLMoeVisionConfig, "text_config": Ernie4_5_VLMoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_start_token_id: int = 101304
    image_end_token_id: int = 101305
    image_token_id: int = 100295
    video_start_token_id: int = 101306
    video_end_token_id: int = 101307
    video_token_id: int = 103367
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        super().__post_init__(**kwargs)


class Ernie4_5_VLMoeTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            raise ValueError(f"Ernie 4.5 VL requires the `default` rope type, but found {self.rope_type} instead.")
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

        self.mrope_section = config.rope_parameters.get("mrope_section", [22, 22, 20])

    @staticmethod
    def compute_default_rope_parameters(
        config: Ernie4_5_VLMoeTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )

        # Special to ernie, we prerotate on the hw dim
        mrope_section = config.rope_parameters.get("mrope_section", [22, 22, 20])
        hw_dim = mrope_section[0] + mrope_section[1]
        t_dim = mrope_section[2]

        inv_freq_3d = torch.empty_like(inv_freq)
        # (Pre-)Rotate to avoid another rotation during the forward
        inv_freq_3d[:hw_dim] = torch.cat([inv_freq[:-t_dim][0::2], inv_freq[:-t_dim][1::2]])
        inv_freq_3d[-t_dim:] = inv_freq[-t_dim:]

        return inv_freq_3d, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        sin = self.recomposition_to_3d(sin)
        cos = self.recomposition_to_3d(cos)

        return cos, sin

    def recomposition_to_3d(self, freq):
        freq_h, freq_w, freq_t = (m[(i + 1) % 3] for i, m in enumerate(freq.split([*self.mrope_section], dim=-1)))
        freq_hw = torch.stack([freq_h, freq_w], dim=-1).flatten(-2)
        freq_hwt = torch.cat([freq_hw, freq_t], dim=-1)
        return freq_hwt.repeat_interleave(2, dim=-1)


def rotate_half_text(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    original_dtype = q.dtype

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q.float() * cos) + (rotate_half_text(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_half_text(k).float() * sin)

    return q_embed.to(original_dtype), k_embed.to(original_dtype)


class Ernie4_5_VLMoeTextAttention(Ernie4_5_MoeAttention):
    pass


class Ernie4_5_VLMoeRMSNorm(Ernie4_5_MoeRMSNorm):
    pass


class Ernie4_5_VLMoeMLP(Ernie4_5_MoeMLP):
    pass


class Ernie4_5_VLMoeMoeStatics(Ernie4_5_MoeStatics):
    pass


class Ernie4_5_VLMoeMoeTopKRouter(Ernie4_5_MoeTopKRouter):
    def __init__(self, config):
        super().__init__(config)
        self.moe_statics = Ernie4_5_VLMoeMoeStatics(config)


class Ernie4_5_VLMoeMoeExperts(Ernie4_5_MoeExperts):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config)
        self.intermediate_dim = config.moe_intermediate_size if intermediate_size is None else intermediate_size


class Ernie4_5_VLMoeSparseMoeBlock(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_k
        self.gate = Ernie4_5_VLMoeMoeTopKRouter(config)
        self.experts = Ernie4_5_VLMoeMoeExperts(config, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        router_logits, top_k_index, top_k_weights = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)

        # moe results are changed to a flattened shape to ease the modality isolated assigning of results
        return final_hidden_states.flatten(), router_logits.flatten()


class Ernie4_5_VLMoeMoeBlock(nn.Module):
    """
    Similar to `Ernie4_5_Moe` where we have modality isolated experts:
        - A set of text experts that are only run on text tokens
        - A set of vision experts that are only run on vision (image/video) tokens

    This modality isolation is unique to the Ernie 4.5 VL Moe models.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts

        self.text_moe = Ernie4_5_VLMoeSparseMoeBlock(config, intermediate_size=config.moe_intermediate_size[0])
        self.vision_moe = Ernie4_5_VLMoeSparseMoeBlock(config, intermediate_size=config.moe_intermediate_size[1])

        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_VLMoeMLP(
                config, config.moe_intermediate_size[0] * config.moe_num_shared_experts
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_mm_token_type_ids: torch.IntTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # (Optional) shared experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        if moe_mm_token_type_ids is not None and moe_mm_token_type_ids.any():
            final_hidden_states = torch.zeros_like(hidden_states)
            router_logits = torch.zeros(
                size=(batch_size * sequence_length, self.num_experts),
                device=final_hidden_states.device,
                dtype=torch.float,
            )

            # True (1 or 2) == vision, False (0) == text tokens
            moe_mm_token_type_ids = moe_mm_token_type_ids.bool()
            token_type_ids_router = moe_mm_token_type_ids.reshape(-1)[:, None].expand(-1, self.num_experts)
            token_type_ids_states = moe_mm_token_type_ids[..., None].expand(-1, -1, hidden_dim)

            # Run moe on each modality and assign their results to the original token positions
            final_hidden_states[~token_type_ids_states], router_logits[~token_type_ids_router] = self.text_moe(
                hidden_states[~token_type_ids_states]
            )
            final_hidden_states[token_type_ids_states], router_logits[token_type_ids_router] = self.vision_moe(
                hidden_states[token_type_ids_states]
            )
        else:
            final_hidden_states, router_logits = self.text_moe(hidden_states)
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            router_logits = router_logits.reshape(-1, self.num_experts)

        # Add (optional) shared experts to the result
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, router_logits


class Ernie4_5_VLMoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Ernie4_5_VLMoeTextAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Ernie4_5_VLMoeMoeBlock(config)
        else:
            self.mlp = Ernie4_5_VLMoeMLP(config)

        self.input_layernorm = Ernie4_5_VLMoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_VLMoeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        moe_mm_token_type_ids: torch.IntTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, Ernie4_5_VLMoeMoeBlock):
            hidden_states, _ = self.mlp(hidden_states, moe_mm_token_type_ids)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class Ernie4_5_VLMoeVisionAttention(Qwen2_5_VLVisionAttention):
    pass


class Ernie4_5_VLMoeVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config, None)

        self.norm1 = nn.LayerNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Ernie4_5VLVisionMLP(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
        )


class Ernie4_5_VLMoePreTrainedModel(Qwen2_5_VLPreTrainedModel):
    _can_compile_fullgraph = False

    _can_record_outputs = {
        "router_logits": OutputRecorder(Ernie4_5_VLMoeMoeBlock, index=1),
        "hidden_states": Ernie4_5_VLMoeDecoderLayer,
        "attentions": Ernie4_5_VLMoeTextAttention,
    }
    _keep_in_fp32_modules_strict = ["gate.weight", "moe_statics"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Ernie4_5_VLMoeMoeTopKRouter):
            init.zeros_(module.moe_statics.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Ernie4_5_VLMoeMoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Ernie4_5_VLMoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Ernie4_5_VLMoeTextModel(Ernie4_5_MoeModel):
    config: Ernie4_5_VLMoeTextConfig

    def __init__(self, config: Ernie4_5_VLMoeTextConfig):
        super().__init__(config)
        self.rotary_emb = Ernie4_5_VLMoeTextRotaryEmbedding(config=config)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        moe_mm_token_type_ids: torch.IntTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        moe_mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The same as `mm_token_type_ids` while additionally considering start/end image/video tokens as respective vision tokens.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Ernie 4.5 VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are is only one scenario when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                moe_mm_token_type_ids=moe_mm_token_type_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Ernie4_5VLVisionMLP(VisionMlp):
    pass


class Ernie4_5_VLMoePatchEmbed(Qwen2_5_VisionPatchEmbed):
    def __init__(
        self,
        patch_size: int | list[int] | tuple[int, int] = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__(patch_size, in_channels, embed_dim)

        del self.temporal_patch_size
        del kernel_size  # noqa: F821
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        return self.proj(hidden_states.to(target_dtype))


class Ernie4_5_VLMoeVisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class Ernie4_5_VLMoeVisionTransformerPretrainedModel(Qwen2VisionTransformerPretrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(Ernie4_5_VLMoeMoeBlock, index=1),
        "hidden_states": Ernie4_5_VLMoeVisionBlock,
        "attentions": Ernie4_5_VLMoeVisionAttention,
    }

    def __init__(self, config) -> None:
        super().__init__(config)

        del self.merger

        self.patch_embed = Ernie4_5_VLMoePatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Ernie4_5_VLMoeVisionRotaryEmbedding(head_dim // 2)

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_dtype(self):
        raise AttributeError("Ernie 4.5 VL Moe does not need this!")

    def get_device(self):
        raise AttributeError("Ernie 4.5 VL Moe does not need this!")

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.ln(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class Ernie4_5_VLMoeVisionMLP(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim, eps=config.vision_config.rms_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.ln(hidden_states)
        return hidden_states


class Ernie4_5_VLMoeVariableResolutionResamplerModel(nn.Module):
    def __init__(self, config: Ernie4_5_VLMoeConfig):
        super().__init__()
        self.config = config

        self.in_dim = config.vision_config.hidden_size
        self.out_dim = config.text_config.hidden_size
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.temporal_merge_size = config.vision_config.temporal_merge_size

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_merge_size**2
        # compress 3d conv(video) to 1d
        self.temporal_dim = self.in_dim * self.spatial_merge_size**2 * self.temporal_merge_size

        self.spatial_linear = Ernie4_5_VLMoeVisionMLP(config, self.spatial_dim, self.spatial_dim)
        self.temporal_linear = Ernie4_5_VLMoeVisionMLP(config, self.temporal_dim, self.spatial_dim)

        self.mlp = nn.Linear(self.spatial_dim, self.out_dim)
        self.after_norm = Ernie4_5_VLMoeRMSNorm(self.out_dim, config.text_config.rms_norm_eps)

    def _temporal_slicing(self, hidden_states, grid_thw):
        """
        Slices along the temporal dimension in even/odd patterns (usually if we have a video input)
        or duplicates along temporal dimension (usually if we have an image input).

        Example:
            Video input with temporal pattern of [1, -1, 2, -2, 3, -3]
                > Even input [1, 2, 3], odd input [-1, -2, -3]
                > Reorderd via slices to [1, 2, 3, -1, -2, -3]
            Image input with temporal pattern [1]
                > Duplicate input [1], [1]
                > Reordered to [1, 1]

        NOTE: This is hard-coded for `temporal_merge_size == 2` and won't work otherwise.
        """
        # Calculating offsets on spatial dim (based on flattened tensors)
        grid_t, grid_hw = grid_thw[:, 0], grid_thw[:, 1:]
        grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_merge_size**2)

        # Calculating offsets on batch dim (based on flattened tensors)
        tokens_per_img_or_vid = (grid_thw.prod(-1) // (self.spatial_merge_size**2)).flatten()
        batch_offsets = torch.empty(tokens_per_img_or_vid.size(), dtype=tokens_per_img_or_vid.dtype)
        batch_offsets[0] = 0
        batch_offsets[1:] = tokens_per_img_or_vid.cumsum(dim=0)[:-1]

        first_slice_offsets = []
        second_slice_offsets = []
        for temporal_size, spatial_size, batch_offset in zip(grid_t, grid_hw_after_conv, batch_offsets):
            # Depending on temporal, we may interleave:
            #   - Images have temporal == 1 --> same offsets (duplicate "frame" image)
            #   - Videos have temporal > 1 --> different offsets (even, odd)
            first_offset_range = range(0, temporal_size, 2)
            second_offset_range = range(1 if temporal_size > 1 else 0, temporal_size, 2)

            for temporal_offset_even, temporal_offset_odd in zip(first_offset_range, second_offset_range):
                first_slice_offsets.append(
                    torch.arange(
                        batch_offset + (temporal_offset_even) * spatial_size,
                        batch_offset + (temporal_offset_even + 1) * spatial_size,
                    )
                )
                second_slice_offsets.append(
                    torch.arange(
                        batch_offset + (temporal_offset_odd) * spatial_size,
                        batch_offset + (temporal_offset_odd + 1) * spatial_size,
                    )
                )

        # Input: [1, -1, 2, -2, 3, -3] or [1]
        # Indices: [0, 2, 4] (even) or [0] (duplicate)
        first_slice_offsets = torch.cat(first_slice_offsets, dim=-1).to(hidden_states.device)
        # Indices: [1, 3, 5] (odd) or [0] (duplicate)
        second_slice_offsets = torch.cat(second_slice_offsets, dim=-1).to(hidden_states.device)

        # Output: [1, 2, 3, -1, -2, -3] or [1, 1]
        return torch.concat(
            [
                torch.index_select(hidden_states, dim=0, index=first_slice_offsets),
                torch.index_select(hidden_states, dim=0, index=second_slice_offsets),
            ],
            dim=-1,
        )

    def forward(self, hidden_states, grid_thw):
        # image spatial
        # reshape imitates convolution via linear projection
        hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1] * (self.spatial_merge_size**2)])
        hidden_states = self.spatial_linear(hidden_states)

        # video temporal
        hidden_states = self._temporal_slicing(hidden_states, grid_thw)
        hidden_states = self.temporal_linear(hidden_states)

        # final mlp
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.after_norm(hidden_states)

        return hidden_states


class Ernie4_5_VLMoeModel(Qwen2VLModel):
    config: Ernie4_5_VLMoeConfig
    _no_split_modules = ["Ernie4_5_VLMoeDecoderLayer", "Ernie4_5_VLMoeVisionBlock"]

    def __init__(self, config: Ernie4_5_VLMoeConfig):
        super().__init__(config)

        del self.visual
        self.vision_tower = Ernie4_5_VLMoeVisionTransformerPretrainedModel._from_config(config.vision_config)
        self.resampler_model = Ernie4_5_VLMoeVariableResolutionResamplerModel(config)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's sizes. The utility expects a `vision + text`
        sequence and will error out otherwise. For pure text sequence, please rely on model's auto-inferred
        position ids. In a mixed vision + text sequence, vision tokens use 3D RoPE (temporal, height, width)
        while text tokens use standard 1D RoPE.

        Example:
            Temporal patches: 3; Height patches: 2; Width patches: 2
            Each vision input results in (temporal x height × width) positions. Here: 3 x 2 × 2 = 12 positions total.

            Temporal position IDs are spaced by:
                `interval = tokens_per_second * temporal_patch_size / fps`

                If fps = 1; tokens_per_second = 25; temporal_patch_size = 2, temporal IDs increase by 50 for each temporal patch:
                `[0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]`

            Height IDs repeat per row: `[0, 0, 1, 1, ...]`
            Width IDs alternate per column: `[0, 1, 0, 1, ...]`
            Text tokens follow standard 1D RoPE and the position IDs grow consequently with a step of `1`

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        temporal_merge_size = self.config.vision_config.temporal_merge_size
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                # image == 1, video == 2
                else:
                    grid_thw = next(grid_iters[modality_type])
                    t_merge_size = 1 if modality_type == 1 else temporal_merge_size
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, t_merge_size, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        video_outputs = self.vision_tower(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs)
        video_embeds = self.resampler_model(video_outputs.last_hidden_state, video_grid_thw)
        split_sizes = (
            video_grid_thw.prod(-1)
            // self.vision_tower.spatial_merge_size**2
            // self.resampler_model.temporal_merge_size
        ).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        video_outputs.pooler_output = video_embeds
        return video_outputs

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        image_outputs = self.vision_tower(pixel_values, image_grid_thw, return_dict=True, **kwargs)
        image_embeds = self.resampler_model(image_outputs.last_hidden_state, image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        image_outputs.pooler_output = image_embeds
        return image_outputs

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        moe_mm_token_type_ids: torch.IntTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeModelOutputWithPast:
        r"""
        mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
        moe_mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The same as `mm_token_type_ids` while additionally considering start/end image/video tokens as respective vision tokens.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            moe_mm_token_type_ids=moe_mm_token_type_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        return MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Ernie4_5_VLMoeForConditionalGeneration(Glm4vForConditionalGeneration, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)

        self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
        self.num_experts = config.text_config.moe_num_experts
        self.num_experts_per_tok = config.text_config.moe_k

    @auto_docstring
    def get_video_features(self, **super_kwargs):
        return super().get_video_features(**super_kwargs)

    @auto_docstring
    def get_image_features(self, **super_kwargs):
        return super().get_image_features(**super_kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        image_grid_thw=None,
        video_grid_thw=None,
        use_cache=True,
        is_first_iteration=False,
        position_ids=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            position_ids=position_ids,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["mm_token_type_ids"] = None
            model_inputs["moe_mm_token_type_ids"] = None

        return model_inputs

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        moe_mm_token_type_ids: torch.IntTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
        moe_mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The same as `mm_token_type_ids` while additionally considering start/end image/video tokens as respective vision tokens.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            moe_mm_token_type_ids=moe_mm_token_type_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Ernie4_5_VLMoeImageProcessorKwargs(Glm4vImageProcessorKwargs):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*):
        The temporal patch size of the vision encoder. Unused in the image processor, only used for videos.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """


class Ernie4_5_VLMoeImageProcessorPil(Glm4vImageProcessorPil):
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 6177}
    temporal_patch_size = None  # Unused

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ):
        """
        Preprocess images one by one for PIL backend.
        """
        processed_images = []
        processed_grids = []

        for image in images:
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height=height,
                    width=width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = self.resize(
                    image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )

            # Rescale and normalize
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            # Ensure float32 for patch processing
            image_array = np.asarray(image, dtype=np.float32)
            if image_array.ndim == 3:  # (C, H, W)
                image_array = np.expand_dims(image_array, axis=0)  # (1, C, H, W)
            if image_array.ndim == 4:  # (B, C, H, W)
                image_array = np.expand_dims(image_array, axis=1)  # (B, T=1, C, H, W)

            resized_height, resized_width = image_array.shape[-2:]
            batch_size, grid_t, channel = image_array.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = image_array.reshape(
                batch_size,
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # [batch, grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
            patches = np.transpose(patches, (0, 1, 3, 6, 4, 7, 2, 5, 8))

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * patch_size * patch_size,
            )

            # Remove batch dimension and append: shape is (seq_len, hidden_dim)
            processed_images.append(flatten_patches.squeeze(0))
            processed_grids.append([grid_t, grid_h, grid_w])

        # Concatenate all images along sequence dimension: (total_seq_len, hidden_dim)
        pixel_values = np.concatenate(processed_images, axis=0)
        image_grid_thw = np.array(processed_grids)

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
        min_pixels = self.size["shortest_edge"]
        max_pixels = self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class Ernie4_5_VLMoeImageProcessor(Glm4vImageProcessor):
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 6177}
    temporal_patch_size = None  # Unused

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)

            # Main difference to Qwen2 VL - no temporal patches
            batch_size, grid_t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # [batch, grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
            patches = patches.permute(0, 1, 3, 6, 4, 7, 2, 5, 8)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * patch_size * patch_size,
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
        min_pixels = self.size["shortest_edge"]
        max_pixels = self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


# Keep aliases for BC
class Ernie4_5_VL_MoeForConditionalGeneration(Ernie4_5_VLMoeForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeForConditionalGeneration` is deprecated; "
            "please use `Ernie4_5_VLMoeForConditionalGeneration` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeConfig(Ernie4_5_VLMoeConfig):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeConfig` is deprecated; please use `Ernie4_5_VLMoeConfig` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeTextConfig(Ernie4_5_VLMoeTextConfig):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeTextConfig` is deprecated; please use `Ernie4_5_VLMoeTextConfig` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeVisionConfig(Ernie4_5_VLMoeVisionConfig):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeVisionConfig` is deprecated; please use `Ernie4_5_VLMoeVisionConfig` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoePreTrainedModel(Ernie4_5_VLMoePreTrainedModel):
    def post_init(self):
        logger.warning_once(
            "`Ernie4_5_VL_MoePreTrainedModel` is deprecated; please use `Ernie4_5_VLMoePreTrainedModel` instead.",
        )
        super().post_init()


class Ernie4_5_VL_MoeModel(Ernie4_5_VLMoeModel):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeModel` is deprecated; please use `Ernie4_5_VLMoeModel` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeTextModel(Ernie4_5_VLMoeTextModel):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeTextModel` is deprecated; please use `Ernie4_5_VLMoeTextModel` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeVisionTransformerPretrainedModel(Ernie4_5_VLMoeVisionTransformerPretrainedModel):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeVisionTransformerPretrainedModel` is deprecated; "
            "please use `Ernie4_5_VLMoeVisionTransformerPretrainedModel` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeVariableResolutionResamplerModel(Ernie4_5_VLMoeVariableResolutionResamplerModel):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeVariableResolutionResamplerModel` is deprecated; "
            "please use `Ernie4_5_VLMoeVariableResolutionResamplerModel` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeImageProcessor(Ernie4_5_VLMoeImageProcessor):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeImageProcessor` is deprecated; please use `Ernie4_5_VLMoeImageProcessor` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeImageProcessorPil(Ernie4_5_VLMoeImageProcessorPil):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeImageProcessorPil` is deprecated; please use `Ernie4_5_VLMoeImageProcessorPil` instead.",
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "Ernie4_5_VL_MoeConfig",
    "Ernie4_5_VL_MoeTextConfig",
    "Ernie4_5_VL_MoeVisionConfig",
    "Ernie4_5_VL_MoePreTrainedModel",
    "Ernie4_5_VL_MoeForConditionalGeneration",
    "Ernie4_5_VL_MoeModel",
    "Ernie4_5_VL_MoeTextModel",
    "Ernie4_5_VL_MoeVisionTransformerPretrainedModel",
    "Ernie4_5_VL_MoeVariableResolutionResamplerModel",
    "Ernie4_5_VL_MoeImageProcessor",
    "Ernie4_5_VL_MoeImageProcessorPil",
    "Ernie4_5_VLMoeConfig",
    "Ernie4_5_VLMoeTextConfig",
    "Ernie4_5_VLMoeVisionConfig",
    "Ernie4_5_VLMoePreTrainedModel",
    "Ernie4_5_VLMoeForConditionalGeneration",
    "Ernie4_5_VLMoeModel",
    "Ernie4_5_VLMoeTextModel",
    "Ernie4_5_VLMoeVisionTransformerPretrainedModel",
    "Ernie4_5_VLMoeVariableResolutionResamplerModel",
    "Ernie4_5_VLMoeImageProcessor",
    "Ernie4_5_VLMoeImageProcessorPil",
]

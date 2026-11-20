# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""GR00T N1.7 model: Qwen3-VL backbone + flow-matching DiT action head for robot action prediction."""

import functools
import math
import re
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict, make_nested_list_of_images
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import requires
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoConfig, AutoModel
from ..qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5_OmniAdaLayerNormZero_Final
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs, smart_resize
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from ..siglip.modeling_siglip import SiglipMLP, eager_attention_forward


logger = logging.get_logger(__name__)

GR00T_LAYER_TYPES = ("full_attention", "text_cross_attention", "image_cross_attention")


@auto_docstring(checkpoint="nvidia/GR00T-N1.7-3B")
@strict
class Gr00tVLRefinerConfig(PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 2048):
        Hidden size of the refiner. Must match the hidden size of the backbone features it refines.
    intermediate_size (`int`, *optional*, defaults to 8192):
        Dimension of the MLP of each refiner block.
    num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of self-attention blocks refining the backbone features before the DiT.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads in each refiner block.
    hidden_act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
        The non-linear activation function of the MLP.
    dropout (`float`, *optional*, defaults to 0.2):
        Dropout probability of the attention outputs, the residual branches and the MLP.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the normal initializer for the weight matrices.
    """

    model_type = "gr00t_vl_refiner"
    base_config_key = "refiner_config"
    base_model_tp_plan = {
        "layers.*.attention.q_proj": "colwise",
        "layers.*.attention.k_proj": "colwise",
        "layers.*.attention.v_proj": "colwise",
        "layers.*.attention.o_proj": "rowwise",
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
    }

    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 4
    num_attention_heads: int = 32
    hidden_act: str = "gelu_pytorch_tanh"
    dropout: float = 0.2
    initializer_range: float = 0.02

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_attention_heads` "
                f"({self.num_attention_heads})."
            )


@auto_docstring(checkpoint="nvidia/GR00T-N1.7-3B")
@strict
class Gr00tActionHeadConfig(PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 1536):
        Hidden size of the DiT, and size of the state and action embeddings it denoises.
    intermediate_size (`int`, *optional*, defaults to 6144):
        Dimension of the MLP of each DiT block.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of DiT blocks.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads in each DiT block.
    hidden_act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
        The non-linear activation function of the MLP.
    layer_types (`list[str]`, *optional*):
        Attention of each DiT block: `"full_attention"` (self-attention over the state and action tokens), or
        `"text_cross_attention"` / `"image_cross_attention"` to the text / image backbone tokens. Defaults to the
        GR00T N1.7 layout: cross-attention blocks alternating between the text and the image tokens, each followed by a
        self-attention block.
    backbone_hidden_size (`int`, *optional*, defaults to 2048):
        Hidden size of the backbone features the cross-attention blocks attend to. Must match
        `vlm_config.text_config.hidden_size`.
    output_size (`int`, *optional*, defaults to 1024):
        Size of the DiT outputs fed to the action decoder. Must match `projector_hidden_size`.
    timestep_embedding_dim (`int`, *optional*, defaults to 256):
        Number of sinusoidal channels used to embed the denoising timestep.
    dropout (`float`, *optional*, defaults to 0.2):
        Dropout probability of the attention outputs, the residual branches and the MLP.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the normal initializer for the weight matrices.
    """

    model_type = "gr00t_action_head"
    base_config_key = "action_head_config"
    base_model_tp_plan = {
        "layers.*.attention.q_proj": "colwise",
        "layers.*.attention.k_proj": "colwise",
        "layers.*.attention.v_proj": "colwise",
        "layers.*.attention.o_proj": "rowwise",
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
    }

    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_types: list[str] | None = None
    backbone_hidden_size: int = 2048
    output_size: int = 1024
    timestep_embedding_dim: int = 256
    dropout: float = 0.2
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 2 else ("text_cross_attention" if i % 4 == 0 else "image_cross_attention")
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_attention_heads` "
                f"({self.num_attention_heads})."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` has {len(self.layer_types)} entries, expected `num_hidden_layers` "
                f"({self.num_hidden_layers})."
            )
        if invalid := set(self.layer_types) - set(GR00T_LAYER_TYPES):
            raise ValueError(f"Unknown `layer_types` {sorted(invalid)}, expected values in {GR00T_LAYER_TYPES}.")


@auto_docstring(checkpoint="nvidia/GR00T-N1.7-3B")
@strict
class Gr00tConfig(PreTrainedConfig):
    r"""
    vlm_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration of the Qwen3-VL vision-language backbone. Defaults to the Cosmos-Reason2-2B layout, with the
        language model truncated to the 16 decoder layers used by GR00T N1.7. Its `text_config` must set
        `tie_last_hidden_states=False`: the action head reads the last decoder layer before the final norm.
    refiner_config (`dict` or [`Gr00tVLRefinerConfig`], *optional*):
        Configuration of the self-attention blocks refining the backbone features before the DiT.
    action_head_config (`dict` or [`Gr00tActionHeadConfig`], *optional*):
        Configuration of the DiT.
    action_horizon (`int`, *optional*, defaults to 40):
        Number of action steps predicted per chunk.
    max_state_dim (`int`, *optional*, defaults to 132):
        Maximum robot state dimension. Shorter state vectors are zero-padded.
    max_action_dim (`int`, *optional*, defaults to 132):
        Maximum action dimension. Shorter action vectors are zero-padded.
    max_num_embodiments (`int`, *optional*, defaults to 32):
        Number of embodiment slots in the embodiment-specific state/action encoders and decoder.
    projector_hidden_size (`int`, *optional*, defaults to 1024):
        Hidden size of the embodiment-specific state encoder and action decoder, and output size of the DiT.
    num_timestep_buckets (`int`, *optional*, defaults to 1000):
        Number of buckets the continuous flow-matching time in `[0, 1]` is discretized into.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        Maximum number of action tokens the learned action position embedding supports.
    num_inference_steps (`int`, *optional*, defaults to 4):
        Number of Euler steps used by `sample_actions`.
    time_sampling_beta_alpha (`float`, *optional*, defaults to 1.5):
        Alpha parameter of the Beta distribution used to sample the flow-matching time during training.
    time_sampling_beta_beta (`float`, *optional*, defaults to 1.0):
        Beta parameter of the Beta distribution used to sample the flow-matching time during training.
    time_sampling_scale (`float`, *optional*, defaults to 0.999):
        The training time is `(1 - Beta(alpha, beta)) * time_sampling_scale`.
    state_dropout_prob (`float`, *optional*, defaults to 0.2):
        Probability of zeroing the whole state embedding of a sample during training.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the normal initializer for the weight matrices.
    loss_reduction (`str`, *optional*, defaults to `"mean"`):
        Reduction applied to the masked flow-matching loss, either `"mean"` or `"none"`.

    Example:

    ```python
    >>> from transformers import Gr00tConfig, Gr00tForConditionalGeneration

    >>> config = Gr00tConfig()
    >>> model = Gr00tForConditionalGeneration(config)
    ```
    """

    model_type = "gr00t"
    sub_configs_defaults = {
        "vlm_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="qwen3_vl",
            init_kwargs={
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 16,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "max_position_embeddings": 262144,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 5000000.0,
                        "mrope_section": [24, 20, 20],
                        "mrope_interleaved": True,
                    },
                    "tie_last_hidden_states": False,
                },
                "vision_config": {
                    "model_type": "qwen3_vl",
                    "depth": 24,
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_heads": 16,
                    "out_hidden_size": 2048,
                    "deepstack_visual_indexes": [5, 11, 17],
                },
            },
        ),
        "refiner_config": Gr00tVLRefinerConfig,
        "action_head_config": Gr00tActionHeadConfig,
    }

    vlm_config: dict | PreTrainedConfig | None = None
    refiner_config: dict | PreTrainedConfig | None = None
    action_head_config: dict | PreTrainedConfig | None = None
    action_horizon: int = 40
    max_state_dim: int = 132
    max_action_dim: int = 132
    max_num_embodiments: int = 32
    projector_hidden_size: int = 1024
    num_timestep_buckets: int = 1000
    max_position_embeddings: int = 1024
    num_inference_steps: int = 4
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    state_dropout_prob: float = 0.2
    initializer_range: float = 0.02
    loss_reduction: str = "mean"

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.loss_reduction not in ("mean", "none"):
            raise ValueError(f"`loss_reduction` must be 'mean' or 'none', got {self.loss_reduction}.")
        text_config = self.vlm_config.text_config
        if text_config.tie_last_hidden_states is not False:
            raise ValueError(
                "`vlm_config.text_config.tie_last_hidden_states` must be False: the action head reads the last decoder "
                "layer before the final norm."
            )
        if self.refiner_config.hidden_size != text_config.hidden_size:
            raise ValueError("`refiner_config.hidden_size` must match `vlm_config.text_config.hidden_size`.")
        if self.action_head_config.backbone_hidden_size != text_config.hidden_size:
            raise ValueError(
                "`action_head_config.backbone_hidden_size` must match `vlm_config.text_config.hidden_size`."
            )
        if self.action_head_config.output_size != self.projector_hidden_size:
            raise ValueError("`action_head_config.output_size` must match `projector_hidden_size`.")


class Gr00tCategorySpecificLinear(nn.Module):
    """Linear layer holding one weight matrix per embodiment, selected with `embodiment_id`."""

    def __init__(self, num_categories: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_categories, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_categories, out_features))

    def forward(self, hidden_states: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        return torch.bmm(hidden_states, self.weight[embodiment_id]) + self.bias[embodiment_id].unsqueeze(1)


class Gr00tCategorySpecificMLP(nn.Module):
    """Two-layer ReLU MLP with embodiment-specific weights."""

    def __init__(self, num_categories: int, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = Gr00tCategorySpecificLinear(num_categories, in_features, hidden_features)
        self.fc2 = Gr00tCategorySpecificLinear(num_categories, hidden_features, out_features)

    def forward(self, hidden_states: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        hidden_states = F.relu(self.fc1(hidden_states, embodiment_id))
        return self.fc2(hidden_states, embodiment_id)


class Gr00tSinusoidalEmbedding(nn.Module):
    """
    Sinusoidal encoding of the discretized timestep. The action encoder and the DiT timestep embedding use two variants
    of it, set by `downscale_freq_shift` and `flip_sin_to_cos`.
    """

    def __init__(self, embedding_dim: int, downscale_freq_shift: int = 0, flip_sin_to_cos: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freqs = nn.Buffer(self.compute_freqs(), persistent=False)

    def compute_freqs(self) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32)
        return (exponent / (half_dim - self.downscale_freq_shift)).exp()

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        freqs = timestep.float()[..., None] * self.freqs
        sin, cos = freqs.sin(), freqs.cos()
        return torch.cat([cos, sin] if self.flip_sin_to_cos else [sin, cos], dim=-1)


class Gr00tActionEncoder(nn.Module):
    """Embeds noisy actions together with a sinusoidal encoding of the denoising timestep."""

    def __init__(self, config: Gr00tConfig):
        super().__init__()
        embedding_dim = config.action_head_config.hidden_size
        self.fc1 = Gr00tCategorySpecificLinear(config.max_num_embodiments, config.max_action_dim, embedding_dim)
        self.fc2 = Gr00tCategorySpecificLinear(config.max_num_embodiments, 2 * embedding_dim, embedding_dim)
        self.fc3 = Gr00tCategorySpecificLinear(config.max_num_embodiments, embedding_dim, embedding_dim)
        self.time_embedding = Gr00tSinusoidalEmbedding(embedding_dim)

    def forward(self, actions: torch.Tensor, timestep: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        action_embeds = self.fc1(actions, embodiment_id)
        time_embeds = self.time_embedding(timestep[:, None].expand(-1, actions.shape[1]))
        hidden_states = torch.cat([action_embeds, time_embeds.to(action_embeds.dtype)], dim=-1)
        hidden_states = F.silu(self.fc2(hidden_states, embodiment_id))
        return self.fc3(hidden_states, embodiment_id)


class Gr00tTimestepEmbedding(nn.Module):
    """Sinusoidal timestep projection followed by a two-layer MLP, conditioning the adaptive norms of the DiT."""

    def __init__(self, config: Gr00tActionHeadConfig):
        super().__init__()
        self.sinusoids = Gr00tSinusoidalEmbedding(
            config.timestep_embedding_dim, downscale_freq_shift=1, flip_sin_to_cos=True
        )
        self.linear_1 = nn.Linear(config.timestep_embedding_dim, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_proj = self.sinusoids(timestep).to(self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(time_proj)))


class Gr00tAdaLayerNorm(Qwen2_5_OmniAdaLayerNormZero_Final):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)


class Gr00tAttention(nn.Module):
    """Multi-headed self-attention without positional encoding, used by the refiner and the DiT."""

    def __init__(self, config: Gr00tActionHeadConfig | Gr00tVLRefinerConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Gr00tCrossAttention(nn.Module):
    """Cross-attention from the state and action tokens of the DiT to the backbone features."""

    def __init__(self, config: Gr00tActionHeadConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.backbone_hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.backbone_hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        kv_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(encoder_hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.v_proj(encoder_hidden_states).view(kv_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Gr00tMLP(SiglipMLP):
    def __init__(self, config: Gr00tActionHeadConfig | Gr00tVLRefinerConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.activation_fn(self.fc1(hidden_states)))
        return self.dropout(self.fc2(hidden_states))


class Gr00tVLRefinerLayer(GradientCheckpointingLayer):
    """Pre-norm self-attention block refining the backbone features."""

    def __init__(self, config: Gr00tVLRefinerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attention = Gr00tAttention(config)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = Gr00tMLP(config)
        # As in the original, the attention output goes through its own dropout, then the residual one
        self.attention_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        # The refiner attends to every backbone token, padding included, as in the original implementation
        attn_output, _ = self.attention(self.norm1(hidden_states), **kwargs)
        hidden_states = hidden_states + self.dropout(self.attention_dropout(attn_output))
        return hidden_states + self.mlp(self.norm3(hidden_states))


class Gr00tDiTLayer(GradientCheckpointingLayer):
    """DiT block: adaptive-norm attention (cross-attention to the backbone features or self-attention) and an MLP."""

    def __init__(self, config: Gr00tActionHeadConfig, layer_idx: int):
        super().__init__()
        self.norm1 = Gr00tAdaLayerNorm(config.hidden_size)
        if config.layer_types[layer_idx] == "full_attention":
            self.attention = Gr00tAttention(config)
        else:
            self.attention = Gr00tCrossAttention(config)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=1e-5, elementwise_affine=False)
        self.mlp = Gr00tMLP(config)
        # As in the original, the attention output goes through its own dropout, then the residual one
        self.attention_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        normed_hidden_states = self.norm1(hidden_states, time_embeds)
        if encoder_hidden_states is None:
            attn_output, _ = self.attention(normed_hidden_states, **kwargs)
        else:
            attn_output, _ = self.attention(
                normed_hidden_states, encoder_hidden_states, attention_mask=encoder_attention_mask, **kwargs
            )
        hidden_states = hidden_states + self.dropout(self.attention_dropout(attn_output))
        return hidden_states + self.mlp(self.norm3(hidden_states))


@auto_docstring
class Gr00tPreTrainedModel(PreTrainedModel):
    config: Gr00tConfig
    base_model_prefix = "model"
    main_input_name = "state"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gr00tDiTLayer", "Gr00tVLRefinerLayer"]
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    # Flash Attention drops the text/image cross-attention masks: they mask keys only, which its unpadding mistakes for
    # query padding when the query and key lengths differ. It gives wrong actions on the original checkpoint, as in PI0
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Gr00tCategorySpecificLinear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.bias)
        elif isinstance(module, Gr00tSinusoidalEmbedding):
            init.copy_(module.freqs, module.compute_freqs())


@auto_docstring(
    custom_intro="""
    Self-attention blocks refining the backbone features before the DiT cross-attends to them.
    """
)
class Gr00tVLRefiner(Gr00tPreTrainedModel):
    config: Gr00tVLRefinerConfig

    def __init__(self, config: Gr00tVLRefinerConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Gr00tVLRefinerLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    @auto_docstring
    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        r"""
        hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The normalized backbone features.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring(
    custom_intro="""
    Diffusion transformer denoising the action chunk, conditioned on the backbone features and the timestep.
    """
)
class Gr00tDiT(Gr00tPreTrainedModel):
    config: Gr00tActionHeadConfig
    # Like PI0's `action_out_proj`, the output projection is sharded and its output gathered for the action decoder
    _tp_plan = {"proj_out_2": "colwise_gather_output"}
    _can_record_outputs = {
        "hidden_states": OutputRecorder(Gr00tDiTLayer),
        "attentions": [OutputRecorder(Gr00tAttention, index=1), OutputRecorder(Gr00tCrossAttention, index=1)],
    }

    def __init__(self, config: Gr00tActionHeadConfig):
        super().__init__(config)
        self.time_embedding = Gr00tTimestepEmbedding(config)
        self.layers = nn.ModuleList(
            [Gr00tDiTLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.proj_out_2 = nn.Linear(config.hidden_size, config.output_size)
        self.post_init()

    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        image_token_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        inputs_embeds (`torch.Tensor` of shape `(batch_size, 1 + action_horizon, hidden_size)`):
            The embedded robot state followed by the embedded noisy action chunk.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, backbone_hidden_size)`):
            The refined backbone features the cross-attention blocks attend to.
        timestep (`torch.LongTensor` of shape `(batch_size,)`):
            The discretized flow-matching timestep, in `[0, num_timestep_buckets)`.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Padding mask of the backbone tokens.
        image_token_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Whether each backbone token is an image token.
        """
        padding_mask = encoder_attention_mask.bool()
        attention_masks = {
            "text_cross_attention": create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=padding_mask & ~image_token_mask,
                encoder_hidden_states=encoder_hidden_states,
            ),
            "image_cross_attention": create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=padding_mask & image_token_mask,
                encoder_hidden_states=encoder_hidden_states,
            ),
            "full_attention": None,
        }

        time_embeds = self.time_embedding(timestep)
        hidden_states = inputs_embeds
        for layer, layer_type in zip(self.layers, self.config.layer_types):
            hidden_states = layer(
                hidden_states,
                time_embeds,
                encoder_hidden_states=None if layer_type == "full_attention" else encoder_hidden_states,
                encoder_attention_mask=attention_masks[layer_type],
                **kwargs,
            )

        # Note the order: the output projection predicts the shift first, unlike the adaptive norms of the blocks
        shift, scale = self.proj_out_1(F.silu(time_embeds)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return BaseModelOutput(last_hidden_state=self.proj_out_2(hidden_states))


@auto_docstring
class Gr00tModel(Gr00tPreTrainedModel):
    def __init__(self, config: Gr00tConfig):
        super().__init__(config)
        self.vlm = AutoModel.from_config(config.vlm_config)
        self.vl_layer_norm = nn.LayerNorm(config.refiner_config.hidden_size, eps=1e-5)
        self.vl_refiner = Gr00tVLRefiner(config.refiner_config)
        self.dit = Gr00tDiT(config.action_head_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    def get_vlm_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.IntTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Runs the backbone and returns the refined features of its last decoder layer, before the final norm, that the
        DiT cross-attends to. They do not depend on the noisy actions, so `sample_actions` computes them only once.
        """
        # Only the action head outputs are recorded, the backbone is run for its last layer features
        kwargs = {
            key: value for key, value in kwargs.items() if key not in ("output_hidden_states", "output_attentions")
        }
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=-1,
            **kwargs,
        )
        return self.vl_refiner(self.vl_layer_norm(vlm_outputs.hidden_states[-1])).last_hidden_state

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        action_embeds: torch.Tensor,
        timestep: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.IntTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        action_embeds (`torch.Tensor` of shape `(batch_size, 1 + action_horizon, action_head_config.hidden_size)`):
            The embedded robot state followed by the embedded noisy action chunk.
        timestep (`torch.LongTensor` of shape `(batch_size,)`):
            The discretized flow-matching timestep, in `[0, num_timestep_buckets)`.
        """
        vlm_features = self.get_vlm_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        return self.dit(
            inputs_embeds=action_embeds,
            encoder_hidden_states=vlm_features,
            timestep=timestep,
            encoder_attention_mask=attention_mask,
            image_token_mask=mm_token_type_ids == 1,
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    GR00T N1.7 with the embodiment-specific state and action projections, trained with a flow-matching objective and
    generating action chunks with `sample_actions`.
    """
)
class Gr00tForConditionalGeneration(Gr00tPreTrainedModel):
    def __init__(self, config: Gr00tConfig):
        super().__init__(config)
        self.model = Gr00tModel(config)
        self.state_encoder = Gr00tCategorySpecificMLP(
            config.max_num_embodiments,
            config.max_state_dim,
            config.projector_hidden_size,
            config.action_head_config.hidden_size,
        )
        self.action_encoder = Gr00tActionEncoder(config)
        self.action_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.action_head_config.hidden_size
        )
        self.action_decoder = Gr00tCategorySpecificMLP(
            config.max_num_embodiments,
            config.projector_hidden_size,
            config.projector_hidden_size,
            config.max_action_dim,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        alpha = torch.tensor(self.config.time_sampling_beta_alpha, dtype=torch.float32)
        beta = torch.tensor(self.config.time_sampling_beta_beta, dtype=torch.float32)
        sample = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        return (1 - sample) * self.config.time_sampling_scale

    def embed_state_and_actions(
        self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor, embodiment_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the DiT input embeddings of the state and the noisy actions, and the discretized timestep."""
        state_embeds = self.state_encoder(state[:, None, :], embodiment_id)
        if self.training and self.config.state_dropout_prob > 0:
            keep_state = torch.rand(state.shape[0], device=state.device) >= self.config.state_dropout_prob
            state_embeds = state_embeds * keep_state[:, None, None].to(state_embeds.dtype)

        bucketed_timestep = (timestep * self.config.num_timestep_buckets).long()
        action_embeds = self.action_encoder(noisy_actions, bucketed_timestep, embodiment_id)
        position_ids = torch.arange(action_embeds.shape[1], device=state.device)
        action_embeds = action_embeds + self.action_position_embeddings(position_ids)[None]
        return torch.cat([state_embeds, action_embeds], dim=1), bucketed_timestep

    def decode_velocity(self, dit_outputs: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """Decodes the DiT outputs of the action tokens into the predicted velocity."""
        return self.action_decoder(dit_outputs, embodiment_id)[:, -self.config.action_horizon :]

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        state: torch.Tensor,
        embodiment_id: torch.LongTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.IntTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        noise: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        state (`torch.Tensor` of shape `(batch_size, max_state_dim)`):
            Normalized and zero-padded robot state.
        embodiment_id (`torch.LongTensor` of shape `(batch_size,)`):
            Index of the embodiment, selecting the embodiment-specific state/action projections.
        noise (`torch.Tensor` of shape `(batch_size, action_horizon, max_action_dim)`, *optional*):
            Noisy action chunk at time `timestep`. During training, it is the pure noise the actions are interpolated
            with. Sampled from a standard normal distribution if not provided.
        timestep (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Flow-matching time in `[0, 1]`, where `0` is pure noise and `1` the clean actions. Sampled from the
            training distribution if not provided.
        actions (`torch.Tensor` of shape `(batch_size, action_horizon, max_action_dim)`, *optional*):
            Normalized ground-truth action chunk. When provided, the flow-matching loss is computed.
        action_mask (`torch.Tensor` of shape `(batch_size, action_horizon, max_action_dim)`, *optional*):
            Mask of the valid (non-padded) action entries the loss is computed on.
        """
        batch_size = state.shape[0]
        device = state.device
        # The processor returns float32 tensors: run them in the dtype of the model, as the original GR00T does
        state = state.to(self.dtype)
        noise = noise.to(self.dtype) if noise is not None else None
        actions = actions.to(self.dtype) if actions is not None else None

        if timestep is None:
            timestep = self.sample_time(batch_size, device)
        if noise is None:
            noise = torch.randn(
                batch_size, self.config.action_horizon, self.config.max_action_dim, device=device, dtype=state.dtype
            )

        if actions is not None:
            time_expanded = timestep[:, None, None].to(actions.dtype)
            noisy_actions = (1 - time_expanded) * noise + time_expanded * actions
            target_velocity = actions - noise
        else:
            noisy_actions = noise

        inputs_embeds, bucketed_timestep = self.embed_state_and_actions(state, noisy_actions, timestep, embodiment_id)
        outputs = self.model(
            action_embeds=inputs_embeds,
            timestep=bucketed_timestep,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        predicted_velocity = self.decode_velocity(outputs.last_hidden_state, embodiment_id)

        loss = None
        if actions is not None:
            if action_mask is None:
                action_mask = torch.ones_like(actions)
            loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none") * action_mask
            if self.config.loss_reduction == "mean":
                loss = loss.sum() / (action_mask.sum() + 1e-6)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=predicted_velocity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        embodiment_id: torch.LongTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.IntTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Generates a normalized action chunk of shape `(batch_size, action_horizon, max_action_dim)` by integrating the
        predicted velocity field from noise (`t=0`) to actions (`t=1`) with `num_steps` Euler steps.
        """
        num_steps = num_steps or self.config.num_inference_steps
        batch_size = state.shape[0]
        state = state.to(self.dtype)
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.action_horizon,
                self.config.max_action_dim,
                device=state.device,
                dtype=state.dtype,
            )
        noise = noise.to(self.dtype)

        # The backbone features don't depend on the noisy actions: run the backbone once, then only the action head
        vlm_features = self.model.get_vlm_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        image_token_mask = mm_token_type_ids == 1

        actions = noise
        dt = 1.0 / num_steps
        for step in range(num_steps):
            timestep = torch.full((batch_size,), step * dt, dtype=torch.float32, device=state.device)
            inputs_embeds, bucketed_timestep = self.embed_state_and_actions(state, actions, timestep, embodiment_id)
            dit_outputs = self.model.dit(
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=vlm_features,
                timestep=bucketed_timestep,
                encoder_attention_mask=attention_mask,
                image_token_mask=image_token_mask,
            )
            actions = actions + dt * self.decode_velocity(dit_outputs.last_hidden_state, embodiment_id)
        return actions


# Vendored from OpenCV (cv2.resize, INTER_AREA), Apache-2.0:
# https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp
@functools.lru_cache(maxsize=16)
def _area_downscale_weights(in_size: int, out_size: int) -> torch.Tensor:
    """Weights, of shape `(out_size, in_size)`, of the fractional pixel areas averaged by `cv2.INTER_AREA`."""
    weights = torch.zeros(out_size, in_size, dtype=torch.float32)
    scale = in_size / out_size
    for dst in range(out_size):
        start, end = dst * scale, dst * scale + scale
        cell_width = min(scale, in_size - start)
        first, last = math.ceil(start), min(math.floor(end), in_size - 1)
        first = min(first, last)
        if first - start > 1e-3:
            weights[dst, first - 1] = (first - start) / cell_width
        weights[dst, first:last] = 1.0 / cell_width
        if end - last > 1e-3:
            weights[dst, last] = min(min(end - last, 1.0), cell_width) / cell_width
    return weights


@functools.lru_cache(maxsize=16)
def _area_upscale_coefficients(in_size: int, out_size: int) -> tuple[torch.Tensor, ...]:
    """
    Source indices and fixed-point (11 bits) weights of the linear interpolation `cv2.INTER_AREA` uses to upscale.
    """
    scale, inv_scale = in_size / out_size, out_size / in_size
    sources, fractions = [], []
    for dst in range(out_size):
        src = math.floor(dst * scale)
        fraction = (dst + 1) - (src + 1) * inv_scale
        fraction = 0.0 if fraction <= 0 else fraction - math.floor(fraction)
        if src >= in_size - 1:
            src, fraction = in_size - 1, 0.0
        sources.append(src)
        fractions.append(fraction)
    first = torch.tensor(sources)
    fractions = torch.tensor(fractions, dtype=torch.float32)
    return (
        first,
        (first + 1).clamp(max=in_size - 1),
        ((1 - fractions) * 2048).round().long(),
        (fractions * 2048).round().long(),
    )


def _area_resize(images: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resizes a batch of square `uint8` images of shape `(batch_size, channels, height, width)` to `size` exactly like
    OpenCV's `cv2.resize(..., interpolation=cv2.INTER_AREA)`, including its integer rounding.
    """
    in_size = images.shape[-1]
    if in_size == size:
        return images
    if in_size == 2 * size:
        summed = images.long().reshape(*images.shape[:2], size, 2, size, 2).sum(dim=(3, 5))
        return ((summed + 2) // 4).to(torch.uint8)
    if in_size > size:
        weights = _area_downscale_weights(in_size, size).to(images.device)
        resized = weights @ (images.float() @ weights.T)
        return resized.round().clamp(0, 255).to(torch.uint8)
    first, second, first_weight, second_weight = (
        t.to(images.device) for t in _area_upscale_coefficients(in_size, size)
    )
    pixels = images.long()
    rows = pixels[..., first] * first_weight + pixels[..., second] * second_weight
    resized = (
        ((first_weight[:, None] * (rows[..., first, :] >> 4)) >> 16)
        + ((second_weight[:, None] * (rows[..., second, :] >> 4)) >> 16)
        + 2
    ) >> 2
    return resized.clamp(0, 255).to(torch.uint8)


class Gr00tImageProcessorKwargs(Qwen2VLImageProcessorKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `256 * 256`):
        The min pixels of the image to resize the image.
    max_pixels (`int`, *optional*, defaults to `4096 * 4096`):
        The max pixels of the image to resize the image.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    image_resolution (`int`, *optional*, defaults to `self.image_resolution`):
        Size of the square images before and after the center crop. Non-square images are first padded to a square
        with black bars.
    crop_fraction (`float`, *optional*, defaults to `self.crop_fraction`):
        Fraction of each side of the image kept by the center crop.
    """

    image_resolution: int
    crop_fraction: float


@auto_docstring
class Gr00tImageProcessor(Qwen2VLImageProcessor):
    size = {"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    image_resolution = 256
    crop_fraction = 0.95
    valid_kwargs = Gr00tImageProcessorKwargs

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        Returns the number of patches of an image. Every image is cropped and resized to `image_resolution`, whatever
        its `height` and `width`.
        """
        images_kwargs = images_kwargs or {}
        image_resolution = images_kwargs.get("image_resolution") or self.image_resolution
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        resized_height, resized_width = smart_resize(
            image_resolution,
            image_resolution,
            factor=patch_size * merge_size,
            min_pixels=images_kwargs.get("min_pixels", self.size["shortest_edge"]),
            max_pixels=images_kwargs.get("max_pixels", self.size["longest_edge"]),
        )
        return (resized_height // patch_size) * (resized_width // patch_size)

    def crop_and_resize(self, images: torch.Tensor, image_resolution: int, crop_fraction: float) -> torch.Tensor:
        """
        Pads the images to a square, resizes them to `image_resolution`, center-crops `crop_fraction` of each side and
        resizes them back to `image_resolution`, all with area interpolation, as the original GR00T evaluation
        transform does.
        """
        height, width = images.shape[-2:]
        if height != width:
            size = max(height, width)
            pad_top, pad_left = (size - height) // 2, (size - width) // 2
            images = F.pad(images, (pad_left, size - width - pad_left, pad_top, size - height - pad_top))
        images = _area_resize(images, image_resolution)
        crop_size = max(1, int(image_resolution * crop_fraction))
        offset = (image_resolution - crop_size) // 2
        images = images[..., offset : offset + crop_size, offset : offset + crop_size]
        return _area_resize(images, image_resolution)

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
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        image_resolution: int = 256,
        crop_fraction: float = 0.95,
        **kwargs,
    ) -> BatchFeature:
        # The crop is computed on `uint8` pixels, like the original OpenCV transform. Float images (in [0, 1]) are
        # converted back afterwards
        images = [
            self.crop_and_resize(image[None], image_resolution, crop_fraction)[0]
            if image.dtype == torch.uint8
            else self.crop_and_resize((image * 255).round().to(torch.uint8)[None], image_resolution, crop_fraction)[
                0
            ].to(image.dtype)
            / 255
            for image in images
        ]
        return super()._preprocess(
            images,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            disable_grouping=disable_grouping,
            return_tensors=return_tensors,
            **kwargs,
        )


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Converts 6D rotations (the first two rows of the rotation matrix) to rotation matrices with Gram-Schmidt."""
    row1 = rot6d[..., :3] / torch.linalg.norm(rot6d[..., :3], dim=-1, keepdim=True)
    row2 = rot6d[..., 3:6] - (row1 * rot6d[..., 3:6]).sum(-1, keepdim=True) * row1
    row2 = row2 / torch.linalg.norm(row2, dim=-1, keepdim=True)
    return torch.stack([row1, row2, torch.linalg.cross(row1, row2, dim=-1)], dim=-2)


@auto_docstring
@requires(backends=("vision", "torch"))
class Gr00tProcessor(Qwen3VLProcessor):
    text_kwargs = {
        "padding": True,
        "padding_side": "left",
        "return_token_type_ids": False,
    }
    videos_kwargs = AttributeError()

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        modality_configs: dict | None = None,
        statistics: dict | None = None,
        embodiment_id_mapping: dict[str, int] | None = None,
        max_state_dim: int = 132,
        max_action_dim: int = 132,
        max_action_horizon: int = 40,
        formalize_language: bool = True,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        use_relative_action: bool = True,
        **kwargs,
    ):
        r"""
        modality_configs (`dict`, *optional*):
            Per-embodiment description of the inputs and outputs: the camera views, the state and action joint groups,
            the action horizon (`delta_indices`) and how each action group is represented (`action_configs`).
        statistics (`dict`, *optional*):
            Per-embodiment dataset statistics of the state and action joint groups, used to normalize them.
        embodiment_id_mapping (`dict[str, int]`, *optional*):
            Index of the embodiment-specific projections of the model for each embodiment tag.
        max_state_dim (`int`, *optional*, defaults to 132):
            Size the normalized state is zero-padded to.
        max_action_dim (`int`, *optional*, defaults to 132):
            Size the normalized actions are zero-padded to.
        max_action_horizon (`int`, *optional*, defaults to 40):
            Number of action steps predicted by the model. Shorter action chunks are zero-padded to it.
        formalize_language (`bool`, *optional*, defaults to `True`):
            Whether to lowercase the instruction and remove its punctuation.
        use_percentiles (`bool`, *optional*, defaults to `True`):
            Whether the state and absolute actions are scaled with their 1st/99th percentiles instead of min/max.
        clip_outliers (`bool`, *optional*, defaults to `True`):
            Whether to clip the normalized state and actions to `[-1, 1]`.
        use_relative_action (`bool`, *optional*, defaults to `True`):
            Whether the action groups declared as relative are predicted relative to the current state.
        """
        self.modality_configs = modality_configs or {}
        self.statistics = statistics or {}
        self.embodiment_id_mapping = embodiment_id_mapping or {}
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        self.formalize_language = formalize_language
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.use_relative_action = use_relative_action
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.image_token_id = getattr(tokenizer, "image_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.image_token
        )
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        # GR00T has no video inputs, so the Qwen3-VL processor init (which expects a video processor) is skipped
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _check_embodiment(self, embodiment_tag: str):
        if embodiment_tag not in self.modality_configs or embodiment_tag not in self.statistics:
            raise ValueError(
                f"Unknown embodiment tag {embodiment_tag!r}. Embodiments with statistics: {sorted(self.statistics)}."
            )

    def _format_prompt(self, instruction: str, num_images: int) -> str:
        if self.formalize_language:
            instruction = re.sub(r"[^\w\s]", "", instruction.lower())
        conversation = [
            {"role": "user", "content": [{"type": "image"}] * num_images + [{"type": "text", "text": instruction}]}
        ]
        return self.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

    def _normalization_params(self, embodiment_tag: str, modality: str, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The `(low, high)` bounds a joint group is scaled to `[-1, 1]` with."""
        stats = self.statistics[embodiment_tag][modality][key]
        if modality == "action" and self._is_relative(embodiment_tag, key):
            # The relative-action statistics replace the action ones as a whole, so their min/max are used
            stats = self.statistics[embodiment_tag]["relative_action"][key]
            return torch.tensor(stats["min"], dtype=torch.float64), torch.tensor(stats["max"], dtype=torch.float64)
        low, high = ("q01", "q99") if self.use_percentiles else ("min", "max")
        return torch.tensor(stats[low], dtype=torch.float64), torch.tensor(stats[high], dtype=torch.float64)

    def _uses_mean_std(self, embodiment_tag: str, modality: str, key: str) -> bool:
        return key in (self.modality_configs[embodiment_tag][modality].get("mean_std_embedding_keys") or [])

    def _action_configs(self, embodiment_tag: str) -> dict[str, dict]:
        action_config = self.modality_configs[embodiment_tag]["action"]
        return dict(zip(action_config["modality_keys"], action_config.get("action_configs") or []))

    def _is_relative(self, embodiment_tag: str, key: str) -> bool:
        config = self._action_configs(embodiment_tag).get(key)
        return self.use_relative_action and config is not None and config["rep"] == "RELATIVE"

    def _normalize(self, values: torch.Tensor, embodiment_tag: str, modality: str, key: str) -> torch.Tensor:
        """Normalizes the raw `values` of a state or action joint group, as the original GR00T processor does."""
        stats = self.statistics[embodiment_tag][modality][key]
        if self._uses_mean_std(embodiment_tag, modality, key):
            mean = torch.tensor(stats["mean"], dtype=torch.float64)
            std = torch.tensor(stats["std"], dtype=torch.float64)
            # Features with a zero std keep their raw value, and only actions are clipped
            has_std = std != 0
            scaled = torch.where(has_std, (values - mean) / torch.where(has_std, std, 1.0), values)
            return scaled.clamp(-1.0, 1.0) if self.clip_outliers and modality == "action" else scaled
        low, high = self._normalization_params(embodiment_tag, modality, key)
        # Joints with a constant value in the dataset are mapped to 0
        varying = ~torch.isclose(high, low).expand_as(values)
        low, high = low.expand_as(values), high.expand_as(values)
        scaled = torch.zeros_like(values)
        scaled[varying] = 2 * (values[varying] - low[varying]) / (high[varying] - low[varying]) - 1
        return scaled.clamp(-1.0, 1.0) if self.clip_outliers else scaled

    def normalize_state(self, state: dict, embodiment_tag: str) -> torch.Tensor:
        """
        Normalizes each joint group of the raw `state` with the statistics of `embodiment_tag`, concatenates them in
        the order of the embodiment's modality config and zero-pads the result to `max_state_dim`.

        Args:
            state (`dict[str, np.ndarray | torch.Tensor]`):
                Raw state of each joint group, of shape `(batch_size, dim)` or `(batch_size, 1, dim)`.
            embodiment_tag (`str`):
                The embodiment the state comes from.

        Returns:
            `torch.Tensor` of shape `(batch_size, max_state_dim)`.
        """
        self._check_embodiment(embodiment_tag)
        normalized = []
        for key in self.modality_configs[embodiment_tag]["state"]["modality_keys"]:
            values = torch.as_tensor(np.asarray(state[key]), dtype=torch.float64)
            normalized.append(self._normalize(values.reshape(values.shape[0], -1), embodiment_tag, "state", key))
        normalized = torch.cat(normalized, dim=-1)
        if normalized.shape[-1] > self.max_state_dim:
            raise ValueError(f"State dimension {normalized.shape[-1]} exceeds `max_state_dim` ({self.max_state_dim}).")
        return F.pad(normalized, (0, self.max_state_dim - normalized.shape[-1])).float()

    def _reference_state(self, state: dict | None, key: str, config: dict) -> torch.Tensor:
        """The last raw state of the joint group a relative action group is expressed in, of shape `(batch_size, dim)`."""
        if state is None:
            raise ValueError(f"`state` is required to convert the relative action group {key!r}.")
        reference = torch.as_tensor(np.asarray(state[config.get("state_key") or key]), dtype=torch.float64)
        return reference.reshape(reference.shape[0], -1, reference.shape[-1])[:, -1]

    def normalize_actions(
        self, actions: dict, embodiment_tag: str, state: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares raw action chunks as training targets: converts the relative action groups to the current state's
        frame, normalizes each group with the statistics of `embodiment_tag`, concatenates them and zero-pads the result
        to `max_action_horizon` steps and `max_action_dim` dimensions.

        Args:
            actions (`dict[str, np.ndarray | torch.Tensor]`):
                Raw absolute actions of each joint group, of shape `(batch_size, horizon, dim)`, where `horizon` is the
                action horizon of the embodiment.
            embodiment_tag (`str`):
                The embodiment the actions come from.
            state (`dict[str, np.ndarray | torch.Tensor]`, *optional*):
                The raw state of each joint group. Required when the embodiment has relative action groups.

        Returns:
            `tuple(torch.Tensor, torch.Tensor)`: The normalized actions and the mask of their valid entries, both of shape
            `(batch_size, max_action_horizon, max_action_dim)`.
        """
        self._check_embodiment(embodiment_tag)
        action_config = self.modality_configs[embodiment_tag]["action"]
        horizon = len(action_config["delta_indices"])
        configs = self._action_configs(embodiment_tag)
        normalized = []
        for key in action_config["modality_keys"]:
            values = torch.as_tensor(np.asarray(actions[key]), dtype=torch.float64)
            if values.ndim != 3 or values.shape[1] != horizon:
                raise ValueError(
                    f"Expected the action group {key!r} of shape (batch_size, {horizon}, dim), got {tuple(values.shape)}."
                )
            if values.shape[-1] == 0:
                normalized.append(values)
                continue
            if self._is_relative(embodiment_tag, key):
                reference = self._reference_state(state, key, configs[key])
                values = self._absolute_to_relative(values, reference, configs[key])
            normalized.append(self._normalize(values, embodiment_tag, "action", key))
        normalized = torch.cat(normalized, dim=-1)
        batch_size, _, action_dim = normalized.shape
        if action_dim > self.max_action_dim or horizon > self.max_action_horizon:
            raise ValueError(
                f"Actions of {horizon} steps and {action_dim} dimensions exceed `max_action_horizon` "
                f"({self.max_action_horizon}) or `max_action_dim` ({self.max_action_dim})."
            )
        padded = normalized.new_zeros(batch_size, self.max_action_horizon, self.max_action_dim)
        padded[:, :horizon, :action_dim] = normalized
        action_mask = torch.zeros_like(padded)
        action_mask[:, :horizon, :action_dim] = 1
        return padded.float(), action_mask.float()

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] | None = None,
        text: TextInput | list[TextInput] | None = None,
        state: dict | None = None,
        embodiment_tag: str | None = None,
        actions: dict | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        r"""
        images (`ImageInput`, `list[ImageInput]` or `list[list[ImageInput]]`):
            The camera images of each sample, ordered by frame then by camera view as in the embodiment's modality
            config.
        text (`str` or `list[str]`):
            The language instruction of each sample. Unless it already contains image placeholders, it is formatted
            into the GR00T prompt with one image placeholder per image of the sample.
        state (`dict[str, np.ndarray | torch.Tensor]`, *optional*):
            Raw state of each joint group of the embodiment, of shape `(batch_size, dim)`. It is normalized and padded
            with [`~Gr00tProcessor.normalize_state`].
        embodiment_tag (`str`, *optional*):
            The embodiment of the batch. Required to process `state` and `actions` and to return `embodiment_id`.
        actions (`dict[str, np.ndarray | torch.Tensor]`, *optional*):
            Raw absolute action chunks of each joint group, of shape `(batch_size, horizon, dim)`, used as training
            targets. They are prepared with [`~Gr00tProcessor.normalize_actions`].

        Returns:
            [`BatchFeature`]: The Qwen3-VL inputs (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` and
            `mm_token_type_ids`), plus `state` and `embodiment_id` when `state` and `embodiment_tag` are given, and
            `actions` and `action_mask` when `actions` are given.
        """
        if isinstance(text, str):
            text = [text]
        # Plain instructions are formatted into the GR00T prompt. Prompts that already contain image placeholders,
        # e.g. from `apply_chat_template`, are processed as they are
        if text is not None and images is not None and not any(self.image_token in sample for sample in text):
            batched_images = make_nested_list_of_images(images)
            if len(batched_images) != len(text):
                raise ValueError(f"Got {len(batched_images)} samples of images for {len(text)} instructions.")
            text = [
                self._format_prompt(sample, len(sample_images)) for sample, sample_images in zip(text, batched_images)
            ]
            images = [image for sample_images in batched_images for image in sample_images]

        return_tensors = kwargs.pop("return_tensors", None)
        inputs = super().__call__(images=images, text=text, **kwargs)
        if embodiment_tag is not None:
            self._check_embodiment(embodiment_tag)
            batch_size = len(inputs["input_ids"]) if "input_ids" in inputs else len(images)
            inputs["embodiment_id"] = torch.full((batch_size,), self.embodiment_id_mapping[embodiment_tag])
        if state is not None:
            if embodiment_tag is None:
                raise ValueError("`embodiment_tag` is required to normalize `state`.")
            inputs["state"] = self.normalize_state(state, embodiment_tag)
        if actions is not None:
            if embodiment_tag is None:
                raise ValueError("`embodiment_tag` is required to normalize `actions`.")
            inputs["actions"], inputs["action_mask"] = self.normalize_actions(actions, embodiment_tag, state)
        return BatchFeature(data=dict(inputs), tensor_type=return_tensors)

    def post_process_actions(
        self, actions: torch.Tensor, embodiment_tag: str, state: dict | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Converts the normalized action chunks generated by [`Gr00tForConditionalGeneration.sample_actions`] to the
        action space of the embodiment: keeps its action horizon and dimensions, splits them into joint groups,
        un-normalizes them and converts the relative ones to absolute actions with the current state.

        Args:
            actions (`torch.Tensor` of shape `(batch_size, action_horizon, max_action_dim)`):
                The normalized action chunks.
            embodiment_tag (`str`):
                The embodiment the actions are generated for.
            state (`dict[str, np.ndarray | torch.Tensor]`, *optional*):
                The raw state of each joint group passed to the processor. Required when the embodiment has relative
                action groups.

        Returns:
            `dict[str, torch.Tensor]`: The actions of each joint group, of shape `(batch_size, horizon, dim)`.
        """
        self._check_embodiment(embodiment_tag)
        action_config = self.modality_configs[embodiment_tag]["action"]
        horizon = len(action_config["delta_indices"])
        actions = actions.detach().to("cpu", torch.float64)[:, :horizon]
        configs = self._action_configs(embodiment_tag)

        outputs, start = {}, 0
        for key in action_config["modality_keys"]:
            stats = self.statistics[embodiment_tag]["action"][key]
            dim = len(stats["min"])
            values = actions[..., start : start + dim]
            start += dim
            if dim == 0:
                # Joint groups some embodiments disable, e.g. the end-effector poses of `xdof_relative_eef_relative_joint`
                outputs[key] = values
                continue
            if self._uses_mean_std(embodiment_tag, "action", key):
                mean, std = (
                    torch.tensor(stats["mean"], dtype=torch.float64),
                    torch.tensor(stats["std"], dtype=torch.float64),
                )
                values = torch.where(std != 0, values * std + mean, values)
            else:
                low, high = self._normalization_params(embodiment_tag, "action", key)
                values = (values.clamp(-1.0, 1.0) + 1.0) / 2.0 * (high - low) + low

            if self._is_relative(embodiment_tag, key):
                reference = self._reference_state(state, key, configs[key])
                values = self._relative_to_absolute(values, reference, configs[key])
            outputs[key] = values
        return outputs

    @staticmethod
    def _absolute_to_relative(actions: torch.Tensor, reference: torch.Tensor, config: dict) -> torch.Tensor:
        """Expresses absolute actions in the frame of the `reference` state: the inverse of `_relative_to_absolute`."""
        if config["type"] == "NON_EEF":
            return actions - reference[:, None]
        if config["type"] == "EEF" and config["format"] == "XYZ_ROT6D":
            # Poses compose in SE(3): the relative pose is inv(T_reference) @ T_action
            inverse_rotation = _rot6d_to_matrix(reference[:, 3:9]).transpose(-1, -2)
            rotation = inverse_rotation[:, None] @ _rot6d_to_matrix(actions[..., 3:9])
            translation = (inverse_rotation[:, None] @ (actions[..., :3] - reference[:, None, :3])[..., None])[..., 0]
            return torch.cat([translation, rotation[..., 0, :], rotation[..., 1, :]], dim=-1)
        raise NotImplementedError(f"Relative actions of type {config['type']} and format {config['format']}.")

    @staticmethod
    def _relative_to_absolute(actions: torch.Tensor, reference: torch.Tensor, config: dict) -> torch.Tensor:
        if config["type"] == "NON_EEF":
            return actions + reference[:, None]
        if config["type"] == "EEF" and config["format"] == "XYZ_ROT6D":
            reference_rotation = _rot6d_to_matrix(reference[:, 3:9])
            rotation = reference_rotation[:, None] @ _rot6d_to_matrix(actions[..., 3:9])
            translation = (reference_rotation[:, None] @ actions[..., :3, None])[..., 0] + reference[:, None, :3]
            return torch.cat([translation, rotation[..., 0, :], rotation[..., 1, :]], dim=-1)
        raise NotImplementedError(f"Relative actions of type {config['type']} and format {config['format']}.")

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]


__all__ = [
    "Gr00tConfig",
    "Gr00tVLRefinerConfig",
    "Gr00tActionHeadConfig",
    "Gr00tPreTrainedModel",
    "Gr00tVLRefiner",
    "Gr00tDiT",
    "Gr00tModel",
    "Gr00tForConditionalGeneration",
    "Gr00tImageProcessor",
    "Gr00tProcessor",
]

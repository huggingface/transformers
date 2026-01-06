# coding=utf-8
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
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...generation import GenerationMixin
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_processing_utils_fast import (
    group_images_by_shape,
    reorder_images,
)
from ...image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
)
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from ...utils.generic import OutputRecorder, check_model_inputs, maybe_autocast
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
from ..glm4v.image_processing_glm4v_fast import Glm4vImageProcessorFast
from ..glm4v.modeling_glm4v import Glm4vForConditionalGeneration
from ..mixtral.modeling_mixtral import load_balancing_loss_func
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLVisionBlock,
)
from ..qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from ..qwen2_vl.image_processing_qwen2_vl import smart_resize
from ..qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, VisionMlp


logger = logging.get_logger(__name__)


class Ernie4_5_VL_MoeVisionConfig(Qwen2VLVisionConfig):
    r"""
    This is the configuration class to store the configuration of the [`Ernie4_5_VL_MoeVisionTransformerPretrainedModel`].
    It is used to instantiate the vision models portion of the complete Ernie4.5-VL Moe model according to the specified
    arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        depth (`int`, *optional*, defaults to 32):
            Number of layers (depth) in the model.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
        temporal_merge_size (`int`, *optional*, defaults to 2):
            The size used for merge along the temporal dimension.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "ernie4_5_vl_moe_vision"

    base_model_tp_plan = {
        "blocks.*.attn.qkv": "colwise",
        "blocks.*.attn.proj": "rowwise",
        "blocks.*.mlp.fc1": "colwise",
        "blocks.*.mlp.fc2": "rowwise",
    }

    def __init__(
        self,
        depth=32,
        hidden_size=1280,
        hidden_act="quick_gelu",
        intermediate_size=4 * 1280,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_merge_size=2,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(
            depth=depth,
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            in_channels=in_channels,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_merge_size=temporal_merge_size,
            rms_norm_eps=rms_norm_eps,
            initializer_range=initializer_range,
            **kwargs,
        )

        del self.embed_dim  # noqa: F821
        del self.mlp_ratio  # noqa: F821
        del self.temporal_patch_size  # noqa: F821

        self.intermediate_size = intermediate_size
        self.temporal_merge_size = temporal_merge_size
        self.rms_norm_eps = rms_norm_eps


class Ernie4_5_VL_MoeTextConfig(Ernie4_5_MoeConfig, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ernie4_5_VL_MoeTextModel`]. It is used to instantiate a
    the text model portion of the complete Ernie4.5-VL Moe model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 103424):
            Vocabulary size of the Ernie 4.5 VL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Ernie4_5_VL_MoeTextModel`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `4`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in any of the projections including mlp and attention for example.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        mlp_layer_types (`list`, *optional*):
            MLP (Moe vs Dense) pattern for each layer.
        moe_intermediate_size (`list[int]`, *optional*, defaults to `[1536, 512]`):
            Intermediate size of the routed experts; differs between text (first) and image (second) experts.
        moe_k (`int`, *optional*, defaults to 6):
            Number of selected experts.
        moe_num_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        moe_num_shared_experts (`int`, *optional*, defaults to 2):
            The number of experts that are shared for all MoE forwards.
        moe_norm_min (`float`, *optional*, defaults to 1e-12):
            Minimum division value during routing normalization.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
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

    def __init__(
        self,
        vocab_size=103424,
        hidden_size=2560,
        intermediate_size=12288,
        num_hidden_layers=28,
        num_attention_heads=20,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        use_bias=False,
        tie_word_embeddings=True,
        rope_parameters=None,
        mlp_layer_types=None,
        moe_intermediate_size=None,
        moe_k=6,
        moe_num_experts=64,
        moe_num_shared_experts=2,
        moe_norm_min=1e-12,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_bias = use_bias
        self.rope_parameters = rope_parameters

        # Default to MoE from the second layer and on
        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        layer_type_validation(self.mlp_layer_types, self.num_hidden_layers, attention=False)

        self.moe_intermediate_size = moe_intermediate_size
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = [1536, 512]
        self.moe_k = moe_k
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_norm_min = moe_norm_min
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        PreTrainedConfig.__init__(
            tie_word_embeddings=tie_word_embeddings, ignore_keys_at_rope_validation={"mrope_section"}, **kwargs
        )


class Ernie4_5_VL_MoeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ernie4_5_VL_MoeModel`]. It is used to instantiate a
    Ernie4.5-VL MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Ernie 4.5 VL 28B A3B [baidu/ERNIE-4.5-VL-28B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Ernie4_5_VL_MoeTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Ernie4_5_VL_MoeVisionConfig`):
            The config object or dictionary of the vision backbone.
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

    ```python
    >>> from transformers import Ernie4_5_VL_MoeForConditionalGeneration, Ernie4_5_VL_MoeConfig

    >>> # Initializing a Ernie4_5_VL_Moe style configuration
    >>> configuration = Ernie4_5_VL_MoeConfig()

    >>> # Initializing a model from the Ernie 4.5 VL 28B A3B configuration
    >>> model = Ernie4_5_VL_MoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie4_5_vl_moe"
    sub_configs = {"vision_config": Ernie4_5_VL_MoeVisionConfig, "text_config": Ernie4_5_VL_MoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_start_token_id=101304,
        image_end_token_id=101305,
        image_token_id=100295,
        video_start_token_id=101306,
        video_end_token_id=101307,
        video_token_id=103367,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, Ernie4_5_VL_MoeVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif isinstance(text_config, Ernie4_5_VL_MoeTextConfig):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


class Ernie4_5_VL_MoeTextRotaryEmbedding(nn.Module):
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
        config: Optional[Ernie4_5_VL_MoeTextConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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


class Ernie4_5_VL_MoeTextAttention(Ernie4_5_MoeAttention):
    pass


class Ernie4_5_VL_MoeRMSNorm(Ernie4_5_MoeRMSNorm):
    pass


class Ernie4_5_VL_MoeMLP(Ernie4_5_MoeMLP):
    pass


class Ernie4_5_VL_MoeMoeStatics(Ernie4_5_MoeStatics):
    pass


class Ernie4_5_VL_MoeMoeTopKRouter(Ernie4_5_MoeTopKRouter):
    def __init__(self, config):
        super().__init__(config)
        self.moe_statics = Ernie4_5_VL_MoeMoeStatics(config)


class Ernie4_5_VL_MoeMoeExperts(Ernie4_5_MoeExperts):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config)
        self.intermediate_dim = config.moe_intermediate_size if intermediate_size is None else intermediate_size


class Ernie4_5_VL_MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_k
        self.gate = Ernie4_5_VL_MoeMoeTopKRouter(config)
        self.experts = Ernie4_5_VL_MoeMoeExperts(config, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        router_logits, top_k_index, top_k_weights = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)

        # moe results are changed to a flattened shape to ease the modality isolated assigning of results
        return final_hidden_states.flatten(), router_logits.flatten()


class Ernie4_5_VL_MoeMoeBlock(nn.Module):
    """
    Similar to `Ernie4_5_Moe` where we have modality isolated experts:
        - A set of text experts that are only run on text tokens
        - A set of vision experts that are only run on vision (image/video) tokens

    This modality isolation is unique to the Ernie 4.5 VL Moe models.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts

        self.text_moe = Ernie4_5_VL_MoeSparseMoeBlock(config, intermediate_size=config.moe_intermediate_size[0])
        self.vision_moe = Ernie4_5_VL_MoeSparseMoeBlock(config, intermediate_size=config.moe_intermediate_size[1])

        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_VL_MoeMLP(
                config, config.moe_intermediate_size[0] * config.moe_num_shared_experts
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        moe_mm_token_type_ids: Optional[torch.IntTensor] = None,
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


class Ernie4_5_VL_MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Ernie4_5_VL_MoeTextAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Ernie4_5_VL_MoeMoeBlock(config)
        else:
            self.mlp = Ernie4_5_VL_MoeMLP(config)

        self.input_layernorm = Ernie4_5_VL_MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_VL_MoeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        moe_mm_token_type_ids: Optional[torch.IntTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, Ernie4_5_VL_MoeMoeBlock):
            hidden_states, _ = self.mlp(hidden_states, moe_mm_token_type_ids)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class Ernie4_5_VL_MoePreTrainedModel(Qwen2_5_VLPreTrainedModel):
    _can_compile_fullgraph = False

    _can_record_outputs = {
        "router_logits": OutputRecorder(Ernie4_5_VL_MoeMoeBlock, index=1),
        "hidden_states": Ernie4_5_VL_MoeDecoderLayer,
        "attentions": Ernie4_5_VL_MoeTextAttention,
    }
    _keep_in_fp32_modules_strict = ["gate.weight", "moe_statics"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Ernie4_5_VL_MoeMoeTopKRouter):
            init.zeros_(module.moe_statics.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Ernie4_5_VL_MoeMoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Ernie4_5_VL_MoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Ernie4_5_VL_MoeTextModel(Ernie4_5_MoeModel):
    config: Ernie4_5_VL_MoeTextConfig

    def __init__(self, config: Ernie4_5_VL_MoeTextConfig):
        super().__init__(config)
        self.rotary_emb = Ernie4_5_VL_MoeTextRotaryEmbedding(config=config)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        moe_mm_token_type_ids: Optional[torch.IntTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
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
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
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
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Ernie4_5VLVisionMLP(VisionMlp):
    pass


class Ernie4_5_VL_MoePatchEmbed(Qwen2_5_VisionPatchEmbed):
    def __init__(
        self,
        patch_size: int = 14,
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


class Ernie4_5_VL_MoeVisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class Ernie4_5_VL_MoeVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config, None)

        self.norm1 = nn.LayerNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Ernie4_5VLVisionMLP(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            hidden_act=config.hidden_act,
        )


class Ernie4_5_VL_MoeVisionTransformerPretrainedModel(Qwen2VisionTransformerPretrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)

        del self.merger

        self.patch_embed = Ernie4_5_VL_MoePatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Ernie4_5_VL_MoeVisionRotaryEmbedding(head_dim // 2)

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_dtype(self):
        raise AttributeError("Ernie 4.5 VL Moe does not need this!")

    def get_device(self):
        raise AttributeError("Ernie 4.5 VL Moe does not need this!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
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
        return hidden_states


class Ernie4_5_VL_MoeVisionMLP(nn.Module):
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


class Ernie4_5_VL_MoeVariableResolutionResamplerModel(nn.Module):
    def __init__(self, config: Ernie4_5_VL_MoeConfig):
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

        self.spatial_linear = Ernie4_5_VL_MoeVisionMLP(config, self.spatial_dim, self.spatial_dim)
        self.temporal_linear = Ernie4_5_VL_MoeVisionMLP(config, self.temporal_dim, self.spatial_dim)

        self.mlp = nn.Linear(self.spatial_dim, self.out_dim)
        self.after_norm = Ernie4_5_VL_MoeRMSNorm(self.out_dim, config.text_config.rms_norm_eps)

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


class Ernie4_5_VL_MoeModel(Qwen2_5_VLModel):
    _checkpoint_conversion_mapping = {"^norm": "language_model.norm"}

    def __init__(self, config: Ernie4_5_VL_MoeConfig):
        super().__init__(config)

        del self.visual
        self.vision_tower = Ernie4_5_VL_MoeVisionTransformerPretrainedModel._from_config(config.vision_config)
        self.resampler_model = Ernie4_5_VL_MoeVariableResolutionResamplerModel(config)

    # TODO: Should be moved to generation loop instead in the future
    # Relevant PR(s): https://github.com/huggingface/transformers/pull/42088
    def get_position_ids(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
    ):
        """
        Calculating the 3D position ids with a custom mechanism / caching
            - First forward calculates the initial positions and the respective
              deltas (offset) for subsequent positions. See `get_rope_index` for
              more details.
            - Second and on (generation), uses the cache position combined with the
              cached deltas to determine the current position.

        NOTE: We assume that the position ids are `None` and recalculate them here in any case.
        """
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            if input_ids is not None:
                batch_size, seq_length, device = input_ids.shape[0], 1, input_ids.device
            elif inputs_embeds is not None:
                batch_size, seq_length, device = inputs_embeds.shape[0], 1, inputs_embeds.device
            else:
                raise ValueError(
                    "Cannot calculate position ids without any input to the model. "
                    "Need either `input_ids` or `inputs_embeds`!"
                )

            delta = (cache_position[0] + self.rope_deltas).to(device) if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return position_ids

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        temporal_merge_size = self.config.vision_config.temporal_merge_size
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                # If we don't have `mm_token_type_ids`, then we have text tokens only (== 0)
                if mm_token_type_ids is None:
                    input_token_type = torch.zeros_like(input_ids)[attention_mask[i] == 1].tolist()
                else:
                    input_token_type = mm_token_type_ids[i, attention_mask[i] == 1].tolist()

                input_type_group = []
                for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                    group = list(group)
                    start_index = group[0][0]
                    end_index = group[-1][0] + 1
                    input_type_group.append((key, start_index, end_index))

                llm_pos_ids_list = []
                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    # text == 0
                    if modality_type == 0:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # image == 1, video == 2
                    else:
                        grid_thw = image_grid_thw if modality_type == 1 else video_grid_thw
                        mm_index = image_index if modality_type == 1 else video_index
                        t_merge_size = 1 if modality_type == 1 else temporal_merge_size

                        t, h, w = (
                            grid_thw[mm_index][0],
                            grid_thw[mm_index][1],
                            grid_thw[mm_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item() // t_merge_size,
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        for t_idx in range(llm_grid_t):
                            t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        if modality_type == 1:
                            image_index += 1
                        else:
                            video_index += 1

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        video_embeds = self.vision_tower(pixel_values_videos, video_grid_thw)
        video_embeds = self.resampler_model(video_embeds, video_grid_thw)
        split_sizes = (
            video_grid_thw.prod(-1)
            // self.vision_tower.spatial_merge_size**2
            // self.resampler_model.temporal_merge_size
        ).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        image_embeds = self.vision_tower(pixel_values, image_grid_thw)
        image_embeds = self.resampler_model(image_embeds, image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        moe_mm_token_type_ids: Optional[torch.IntTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MoeModelOutputWithPast]:
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
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.get_position_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                mm_token_type_ids=mm_token_type_ids,
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
            cache_position=cache_position,
            **kwargs,
        )

        return MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Ernie4_5_VL_MoeForConditionalGeneration(Glm4vForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = {"^model.norm": "model.language_model.norm"}

    def __init__(self, config):
        super().__init__(config)

        self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
        self.num_experts = config.text_config.moe_num_experts
        self.num_experts_per_tok = config.text_config.moe_k

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        cache_position=None,
        past_key_values=None,
        image_grid_thw=None,
        video_grid_thw=None,
        use_cache=True,
        is_first_iteration=False,
        # Intentionally ignore position ids to force custom cache logic
        position_ids=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Using our own caching with rope delta
        model_inputs["position_ids"] = self.model.get_position_ids(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
            past_key_values=model_inputs.get("past_key_values"),
            inputs_embeds=model_inputs.get("inputs_embeds"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            cache_position=model_inputs.get("cache_position"),
            mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        moe_mm_token_type_ids: Optional[torch.IntTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
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
            cache_position=cache_position,
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


class Ernie4_5_VL_MoeImageProcessorKwargs(Glm4vImageProcessorKwargs):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*):
        The temporal patch size of the vision encoder. Unused in the image processor, only used for videos.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """


class Ernie4_5_VL_MoeImageProcessor(Glm4vImageProcessor):
    r"""
    Constructs a Ernie 4.5 VL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]`, *optional*, defaults to `{"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 6177}`):
            Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel
            in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*):
            The temporal patch size of the vision encoder. Unused in the image processor, only used for videos.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 14,
        temporal_patch_size: Optional[int] = None,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        BaseImageProcessor.__init__(**kwargs)
        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            size = {"shortest_edge": size["shortest_edge"], "longest_edge": size["longest_edge"]}
        else:
            size = {"shortest_edge": 56 * 56, "longest_edge": 6177 * 28 * 28}
        self.size = size

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`list[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*):
                The temporal patch size of the vision encoder. Unused in the image processor, only used for videos.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose([0, 3, 1, 2])

        # Main difference to Qwen2 VL - no temporal patches
        channel = patches.shape[1]
        grid_t = patches.shape[0]
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            [
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            ]
        )
        # [grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
        patches = patches.transpose([0, 2, 5, 3, 6, 1, 4, 7])
        flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel * patch_size * patch_size)

        return flatten_patches, (grid_t, grid_h, grid_w)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

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


class Ernie4_5_VL_MoeImageProcessorFast(Glm4vImageProcessorFast):
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 6177}
    temporal_patch_size = None  # Unused

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
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
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
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
    "Ernie4_5_VL_MoeImageProcessorFast",
]

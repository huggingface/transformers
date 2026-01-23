# Copyright 2026 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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

import math

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..clip.modeling_clip import (
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionTransformer,
)
from ..deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2Model,
    DeepseekV2RMSNorm,
)
from ..llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from ..llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
)
from ..sam.modeling_sam import SamPatchEmbeddings, SamVisionAttention, SamVisionEncoder, SamVisionNeck


logger = logging.get_logger(__name__)

DEEPSEEK_OCR_DEFAULT_IMAGE_TOKEN_ID = 128815


class DeepseekOcrPatchEmbeddings(SamPatchEmbeddings):
    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height % self.patch_size[0] != 0 or width % self.patch_size[1] != 0:
            raise ValueError(
                "Input height and width must be divisible by the patch size "
                f"({self.patch_size[0]}x{self.patch_size[1]}). Received {height}x{width}."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class DeepseekOcrSamConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_sam_vision"
    base_config_key = "sam_config"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=None,
        output_channels=256,
        downsample_channels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = "deepseek_ocr"
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes if global_attn_indexes is not None else [2, 5, 8, 11]
        self.output_channels = output_channels
        self.downsample_channels = downsample_channels if downsample_channels is not None else [512, 1024]
        self.mlp_dim = int(hidden_size * 4.0)
        self.out_channels = output_channels


class DeepseekOcrCLIPTextConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_clip_text"
    base_config_key = "text_config"


class DeepseekOcrCLIPVisionConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_clip_vision"
    base_config_key = "clip_vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range


class DeepseekOcrCLIPConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_clip"
    sub_configs = {"text_config": DeepseekOcrCLIPTextConfig, "vision_config": DeepseekOcrCLIPVisionConfig}


class DeepseekOcrCLIPPreTrainedModel(PreTrainedModel):
    config_class = DeepseekOcrCLIPConfig

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DeepseekOcrVisionEmbeddings):
            num_positions = module.position_embedding.num_embeddings
            position_ids = torch.arange(num_positions, device=module.position_embedding.weight.device).unsqueeze(0)
            module.position_ids = position_ids


class DeepseekOcrProjectorConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_projector"
    base_config_key = "projector_config"

    def __init__(
        self,
        input_dim=2048,
        n_embed=1280,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_embed = n_embed


class DeepseekOcrVisionConfig(PreTrainedConfig):
    model_type = "deepseek_ocr_vision"
    base_config_key = "vision_config"
    sub_configs = {
        "sam_config": DeepseekOcrSamConfig,
        "clip_config": DeepseekOcrCLIPVisionConfig,
    }

    def __init__(self, sam_config=None, clip_config=None, **kwargs):
        super().__init__(**kwargs)

        if sam_config is None:
            self.sam_config = DeepseekOcrSamConfig()
        elif isinstance(sam_config, dict):
            self.sam_config = DeepseekOcrSamConfig(**sam_config)
        else:
            self.sam_config = sam_config

        if clip_config is None:
            self.clip_config = DeepseekOcrCLIPVisionConfig()
        elif isinstance(clip_config, dict):
            self.clip_config = DeepseekOcrCLIPVisionConfig(**clip_config)
        else:
            self.clip_config = clip_config

        # Aggregate commonly accessed vision attributes.
        self.image_size = self.sam_config.image_size
        self.patch_size = self.sam_config.patch_size


class DeepseekOcrTextConfig(DeepseekV2Config):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekOcrTextModel`]. It is used to instantiate a DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of DeepSeek-V2-Lite" [deepseek-ai/DeepSeek-V2-Lite"](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite").
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the DeepSeek model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`DeepseekOcrTextModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            The number of key-value heads used to implement Grouped Query Attention (GQA). If
            `num_key_value_heads=num_attention_heads`, the model will use Multi-Head Attention (MHA). If
            `num_key_value_heads=1`, the model will use Multi-Query Attention (MQA). Otherwise, GQA is used.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon value used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/value attentions (useful for inference optimization).
        pad_token_id (`int`, *optional*):
            Padding token ID.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning-of-sequence token ID.
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token ID.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value, and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to attention weights.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias term in the MLP layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in the shallow layers before switching to MoE layers.
        n_group (`int`, *optional*):
            Number of groups for routed experts.
        n_routed_experts (`int`, *optional*, defaults to 64):
            Number of routed experts (None indicates a dense model).
        n_shared_experts (`int`, *optional*, defaults to 2):
            Number of shared experts (None indicates a dense model).
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            The head dimension for QK projections when using RoPE.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts in MoE models.
        topk_group (`int`, *optional*):
            Number of selected groups per token for expert selection.
        topk_method (`str`, *optional*, defaults to `"greedy"`):
            The method used for selecting top-k experts in the routed gate mechanism.
        num_experts_per_tok (`int`, *optional*):
            The number of experts selected per token. If `None`, the model behaves as a dense Transformer.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE (Mixture of Experts) representations.

    ```python
    >>> from transformers import DeepseekOcrTextModel, DeepseekOcrTextConfig
    >>> # Initializing a DeepSeek-V2 style configuration
    >>> configuration = DeepseekOcrTextConfig()
    >>> # Accessing the model configuration
    >>> model = DeepseekOcrTextModel(configuration)
    >>> print(model.config)
    ```
    """

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts": "gather",
    }

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        mlp_bias: bool | None = False,
        first_k_dense_replace: int | None = 0,
        n_group: int | None = None,
        n_routed_experts: int | None = 64,
        n_shared_experts: int | None = 2,
        routed_scaling_factor: float | None = 1.0,
        topk_group: int | None = None,
        topk_method: str | None = "greedy",
        norm_topk_prob: bool | None = False,
        num_experts_per_tok: int | None = None,
        moe_intermediate_size: int | None = 1407,
        **kwargs,
    ):
        rope_theta = kwargs.get("rope_theta", 10_000.0)
        norm_topk_prob = kwargs.get("norm_topk_prob", norm_topk_prob)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        del self.kv_lora_rank
        del self.q_lora_rank
        del self.qk_nope_head_dim
        del self.qk_rope_head_dim
        del self.v_head_dim
        del self.head_dim
        self.norm_topk_prob = norm_topk_prob
        self.rope_theta = getattr(self, "rope_theta", rope_theta)
        self.rope_parameters = getattr(self, "rope_parameters", None) or {}
        self.standardize_rope_params()
        self.validate_rope()


class DeepseekOcrConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekOcrForConditionalGeneration`]. It is used to instantiate a
    DeepseekOCR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DeepseekOCR
    [deepseek-ai/deepseek-ocr](https://huggingface.co/deepseek-ai/deepseek-ocr) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `DeepseekV2Config`):
            The config object or dictionary of the text backbone (DeepSeek-V2).
        vision_config (`DeepseekOcrVisionConfig` or `dict`, *optional*):
            The config object or dictionary of the vision encoders (SAM and CLIP).
        projector_config (`DeepseekOcrProjectorConfig` or `dict`, *optional*):
            The config object or dictionary of the projector that maps vision features to text embedding space.
        image_token_id (`int`, *optional*, defaults to 128815):
            The id of the image token in the model's token vocabulary.

    Example:

    ```python
    >>> from transformers import DeepseekOcrConfig, DeepseekOcrForConditionalGeneration

    >>> # Initializing a DeepseekOCR configuration
    >>> configuration = DeepseekOcrConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DeepseekOcrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_ocr"
    processor_class = "DeepseekOcrProcessor"
    sub_configs = {
        "text_config": DeepseekOcrTextConfig,
        "vision_config": DeepseekOcrVisionConfig,
        "projector_config": DeepseekOcrProjectorConfig,
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        **kwargs,
    ):
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision
        kwargs["token"] = token

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]

        if config_dict.get("model_type") == "deepseek_vl_v2":
            config_dict["model_type"] = cls.model_type

        return cls.from_dict(config_dict, **kwargs)

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projector_config=None,
        image_token_id=DEEPSEEK_OCR_DEFAULT_IMAGE_TOKEN_ID,
        **kwargs,
    ):
        kwargs.pop("auto_map", None)
        language_config = kwargs.pop("language_config", None)
        original_model_type = kwargs.pop("model_type", None)
        image_token_index = kwargs.pop("image_token_index", None)
        if text_config is None and language_config is not None:
            text_config = language_config

        if image_token_index is not None:
            image_token_id = image_token_index
        self.image_token_id = image_token_id

        if text_config is None:
            self.text_config = DeepseekOcrTextConfig(
                hidden_size=1280,
                intermediate_size=6848,
                num_hidden_layers=12,
                num_attention_heads=10,
                num_key_value_heads=10,
                moe_intermediate_size=896,
                n_routed_experts=64,
                n_shared_experts=2,
                num_experts_per_tok=6,
                first_k_dense_replace=1,
                vocab_size=129280,
                max_position_embeddings=8192,
            )
        elif isinstance(text_config, dict):
            text_config.pop("auto_map", None)
            self.text_config = DeepseekOcrTextConfig(**text_config)
        else:
            self.text_config = text_config
        if getattr(self.text_config, "image_token_id", None) is None:
            self.text_config.image_token_id = self.image_token_id

        if vision_config is None:
            self.vision_config = DeepseekOcrVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config.pop("auto_map", None)
            self.vision_config = DeepseekOcrVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if projector_config is None:
            self.projector_config = DeepseekOcrProjectorConfig()
        elif isinstance(projector_config, dict):
            projector_config.pop("auto_map", None)
            self.projector_config = DeepseekOcrProjectorConfig(**projector_config)
        else:
            self.projector_config = projector_config

        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        self.ignore_index = kwargs.pop("ignore_index", -100)

        pad_token_id = kwargs.pop("pad_token_id", getattr(self.text_config, "pad_token_id", None))
        if pad_token_id is None:
            pad_token_id = getattr(self.text_config, "bos_token_id", None)
        bos_token_id = kwargs.pop("bos_token_id", getattr(self.text_config, "bos_token_id", None))
        eos_token_id = kwargs.pop("eos_token_id", getattr(self.text_config, "eos_token_id", None))
        if getattr(self.text_config, "pad_token_id", None) is None:
            self.text_config.pad_token_id = pad_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.original_model_type = original_model_type
        self.model_type = "deepseek_ocr"


class DeepseekOcrPreTrainedModel(PreTrainedModel):
    config_class = DeepseekOcrConfig
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}


class DeepseekOcrProjector(PreTrainedModel):
    """
    Projector that maps concatenated SAM + CLIP features to language model space.
    """

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.Linear(config.input_dim, config.n_embed)
        self.post_init()

    def forward(self, x, **kwargs):
        return self.layers(x)


class DeepseekOcrVisionAttention(SamVisionAttention):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        self.config = config


class DeepseekOcrSamVisionNeck(SamVisionNeck):
    def __init__(self, config):
        super().__init__(config)


class DeepseekOcrModelOutputWithPast(LlavaNextModelOutputWithPast):
    pass


class DeepseekOcrCausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


class DeepseekOcrSamVisionEncoder(SamVisionEncoder):
    """
    SAM ViT-B vision encoder with additional neck layers for Deepseek OCR.
    Wraps the SAM vision encoder and adds downsampling convolutions.
    """

    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        out_channels = config.out_channels
        downsample_channels = config.downsample_channels

        self.patch_embed = DeepseekOcrPatchEmbeddings(config)

        self.net_2 = nn.Conv2d(out_channels, downsample_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            downsample_channels[0], downsample_channels[1], kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed
            if pos_embed.shape[1:3] != hidden_states.shape[1:3]:
                pos_embed = nn.functional.interpolate(
                    pos_embed.permute(0, 3, 1, 2),
                    size=hidden_states.shape[1:3],
                    mode="bicubic",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            hidden_states = hidden_states + pos_embed
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        hidden_states = self.neck(hidden_states)
        hidden_states = self.net_2(hidden_states)
        hidden_states = self.net_3(hidden_states)

        return hidden_states


class DeepseekOcrVisionEmbeddings(CLIPVisionEmbeddings):
    def forward(self, pixel_values, patch_embeds=None, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape

        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)
        if patch_embeds.dim() == 4:
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        else:
            patch_embeds = patch_embeds
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        num_positions = self.position_embedding.num_embeddings
        position_ids = self.position_ids
        if position_ids.shape[-1] != num_positions or position_ids.min() < 0 or position_ids.max() >= num_positions:
            position_ids = torch.arange(num_positions, device=self.position_embedding.weight.device).unsqueeze(0)
            self.position_ids = position_ids
        position_embeddings = self.position_embedding(position_ids)
        if position_embeddings.shape[1] != embeddings.shape[1]:
            class_pos_embed = position_embeddings[:, :1]
            patch_pos_embed = position_embeddings[:, 1:]
            src_size = int(math.sqrt(patch_pos_embed.shape[1]))
            target_tokens = embeddings.shape[1] - 1
            target_size = int(math.sqrt(target_tokens))
            patch_pos_embed = patch_pos_embed.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
            patch_pos_embed = patch_pos_embed.to(torch.float32)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed,
                size=(target_size, target_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, target_tokens, -1)
            position_embeddings = torch.cat([class_pos_embed, patch_pos_embed.to(position_embeddings.dtype)], dim=1)
        embeddings = embeddings + position_embeddings
        return embeddings


class DeepseekOcrEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            **kwargs,
        )


class DeepseekOcrCLIPEncoder(CLIPEncoder):
    def __init__(self, config: DeepseekOcrCLIPVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([DeepseekOcrEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = False,  # TODO get rid of this when we're done with the fwd pass
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds

        all_hidden_states = [] if output_hidden_states else None

        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                **kwargs,
            )

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            all_hidden_states = tuple(all_hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class DeepseekOcrCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.hidden_size
        self.embeddings = DeepseekOcrVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = DeepseekOcrCLIPEncoder(config)
        del self.post_layernorm

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        patch_embeds = kwargs.pop("patch_embeds", None)
        hidden_states = self.embeddings(
            pixel_values,
            patch_embeds,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class DeepseekOcrCLIPVisionModel(CLIPVisionModel):
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    config_class = DeepseekOcrCLIPVisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_model = DeepseekOcrCLIPVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @check_model_inputs(tie_last_hidden_states=False)
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, DeepseekOcrCLIPVisionModel

        >>> model = DeepseekOcrCLIPVisionModel.from_pretrained("openai/deepseek_ocr_c_l_i_p-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/deepseek_ocr_c_l_i_p-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        patch_embeds = kwargs.pop("patch_embeds", None)
        return self.vision_model(
            pixel_values=pixel_values,
            patch_embeds=patch_embeds,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class DeepseekOcrTextMLP(nn.Module):
    def __init__(self, config: DeepseekOcrTextConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepseekOcrTextExperts(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        for _ in range(config.n_routed_experts):
            self.append(DeepseekOcrTextMLP(config, intermediate_size=config.moe_intermediate_size))

    def forward(self, hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        tokens_per_expert = torch.bincount(topk_idx.view(-1), minlength=self.num_experts)

        flat_indices = topk_idx.view(-1)
        sorted_positions = flat_indices.argsort()
        original_token_indices = sorted_positions // self.top_k

        sorted_tokens = hidden_states[original_token_indices]
        combined_results = torch.empty_like(sorted_tokens)

        boundaries = torch.cumsum(tokens_per_expert, dim=0)
        start_indices = torch.cat((torch.tensor([0], device=boundaries.device), boundaries[:-1]))

        for i in range(self.num_experts):
            count = tokens_per_expert[i].item()
            if count == 0:
                continue

            start = start_indices[i].item()
            end = boundaries[i].item()

            combined_results[start:end] = self[i](sorted_tokens[start:end])

        dispatch_buffer = torch.empty_like(combined_results)
        dispatch_buffer.scatter_(0, sorted_positions.unsqueeze(-1).expand_as(combined_results), combined_results)

        dispatch_buffer = dispatch_buffer.view(topk_idx.shape[0], self.top_k, -1)
        weighted = dispatch_buffer.to(topk_weight.dtype) * topk_weight.unsqueeze(-1)

        return weighted.sum(dim=1).to(hidden_states.dtype)


class DeepseekOcrTextMoe(nn.Module):
    def __init__(self, config: DeepseekOcrTextConfig):
        super().__init__()
        self.config = config
        self.experts = DeepseekOcrTextExperts(config)
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekOcrTextMLP(config=config, intermediate_size=intermediate_size)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.top_k = config.num_experts_per_tok
        self.topk_group = config.topk_group
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)

    def route_tokens_to_experts(self, scores):
        if self.top_k is None or self.top_k <= 0:
            raise ValueError("`num_experts_per_tok` must be a positive integer for MoE routing.")

        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            if self.num_group is None or self.topk_group is None:
                raise ValueError("`n_group` and `topk_group` must be provided for group_limited_greedy routing.")
            group_scores = scores.view(scores.shape[0], self.num_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(scores.shape[0], self.num_group, scores.shape[-1] // self.num_group)
                .reshape(scores.shape[0], -1)
            )
            masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)
        else:
            raise ValueError(f"Unsupported topk routing method: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            topk_weight = topk_weight / denominator

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = nn.functional.linear(hidden_states.type(torch.float32), self.gate.weight.type(torch.float32))
        router_scores = router_logits.softmax(dim=-1, dtype=torch.float32)
        router_scores_flat = router_scores.view(-1, router_scores.shape[-1])
        topk_indices, topk_weights = self.route_tokens_to_experts(router_scores_flat)
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        expert_output = self.experts(hidden_states_flat, topk_indices, topk_weights)
        hidden_states = expert_output.view(*orig_shape)

        if hasattr(self, "shared_experts"):
            hidden_states = hidden_states + self.shared_experts(residuals)

        return hidden_states


class DeepseekOcrTextAttention(LlamaAttention):
    pass


class DeepseekOcrTextDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = DeepseekOcrTextAttention(config, layer_idx)
        self.mlp = (
            DeepseekOcrTextMoe(config) if layer_idx >= config.first_k_dense_replace else DeepseekOcrTextMLP(config)
        )


class DeepseekOcrTextRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DeepseekOcrTextRMSNorm(DeepseekV2RMSNorm):
    pass


class DeepseekOcrTextPreTrainedModel(PreTrainedModel):
    config: DeepseekOcrTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DeepseekOcrTextExperts):
            for expert in module:
                init.normal_(expert.gate_proj.weight, mean=0.0, std=self.config.initializer_range)
                init.normal_(expert.up_proj.weight, mean=0.0, std=self.config.initializer_range)
                init.normal_(expert.down_proj.weight, mean=0.0, std=self.config.initializer_range)


class DeepseekOcrTextModel(DeepseekV2Model):
    config: DeepseekOcrTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekOcrTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DeepseekOcrTextDecoderLayer,
        "attentions": DeepseekOcrTextAttention,
    }

    def __init__(self, config):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [DeepseekOcrTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekOcrTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekOcrTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        for module in self.layers:
            if isinstance(module.mlp, DeepseekOcrTextMoe):
                module.mlp.gate.weight.data.normal_(mean=0.0, std=config.initializer_range)


@auto_docstring(
    custom_intro="""
    The Deepseek-OCR model which consists of two vision backbones and a language model without language modeling head.
    """
)
class DeepseekOcrModel(LlavaNextModel):
    _checkpoint_conversion_mapping = {}
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, config: DeepseekOcrConfig):
        super().__init__(config)
        del self.vision_tower
        del self.multi_modal_projector

        self.sam_model = DeepseekOcrSamVisionEncoder._from_config(config.vision_config.sam_config)
        self.clip_model = DeepseekOcrCLIPVisionModel._from_config(config.vision_config.clip_config)

        self.multi_modal_projector = DeepseekOcrProjector._from_config(config.projector_config)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = DeepseekOcrTextModel._from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        embed_std = 1 / math.sqrt(config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.hidden_size) * embed_std)
        self.view_seperator = nn.Parameter(
            torch.randn(config.hidden_size) * embed_std
        )  # TODO the typo is in the checkpoint

        self.post_init()

    def pack_image_features(
        self,
        image_features,
        image_newline=None,
        image_spatial_crops=None,
    ):
        """
        Packs local-crop + global grids into the same newline/separator layout LlavaNext expects.

        Contrary to LlavaNext, DeepSeek-OCR receives a list of feature
        groups where each entry already separates local crops and the global 1024 view. We therefore:
          * reshape each local grid back to (height_crop_num × crop_grid, width_crop_num × crop_grid) and append a
            newline embedding per row,
          * reshape the global feature grid and append its newline,
          * finally, append the learned view separator that delimits image blocks.
        """
        newline_token = image_newline if image_newline is not None else self.image_newline
        new_image_features = []

        for image_idx, features in enumerate(image_features):
            crop_shape = None
            if image_spatial_crops is not None:
                crop_shape = image_spatial_crops[image_idx]
                if isinstance(crop_shape, torch.Tensor):
                    crop_shape = crop_shape.tolist()

            width_crop_num = int(crop_shape[0]) if crop_shape is not None else 1
            height_crop_num = int(crop_shape[1]) if crop_shape is not None else 1
            has_local_crops = width_crop_num > 1 or height_crop_num > 1

            if isinstance(features, list):
                patch_features = features
            else:
                patch_features = [features[i] for i in range(features.shape[0])]

            if has_local_crops and len(patch_features) >= width_crop_num * height_crop_num + 1:
                valid_patch_count = width_crop_num * height_crop_num + 1
            else:
                valid_patch_count = 1 if len(patch_features) > 0 else 0
                has_local_crops = False

            patch_features = patch_features[:valid_patch_count]
            if len(patch_features) == 0:
                new_image_features.append(torch.empty(0, self.config.hidden_size, device=newline_token.device))
                continue

            global_feature = patch_features[-1]
            local_feature_list = patch_features[:-1] if has_local_crops else []
            processed_parts = []

            if local_feature_list:
                local_features = torch.stack(local_feature_list, dim=0)
                local_tokens = local_features.shape[1]
                local_grid = int(math.isqrt(local_tokens))

                if local_grid * local_grid == local_tokens:
                    local_features = local_features.view(
                        height_crop_num,
                        width_crop_num,
                        local_grid,
                        local_grid,
                        -1,
                    )
                    local_features = local_features.permute(0, 2, 1, 3, 4).contiguous()
                    local_features = local_features.view(
                        height_crop_num * local_grid,
                        width_crop_num * local_grid,
                        -1,
                    )
                    newline = (
                        newline_token.unsqueeze(0)
                        .unsqueeze(0)
                        .to(local_features.device, dtype=local_features.dtype)
                        .expand(local_features.shape[0], 1, -1)
                    )
                    local_features = torch.cat((local_features, newline), dim=1)
                    local_features = local_features.view(-1, local_features.shape[-1])
                else:
                    local_features = local_features.view(-1, local_features.shape[-1])
                    newline = newline_token.unsqueeze(0).to(local_features.device, dtype=local_features.dtype)
                    local_features = torch.cat((local_features, newline), dim=0)

                processed_parts.append(local_features)

            global_tokens = global_feature.shape[0]
            global_grid = int(math.isqrt(global_tokens))
            if global_grid * global_grid == global_tokens:
                global_features = global_feature.view(global_grid, global_grid, -1)
                newline = (
                    newline_token.unsqueeze(0)
                    .unsqueeze(0)
                    .to(global_features.device, dtype=global_features.dtype)
                    .expand(global_grid, 1, -1)
                )
                global_features = torch.cat((global_features, newline), dim=1)
                global_features = global_features.view(-1, global_features.shape[-1])
            else:
                global_features = torch.cat(
                    (
                        global_feature,
                        newline_token.unsqueeze(0).to(global_feature.device, dtype=global_feature.dtype),
                    ),
                    dim=0,
                )

            processed_parts.append(global_features)

            combined = torch.cat(processed_parts, dim=0)
            new_image_features.append(combined)

        return new_image_features

    def _project_image_patches(
        self,
        pixel_batch: torch.Tensor,
    ) -> list[torch.Tensor]:
        if pixel_batch.dim() == 3:
            pixel_batch = pixel_batch.unsqueeze(0)

        sam_features = self.sam_model(pixel_batch)
        sam_seq = sam_features.flatten(2).permute(0, 2, 1)

        clip_out = self.clip_model(
            pixel_values=pixel_batch,
            patch_embeds=sam_features,
            interpolate_pos_encoding=True,
        )

        clip_seq = clip_out.last_hidden_state

        clip_seq = clip_seq[:, 1:]

        fused = torch.cat([clip_seq, sam_seq], dim=-1)
        proj = self.multi_modal_projector(fused)
        return [proj[i] for i in range(proj.shape[0])]

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_crops: torch.LongTensor | None = None,
        image_sizes: torch.Tensor | None = None,
        image_spatial_crops: torch.Tensor | None = None,
    ):
        """Wrapper for the two image feature stacks used in deepseek OCR."""
        image_feature_groups: list[list[torch.Tensor]] = []

        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        if num_local_crops is None:
            if image_spatial_crops is not None:
                num_local_crops = (image_spatial_crops[:, 0] * image_spatial_crops[:, 1]).to(dtype=torch.long)
            else:
                num_local_crops = torch.zeros(batch_size, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            patch_features = []
            local_count = num_local_crops[batch_idx].item()

            if local_count > 0 and pixel_values_local is not None and pixel_values_local.shape[1] >= local_count:
                local_pixels = pixel_values_local[batch_idx, :local_count]
                patch_features.extend(self._project_image_patches(local_pixels))

            global_pixels = pixel_values[batch_idx]
            patch_features.extend(self._project_image_patches(global_pixels))

            image_feature_groups.append(patch_features)

        packed_features = self.pack_image_features(
            image_features=image_feature_groups,
            image_newline=self.image_newline,
            image_spatial_crops=image_spatial_crops,
        )

        separator = self.view_seperator
        for i, features in enumerate(packed_features):
            view_sep = separator.unsqueeze(0).to(features.device, dtype=features.dtype)
            packed_features[i] = torch.cat([features, view_sep], dim=0)

        return packed_features

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        num_local_crops: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, 1, num_channels, height, width)`):
            Global view (1024x1024) consumed by SAM + CLIP. This is injected wherever `<image>` placeholders appear.
        pixel_values_local (`torch.FloatTensor` of shape `(batch_size, max_num_crops, num_channels, crop_height, crop_width)`):
            Optional high-resolution (640x640) crops. When provided, they are stitched into the packed feature grid
            ahead of the global features.
        num_local_crops (`torch.LongTensor` of shape `(batch_size,)`):
            Number of valid local crops for each image in the batch.
        """
        image_spatial_crop = kwargs.pop("image_spatial_crop", None)
        pixel_values_local = kwargs.pop("pixel_values_local", pixel_values_local)
        num_local_crops = kwargs.pop("num_local_crops", num_local_crops)

        if image_sizes is None and image_spatial_crop is not None:
            image_sizes = image_spatial_crop

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                image_spatial_crops=image_spatial_crop,
                pixel_values_local=pixel_values_local,
                num_local_crops=num_local_crops,
            )
            image_hidden_states = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_hidden_states
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return DeepseekOcrModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The Deepseek-OCR model which consists of two vision backbones and a deepseek language model with a decoding head.
    """
)
class DeepseekOcrForConditionalGeneration(LlavaNextForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {}
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekOcrModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        num_local_crops: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | DeepseekOcrCausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, 1, num_channels, height, width)`):
            Global view of images downsampled to 1024x1024 for processing by both SAM and CLIP encoders.
        pixel_values_local (`torch.FloatTensor` of shape `(batch_size, max_num_crops, num_channels, crop_height, crop_width)`):
            High-resolution local crops (640x640) extracted from images for detailed OCR processing.
        num_local_crops (`torch.LongTensor` of shape `(batch_size,)`):
            Number of valid local crops for each image in the batch.
        """
        image_spatial_crop = kwargs.pop("image_spatial_crop", None)
        if image_sizes is None and image_spatial_crop is not None:
            image_sizes = image_spatial_crop

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_local=pixel_values_local,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            num_local_crops=num_local_crops,
            cache_position=cache_position,
            image_spatial_crop=image_spatial_crop,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return DeepseekOcrCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_local=None,
        num_local_crops=None,
        image_attention_mask=None,
        image_spatial_crop=None,
        num_img_tokens=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        current_cache_position = model_inputs.get("cache_position", cache_position)
        if current_cache_position is None or current_cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_local"] = pixel_values_local
            model_inputs["num_local_crops"] = num_local_crops
            model_inputs["image_attention_mask"] = image_attention_mask
            model_inputs["image_spatial_crop"] = image_spatial_crop
            model_inputs["num_img_tokens"] = num_img_tokens
            model_inputs["image_sizes"] = image_sizes

        return model_inputs


__all__ = [
    "DeepseekOcrConfig",
    "DeepseekOcrVisionConfig",
    "DeepseekOcrSamConfig",
    "DeepseekOcrCLIPVisionConfig",
    "DeepseekOcrCLIPPreTrainedModel",
    "DeepseekOcrProjectorConfig",
    "DeepseekOcrModelOutputWithPast",
    "DeepseekOcrCausalLMOutputWithPast",
    "DeepseekOcrTextModel",
    "DeepseekOcrTextPreTrainedModel",
    "DeepseekOcrModel",
    "DeepseekOcrForConditionalGeneration",
    "DeepseekOcrPreTrainedModel",
    "DeepseekOcrProjector",
    "DeepseekOcrSamVisionEncoder",
    "DeepseekOcrCLIPVisionModel",
]

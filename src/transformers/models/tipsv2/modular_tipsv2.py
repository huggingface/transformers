# Copyright 2026 Google LLC and the HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import PILImageResampling
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_sentencepiece import SentencePieceBackend
from ...utils import TransformersKwargs, auto_docstring, logging, torch_int
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..clip.modeling_clip import CLIPOutput, CLIPTextEmbeddings, _get_vector_norm, image_text_contrastive_loss
from ..dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersEmbeddings,
    Dinov2WithRegistersEncoder,
    Dinov2WithRegistersModel,
    Dinov2WithRegistersPreTrainedModel,
)
from ..siglip2.configuration_siglip2 import Siglip2TextConfig
from ..speech_to_text.modeling_speech_to_text import Speech2TextSinusoidalPositionalEmbedding


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


@requires(backends=("sentencepiece",))
class Tipsv2Tokenizer(SentencePieceBackend):
    """Tipsv2 tokenizer based on SentencePiece.

    The original Tipsv2 text pipeline lowercases inputs, does not add BOS/EOS tokens, pads to a maximum length of 64,
    and uses padding token id 0.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token: str | None = "<unk>",
        pad_token: str | None = "<pad>",
        bos_token: str | None = None,
        eos_token: str | None = None,
        model_max_length: int = 64,
        do_lower_case: bool = True,
        token_type_ids_pattern: str = "all_zeros",
        **kwargs,
    ) -> None:
        self.do_lower_case = do_lower_case

        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            model_max_length=model_max_length,
            do_lower_case=do_lower_case,
            token_type_ids_pattern=token_type_ids_pattern,
            **kwargs,
        )

        if self.pad_token_id != 0:
            raise ValueError(f"Expected the SentencePiece padding token to have id 0, but got {self.pad_token_id}.")

    def _tokenize(self, text, **kwargs):
        if self.do_lower_case:
            text = text.lower()
        return self.sp_model.encode(text, out_type=str)


@auto_docstring
class Tipsv2ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    do_normalize = False
    do_convert_rgb = True


class Tipsv2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "truncation": True,
            "max_length": 64,
        },
    }


@auto_docstring
class Tipsv2Processor(ProcessorMixin):
    valid_processor_kwargs = Tipsv2ProcessorKwargs

    def __init__(self, image_processor: Tipsv2ImageProcessor | None = None, tokenizer: Tipsv2Tokenizer | None = None):
        super().__init__(image_processor, tokenizer)


@auto_docstring(checkpoint="google/tipsv2-b14")
@strict
class Tipsv2VisionConfig(Dinov2WithRegistersConfig):
    r"""
    layerscale_value (`float`, *optional*, defaults to `1.0`):
        Initial value to use for layer scale.
    use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feedforward neural network.
    num_register_tokens (`int`, *optional*, defaults to `1`):
        Number of register tokens to use.
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        DINOv2-with-registers compatibility flag serialized for config parity. It is not used by
        [`Tipsv2VisionModel`].
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        DINOv2-with-registers compatibility flag serialized for config parity. It is not used by
        [`Tipsv2VisionModel`].
    interpolate_antialias (`bool`, *optional*, defaults to `True`):
        Whether to use antialiasing when interpolating vision position embeddings.
    interpolate_offset (`float`, *optional*, defaults to `0.0`):
        Offset to use when resizing vision position embeddings.

    Example:

    ```python
    >>> from transformers import Tipsv2VisionConfig, Tipsv2VisionModel

    >>> configuration = Tipsv2VisionConfig()
    >>> model = Tipsv2VisionModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "tipsv2_vision_model"
    base_config_key = "vision_config"

    image_size: int | list[int] | tuple[int, int] = 448
    mlp_ratio: int | float = 4  # float required for so400m14 checkpoint
    num_register_tokens: int = 1
    interpolate_antialias: bool = True
    interpolate_offset: float = 0.0


@auto_docstring(checkpoint="google/tipsv2-b14")
@strict
class Tipsv2TextConfig(Siglip2TextConfig):
    r"""
    scale_sqrt_depth (`bool`, *optional*, defaults to `True`):
        Whether to scale token embeddings by `sqrt(hidden_size)` before adding sinusoidal position embeddings.
    pooling_epsilon (`float`, *optional*, defaults to `1e-8`):
        Epsilon added to the valid token count when computing masked mean pooling.

    Example:

    ```python
    >>> from transformers import Tipsv2TextConfig, Tipsv2TextModel

    >>> configuration = Tipsv2TextConfig()
    >>> model = Tipsv2TextModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "tipsv2_text_model"
    base_config_key = "text_config"

    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    scale_sqrt_depth: bool = True
    pooling_epsilon: float = 1e-8

    projection_size = AttributeError()

    def __post_init__(self, **kwargs):
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


@auto_docstring(checkpoint="google/tipsv2-b14")
@strict
class Tipsv2Config(PreTrainedConfig):
    r"""
    text_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`Tipsv2TextConfig`].
    vision_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`Tipsv2VisionConfig`].
    temperature (`float`, *optional*, defaults to `0.01`):
        Temperature used to scale cosine-similarity logits in [`Tipsv2Model`].

    Example:

    ```python
    >>> from transformers import Tipsv2Config, Tipsv2Model

    >>> configuration = Tipsv2Config()
    >>> model = Tipsv2Model(configuration)
    >>> configuration = model.config

    >>> from transformers import Tipsv2TextConfig, Tipsv2VisionConfig

    >>> text_config = Tipsv2TextConfig()
    >>> vision_config = Tipsv2VisionConfig()
    >>> config = Tipsv2Config(text_config=text_config, vision_config=vision_config)
    ```"""

    model_type = "tipsv2"
    sub_configs = {"text_config": Tipsv2TextConfig, "vision_config": Tipsv2VisionConfig}

    text_config: dict | Tipsv2TextConfig | None = None
    vision_config: dict | Tipsv2VisionConfig | None = None
    temperature: float = 0.01

    def __post_init__(self, **kwargs):
        text_config_kwargs = {}
        if isinstance(self.text_config, dict):
            text_config_kwargs.update(self.text_config)
        elif isinstance(self.text_config, Tipsv2TextConfig):
            text_config_kwargs.update(self.text_config.to_dict())

        vision_config_kwargs = {}
        if isinstance(self.vision_config, dict):
            vision_config_kwargs.update(self.vision_config)
        elif isinstance(self.vision_config, Tipsv2VisionConfig):
            vision_config_kwargs.update(self.vision_config.to_dict())

        # Backwards compatibility: convert flat config (old format) to nested config.
        # Old configs stored all parameters at the top level with different key names.
        FLAT_TO_VISION_CONFIG_KEYS = {
            "embed_dim": "hidden_size",
            "img_size": "image_size",
            "init_values": "layerscale_value",
            "num_register_tokens": "num_register_tokens",
            "patch_size": "patch_size",
        }
        FLAT_TO_TEXT_CONFIG_KEYS = {
            "text_hidden_size": "hidden_size",
            "text_mlp_dim": "intermediate_size",
            "max_len": "max_position_embeddings",
            "text_num_heads": "num_attention_heads",
            "text_num_layers": "num_hidden_layers",
            "vocab_size": "vocab_size",
        }
        VISION_FUNCTION_TO_KWARGS = {
            "vit_base": {"num_hidden_layers": 12, "num_attention_heads": 12},
            "vit_large": {"num_hidden_layers": 24, "num_attention_heads": 16},
            "vit_so400m": {"num_hidden_layers": 27, "num_attention_heads": 16},
            "vit_giant2": {"num_hidden_layers": 40, "num_attention_heads": 24},
        }
        FLAT_KEYS = set(FLAT_TO_VISION_CONFIG_KEYS) | set(FLAT_TO_TEXT_CONFIG_KEYS) | {"ffn_layer", "vision_fn"}
        if any(key in kwargs for key in FLAT_KEYS):
            vision_kwargs = {new: kwargs.pop(old) for old, new in FLAT_TO_VISION_CONFIG_KEYS.items() if old in kwargs}
            if "ffn_layer" in kwargs:
                vision_kwargs["use_swiglu_ffn"] = kwargs.pop("ffn_layer") == "swiglu"
            if "vision_fn" in kwargs:
                vision_kwargs.update(VISION_FUNCTION_TO_KWARGS.get(kwargs.pop("vision_fn"), {}))

            text_kwargs = {new: kwargs.pop(old) for old, new in FLAT_TO_TEXT_CONFIG_KEYS.items() if old in kwargs}

            # The vision MLP intermediate size equals text_mlp_dim; derive mlp_ratio for non-integer cases (e.g. so400m).
            if "intermediate_size" in text_kwargs and "hidden_size" in vision_kwargs:
                vision_kwargs.setdefault("mlp_ratio", text_kwargs["intermediate_size"] / vision_kwargs["hidden_size"])

            for key, value in vision_kwargs.items():
                if key in vision_config_kwargs and value != vision_config_kwargs[key]:
                    logger.info(
                        f"`{key}` is found in both the main config and `vision_config` but with different "
                        f"values. The value from the main config (`{value}`) will be used instead."
                    )
            vision_config_kwargs.update(vision_kwargs)

            for key, value in text_kwargs.items():
                if key in text_config_kwargs and value != text_config_kwargs[key]:
                    logger.info(
                        f"`{key}` is found in both the main config and `text_config` but with different "
                        f"values. The value from the main config (`{value}`) will be used instead."
                    )
            text_config_kwargs.update(text_kwargs)

        self.text_config = Tipsv2TextConfig(**text_config_kwargs)
        self.vision_config = Tipsv2VisionConfig(**vision_config_kwargs)
        super().__post_init__(**kwargs)


class Tipsv2Output(CLIPOutput):
    pass


class Tipsv2VisionEmbeddings(Dinov2WithRegistersEmbeddings):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        embeddings_dtype = embeddings.dtype
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings.to(dtype=embeddings_dtype)

        position_embeddings = self.position_embeddings.float()
        class_pos_embed = position_embeddings[:, 0]
        patch_pos_embed = position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        patch_size = self.config.patch_size
        if isinstance(patch_size, int):
            patch_height = patch_width = patch_size
        else:
            patch_height, patch_width = patch_size
        height = height // patch_height
        width = width // patch_width

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        interpolate_kwargs = {}
        if self.config.interpolate_offset:
            scale_height = float(height + self.config.interpolate_offset) / sqrt_num_positions
            scale_width = float(width + self.config.interpolate_offset) / sqrt_num_positions
            interpolate_kwargs["scale_factor"] = (scale_height, scale_width)
        else:
            interpolate_kwargs["size"] = (torch_int(height), torch_int(width))

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            mode="bilinear",
            antialias=self.config.interpolate_antialias,
            **interpolate_kwargs,
        )

        if not torch.jit.is_tracing():
            if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
                raise ValueError("Width or height does not match with the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(dtype=embeddings_dtype)


class Tipsv2VisionPreTrainedModel(Dinov2WithRegistersPreTrainedModel):
    config: Tipsv2VisionConfig
    base_model_prefix = "vision_model"


class Tipsv2VisionEncoder(Dinov2WithRegistersEncoder):
    pass


class Tipsv2VisionModel(Dinov2WithRegistersModel):
    pass


# Identical to CLIP's implementation but couldn't import it because the vision model
# already requires eager_attention_forward from DINOv2 which results in modular
# conflicts.
def text_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Tipsv2SinusoidalPositionalEmbedding(Speech2TextSinusoidalPositionalEmbedding):
    def __init__(self, config: Tipsv2TextConfig):
        nn.Module.__init__(self)
        self.make_weights(
            num_embeddings=config.max_position_embeddings, embedding_dim=config.hidden_size, padding_idx=None
        )

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.weights[position_ids]

    def create_position_ids_from_input_ids(self):
        raise AttributeError("Not needed")


class Tipsv2TextEmbeddings(CLIPTextEmbeddings):
    def __init__(self, config: Tipsv2TextConfig):
        super().__init__(config)
        self.position_embedding = Tipsv2SinusoidalPositionalEmbedding(config)
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_sqrt_depth else 1.0

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_ids.shape[-1]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        inputs_embeds = inputs_embeds * self.embed_scale
        position_embeddings = self.position_embedding(position_ids).to(dtype=inputs_embeds.dtype)
        return inputs_embeds + position_embeddings


class Tipsv2TextAttention(nn.Module):
    def __init__(self, config: Tipsv2TextConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, text_eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Tipsv2TextMLP(nn.Module):
    def __init__(self, config: Tipsv2TextConfig):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = hidden_states * valid_mask[..., None]
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states * valid_mask[..., None]
        return hidden_states


class Tipsv2TextEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Tipsv2TextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Tipsv2TextAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Tipsv2TextMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        valid_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, valid_mask)
        hidden_states = residual + hidden_states
        return hidden_states


class Tipsv2TextEncoder(nn.Module):
    def __init__(self, config: Tipsv2TextConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Tipsv2TextEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        if valid_mask is None:
            valid_mask = hidden_states.new_ones(hidden_states.shape[:-1])

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                valid_mask=valid_mask,
                **kwargs,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class Tipsv2TextPreTrainedModel(PreTrainedModel):
    config: Tipsv2TextConfig
    base_model_prefix = "text_model"
    main_input_name = "input_ids"
    input_modalities = ("text",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["Tipsv2TextEmbeddings", "Tipsv2TextEncoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Tipsv2TextEncoderLayer,
        "attentions": Tipsv2TextAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Tipsv2TextEmbeddings):
            init.normal_(module.token_embedding.weight, mean=0.0, std=self.config.initializer_range)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, Tipsv2SinusoidalPositionalEmbedding):
            num_embeddings, embedding_dim = module.weights.shape
            embedding_weights = module.get_embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                padding_idx=None,
            )
            init.copy_(module.weights, embedding_weights)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)


@auto_docstring(
    custom_intro="""
    The TIPSv2 text tower without any projection head on top.
    """
)
class Tipsv2TextModel(Tipsv2TextPreTrainedModel):
    _input_embed_layer = "token_embedding"

    def __init__(self, config: Tipsv2TextConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = Tipsv2TextEmbeddings(config)
        self.encoder = Tipsv2TextEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.embeddings.token_embedding = value

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Official TIPSv2-style padding mask where `1` marks padding tokens and `0` marks valid tokens. Mutually
            exclusive with the Hugging Face-style `attention_mask`, where `1` marks valid tokens.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if attention_mask is not None and padding_mask is not None:
            raise ValueError("You cannot specify both attention_mask and padding_mask")

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            input_shape = inputs_embeds.size()[:-1]
            inputs_embeds = inputs_embeds.view(-1, input_shape[-1], inputs_embeds.shape[-1])

        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
        elif padding_mask is not None:
            attention_mask = padding_mask.view(-1, input_shape[-1]) == 0
        elif input_ids is not None and self.config.pad_token_id is not None:
            attention_mask = input_ids != self.config.pad_token_id
        else:
            attention_mask = torch.ones(input_shape, dtype=torch.bool, device=self.device).view(-1, input_shape[-1])

        if attention_mask.dim() != 2:
            raise ValueError("`attention_mask` must be a 2D mask with 1 for valid tokens and 0 for padding tokens.")

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        valid_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            valid_mask=valid_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        masked_hidden_state = torch.where(
            valid_mask[..., None] > 0, last_hidden_state, torch.zeros_like(last_hidden_state)
        )
        pooled_output = masked_hidden_state.sum(dim=1) / (
            valid_mask.sum(dim=1, keepdim=True) + self.config.pooling_epsilon
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


@auto_docstring
class Tipsv2PreTrainedModel(PreTrainedModel):
    config: Tipsv2Config
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Tipsv2TextEmbeddings",
        "Tipsv2TextEncoderLayer",
        "Tipsv2VisionEmbeddings",
        "Tipsv2VisionLayer",
    ]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        pass


@auto_docstring
class Tipsv2Model(Tipsv2PreTrainedModel):
    def __init__(self, config: Tipsv2Config):
        super().__init__(config)
        if config.text_config.hidden_size != config.vision_config.hidden_size:
            raise ValueError(
                "TIPSv2 does not define projection layers, so `text_config.hidden_size` and "
                "`vision_config.hidden_size` must match."
            )
        if config.temperature <= 0:
            raise ValueError("`temperature` must be strictly positive.")

        self.text_model = Tipsv2TextModel._from_config(config.text_config)
        self.vision_model = Tipsv2VisionModel._from_config(config.vision_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Official TIPSv2-style padding mask where `1` marks padding tokens and `0` marks valid tokens. Mutually
            exclusive with the Hugging Face-style `attention_mask`, where `1` marks valid tokens.
        """
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        return text_outputs

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            return_dict=True,
            **kwargs,
        )
        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        bool_masked_pos: torch.Tensor | None = None,
        return_loss: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tipsv2Output:
        r"""
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Official TIPSv2-style padding mask where `1` marks padding tokens and `0` marks valid tokens. Mutually
            exclusive with the Hugging Face-style `attention_mask`, where `1` marks valid tokens.
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss when both image and text inputs are provided.
        """
        if pixel_values is None and input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify pixel_values, input_ids, or inputs_embeds")

        vision_outputs = None
        image_embeds = None
        if pixel_values is not None:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
                return_dict=True,
                **kwargs,
            )
            image_embeds = vision_outputs.pooler_output
            image_embeds = image_embeds / _get_vector_norm(image_embeds)

        text_outputs = None
        text_embeds = None
        if input_ids is not None or inputs_embeds is not None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                **kwargs,
            )
            text_embeds = text_outputs.pooler_output
            text_embeds = text_embeds / _get_vector_norm(text_embeds)

        logits_per_text = None
        logits_per_image = None
        loss = None
        if image_embeds is not None and text_embeds is not None:
            logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
            logits_per_text = logits_per_text / self.config.temperature
            logits_per_image = logits_per_text.t()
            if return_loss:
                loss = image_text_contrastive_loss(logits_per_text)

        return Tipsv2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


__all__ = [
    "Tipsv2ImageProcessor",
    "Tipsv2Processor",
    "Tipsv2Tokenizer",
    "Tipsv2Config",
    "Tipsv2Model",
    "Tipsv2PreTrainedModel",
    "Tipsv2TextConfig",
    "Tipsv2TextModel",
    "Tipsv2TextPreTrainedModel",
    "Tipsv2VisionConfig",
    "Tipsv2VisionModel",
    "Tipsv2VisionPreTrainedModel",
]

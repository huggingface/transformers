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
from dataclasses import dataclass
from typing import Any

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
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging, torch_int
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ..dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersEmbeddings,
    Dinov2WithRegistersEncoder,
    Dinov2WithRegistersModel,
    Dinov2WithRegistersPreTrainedModel,
)


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
        Offset used by the original TIPSv2 implementation when resizing vision position embeddings.

    Example:

    ```python
    >>> from transformers import Tipsv2VisionConfig, Tipsv2VisionModel

    >>> configuration = Tipsv2VisionConfig()
    >>> model = Tipsv2VisionModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "tipsv2_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: float | int = 4.0
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 448
    patch_size: int | list[int] | tuple[int, int] = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    use_swiglu_ffn: bool = False
    num_register_tokens: int = 1
    interpolate_antialias: bool = True
    interpolate_offset: float = 0.0


@auto_docstring(checkpoint="google/tipsv2-b14")
@strict
class Tipsv2TextConfig(PreTrainedConfig):
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

    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 64
    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    scale_sqrt_depth: bool = True
    pooling_epsilon: float = 1e-8

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

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    temperature: float = 0.01

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Tipsv2TextConfig()
            logger.info("`text_config` is `None`. Initializing the `Tipsv2TextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = Tipsv2TextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Tipsv2VisionConfig()
            logger.info("`vision_config` is `None`. Initializing the `Tipsv2VisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Tipsv2VisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def image_text_contrastive_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


@auto_docstring
@dataclass
class Tipsv2Output(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for image-text similarity.
    logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`, *optional*):
        The cosine-similarity scores between `image_embeds` and `text_embeds`, scaled by the inverse temperature.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`, *optional*):
        The cosine-similarity scores between `text_embeds` and `image_embeds`, scaled by the inverse temperature.
    text_embeds (`torch.FloatTensor` of shape `(text_batch_size, hidden_size)`, *optional*):
        The normalized text embeddings obtained from masked mean pooling over [`Tipsv2TextModel`].
    image_embeds (`torch.FloatTensor` of shape `(image_batch_size, hidden_size)`, *optional*):
        The normalized image embeddings obtained from the CLS token of [`Tipsv2VisionModel`].
    patch_tokens (`torch.FloatTensor` of shape `(image_batch_size, num_patches, hidden_size)`, *optional*):
        The vision tower patch-token sequence before global pooling.
    register_tokens (`torch.FloatTensor` of shape `(image_batch_size, num_register_tokens, hidden_size)`, *optional*):
        The vision tower register-token sequence.
    text_model_output (`BaseModelOutputWithPooling`, *optional*):
        The output of the [`Tipsv2TextModel`].
    vision_model_output (`BaseModelOutputWithPooling`, *optional*):
        The output of the [`Tipsv2VisionModel`].
    """

    loss: torch.FloatTensor | None = None
    logits_per_image: torch.FloatTensor | None = None
    logits_per_text: torch.FloatTensor | None = None
    text_embeds: torch.FloatTensor | None = None
    image_embeds: torch.FloatTensor | None = None
    patch_tokens: torch.FloatTensor | None = None
    register_tokens: torch.FloatTensor | None = None
    text_model_output: BaseModelOutputWithPooling | None = None
    vision_model_output: BaseModelOutputWithPooling | None = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(v.to_tuple() if isinstance(v, ModelOutput) else v for v in self.values())


class Tipsv2VisionEmbeddings(Dinov2WithRegistersEmbeddings):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        previous_dtype = embeddings.dtype
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings.to(dtype=previous_dtype)

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
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(dtype=previous_dtype)


class Tipsv2VisionPreTrainedModel(Dinov2WithRegistersPreTrainedModel):
    config: Tipsv2VisionConfig
    base_model_prefix = "tipsv2_vision_model"


class Tipsv2VisionEncoder(Dinov2WithRegistersEncoder):
    pass


@auto_docstring(
    custom_intro="""
    The TIPSv2 vision tower without any projection head on top.
    """
)
class Tipsv2VisionModel(Dinov2WithRegistersModel):
    def __init__(self, config: Tipsv2VisionConfig):
        super().__init__(config)
        self.apply_layernorm = config.apply_layernorm
        self.reshape_hidden_states = config.reshape_hidden_states


def tipsv2_text_eager_attention_forward(
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


class Tipsv2TextEmbeddings(nn.Module):
    min_timescale: int = 1
    max_timescale: int = 10_000

    def __init__(self, config: Tipsv2TextConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_sqrt_depth else 1.0
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def _create_sinusoidal_position_embedding(
        self, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        position = position_ids.to(device=device, dtype=torch.float32)
        num_timescales = self.embedding_dim // 2
        denominator = torch.maximum(
            torch.tensor(num_timescales, dtype=torch.float32, device=device) - 1,
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        log_timescale_increment = (
            torch.log(
                torch.tensor(float(self.max_timescale) / float(self.min_timescale), dtype=torch.float32, device=device)
            )
            / denominator
        )
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32, device=device) * -log_timescale_increment
        )
        scaled_time = position[:, :, None] * inv_timescales[None, None, :]
        signal = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=2)
        signal = nn.functional.pad(signal, (0, self.embedding_dim % 2, 0, 0, 0, 0))
        return signal.to(dtype=dtype)

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
        position_embeddings = self._create_sinusoidal_position_embedding(
            position_ids=position_ids,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
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
            self.config._attn_implementation, tipsv2_text_eager_attention_forward
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
    base_model_prefix = "tipsv2_text_model"
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
        if isinstance(module, Tipsv2TextEmbeddings):
            init.normal_(module.token_embedding.weight, mean=0.0, std=self.config.initializer_range)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
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
    base_model_prefix = "tipsv2"
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
        patch_tokens = None
        register_tokens = None
        if pixel_values is not None:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
                return_dict=True,
                **kwargs,
            )
            image_embeds = vision_outputs.pooler_output
            image_embeds = image_embeds / _get_vector_norm(image_embeds)

            sequence_output = vision_outputs.last_hidden_state
            num_register_tokens = self.config.vision_config.num_register_tokens
            register_tokens = sequence_output[:, 1 : 1 + num_register_tokens]
            patch_tokens = sequence_output[:, 1 + num_register_tokens :]

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
            patch_tokens=patch_tokens,
            register_tokens=register_tokens,
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

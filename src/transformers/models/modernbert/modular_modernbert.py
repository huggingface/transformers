# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.
#
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
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...integrations import use_kernel_func_from_hub, use_kernelized_func
from ...masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, check_model_inputs
from ..align.modeling_align import eager_attention_forward
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding, rotate_half


logger = logging.get_logger(__name__)


class ModernBertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModernBertModel`]. It is used to instantiate an ModernBert
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ModernBERT-base.
    e.g. [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50368):
            Vocabulary size of the ModernBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModernBertModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1152):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu"`
            if not specified.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
            The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        norm_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the normalization layers.
        pad_token_id (`int`, *optional*, defaults to 50283):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 50282):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 50281):
            Beginning of stream token id.
        cls_token_id (`int`, *optional*, defaults to 50281):
            Classification token id.
        sep_token_id (`int`, *optional*, defaults to 50282):
            Separation token id.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        local_attention (`int`, *optional*, defaults to 128):
            The window size for local attention.
        embedding_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layers.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the MLP layers.
        decoder_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the decoder layers.
        classifier_pooling (`str`, *optional*, defaults to `"cls"`):
            The pooling method for the classifier. Should be either `"cls"` or `"mean"`. In local attention layers, the
            CLS token doesn't attend to all tokens on long sequences.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the classifier.
        classifier_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the classifier.
        classifier_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the classifier.
        deterministic_flash_attn (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic flash attention. If `False`, inference will be faster but not deterministic.
        sparse_prediction (`bool`, *optional*, defaults to `False`):
            Whether to use sparse prediction for the masked language model instead of returning the full dense logits.
        sparse_pred_ignore_index (`int`, *optional*, defaults to -100):
            The index to ignore for the sparse prediction.
        reference_compile (`bool`, *optional*):
            Whether to compile the layers of the model which were compiled during pretraining. If `None`, then parts of
            the model will be compiled if 1) `triton` is installed, 2) the model is not on MPS, 3) the model is not
            shared between devices, and 4) the model is not resized after initialization. If `True`, then the model may
            be faster in some scenarios. This argument is deprecated and will be removed in a future version.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings

    Examples:

    ```python
    >>> from transformers import ModernBertModel, ModernBertConfig

    >>> # Initializing a ModernBert style configuration
    >>> configuration = ModernBertConfig()

    >>> # Initializing a model from the modernbert-base style configuration
    >>> model = ModernBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "modernbert"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = {"global": 160_000.0, "local": 10_000.0}

    def __setattr__(self, name, value):
        if name == "reference_compile" and value is not None:
            logger.warning_once(
                "The `reference_compile` argument is deprecated and will be removed in a future version. "
                "Use `torch.compile()` directly on the model instead."
            )
            value = None
        super().__setattr__(name, value)

    def __init__(
        self,
        vocab_size: int | None = 50368,
        hidden_size: int | None = 768,
        intermediate_size: int | None = 1152,
        num_hidden_layers: int | None = 22,
        num_attention_heads: int | None = 12,
        hidden_activation: str | None = "gelu",
        max_position_embeddings: int | None = 8192,
        initializer_range: float | None = 0.02,
        initializer_cutoff_factor: float | None = 2.0,
        norm_eps: int | None = 1e-5,
        norm_bias: bool | None = False,
        pad_token_id: int | None = 50283,
        eos_token_id: int | None = 50282,
        bos_token_id: int | None = 50281,
        cls_token_id: int | None = 50281,
        sep_token_id: int | None = 50282,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        layer_types: list[str] | None = None,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        local_attention: int | None = 128,
        embedding_dropout: float | None = 0.0,
        mlp_bias: bool | None = False,
        mlp_dropout: float | None = 0.0,
        decoder_bias: bool | None = True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: float | None = 0.0,
        classifier_bias: bool | None = False,
        classifier_activation: str | None = "gelu",
        deterministic_flash_attn: bool | None = False,
        sparse_prediction: bool | None = False,
        sparse_pred_ignore_index: int | None = -100,
        reference_compile: bool | None = None,  # Deprecated
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.local_attention = local_attention
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )

        self.layer_types = layer_types

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self.global_attn_every_n_layers = kwargs.get("global_attn_every_n_layers", 3)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool(i % self.global_attn_every_n_layers) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`. If we find `rope_parameters`
        # as arg in the inputs, we can safely assume that it is in the new format. New naming used -> new format
        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)
            self.rope_parameters["sliding_attention"].update(rope_scaling)
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("global_rope_theta", self.default_theta["global"])
        )
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("local_rope_theta", self.default_theta["local"])
        )

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output

    @property
    def sliding_window(self):
        """Half-window size: `local_attention` is the total window, so we divide by 2."""
        return self.local_attention // 2

    @sliding_window.setter
    def sliding_window(self, value):
        """Set sliding_window by updating local_attention to 2 * value."""
        self.local_attention = value * 2


class ModernBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(
        self, input_ids: torch.LongTensor | None = None, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = self.drop(self.norm(self.tok_embeddings(input_ids)))
        return hidden_states


class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: ModernBertConfig, device=None):
        super().__init__(config, device)

    @staticmethod
    def compute_default_rope_parameters(
        config: ModernBertConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        return super().compute_default_rope_parameters(config, device, seq_len, layer_type)


@use_kernel_func_from_hub("rotary_pos_emb")
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
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(original_dtype), k_embed.to(original_dtype)


@use_kernelized_func(apply_rotary_pos_emb)
class ModernBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(self, config: ModernBertConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * self.head_dim * config.num_attention_heads, bias=config.attention_bias
        )

        if layer_idx % config.global_attn_every_n_layers != 0:
            # config.sliding_window = local_attention // 2 (half-window size, e.g. 64 for local_attention=128)
            # +1 is needed because flash attention sets inclusive boundaries (see modeling_flash_attention_utils.py)
            self.sliding_window = config.sliding_window + 1
        else:
            self.sliding_window = None

        self.is_causal = False

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]

        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(*input_shape, 3, -1, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(dim=-3)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.head_dim**-0.5,
            sliding_window=self.sliding_window,
            deterministic=self.deterministic_flash_attn,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_drop(self.Wo(attn_output))
        return attn_output, attn_weights


class ModernBertEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ModernBertConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_idx=layer_idx)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_output, _ = self.attn(
            self.attn_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


@auto_docstring
class ModernBertPreTrainedModel(PreTrainedModel):
    config: ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": ModernBertEncoderLayer,
        "attentions": ModernBertAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, ModernBertEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertForMaskedLM):
            init_weight(module.decoder, stds["out"])
        elif isinstance(
            module,
            (
                ModernBertForSequenceClassification,
                ModernBertForMultipleChoice,
                ModernBertForTokenClassification,
                ModernBertForQuestionAnswering,
            ),
        ):
            init_weight(module.classifier, stds["final_out"])
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, ModernBertRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False
    ) -> str:
        """
        Checks and dispatches to hhe requested attention implementation.
        """
        # If the user didn't specify anything, try to use flash_attention_2 if available.
        # Otherwise we fall back to the default SDPA -> Eager from the super() method.

        try:
            attn_implementation = (
                "flash_attention_2"
                if attn_implementation is None and self._flash_attn_2_can_dispatch()
                else attn_implementation
            )
        except (ValueError, ImportError):
            pass
        return super()._check_and_adjust_attn_implementation(
            attn_implementation=attn_implementation, is_init_check=is_init_check
        )


@auto_docstring
class ModernBertModel(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.rotary_emb = ModernBertRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if not isinstance(attention_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask_mapping[encoder_layer.attention_type],
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a decoder head on top that is used for masked language modeling.
    """
)
class ModernBertForMaskedLM(ModernBertPreTrainedModel):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = self.decoder(self.head(last_hidden_state))

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a sequence classification head on top that performs pooling.
    """
)
class ModernBertForSequenceClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            if attention_mask is None:
                attention_mask = torch.ones(
                    last_hidden_state.shape[:2], device=last_hidden_state.device, dtype=torch.bool
                )
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.
    """
)
class ModernBertForTokenClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class ModernBertForQuestionAnswering(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.
    """
)
class ModernBertForMultipleChoice(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors.
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs[0]  # shape (num_choices, seq_len, hidden_size)

        # If classifier_pooling is "cls", isolate the <cls> token
        if self.config.classifier_pooling == "cls":
            indices_0 = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            # for left or right padding, <cls> is the first non-pad token
            if attention_mask is not None:
                cls_mask = attention_mask.argmax(dim=-1).to(last_hidden_state.device)
            # if no pad, <cls> is the first token
            else:
                cls_mask = torch.tensor(0, dtype=torch.long, device=last_hidden_state.device)
            # extract the <cls> token for the logits
            last_hidden_state = last_hidden_state[indices_0, cls_mask]

        # If classifier_pooling is "mean", pool the hidden states by averaging over the sequence length
        elif self.config.classifier_pooling == "mean":
            num_non_pad_tokens = attention_mask.sum(dim=1, keepdim=True)
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / num_non_pad_tokens

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ModernBertConfig",
    "ModernBertModel",
    "ModernBertPreTrainedModel",
    "ModernBertForMaskedLM",
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "ModernBertForQuestionAnswering",
    "ModernBertForMultipleChoice",
]

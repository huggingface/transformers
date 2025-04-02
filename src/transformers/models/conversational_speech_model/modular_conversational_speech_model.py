# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin
from ...generation.logits_process import (
    LogitsProcessorList,
)
from ...generation.stopping_criteria import (
    StoppingCriteriaList,
)
from ...generation.utils import GenerateDecoderOnlyOutput, GenerateNonBeamOutput
from ...loss.loss_utils import fixed_cross_entropy
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import rope_config_validation
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import KwargsForCausalLM, LlamaForCausalLM, LlamaModel


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "ConversationalSpeechModelConfig"


class ConversationalSpeechModelDepthDecoderConfig(LlamaConfig):
    def __init__(
        self,
        num_codebooks=32,
        backbone_hidden_size=2048,
        vocab_size=2051,
        hidden_size=1024,
        intermediate_size=8192,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=2050,
        bos_token_id=None,
        eos_token_id=None,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        attn_implementation="sdpa",
        **kwargs,
    ):
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.backbone_hidden_size = backbone_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            attn_implementation=attn_implementation,
            **kwargs,
        )


class ConversationalSpeechModelBackboneConfig(LlamaConfig):
    def __init__(
        self,
        num_codebooks=32,
        codebook_vocab_size=2051,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=128002,
        codebook_pad_token_id=2050,
        codebook_eos_token_id=0,
        bos_token_id=128000,
        eos_token_id=128001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        attn_implementation="sdpa",
        **kwargs,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_vocab_size = codebook_vocab_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        self.codebook_pad_token_id = codebook_pad_token_id
        self.codebook_eos_token_id = codebook_eos_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            attn_implementation=attn_implementation,
            **kwargs,
        )


class ConversationalSpeechModelConfig(PretrainedConfig):
    model_type = "conversational_speech_model"
    sub_model_type = {
        "backbone_config": ConversationalSpeechModelBackboneConfig,
        "depth_decoder_config": ConversationalSpeechModelDepthDecoderConfig,
    }

    def get_text_config(self, decoder=False):
        return self.backbone_config

    def __init__(
        self,
        backbone_config=None,
        depth_decoder_config=None,
        **kwargs,
    ):
        if backbone_config is None:
            self.backbone_config = ConversationalSpeechModelBackboneConfig()
            logger.info("backbone_config is None, using default backbone config.")
        elif isinstance(backbone_config, dict):
            self.backbone_config = ConversationalSpeechModelBackboneConfig(**backbone_config)
        elif isinstance(backbone_config, ConversationalSpeechModelBackboneConfig):
            self.backbone_config = backbone_config

        if depth_decoder_config is None:
            self.depth_decoder_config = ConversationalSpeechModelDepthDecoderConfig()
            logger.info("depth_decoder_config is None, using default depth decoder config.")
        elif isinstance(depth_decoder_config, dict):
            self.depth_decoder_config = ConversationalSpeechModelDepthDecoderConfig(**depth_decoder_config)
        elif isinstance(depth_decoder_config, ConversationalSpeechModelDepthDecoderConfig):
            self.depth_decoder_config = depth_decoder_config

        self.vocab_size = self.backbone_config.codebook_vocab_size
        self.hidden_size = self.backbone_config.hidden_size
        self.num_codebooks = self.backbone_config.num_codebooks
        self.initializer_range = self.backbone_config.initializer_range

        # tie_word_embeddings does not apply here
        kwargs["tie_word_embeddings"] = False
        self.max_position_embeddings = 2048

        super().__init__(**kwargs)


@dataclass
class ConversationalSpeechModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    depth_decoder_loss: Optional[torch.FloatTensor] = None
    depth_decoder_logits: torch.FloatTensor = None
    depth_decoder_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    depth_decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    depth_decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    backbone_loss: Optional[torch.FloatTensor] = None


class ConversationalSpeechModelEmbeddings(nn.Module):
    def __init__(self, num_codebooks, codebook_vocab_size, backbone_hidden_size, codebook_padding_idx):
        super().__init__()
        self.codebook_vocab_size = codebook_vocab_size
        self.embed_audio_tokens = nn.Embedding((num_codebooks * codebook_vocab_size), backbone_hidden_size, codebook_padding_idx)

    def forward(self, input_ids, codebook_idxs):
        """
        Args:
            input_ids (`torch.Tensor`): 
                Codebooks ids of shape (batch_size, seq_length)
            codebook_idxs (`torch.Tensor`): 
                Corresponding codebook indices of shape (batch_size, seq_length)
        """
        offset = codebook_idxs * self.codebook_vocab_size
        return self.embed_audio_tokens(input_ids + offset)

START_DOCSTRING_BASE = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`{config_class}`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CONVERSATIONAL_SPEECH_MODEL_DEPTH_DECODER_START_DOCSTRING = START_DOCSTRING_BASE.format(
    config_class="ConversationalSpeechModelDepthDecoderConfig"
)


CONVERSATIONAL_SPEECH_MODEL_BACKBONE_START_DOCSTRING = START_DOCSTRING_BASE.format(
    config_class="ConversationalSpeechModelBackboneConfig"
)


CONVERSATIONAL_SPEECH_MODEL_START_DOCSTRING = START_DOCSTRING_BASE.format(
    config_class="ConversationalSpeechModelConfig"
)


@add_start_docstrings(
    "The bare ConversationalSpeechModel Model outputting raw hidden-states without any specific head on top.",
    CONVERSATIONAL_SPEECH_MODEL_START_DOCSTRING,
)
class ConversationalSpeechModelPreTrainedModel(PreTrainedModel):
    config_class = ConversationalSpeechModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ConversationalSpeechModelDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

INPUTS_DOCSTRING_BASE = r"""
    Args:
        {input_ids_docstring}
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


DEPTH_DECODER_INPUT_IDS_DOCSTRING = (
    r"""input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)"""
)


BACKBONE_INPUT_IDS_DOCSTRING = (
    r"""input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks + 1)`):
            Indices of input sequence tokens in the vocabulary. The `num_codebooks` first tokens along last dimension are
            the codebook indices in a growing order. The last token along last dimension is the text token. An input
            sequence is the concatenation of audio and text frames. A frames is a vector along the sequence length
            dimension (with shape `(num_codebooks + 1,)`):
            1. an audio frames, which is the concatenation of `num_codebooks` codebook indices, and the text padding idx.
            2. a text frames, which is the concatenation of `num_codebooks` codebook padding indices, and the text token.

            #TODO: complete this
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)"""
)


CONVERSATIONAL_SPEECH_MODEL_DEPTH_DECODER_INPUTS_DOCSTRING = INPUTS_DOCSTRING_BASE.format(
    input_ids_docstring=DEPTH_DECODER_INPUT_IDS_DOCSTRING
)


CONVERSATIONAL_SPEECH_MODEL_BACKBONE_INPUTS_DOCSTRING = INPUTS_DOCSTRING_BASE.format(
    input_ids_docstring=BACKBONE_INPUT_IDS_DOCSTRING
)





@add_start_docstrings(
    "The bare ConversationalSpeechModelDepthDecoder Model outputting raw hidden-states without any specific head on top.",
    CONVERSATIONAL_SPEECH_MODEL_DEPTH_DECODER_START_DOCSTRING,
)
class ConversationalSpeechModelDepthDecoder(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ConversationalSpeechModelDecoderLayer`]

    Args:
        config: ConversationalSpeechModelDepthDecoderConfig
    """

    config_class = ConversationalSpeechModelDepthDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechModelEmbeddings(config.num_codebooks, config.vocab_size, config.backbone_hidden_size, self.padding_idx)
        self.inputs_embeds_projector = nn.Linear(config.backbone_hidden_size, config.hidden_size, bias=False)

    @add_start_docstrings_to_model_forward(CONVERSATIONAL_SPEECH_MODEL_DEPTH_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            inputs_seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device)

        if inputs_embeds is None:
            codebook_idxs = cache_position.unsqueeze(0) - 1
            inputs_embeds = self.embed_tokens(input_ids, codebook_idxs)

        inputs_embeds = self.inputs_embeds_projector(inputs_embeds)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class ConversationalSpeechModelCodebooksHead(nn.Module):
    def __init__(self, hidden_size, num_codebooks, vocab_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.weight = nn.Parameter(
            torch.empty(self.num_codebooks - 1, hidden_size, vocab_size)
        )

    def reset_parameters(self):
        for i in range(self.num_codebooks - 1):
            nn.init.kaiming_uniform_(
                self.weight[i], a=math.sqrt(5)
            )

    def forward(self, hidden_states, last_cache_position):
        if last_cache_position is None:
            codebook_weight = self.weight
        else:
            codebook_weight = self.weight[last_cache_position[-1:] - 1]

        return torch.einsum("bsh,sho->bso", hidden_states, codebook_weight)


@add_start_docstrings(
    """
    The ConversationalSpeechModelDepthDecoder Model transformer, with a ConversationalSpeechModelCodebooksHead on top,
    which can be seen a position-specific language modeling head, allowing to use a different linear layer for each codebook
    (e.g. position 0 is the first codebook and uses the first codebook head, etc.)
    """,
    CONVERSATIONAL_SPEECH_MODEL_DEPTH_DECODER_START_DOCSTRING,
)
class ConversationalSpeechModelDepthDecoderForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = None

    def __init__(self, config):
        super().__init__(config)
        del self.lm_head
        self.codebooks_head = ConversationalSpeechModelCodebooksHead(config.hidden_size, config.num_codebooks, config.vocab_size)
        self.model = ConversationalSpeechModelDepthDecoder(config)

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(CONVERSATIONAL_SPEECH_MODEL_DEPTH_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        #TODO
        ```python
        pass
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if isinstance(logits_to_keep, int):
            if logits_to_keep == 0:
                # skip idx 0 logits since it's for the concatenated backbone last hidden state
                slice_indices = slice(1, None)
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        logits = self.codebooks_head(hidden_states[:, slice_indices, :], cache_position)
        logits = logits.contiguous()

        loss = None
        if labels is not None:
            logits = logits.float()
            labels = labels.to(logits.device)
            shift_labels = labels[..., 1:].contiguous()
            logits = logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(logits.device)
            loss = fixed_cross_entropy(logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ConversationalSpeechBackboneModelEmbeddings(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_codebooks, codebook_vocab_size, text_padding_idx, codebook_padding_idx):
        super().__init__()
        # we do not set them as padding idx for the embeddings
        # indeed, embed_audio_tokens is tied to the depth decoder that might use them
        # nevertheless for the backbone model, they are used to zeros out the padding tokens
        self.text_padding_idx = text_padding_idx
        self.codebook_padding_idx = codebook_padding_idx
        self.embed_text_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_audio_tokens = nn.Embedding((num_codebooks * codebook_vocab_size), hidden_size)
        self.audio_tokens_offsets = torch.arange(num_codebooks) * codebook_vocab_size

    def forward(self, input_ids):
        """
        Args:
            input_ids (`torch.Tensor` of shape (batch_size, seq_length, num_codebooks + 1)):
                On last dimension, first values are codebook tokens, and last value is a text token.
        Returns:
            `torch.Tensor` of shape (batch_size, seq_length, hidden_size):
                Embedded tokens, summed over the last dimension according to input_ids_mask.
        """
        text_tokens = input_ids[:, :, -1:]
        text_tokens_mask = text_tokens != self.text_padding_idx

        audio_tokens = input_ids[:, :, :-1]
        audio_tokens_mask = audio_tokens != self.codebook_padding_idx
        audio_tokens = audio_tokens + self.audio_tokens_offsets.to(audio_tokens.device)

        text_embeds = self.embed_text_tokens(text_tokens)
        text_embeds *= text_tokens_mask.unsqueeze(-1)

        audio_embeds = self.embed_audio_tokens(audio_tokens)
        audio_embeds *= audio_tokens_mask.unsqueeze(-1)

        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=-2)
        inputs_embeds = inputs_embeds.sum(dim=-2)

        return inputs_embeds


@add_start_docstrings(
    "The bare ConversationalSpeechBackboneModel Model outputting raw hidden-states without any specific head on top.",
    CONVERSATIONAL_SPEECH_MODEL_BACKBONE_START_DOCSTRING,
)
class ConversationalSpeechModelBackboneModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ConversationalSpeechModelDecoderLayer`]

    Args:
        config: ConversationalSpeechBackboneModelConfig
    """
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = ConversationalSpeechBackboneModelEmbeddings(
            config.hidden_size,
            config.vocab_size,
            config.num_codebooks,
            config.codebook_vocab_size,
            self.padding_idx,
            config.codebook_pad_token_id,
        )

    @add_start_docstrings_to_model_forward(CONVERSATIONAL_SPEECH_MODEL_BACKBONE_INPUTS_DOCSTRING)
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


# TODO: for some reason modular is not catching this
CONVERSATIONAL_SPEECH_MODEL_INPUTS_DOCSTRING = INPUTS_DOCSTRING_BASE.format(
    input_ids_docstring=BACKBONE_INPUT_IDS_DOCSTRING
)


@add_start_docstrings(
    """
    The ConversationalSpeechModel Model transformer, with ConversationalSpeechModelBackboneModel backbone model that predicts the first codebook token,
    and a ConversationalSpeechModelDepthDecoderForCausalLM depth decoder that predicts the other codebook tokens.
    """,
    CONVERSATIONAL_SPEECH_MODEL_START_DOCSTRING,
)
class ConversationalSpeechModelForCausalLM(LlamaForCausalLM, GenerationMixin):
    _tied_weights_keys = ["backbone_model.embed_tokens.embed_audio_tokens.weight", "depth_decoder.model.embed_tokens.embed_audio_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        del self.model
        self.depth_decoder = ConversationalSpeechModelDepthDecoderForCausalLM._from_config(config.depth_decoder_config)
        self.backbone_model = ConversationalSpeechModelBackboneModel._from_config(config.backbone_config)

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(CONVERSATIONAL_SPEECH_MODEL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks + 1)`, *optional*):
                Labels for computing the masked language modeling loss. The `num_codebooks` first tokens along last dimension are
                the codebook indices in a growing order (see `input_ids` docstring). Indicies should be in:
                1. `[0, ..., config.vocab_size]` for codebook tokens
                2. `[0, ..., config.backbone_config.vocab_size]` for text tokens

                Text frames (see `input_ids` docstring) indices should be all set to `-100`.
                Audio frames that should not intervene in the loss computation for depth decoder should have all tokens expect the
                first one set to `-100`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                Kept for compatibility. Does not support another value than:
                1. `0`, which is equivalent to keeping all logits, used in the training regime
                2. `1`, which is equivalent to keeping only the last logit, used in the generation regime

        Returns:

        Example:

        #TODO
        ```python
        pass
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        backbone_outputs = self.backbone_model(
            input_ids=input_ids,
            input_ids_mask=input_ids_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        backbone_hidden_states = backbone_outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        backbone_logits = self.lm_head(backbone_hidden_states[:, slice_indices, :])

        loss = None
        backbone_loss = None
        depth_decoder_loss = None
        depth_decoder_outputs = None
        if labels is not None:
            # select first codebook as labels for the backbone model
            backbone_labels = labels[:, :, 0] if labels is not None else None
            backbone_loss = self.loss_function(
                logits=backbone_logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs
            )

            # for the depth decoder, we need to select the frames to train on
            # those are frames where the label is not uniformly `ignore_index` along the codebook dimension
            depth_decoder_labels = labels[:, :, : self.config.num_codebooks]
            mask_idxs = (depth_decoder_labels[:, :, 1:] == -100).all(dim=-1)
            train_idxs = (~mask_idxs).nonzero()

            depth_decoder_input_ids = input_ids[train_idxs[:, 0], train_idxs[:, 1], : self.config.num_codebooks - 1]
            backbone_last_hidden_states = backbone_hidden_states[train_idxs[:, 0], train_idxs[:, 1] - 1, :]
            depth_decoder_labels = depth_decoder_labels[train_idxs[:, 0], train_idxs[:, 1], :]

            codebook_idxs = torch.arange(self.config.num_codebooks - 1, device=depth_decoder_input_ids.device, dtype=torch.long)
            codebook_idxs = codebook_idxs.expand(depth_decoder_input_ids.shape[0], -1)
            inputs_embeds = torch.cat(
                [
                    backbone_last_hidden_states.unsqueeze(1),
                    self.depth_decoder.get_input_embeddings()(depth_decoder_input_ids, codebook_idxs)
                ],
                dim=1
            )

            depth_decoder_outputs = self.depth_decoder(
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                labels=depth_decoder_labels,
            )
            depth_decoder_loss = depth_decoder_outputs.loss
            loss = backbone_loss + depth_decoder_loss

        return ConversationalSpeechModelOutputWithPast(
            loss=loss,
            backbone_loss=backbone_loss,
            depth_decoder_loss=depth_decoder_loss,
            logits=backbone_logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
            depth_decoder_logits=depth_decoder_outputs.logits if depth_decoder_outputs is not None else None,
            depth_decoder_past_key_values=depth_decoder_outputs.past_key_values if depth_decoder_outputs is not None else None,
            depth_decoder_hidden_states=depth_decoder_outputs.hidden_states if depth_decoder_outputs is not None else None,
            depth_decoder_attentions=depth_decoder_outputs.attentions if depth_decoder_outputs is not None else None,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        This method overrides [~generation.utils.GenerationMixin._sample].
        To ease maintenance, modifications are marked with the comment "CSM specific".

        Indeed, CSM model requires a custom generation sampling step:
        1. Infer the backbone model to sample the first codebook token
        2. Call generate on the depth decoder with the first codebook token as input_ids to sample the next codebook tokens
        3. Use these generated codebook tokens as input_ids to sample the next first codebook token using the backbone model
        4. Repeat until stopping criteria is met

        CSM supports two stopping criterias:
        - stop when the generated sequence is at max_length
        - stop when all the generated codebook tokens are the codebook_eos_token_id
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # *************** CSM specific ***************
        # expand input_ids to (batch_size, seq_length, num_codebooks)
        input_ids = input_ids.reshape(batch_size, 0, self.config.num_codebooks + 1)
        depth_decoder_generate_kwargs = model_kwargs.pop("depth_decoder_generate_kwargs", {})
        # ============================================

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            is_compileable = is_compileable and not self.generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # *************** CSM specific ***************
            # infer the depth decoder
            first_codebook_ids = next_tokens[:, None]
            backbone_last_hidden_states = outputs.hidden_states[-1][:, -1, :]

            # first input embeds should be concatenation of backbone last hidden state and the embeddings
            # of the first codebook generated by the backbone model
            inputs_embeds = torch.cat(
                [
                    backbone_last_hidden_states.unsqueeze(1),
                    self.depth_decoder.get_input_embeddings()(
                        first_codebook_ids, torch.zeros((batch_size, 1), device=first_codebook_ids.device, dtype=torch.long)
                    )
                ],
                dim=1
            )
            torch.compiler.cudagraph_mark_step_begin()
            depth_decoder_outputs = self.depth_decoder.generate(
                inputs_embeds=inputs_embeds,
                **depth_decoder_generate_kwargs,
            )
            codebook_ids = torch.cat([first_codebook_ids, depth_decoder_outputs], dim=-1)
            next_tokens = torch.cat(
                [
                    codebook_ids,
                    torch.ones((codebook_ids.shape[0], 1), dtype=torch.long, device=codebook_ids.device) * self.backbone_model.padding_idx
                ],
                dim=-1,
            )
            # ============================================

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # *************** CSM specific ***************
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None, :]], dim=1)
            # ============================================
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # *************** CSM specific ***************
            # for the eos stopping criteria, is it expected that the eos token is the same for each codebook !!!!
            unfinished_sequences = unfinished_sequences & ~(
                input_ids[:, -1, :-1] == self.config.backbone_config.codebook_eos_token_id
            ).all(-1)
            # ============================================
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
            # *************** CSM specific ***************
            del depth_decoder_outputs
            # ============================================

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids

    def _validate_model_kwargs(self, model_kwargs):
        """
        This method overrides [~generation.utils.GenerationMixin._validate_model_kwargs].
        We need to pass to generate the depth_decoder_generate_kwargs, yet they are not model_kwargs.
        """
        model_kwargs.pop("depth_decoder_generate_kwargs", None)
        GenerationMixin._validate_model_kwargs(model_kwargs)

    def _validate_depth_decoder_generate_kwargs(self, depth_decoder_generate_kwargs):
        min_new_tokens = depth_decoder_generate_kwargs.get("min_new_tokens", self.config.num_codebooks - 1)
        max_new_tokens = depth_decoder_generate_kwargs.get("max_new_tokens", self.config.num_codebooks - 1)
        if set([min_new_tokens, max_new_tokens]) != set([self.config.num_codebooks - 1]):
            raise ValueError(
                f"depth_decoder_generate_kwargs' min_new_tokens ({min_new_tokens}) and max_new_tokens ({max_new_tokens}) must be equal to self.config.num_codebooks - 1 ({self.config.num_codebooks - 1})"
            )

        depth_decoder_generate_kwargs["min_new_tokens"] = min_new_tokens
        depth_decoder_generate_kwargs["max_new_tokens"] = max_new_tokens
        depth_decoder_generate_kwargs["return_dict_in_generate"] = False

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,  # TODO: to test
        streamer: Optional["BaseStreamer"] = None,  # TODO: to test
        depth_decoder_generate_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        This method overrides [~generation.utils.GenerationMixin.generate] to match the specifics of the CSM model.
        Indeed, CSM model requires a custom generation sampling step:
        1. Infer the backbone model to sample the first codebook token
        2. Call generate on the depth decoder with the first codebook token as input_ids to sample the next codebook tokens
        3. Use these generated codebook tokens as input_ids to sample the next first codebook token using the backbone model
        4. Repeat until stopping criteria is met

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, do_sample=True)`.
        </Tip>

        Parameters:
            inputs_ids (`torch.Tensor` of shape (batch_size, seq_length, num_codebooks + 1), *optional*):
                The sequence used as a prompt for the backbone model. 
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`] or `torch.LongTensor`: A [`~generation.GenerateDecoderOnlyOutput`]
            (if `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.
        """
        # TODO: ensure the user is not requesting an unsupported generation mode (!= greeedy/ sampling)
        # TODO: ensure the user is not using another stopping criteria than max length one

        depth_decoder_generate_kwargs = {} if depth_decoder_generate_kwargs is None else depth_decoder_generate_kwargs
        self._validate_depth_decoder_generate_kwargs(depth_decoder_generate_kwargs)

        # as itself generate does not handle input_ids with a depth dimension (here we have [batch_size, seq_length, num_codebooks])
        # we circumvent this by providing the inputs_embeds directly that allows use to fall back to an handled shape [batch_size, seq_length, hidden_size]
        inputs_embeds = self.backbone_model.get_input_embeddings()(input_ids)

        if kwargs.pop("output_hidden_states", None) is False:
            logger.warning("CSM does not support `output_hidden_states=False`, this will be ignored.")

        return GenerationMixin.generate(
            inputs_embeds=inputs_embeds,
            depth_decoder_generate_kwargs=depth_decoder_generate_kwargs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            streamer=streamer,
            output_hidden_states=True,
            **kwargs
        )


__all__ = [
    "ConversationalSpeechModelDepthDecoderConfig",
    "ConversationalSpeechModelConfig",
    "ConversationalSpeechModelDepthDecoder",
    "ConversationalSpeechModelDepthDecoderForCausalLM",
    "ConversationalSpeechModelBackboneModel",
    "ConversationalSpeechModelForCausalLM",
]

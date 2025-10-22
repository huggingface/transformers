# coding=utf-8
# Copyright 2024 Convai Innovations and The HuggingFace Inc. team. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2023 The Llama Authors released the Llama v2 model.
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
"""PyTorch ConvaiCausalLM model using the modular approach."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

# Import necessary components from base classes and utilities
from ...cache_utils import Cache

# Import GenerationMixin explicitly
from ...generation.utils import GenerationMixin
from ...integrations import use_kernel_forward_from_hub  # Added for RMSNorm decorator
from ...modeling_layers import GradientCheckpointingLayer  # Added
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...utils import (  # Added add_start_docstrings
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_flex_attn_available  # Added is_torch_flex_attn_available

# Import Llama components ONLY for inheritance of sub-modules and potentially types/docstrings
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,  # Inherit base class features
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    # LLAMA_INPUTS_DOCSTRING, # Do not import this
)

# Import configuration class directly from this model's definition
from .configuration_convaicausallm import ConvaiCausalLMConfig


# --- Copied Imports required by Llama components ---
if is_torch_flex_attn_available():
    pass


logger = logging.get_logger(__name__)

# Define necessary variables for docstrings
_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-causal-lm"
_CONFIG_FOR_DOC = "ConvaiCausalLMConfig"

# ==== Docstring Variables (Define locally) ====
# Copied and adapted from LLAMA_START_DOCSTRING
CONVAI_CAUSAL_L_M_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ConvaiCausalLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Copied and adapted from LLAMA_INPUTS_DOCSTRING
CONVAI_CAUSAL_L_M_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length) or `BlockMask`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If the model is configured to use flex_attention, it will attempt to convert the mask Tensor into a BlockMask,
            but you can also pass a `BlockMask` object directly here.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`~modeling_attn_mask_utils.AttentionMaskConverter._prepare_4d_causal_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

# ==== Helper Functions (Copied from Llama) ====
# Needed by inherited components if not overridden


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key-value heads for Grouped Query Attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention implementation helper."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Ensure mask slicing matches kv sequence length
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ==== Renamed Inherited Components ====


@use_kernel_forward_from_hub("RMSNorm")  # Keep decorator if applicable
class ConvaiCausalLMRMSNorm(LlamaRMSNorm):
    pass


class ConvaiCausalLMRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class ConvaiCausalLMAttention(LlamaAttention):
    def __init__(self, config: ConvaiCausalLMConfig, layer_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size mismatch")
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    # Forward pass is inherited from LlamaAttention.
    # It uses the helper functions defined above (apply_rotary_pos_emb, repeat_kv, eager_attention_forward)
    # and the attributes set in __init__.


class ConvaiCausalLMMLP(LlamaMLP):
    pass


class ConvaiCausalLMDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: ConvaiCausalLMConfig, layer_idx: int):
        # Explicitly call GradientCheckpointingLayer init if needed, or just nn.Module
        super(GradientCheckpointingLayer, self).__init__()  # Call grandparent if needed
        # Or just: nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = ConvaiCausalLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = ConvaiCausalLMMLP(config)
        self.input_layernorm = ConvaiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ConvaiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Forward pass is inherited from LlamaDecoderLayer.
    # It uses the components defined above (self_attn, mlp, norms).


# ==== PreTrainedModel Base ====
@add_start_docstrings(
    "The bare ConvaiCausalLM Model outputting raw hidden-states without any specific head on top.",
    CONVAI_CAUSAL_L_M_START_DOCSTRING,  # Use locally defined docstring
)
class ConvaiCausalLMPreTrainedModel(LlamaPreTrainedModel):  # Inherit Llama's base functionality
    config_class = ConvaiCausalLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ConvaiCausalLMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    # Inherit other flags like _supports_flash_attn_2 etc.
    # Ensure _init_weights handles ConvaiCausalLMRMSNorm correctly (it should as it checks isinstance)


# ==== Main Model ====
# Inherit from LlamaModel only to simplify init and grab its forward structure
@add_start_docstrings(
    "The bare ConvaiCausalLM Model outputting raw hidden-states without any specific head on top.",
    CONVAI_CAUSAL_L_M_START_DOCSTRING,  # Use locally defined docstring
)
class ConvaiCausalLMModel(ConvaiCausalLMPreTrainedModel, LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ConvaiCausalLMDecoderLayer`]
    Args:
        config: ConvaiCausalLMConfig
    """

    def __init__(self, config: ConvaiCausalLMConfig):
        super(ConvaiCausalLMModel, self).__init__(config)  # Calls LlamaModel's init via PreTrainedModel hierarchy
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ConvaiCausalLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ConvaiCausalLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = ConvaiCausalLMRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    # Forward pass is inherited from LlamaModel and uses the overridden components.
    # We need to ensure _update_causal_mask and _prepare_4d_causal_attention_mask_with_cache_position
    # are available if LlamaModel.forward uses them. Since we inherit LlamaModel, they should be.


# ==== Causal LM Head Model ====
# Inherit from our PreTrainedModel and GenerationMixin
@add_start_docstrings(
    """
    The ConvaiCausalLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    CONVAI_CAUSAL_L_M_START_DOCSTRING,  # Use locally defined docstring
)
class ConvaiCausalLMForCausalLM(ConvaiCausalLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ConvaiCausalLMConfig):
        super().__init__(config)
        self.model = ConvaiCausalLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        if not isinstance(decoder, ConvaiCausalLMModel):
            logger.warning(f"Setting decoder of type {type(decoder)}, expected ConvaiCausalLMModel.")
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Manually define the forward method body (Copied & Adapted from Llama)
    @add_start_docstrings_to_model_forward(CONVAI_CAUSAL_L_M_INPUTS_DOCSTRING)  # Use locally defined docstring
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                 Controls which logits to compute to save memory. See Llama documentation.

        Returns: CausalLMOutputWithPast

        Example:
        ```python
        >>> from transformers import AutoTokenizer, ConvaiCausalLMForCausalLM
        >>> import torch

        >>> model_id = "convaiinnovations/hindi-causal-lm"
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id)
        >>> model = ConvaiCausalLMForCausalLM.from_pretrained(model_id)
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model.to(device)

        >>> prompt = "भारत एक विशाल देश है"
        >>> inputs = tokenizer(prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.8, top_k=50, do_sample=True)
        >>> print(tokenizer.decode(outputs, skip_special_tokens=True))
        # Example output: भारत एक विशाल देश है। यहाँ विभिन्न प्रकार की भाषाएँ और संस्कृतियाँ पाई जाती हैं। यह दुनिया का
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "cache_position": cache_position,
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        outputs: BaseModelOutputWithPast = self.model(input_ids=input_ids, **model_kwargs)

        hidden_states = outputs.last_hidden_state

        if isinstance(logits_to_keep, int) and logits_to_keep != 0:
            slice_indices = slice(-logits_to_keep, None)
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        elif isinstance(logits_to_keep, torch.Tensor):
            slice_indices = logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    # Manually define prepare_inputs_for_generation (Copied & Adapted from Llama)
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        use_cache = kwargs.get("use_cache")

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if past_key_values is not None:
            if inputs_embeds is None:  # Only slice input_ids if not using embeds for the current step
                # The standard behavior for cached generation is to take only the last token id.
                input_ids = input_ids[:, -1:]
                # Update model_inputs with the potentially sliced input_ids
                model_inputs["input_ids"] = input_ids

        # Prepare cache position
        if past_key_values is not None:
            if cache_position is None:
                past_length = past_key_values.get_seq_length(self.config.num_hidden_layers - 1)
                current_length = (
                    input_ids.shape[1]
                    if input_ids is not None
                    else (inputs_embeds.shape[1] if inputs_embeds is not None else 1)
                )  # Default current length to 1 if only past is provided
                device = (
                    input_ids.device
                    if input_ids is not None
                    else (inputs_embeds.device if inputs_embeds is not None else next(self.parameters()).device)
                )
                cache_position = torch.arange(past_length, past_length + current_length, device=device)
            model_inputs["cache_position"] = cache_position
        else:
            model_inputs["cache_position"] = None

        # Prepare position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                # For cached generation, position_ids needs to be for the new token(s)
                current_input_length = (
                    input_ids.shape[1]
                    if input_ids is not None
                    else (inputs_embeds.shape[1] if inputs_embeds is not None else 1)
                )
                # Take the last `current_input_length` position ids
                position_ids = position_ids[:, -current_input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,  # Pass full mask, model forward handles slicing/causal
            }
        )
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        return model_inputs

    # Ensure _reorder_cache exists if using beam search etc. (Copy from Llama if needed)
    # This static method definition should be fine within the class body.
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # Handle Cache objects (common case now)
            if isinstance(layer_past, Cache):
                # DynamicCache and StaticCache implement reorder_cache
                if hasattr(layer_past, "reorder_cache"):
                    reordered_past += (layer_past.reorder_cache(beam_idx),)
                else:
                    # Fallback/Warning for unknown Cache types without reorder_cache
                    logger.warning(
                        f"Cache type {type(layer_past)} does not implement reorder_cache. Beam search may fail."
                    )
                    reordered_past += (layer_past,)  # Pass through, hoping for the best
            # Handle older tuple-based caches (less common now but for BC)
            elif isinstance(layer_past, tuple):
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            else:
                logger.warning(f"Unexpected cache structure type: {type(layer_past)}")
                reordered_past += (layer_past,)  # Pass through

        return reordered_past

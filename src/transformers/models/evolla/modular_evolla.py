"""PyTorch Evolla model."""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings_to_model_forward,
    logging,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from .configuration_evolla import EvollaConfig
from .protein import ProteinEncoderModelOutput, SaProtProteinEncoder
from .sequence_aligner import CrossAttention
from .sequence_compressor import SequenceCompressorResampler


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EvollaConfig"


class EvollaRMSNorm(LlamaRMSNorm):
    pass


ALL_LAYERNORM_LAYERS.append(EvollaRMSNorm)


class EvollaRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class EvollaMLP(LlamaMLP):
    pass


class EvollaAttention(LlamaAttention):
    pass


# this was adapted from LlamaDecoderLayer
class EvollaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: EvollaConfig, layer_idx: int, adapter: CrossAttention = None):
        super().__init__(config, layer_idx)
        if adapter is not None:
            self.adapter = adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        protein_kv_states: Optional[torch.Tensor] = None,
        structure_kv_states: Optional[torch.Tensor] = None,
        msa_kv_states: Optional[torch.Tensor] = None,
        protein_batch_mask: Optional[torch.Tensor] = None,
        structure_batch_mask: Optional[torch.Tensor] = None,
        msa_batch_mask: Optional[torch.Tensor] = None,
        query_attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if hasattr(self, "adapter"):
            hidden_states = self.adapter(
                query_states=hidden_states,
                protein_kv_states=protein_kv_states,
                structure_kv_states=structure_kv_states,
                msa_kv_states=msa_kv_states,
                query_attn_mask=query_attn_mask,
                protein_batch_mask=protein_batch_mask,
                structure_batch_mask=structure_batch_mask,
                msa_batch_mask=msa_batch_mask,
            )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# this was adapted from transformers.models.idefics.modeling_idefics.IdeficsPreTrainedModel with Idefics->Evolla
class EvollaPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        # important: this ported version of Evolla isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the m4 code
        # base should be used for training from scratch and it contains the correct code.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class EvollaProteinEncoder(nn.Module):
    def __init__(
        self,
        config: EvollaConfig,
        add_pooling_layer: bool = False,
    ):
        super().__init__()
        self.model = SaProtProteinEncoder(
            vocab_size=config.protein_vocab_size,
            mask_token_id=config.protein_mask_token_id,
            pad_token_id=config.protein_pad_token_id,
            hidden_size=config.protein_hidden_size,
            num_hidden_layers=config.protein_num_hidden_layers,
            num_attention_heads=config.protein_num_attention_heads,
            intermediate_size=config.protein_intermediate_size,
            hidden_dropout_prob=config.protein_hidden_dropout_prob,
            attention_probs_dropout_prob=config.protein_attention_probs_dropout_prob,
            max_position_embeddings=config.protein_max_position_embeddings,
            layer_norm_eps=config.protein_layer_norm_eps,
            position_embedding_type=config.protein_position_embedding_type,
            emb_layer_norm_before=config.protein_emb_layer_norm_before,
            token_dropout=config.protein_token_dropout,
            add_pooling_layer=add_pooling_layer,
        )

        self.sequence_compressor_resampler = SequenceCompressorResampler(
            protein_repr_dim=config.protein_hidden_size,
            output_repr_dim=config.hidden_size,
            depth=config.resampler_depth,
            dim_head=config.resampler_dim_head,
            heads=config.resampler_heads,
            num_latents=config.resampler_num_latents,
            ff_mult=config.resampler_ff_mult,
        )

    def sequence_encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        sequence_repr = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return sequence_repr

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        protein_output = self.sequence_encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # TODO: could be replaced by last hidden state
        protein_embeds = protein_output.last_hidden_state

        sequence_repr = self.sequence_compressor_resampler(protein_embeds, attention_mask)

        if not return_dict:
            return sequence_repr, protein_embeds, attention_mask

        return ProteinEncoderModelOutput(
            sequence_compressor_output=sequence_repr,
            last_hidden_state=protein_output.last_hidden_state,
            hidden_states=protein_output.hidden_states,
            attentions=protein_output.attentions,
        )


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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


class EvollaLLM(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(
        self,
        config: EvollaConfig,
    ):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [
                EvollaDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    adapter=CrossAttention(
                        hidden_size=config.hidden_size,
                        num_attention_heads=config.num_attention_heads,
                        attention_probs_dropout_prob=config.aligner_attention_probs_dropout_prob,
                        enable_bias=config.aligner_enable_bias,
                        ffn_mult=config.aligner_ffn_mult,
                        protein_encoder_dim=config.hidden_size,
                    ),
                )
                if (layer_idx + 1) % max(config.num_hidden_layers // config.aligner_num_add_layers, 1) == 0
                else EvollaDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = EvollaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = EvollaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # self.use_cache = config.use_cache
        # self.use_return_dict = config.use_return_dict
        self.config = config

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        protein_feats: Optional[torch.FloatTensor] = None,
        structure_feats: Optional[torch.FloatTensor] = None,
        msa_feats: Optional[torch.FloatTensor] = None,
        protein_batch_mask: Optional[torch.Tensor] = None,
        structure_batch_mask: Optional[torch.Tensor] = None,
        msa_batch_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length, _ = inputs_embeds.shape
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_length_with_past = seq_length + past_key_values_length

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.int64, device=inputs_embeds.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

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

        for decoder_layer in self.layers:
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
                    protein_kv_states=protein_feats,
                    structure_kv_states=structure_feats,
                    msa_kv_states=msa_feats,
                    protein_batch_mask=protein_batch_mask,
                    structure_batch_mask=structure_batch_mask,
                    msa_batch_mask=msa_batch_mask,
                    query_attn_mask=attention_mask,
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
                    protein_kv_states=protein_feats,
                    structure_kv_states=structure_feats,
                    msa_kv_states=msa_feats,
                    protein_batch_mask=protein_batch_mask,
                    structure_batch_mask=structure_batch_mask,
                    msa_batch_mask=msa_batch_mask,
                    query_attn_mask=attention_mask,
                    **kwargs,
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


class EvollaModel(EvollaPreTrainedModel):
    r""" """

    def __init__(self, config: EvollaConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)

        self.protein_encoder = EvollaProteinEncoder(
            config=self.config,
            add_pooling_layer=False,
        )

        self.llm = EvollaLLM(
            config=self.config,
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # text input ids
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        inputs_embeds: Optional[torch.FloatTensor] = None,  # text input embeddings
        protein_input_ids: torch.LongTensor = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if protein_input_ids is None:
            raise ValueError("protein_input_ids is required")

        text_input_ids = input_ids
        text_attention_mask = attention_mask
        text_inputs_embeds = inputs_embeds

        # create batch mask for seqs
        protein_batch_mask = torch.tensor([True] * protein_input_ids.shape[0])

        protein_outputs = self.protein_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        text_outputs = self.llm(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            inputs_embeds=text_inputs_embeds,
            protein_feats=protein_outputs.sequence_compressor_output,
            protein_batch_mask=protein_batch_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if output_hidden_states:
            decoder_hidden_states = text_outputs.hidden_states
        else:
            decoder_hidden_states = None

        last_hidden_state = text_outputs.last_hidden_state

        if output_attentions:
            decoder_attentions = text_outputs.attentions
        else:
            decoder_attentions = None

        # change the output to BaseModelOutputWithPast
        output = BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            hidden_states=decoder_hidden_states,
            attentions=decoder_attentions,
        )
        return output if return_dict else output.to_tuple()

    def embed_tokens(self, input_ids, **kwargs):
        return self.llm.embed_tokens(input_ids, **kwargs)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)


# this was adapted from modeling_idefics.IdeficsForVisionText2Text
class EvollaForProteinText2Text(EvollaPreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.model = EvollaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # text input ids
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        inputs_embeds: Optional[torch.FloatTensor] = None,  # text input embeddings
        labels: Optional[torch.LongTensor] = None,
        protein_input_ids: torch.LongTensor = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Args:

        Returns:

        Example:

        ```python
        >>> from transformers import EvollaProcessor, EvollaForProteinText2Text
        >>> model = EvollaForProteinText2Text.from_pretrained("westlake/Evolla-10B-hf")
        >>> processor = EvollaProcessor.from_pretrained("westlake/Evolla-10B-hf")

        >>> protein_information = {
            "aa_seq": "your amino acid sequence",
            "foldseek": "your foldseek sequence",
        }
        >>> question = "What is the function of this protein?"
        >>> message = [
            {"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
            {"role": "user", "content": question},
        ]

        >>> inputs = processor(proteins=[protein_information], messages_list=[message], return_tensors="pt", padding="longest")
        >>> outputs = model.generate(**inputs)

        >>> print(processor.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        lm_outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return lm_outputs if return_dict else lm_outputs.to_tuple()


__all__ = ["EvollaForProteinText2Text", "EvollaModel", "EvollaPreTrainedModel"]

# coding=utf-8
# Copyright 2022, UCLA NLP, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch PLBART model."""

import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, is_torch_flex_attn_available
from ..bart.modeling_bart import (
    BartClassificationHead,
    BartDecoder,
    BartEncoder,
    BartForCausalLM,
    BartScaledWordEmbedding,
)
from ..bigbird_pegasus.modeling_bigbird_pegasus import BigBirdPegasusForSequenceClassification
from ..mbart.modeling_mbart import shift_tokens_right
from .configuration_plbart import PLBartConfig


if is_torch_flex_attn_available():
    from ...integrations.flex_attention import BlockMask, make_flex_block_causal_mask


class PLBartScaledWordEmbedding(BartScaledWordEmbedding):
    pass


@auto_docstring
class PLBartPreTrainedModel(PreTrainedModel):
    config: PLBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PLBartDecoderLayer", "PLBartEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Optional[Union[torch.Tensor, "BlockMask"]],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            # Other attention flavors support in-built causal (when `mask is None`)
            # while we need to create our specific block mask regardless
            elif attention_mask is None:
                attention_mask = make_flex_block_causal_mask(
                    torch.ones(
                        size=(input_tensor.shape[0], input_tensor.shape[1]),
                        device=attention_mask.device,
                    )
                )
            return attention_mask

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_cross_attn_mask
    def _update_cross_attn_mask(
        self,
        encoder_hidden_states: Union[torch.Tensor, None],
        encoder_attention_mask: Union[torch.Tensor, None],
        input_shape: torch.Size,
        inputs_embeds: torch.Tensor,
    ):
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(encoder_attention_mask, torch.Tensor):
                    encoder_attention_mask = make_flex_block_causal_mask(
                        encoder_attention_mask,
                        query_length=input_shape[-1],
                        is_causal=False,
                    )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        return encoder_attention_mask


class PLBartEncoder(BartEncoder):
    pass


class PLBartDecoder(BartDecoder):
    pass


@auto_docstring
class PLBartModel(PLBartPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: PLBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = PLBartScaledWordEmbedding(vocab_size, config.d_model, padding_idx, embed_scale=embed_scale)

        self.encoder = PLBartEncoder(config, self.shared)
        self.decoder = PLBartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`] or [`PLBartMultiTokenizer`] depending on the checkpoint.
            See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            PLBart uses a specific language id token as the starting token for `decoder_input_ids` generation that
            varies according to source and target language, *e.g.* 50003 for *en_XX*, and 50001 for *java*. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (:
            obj:*torch.LongTensor* of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior:
            generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
        cross_attn_head_mask (:
            obj:*torch.Tensor* of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify
            selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, PLBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The PLBART Model with a language modeling head. Can be used for code-to-text, text-to-code and code-to-code.
    """
)
class PLBartForConditionalGeneration(PLBartPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: PLBartConfig):
        super().__init__(config)
        self.model = PLBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None, mean_resizing: bool = True
    ) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`] or [`PLBartMultiTokenizer`] depending on the checkpoint.
            See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            PLBart uses a specific language id token as the starting token for `decoder_input_ids` generation that
            varies according to source and target language, *e.g.* 50003 for *en_XX*, and 50001 for *java*. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (:
            obj:*torch.LongTensor* of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior:
            generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
        cross_attn_head_mask (:
            obj:*torch.Tensor* of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify
            selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example Mask-filling:

        ```python
        >>> from transformers import AutoTokenizer, PLBartForConditionalGeneration

        >>> model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")

        >>> # en_XX is the language symbol id <LID> for English
        >>> TXT = "<s> Is 0 the <mask> Fibonacci number ? </s> en_XX"
        >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="pt").input_ids

        >>> logits = model(input_ids).logits
        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['first', 'same', 'highest', 'result', 'number']
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)


class PLBartClassificationHead(BartClassificationHead):
    pass


class PLBartForSequenceClassification(BigBirdPegasusForSequenceClassification):
    def forward(**super_kwargs):
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`] or [`PLBartMultiTokenizer`] depending on the checkpoint.
            See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            PLBart uses a specific language id token as the starting token for `decoder_input_ids` generation that
            varies according to source and target language, *e.g.* 50003 for *en_XX*, and 50001 for *java*. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For translation and summarization training, `decoder_input_ids` should be provided. If no
            `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
            for denoising pre-training following the paper.
        decoder_attention_mask (:
            obj:*torch.LongTensor* of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior:
            generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
        cross_attn_head_mask (:
            obj:*torch.Tensor* of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify
            selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        super().forward(**super_kwargs)


class PLBartForCausalLM(BartForCausalLM):
    @auto_docstring
    def forward(**super_kwargs):
        r"""
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PLBartForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
        >>> model = PLBartForCausalLM.from_pretrained("uclanlp/plbart-base", add_cross_attention=False)
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
        >>> list(logits.shape) == expected_shape
        True
        ```"""
        super().forward(**super_kwargs)


__all__ = [
    "PLBartForCausalLM",
    "PLBartForConditionalGeneration",
    "PLBartForSequenceClassification",
    "PLBartModel",
    "PLBartPreTrainedModel",
]

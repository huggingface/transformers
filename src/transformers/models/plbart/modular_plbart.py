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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import initialization as init
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
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

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, PLBartForConditionalGeneration):
            init.zeros_(module.final_logits_bias)


class PLBartEncoder(BartEncoder):
    pass


class PLBartDecoder(BartDecoder):
    pass


@auto_docstring
class PLBartModel(PLBartPreTrainedModel):
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }

    def __init__(self, config: PLBartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = PLBartScaledWordEmbedding(vocab_size, config.d_model, padding_idx, embed_scale=embed_scale)

        self.encoder = PLBartEncoder(config)
        self.decoder = PLBartDecoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: list[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | Seq2SeqModelOutput:
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
    _tied_weights_keys = {
        "lm_head.weight": "model.shared.weight",
    }

    def __init__(self, config: PLBartConfig):
        super().__init__(config)
        self.model = PLBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.post_init()

    def resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: int | None = None, mean_resizing: bool = True
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
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: list[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput:
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        super().forward(**super_kwargs)


class PLBartForCausalLM(BartForCausalLM):
    @auto_docstring
    def forward(**super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PLBartForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
        >>> model = PLBartForCausalLM.from_pretrained("uclanlp/plbart-base")
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

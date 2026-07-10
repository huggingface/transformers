# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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


from dataclasses import dataclass

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import CompileConfig, GenerationMixin
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutput,
    CausalLMOutputWithPast,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ..auto import AutoModel
from ..parakeet.modeling_parakeet import ParakeetCTCGenerateOutput, ParakeetForCTC
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2Encoder,
    Wav2Vec2EncoderLayer,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from .configuration_omniasr import OmniASRConfig, OmniASRCTCConfig, OmniASREncoderConfig


# Different from Wav2Vec2PositionalConvEmbedding: no weight norm, has residual, uses remove_pad instead of SamePadLayer
class OmniASRPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.remove_pad = config.num_conv_pos_embeddings % 2 == 0
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        if self.remove_pad:
            hidden_states = hidden_states[:, :, :-1]
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states + residual


# NOTE: need to overwrite config name
class OmniASRAttention(Wav2Vec2Attention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: OmniASREncoderConfig | None = None,
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias, is_causal, config)


class OmniASREncoderLayer(Wav2Vec2EncoderLayer):
    # NOTE: original: https://github.com/facebookresearch/fairseq2/blob/a1f0c565a99d3cd3e3157678b5c48653e3d439f4/src/fairseq2/models/transformer/encoder_layer.py#L141
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # Self-attention block with pre-norm (layer_norm_pre=True in config)
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)  # Pre-norm: normalize BEFORE attention
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states  # Add residual

        # FFN block with pre-norm
        ffn_residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)  # Pre-norm: normalize BEFORE FFN
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = ffn_residual + hidden_states  # Add residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class OmniASREncoder(Wav2Vec2Encoder):
    @can_return_tuple
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        # NOTE (ebezzam): residual and layer norm removed here (wrt Wav2Vec2Encoder)
        hidden_states = self.pos_conv_embed(hidden_states)
        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # NOTE (ebezzam): layer norm shifted here (wrt Wav2Vec2Encoder)
        hidden_states = self.layer_norm(hidden_states)
        if self.training:
            hidden_states = self.dropout(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class OmniASRPreTrainedModel(Wav2Vec2PreTrainedModel):
    config: OmniASREncoderConfig
    base_model_prefix = "model"
    _no_split_modules = ["OmniASREncoderLayer"]

    def _init_weights(self, module):
        # TODO upstream change to `Wav2Vec2PreTrainedModel` for standard layers?
        PreTrainedModel._init_weights(self, module)

    def _get_audio_encoder(self):
        # OmniASRForCTC exposes the audio encoder as `encoder`, OmniASRForConditionalGeneration as `model.encoder`.
        model = getattr(self, "model", None)
        return model.encoder if model is not None else self.encoder

    def apply_weight_norm(self, legacy=True):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            weight_norm = nn.utils.parametrizations.weight_norm

        pos_conv_embed = self._get_audio_encoder().encoder.pos_conv_embed
        pos_conv = pos_conv_embed.conv
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(pos_conv.weight, modifier_rank=0):
                weight_norm(pos_conv, name="weight", dim=2)
            if hasattr(pos_conv, "parametrizations"):
                weight_g = pos_conv.parametrizations.weight.original0
                weight_v = pos_conv.parametrizations.weight.original1
            else:
                weight_g = pos_conv.weight_g
                weight_v = pos_conv.weight_v
            deepspeed.zero.register_external_parameter(pos_conv_embed, weight_v)
            deepspeed.zero.register_external_parameter(pos_conv_embed, weight_g)
        else:
            weight_norm(pos_conv, name="weight", dim=2)

    def remove_weight_norm(self, legacy=True):
        remove_weight_norm = nn.utils.remove_weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            remove_weight_norm = torch.nn.utils.parametrize.remove_parametrizations

        # TODO deepspeed zero3 case
        remove_weight_norm(self._get_audio_encoder().encoder.pos_conv_embed.conv, name="weight")

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.LongTensor | int, add_adapter: bool = False
    ) -> torch.LongTensor | int:
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # OmniASRForCTC wraps an encoder_config; OmniASRSpeechEncoder uses the config directly.
        encoder_config = getattr(self.config, "encoder_config", self.config)
        for kernel_size, stride in zip(encoder_config.conv_kernel, encoder_config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_output_attention_mask(self, attention_mask: torch.Tensor, target_length: int | None = None):
        """
        Convert the input attention mask to its subsampled form. `target_length` sets the desired output length, useful
        when the attention mask length differs from `sum(-1).max()` (i.e., when the longest sequence in the batch is padded)
        """
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        # Use target_length if provided, otherwise use max length in batch
        max_length = target_length if target_length is not None else output_lengths.max()
        attention_mask = torch.arange(max_length, device=attention_mask.device) < output_lengths[:, None]
        return attention_mask


class OmniASRBaseModelOutput(Wav2Vec2BaseModelOutput):
    pass


class OmniASRCTCGenerateOutput(ParakeetCTCGenerateOutput):
    pass


@auto_docstring(
    custom_intro="""
    The OmniASR speech encoder, which is a Wav2Vec2-style encoder.
    """
)
class OmniASRSpeechEncoder(Wav2Vec2Model):
    def __init__(self, config: OmniASREncoderConfig):
        super().__init__(config)
        self.adapter = None


class OmniASRForCTC(ParakeetForCTC):
    def __init__(self, config: OmniASRCTCConfig):
        super().__init__(config)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)

    # NOTE: `input_values` is used instead of `input_features` (as we use audio values directly). Better way to do modular?
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-ctc-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> outputs = model(**inputs)

        >>> print(outputs.loss)
        ```"""

        if labels is not None:
            kwargs.setdefault("output_attention_mask", True)
        encoder_outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = encoder_outputs.last_hidden_state
        logits = self.ctc_head(hidden_states)

        loss = None
        if labels is not None:
            # TODO use attention mask from encoder (see updated Parakeet)
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            encoder_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100 when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    encoder_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict_in_generate: bool = False,
        compile_config: CompileConfig | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> OmniASRCTCGenerateOutput | torch.LongTensor:
        r"""
        compile_config ([`~generation.CompileConfig`], *optional*):
            If provided, `torch.compile` will be applied to the forward calls in the decoding loop.

        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForCTC
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-ctc-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForCTC.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
        >>> predicted_ids = model.generate(**inputs)
        >>> transcription = processor.decode(predicted_ids, skip_special_tokens=True)

        >>> print(transcription)
        ```
        """
        model_forward = self.get_compiled_call(compile_config) if compile_config is not None else self.__call__

        kwargs["return_dict"] = True
        outputs: CausalLMOutput = model_forward(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        # greedy decoding
        sequences = outputs.logits.argmax(dim=-1)

        # mask out padded tokens
        if attention_mask is not None:
            attention_mask = self._get_output_attention_mask(attention_mask, target_length=sequences.shape[1])
            sequences[~attention_mask] = self.config.pad_token_id

        if return_dict_in_generate:
            return OmniASRCTCGenerateOutput(
                sequences=sequences,
                logits=outputs.logits,
                attentions=outputs.attentions,
                hidden_states=outputs.hidden_states,
            )

        return sequences


@auto_docstring(
    custom_intro="""
    Base class for OmniASR outputs, with hidden states and attentions.
    """
)
@dataclass
class OmniASRModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        Projected audio hidden states.
    """

    audio_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    The OmniASR model, which consists of a Wav2Vec2 encoder, a multi-modal projector and a LLama language model,
    without a language modeling head.
    """
)
class OmniASRModel(OmniASRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = AutoModel.from_config(config.audio_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.audio_config.hidden_size * config.encoder_stacking,
            config.text_config.hidden_size,
            bias=True,
        )

        # TODO better handling
        self.language_token_id = config.language_token_id
        if config.num_special_tokens > 0:
            reserved_language_token_id = config.text_config.vocab_size - config.num_special_tokens
            if self.language_token_id < reserved_language_token_id:
                self.language_token_id = reserved_language_token_id
        self.lang_embeddings = nn.Embedding(config.num_language_embeddings, config.text_config.hidden_size)

        self.post_init()

    def get_audio_features(self, input_values: torch.FloatTensor):
        audio_outputs = self.encoder(input_values)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        return audio_embeds

    # Original: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/model.py#L141
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        # TODO better handling of language ids?
        language_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | OmniASRModelOutputWithPast:
        r"""
        language_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Language IDs for the input audio. If not provided, the model defaults to a language-agnostic mode.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_values is not None:
            # First step: build full audio context (audio | lid_marker | lang_id | bos)
            batch_size = input_values.size(0)

            audio_embeds = self.get_audio_features(input_values)
            dtype = audio_embeds.dtype

            text_embed_fn = self.get_input_embeddings()
            target_device = text_embed_fn.weight.device

            language_id_token_batch = torch.full(
                (batch_size, 1), self.language_token_id, dtype=torch.long, device=target_device
            )
            bos_batch = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=target_device)

            if language_ids is not None:
                language_id_batch = language_ids.to(target_device)
            else:
                language_id_batch = torch.zeros(batch_size, dtype=torch.long, device=target_device)

            if self.training and self.config.language_embedding_probability > 0.0:
                dropout_mask = torch.rand(batch_size, device=target_device) < (
                    1 - self.config.language_embedding_probability
                )
                language_id_batch = language_id_batch.clone()
                language_id_batch[dropout_mask] = 0

            lid_marker_embeds = text_embed_fn(language_id_token_batch).to(dtype)
            bos_embeds = text_embed_fn(bos_batch).to(dtype)
            lang_embeddings_device = self.lang_embeddings.weight.device
            lang_id_embeds = (
                self.lang_embeddings(language_id_batch.unsqueeze(-1).to(lang_embeddings_device))
                .to(dtype)
                .to(target_device)
            )

            inputs_embeds = torch.cat(
                [audio_embeds.to(target_device), lid_marker_embeds, lang_id_embeds, bos_embeds], dim=1
            )

            seq_len = inputs_embeds.shape[1]
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=inputs_embeds.device)

        if labels is not None:
            # Training: append target_text + eos embeddings for teacher forcing
            batch_size = input_values.size(0)
            device = input_values.device
            dtype = inputs_embeds.dtype
            text_embed_fn = self.get_input_embeddings()
            eos_batch = torch.full((batch_size, 1), self.config.eos_token_id, dtype=torch.long, device=device)
            target_embeds = text_embed_fn(labels).to(dtype)
            eos_embeds = text_embed_fn(eos_batch).to(dtype)
            inputs_embeds = torch.cat([inputs_embeds, target_embeds, eos_embeds], dim=1)

        # Build attention mask if not provided
        if attention_mask is None and past_key_values is None:
            batch_size = inputs_embeds.size(0)
            seq_len = inputs_embeds.size(1)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=inputs_embeds.device)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return OmniASRModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
        )


@auto_docstring(
    custom_intro="""
    OmniASR model, which consists of a Wav2Vec2 encoder, a multi-modal projector and a LLama language model.
    """
)
class OmniASRForConditionalGeneration(VoxtralForConditionalGeneration):
    config: OmniASRConfig
    main_input_name = "input_ids"
    _keep_in_fp32_modules_strict = AttributeError()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        # TODO better handling of language ids?
        language_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        language_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Language IDs for the input audio. If not provided, the model defaults to a language-agnostic mode.
        """
        outputs: OmniASRModelOutputWithPast = self.model(
            input_ids=input_ids,
            input_values=input_values,
            language_ids=language_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # TODO check
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

            # # Context length = audio_seq_len + 3 (lid_marker + lang_id + bos)
            # context_seq_len = hidden_states.size(1) - labels.size(1) - 1  # subtract target + eos
            # target_len = labels.size(1) + 1  # +1 for EOS
            # target_logits = logits[:, context_seq_len - 1 : context_seq_len - 1 + target_len, :]

            # batch_size = labels.size(0)
            # device = labels.device
            # eos_batch = torch.full((batch_size, 1), self.config.eos_token_id, dtype=torch.long, device=device)
            # targets = torch.cat([labels, eos_batch], dim=1)

            # loss = nn.functional.cross_entropy(
            #     input=target_logits.reshape(-1, target_logits.size(-1)),
            #     target=targets.reshape(-1),
            #     ignore_index=self.config.pad_token_id,
            #     reduction="mean",
            # )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, is_first_iteration=False, **kwargs):
        raise NotImplementedError("OmniASRForConditionalGeneration does not need this.")

    # TODO: should try to rely on base generate
    def generate(self, input_values=None, language_ids=None, **kwargs):
        """Generate token sequences from audio input."""
        if input_values is None:
            input_values = kwargs.pop("input_values", None)
        if language_ids is None:
            language_ids = kwargs.pop("language_ids", None)

        if input_values is not None:
            batch_size = input_values.size(0)

            audio_embeds = self.get_audio_features(input_values)
            dtype = audio_embeds.dtype
            text_embed_fn = self.get_input_embeddings()
            target_device = text_embed_fn.weight.device

            lid_marker_ids = torch.full(
                (batch_size, 1), self.model.language_token_id, dtype=torch.long, device=target_device
            )
            bos_ids = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=target_device)

            lid_marker_embeds = text_embed_fn(lid_marker_ids).to(dtype)
            bos_embeds = text_embed_fn(bos_ids).to(dtype)

            if language_ids is not None:
                language_id_batch = language_ids.to(target_device)
            else:
                language_id_batch = torch.zeros(batch_size, dtype=torch.long, device=target_device)

            lang_embeddings_device = self.model.lang_embeddings.weight.device
            lang_id_embeds = self.model.lang_embeddings(language_id_batch.unsqueeze(-1).to(lang_embeddings_device)).to(
                dtype, target_device
            )

            inputs_embeds = torch.cat(
                [audio_embeds.to(target_device), lid_marker_embeds, lang_id_embeds, bos_embeds], dim=1
            )
            kwargs.pop("attention_mask", None)

            return GenerationMixin.generate(inputs_embeds=inputs_embeds, **kwargs)

        return GenerationMixin.generate(**kwargs)


__all__ = [
    "OmniASRForCTC",
    "OmniASRForConditionalGeneration",
    "OmniASRModel",
    "OmniASRSpeechEncoder",
    "OmniASRPreTrainedModel",
]

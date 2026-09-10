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


import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import CompileConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    CausalLMOutput,
    CausalLMOutputWithPast,
)
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_compilable_check,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..parakeet.modeling_parakeet import (
    ParakeetCTCGenerateOutput,
    ParakeetEncoderModelOutput,
    ParakeetForCTC,
    ParakeetPreTrainedModel,
)
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration, VoxtralModel, VoxtralModelOutputWithPast
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2EncoderLayer,
    Wav2Vec2LayerNormConvLayer,
)
from .configuration_omniasr import OmniASRConfig, OmniASRCTCConfig, OmniASREncoderConfig


# NOTE: Simplified version of Wav2Vec2PositionalConvEmbedding
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
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        position_embeddings = hidden_states.transpose(1, 2)
        position_embeddings = self.conv(position_embeddings)
        # Instead of `Wav2Vec2SamePadLayer`, in-line removal of padding
        position_embeddings = position_embeddings[:, :, :-1]
        position_embeddings = self.activation(position_embeddings)
        return position_embeddings.transpose(1, 2)


class OmniASRAttention(Wav2Vec2Attention):
    pass


# NOTE: original: https://github.com/facebookresearch/fairseq2/blob/a1f0c565a99d3cd3e3157678b5c48653e3d439f4/src/fairseq2/models/transformer/encoder_layer.py#L141
class OmniASREncoderLayer(Wav2Vec2EncoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)  # Pre-norm: normalize BEFORE attention
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states  # Add residual

        # FFN block with pre-norm
        ffn_residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)  # Pre-norm: normalize BEFORE FFN
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = ffn_residual + hidden_states  # Add residual

        return hidden_states


class OmniASRLayerNormConvLayer(Wav2Vec2LayerNormConvLayer):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        self.activation = ACT2FN[config.hidden_act]


# NOTE: similar to `ParakeetEncoderSubsamplingConv2D` but for 1D, and a replacement for `Wav2Vec2FeatureEncoder` and `Wav2Vec2FeatureProjection`
class OmniASRSubsamplingConv1D(nn.Module):
    def __init__(self, config: OmniASREncoderConfig):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [OmniASRLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden_states = input_values[:, None]

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.layer_norm(hidden_states)
        return self.projection(hidden_states)


@auto_docstring
class OmniASRPreTrainedModel(ParakeetPreTrainedModel):
    main_input_name = "input_values"
    _supports_flash_attn = True
    _no_split_modules = None
    _can_record_outputs = None

    def _init_weights(self, module):
        raise AttributeError("Normal super call")

    def _get_subsampling_output_length(self, input_lengths: torch.LongTensor | int) -> torch.LongTensor | int:
        """Computes the output length of the convolutional layers."""

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # OmniASRForCTC wraps an encoder_config; OmniASREncoder uses the config directly.
        encoder_config = getattr(self.config, "encoder_config", self.config)
        for kernel_size, stride in zip(encoder_config.conv_kernel, encoder_config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths


class OmniASREncoderModelOutput(ParakeetEncoderModelOutput):
    pass


class OmniASRCTCGenerateOutput(ParakeetCTCGenerateOutput):
    pass


@auto_docstring(
    custom_intro="""
    The OmniASR speech encoder, which is a Wav2Vec2-style encoder.
    """
)
class OmniASREncoder(OmniASRPreTrainedModel):
    config: OmniASREncoderConfig
    base_model_prefix = "encoder"
    _no_split_modules = ["OmniASREncoderLayer"]
    _can_record_outputs = {
        "attentions": OmniASRAttention,
        "hidden_states": OmniASREncoderLayer,
    }

    def __init__(self, config: OmniASREncoderConfig):
        super().__init__(config)

        self.gradient_checkpointing = False

        self.hidden_dropout = config.hidden_dropout
        self.layerdrop = config.layerdrop

        self.subsampling = OmniASRSubsamplingConv1D(config)
        self.encode_positions = OmniASRPositionalConvEmbedding(config)
        self.layers = nn.ModuleList([OmniASREncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        output_attention_mask: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> OmniASREncoderModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the subsampled attention mask. Only effective when `attention_mask` is provided.
        """
        hidden_states = self.subsampling(input_values)

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(attention_mask, target_length=hidden_states.shape[1])
            attention_mask = output_mask
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~output_mask[..., None], 0.0)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        position_embeddings = self.encode_positions(hidden_states)
        hidden_states = hidden_states + position_embeddings

        for encoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(hidden_states, attention_mask=attention_mask, **kwargs)

        # NOTE (ebezzam): layer norm applied after the layers (wrt Wav2Vec2Encoder, which applies it before)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.hidden_dropout, training=self.training)

        return OmniASREncoderModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask.int() if output_attention_mask and output_mask is not None else None,
        )


class OmniASRForCTC(ParakeetForCTC):
    def __init__(self, config: OmniASRCTCConfig):
        super().__init__(config)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)

    # NOTE: `input_values` is used instead of `input_features` as we use audio values directly
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
            encoder_lengths = encoder_outputs.attention_mask.sum(-1)

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


class OmniASRModelOutputWithPast(VoxtralModelOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    The OmniASR model, which consists of a Wav2Vec2 encoder, a multi-modal projector and a LLama language model,
    without a language modeling head.
    """
)
class OmniASRModel(VoxtralModel):
    # The base class is annotated and driven for the speech encoder; this one is a multimodal decoder model,
    # configured by `OmniASRConfig` and prompted through `input_ids`.
    config: OmniASRConfig
    main_input_name = "input_ids"
    input_modalities = ("audio", "text")

    def __init__(self, config):
        super().__init__(config)
        self.multi_modal_projector = nn.Linear(
            config.audio_config.hidden_size * config.encoder_stacking,
            config.text_config.hidden_size,
            bias=True,
        )
        self.lang_embeddings = nn.Embedding(config.num_language_embeddings, config.text_config.hidden_size)

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Encodes the raw audio waveform with the speech encoder and projects it into the language model's embedding
        space. Padding frames are dropped, so `pooler_output` holds exactly the frames that the audio placeholders
        of `input_ids` stand for, in row-major order.
        """
    )
    def get_audio_features(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            Float values of the raw audio waveform, as produced by [`OmniASRFeatureExtractor`].
        padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Mask to avoid running the speech encoder over padding samples of `input_values`. Values selected in
            `[0, 1]`: 1 for samples that are **not** padding, 0 for padding.
        """
        audio_outputs = self.audio_tower(input_values, attention_mask=padding_mask, **kwargs)
        audio_embeds = self.multi_modal_projector(audio_outputs.last_hidden_state)

        # `masked_scatter` consumes the features in order, and a padded sample carries as many placeholders as it
        # has valid frames, so the padding frames are dropped here.
        frames_mask = audio_outputs.attention_mask
        audio_embeds = audio_embeds.flatten(0, 1) if frames_mask is None else audio_embeds[frames_mask.bool()]

        return BaseModelOutputWithPooling(
            last_hidden_state=audio_outputs.last_hidden_state,
            pooler_output=audio_embeds,
            hidden_states=audio_outputs.hidden_states,
            attentions=audio_outputs.attentions,
        )

    # TODO remove
    def get_language_features(
        self, language_ids: torch.LongTensor | None, batch_size: int, device: torch.device
    ) -> torch.FloatTensor:
        r"""
        Looks up one language embedding per audio input. Index 0 is the language-agnostic entry, which is both what a
        missing `language_ids` falls back to and what the language is dropped to during training, with probability
        `config.language_embedding_probability`.
        """
        if language_ids is None:
            language_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if self.training and self.config.language_embedding_probability > 0.0:
            dropout_mask = torch.rand(batch_size, device=language_ids.device) < (
                1 - self.config.language_embedding_probability
            )
            language_ids = language_ids.masked_fill(dropout_mask, 0)

        return self.lang_embeddings(language_ids)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        audio_features: torch.FloatTensor | None = None,
        language_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_audio_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_audio_mask = special_audio_mask.all(-1)
            special_language_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.language_embedding_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_language_mask = special_language_mask.all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id
            special_language_mask = input_ids == self.config.language_embedding_token_id

        n_audio_tokens = special_audio_mask.sum()
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if audio_features is not None:
            torch_compilable_check(
                inputs_embeds[special_audio_mask].numel() == audio_features.numel(),
                f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features: {audio_features.shape[0]}",
            )

        n_language_tokens = special_language_mask.sum()
        special_language_mask = special_language_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if language_features is not None:
            torch_compilable_check(
                inputs_embeds[special_language_mask].numel() == language_features.numel(),
                f"Language features and language tokens do not match, tokens: {n_language_tokens}, features: {language_features.shape[0]}",
            )

        return special_audio_mask, special_language_mask

    # Original: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/model.py#L141
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        language_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | OmniASRModelOutputWithPast:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Float values of the raw audio waveform. The decoder context is `audio | lid_marker | language | bos`, so
            `input_ids` holds one [`OmniASRConfig.audio_token_id`] placeholder per encoder frame and one
            [`OmniASRConfig.language_embedding_token_id`] placeholder, both filled in here. Use
            [`OmniASRProcessor.__call__`] to build them.
        padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Mask to avoid running the speech encoder over padding samples of `input_values`, with 1 for samples that
            are **not** padding. Distinct from `attention_mask`, which covers the decoder sequence.
        language_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Index into the language embedding table for each audio input, as produced by
            [`OmniASRProcessor.__call__`] from its `language` argument. Defaults to the language-agnostic entry (0).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_values is not None:
            audio_embeds = self.get_audio_features(input_values, padding_mask, return_dict=True).pooler_output
            language_embeds = self.get_language_features(language_ids, inputs_embeds.shape[0], inputs_embeds.device)

            special_audio_mask, special_language_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                audio_features=audio_embeds,
                language_features=language_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_audio_mask, audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_language_mask, language_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            )

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
    input_modalities = ("audio", "text")
    _keep_in_fp32_modules_strict = AttributeError()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
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
        input_values (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Float values of the raw audio waveform, scattered over the audio placeholders of `input_ids`. See
            [`OmniASRModel.forward`].
        padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*):
            Mask to avoid running the speech encoder over padding samples of `input_values`, with 1 for samples that
            are **not** padding. Distinct from `attention_mask`, which covers the decoder sequence.
        language_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Index into the language embedding table for each audio input, as produced by
            [`OmniASRProcessor.__call__`] from its `language` argument. Defaults to the language-agnostic entry (0).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the causal language modeling loss. They are shifted internally, so they must be
            aligned with the decoder sequence -- i.e. as long as `input_ids` / `inputs_embeds`, with `-100` on the
            positions that should not contribute to the loss (the audio context, and padding).

        Example:

        ```python
        >>> from transformers import AutoProcessor, OmniASRForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> model_id = "bezzam/omniasr-llm-300m-v2"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = OmniASRForConditionalGeneration.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        >>> inputs = processor(ds[0]["audio"]["array"], language="eng_Latn")
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> transcription = processor.decode(generated_ids, skip_special_tokens=True)
        ```"""
        outputs: OmniASRModelOutputWithPast = self.model(
            input_ids=input_ids,
            input_values=input_values,
            padding_mask=padding_mask,
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
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, use_cache=True, is_first_iteration=False, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, use_cache=use_cache, is_first_iteration=is_first_iteration, **kwargs
        )

        if not is_first_iteration and use_cache:
            # The audio context is encoded once, during prefill; afterwards the cache carries it.
            model_inputs["input_values"] = None
            model_inputs["padding_mask"] = None
            model_inputs["language_ids"] = None

        return model_inputs


__all__ = [
    "OmniASRForCTC",
    "OmniASRForConditionalGeneration",
    "OmniASRModel",
    "OmniASREncoder",
    "OmniASRPreTrainedModel",
]

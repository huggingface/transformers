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
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..parakeet.modeling_parakeet import ParakeetCTCGenerateOutput, ParakeetForCTC
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration, VoxtralModel
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention,
    Wav2Vec2Encoder,
    Wav2Vec2EncoderLayer,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2LayerNormConvLayer,
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
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Self-attention block with pre-norm (layer_norm_pre=True in config)
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)  # Pre-norm: normalize BEFORE attention
        hidden_states, _, _ = self.attention(
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
    pass


class OmniASRFeatureEncoder(Wav2Vec2FeatureEncoder):
    # OmniASR always layer-norms the convolutions (`feature_extractor_layer_norm_convs=True` upstream), so the
    # group-norm variant inherited from Wav2Vec2 is dropped.
    def __init__(self, config):
        nn.Module.__init__(self)
        self.conv_layers = nn.ModuleList(
            [OmniASRLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        )
        self.gradient_checkpointing = False
        self._requires_grad = True


class OmniASRFeatureProjection(Wav2Vec2FeatureProjection):
    pass


class OmniASREncoder(Wav2Vec2Encoder):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
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
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                hidden_states = layer(hidden_states, attention_mask=attention_mask, **kwargs)

        # NOTE (ebezzam): layer norm shifted here (wrt Wav2Vec2Encoder)
        hidden_states = self.layer_norm(hidden_states)
        if self.training:
            hidden_states = self.dropout(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class OmniASRPreTrainedModel(PreTrainedModel):
    config: OmniASREncoderConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _no_split_modules = ["OmniASREncoderLayer"]

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor | int) -> torch.LongTensor | int:
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


@auto_docstring(
    custom_intro="""
    Extends [`~modeling_outputs.Wav2Vec2BaseModelOutput`] with the output attention mask, since the convolutional
    feature encoder does not preserve the input sequence length.
    """
)
@dataclass
class OmniASRBaseModelOutput(Wav2Vec2BaseModelOutput):
    r"""
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on padding frames, subsampled to the encoder's output length. Returned
        because that length differs from the length of the input `attention_mask`. Mask values selected in `[0, 1]`:

        - 1 for frames that are **not masked**,
        - 0 for frames that are **masked**.
    """

    attention_mask: torch.Tensor | None = None


class OmniASRCTCGenerateOutput(ParakeetCTCGenerateOutput):
    pass


@auto_docstring(
    custom_intro="""
    The OmniASR speech encoder, which is a Wav2Vec2-style encoder.
    """
)
class OmniASRSpeechEncoder(OmniASRPreTrainedModel):
    _can_record_outputs = {
        "attentions": OmniASRAttention,
        "hidden_states": OmniASREncoderLayer,
    }

    def __init__(self, config: OmniASREncoderConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = OmniASRFeatureEncoder(config)
        self.feature_projection = OmniASRFeatureProjection(config)
        self.encoder = OmniASREncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        output_attention_mask: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> OmniASRBaseModelOutput:
        r"""
        output_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the subsampled attention mask. Only effective when `attention_mask` is provided.

        Encode raw audio into hidden states with the convolutional feature encoder followed by the transformer encoder.
        """
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        output_mask = None
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            output_mask = self._get_output_attention_mask(attention_mask, target_length=extract_features.shape[1])
            attention_mask = output_mask

        hidden_states, extract_features = self.feature_projection(extract_features)

        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, **kwargs)

        return OmniASRBaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            extract_features=extract_features,
            attention_mask=output_mask.int() if output_mask is not None and output_attention_mask else None,
        )


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
            # the encoder already subsampled the input mask; an unpadded batch attends every output frame
            if encoder_outputs.attention_mask is not None:
                encoder_lengths = encoder_outputs.attention_mask.sum(-1)
            else:
                encoder_lengths = torch.full(
                    (logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device
                )

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
class OmniASRModel(VoxtralModel):
    def __init__(self, config):
        super().__init__(config)
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


    def get_placeholder_mask(
            self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, audio_features: torch.FloatTensor
        ):
        raise NotImplementedError("TODO replace build_audio_context with this function, which will be used in the decoder to mask out the audio context")

    def get_audio_features(self, input_values: torch.FloatTensor, attention_mask: torch.Tensor | None = None):
        audio_outputs = self.audio_tower(input_values, attention_mask=attention_mask)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        return audio_embeds

    def _build_audio_context(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        language_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.FloatTensor]:
        """
        Build the decoder context `audio | lid_marker | lang_id | bos` and its attention mask.

        The original implementation packs each sample's markers directly after its own last valid audio frame. Here
        the audio frames are left-padded to do the same, so every sequence in the batch ends with `bos` and the
        relative distance between the audio and the markers does not depend on how much the batch was padded.
        """
        audio_embeds = self.get_audio_features(input_values, attention_mask=attention_mask)
        batch_size, audio_length, hidden_size = audio_embeds.shape
        dtype = audio_embeds.dtype

        text_embed_fn = self.get_input_embeddings()
        target_device = text_embed_fn.weight.device
        audio_embeds = audio_embeds.to(target_device)

        lid_marker_ids = torch.full((batch_size, 1), self.language_token_id, dtype=torch.long, device=target_device)
        bos_ids = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=target_device)

        if language_ids is None:
            language_ids = torch.zeros(batch_size, dtype=torch.long, device=target_device)
        else:
            language_ids = language_ids.to(target_device)
        if self.training and self.config.language_embedding_probability > 0.0:
            dropout_mask = torch.rand(batch_size, device=target_device) < (
                1 - self.config.language_embedding_probability
            )
            language_ids = language_ids.masked_fill(dropout_mask, 0)

        lid_marker_embeds = text_embed_fn(lid_marker_ids).to(dtype)
        bos_embeds = text_embed_fn(bos_ids).to(dtype)
        lang_id_embeds = self.lang_embeddings(language_ids.unsqueeze(-1).to(self.lang_embeddings.weight.device)).to(
            dtype=dtype, device=target_device
        )

        if attention_mask is None:
            audio_attention_mask = torch.ones(batch_size, audio_length, dtype=torch.long, device=target_device)
        else:
            # Shift each sample right by the amount of audio padding it carries, so its valid frames end at
            # `audio_length` and the markers below follow immediately after them.
            audio_lengths = self.audio_tower._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(target_device)
            indices = torch.arange(audio_length, device=target_device) - (audio_length - audio_lengths)[:, None]
            audio_attention_mask = indices >= 0
            gather_indices = indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden_size)
            audio_embeds = audio_embeds.gather(1, gather_indices) * audio_attention_mask.unsqueeze(-1)
            audio_attention_mask = audio_attention_mask.to(torch.long)

        inputs_embeds = torch.cat([audio_embeds, lid_marker_embeds, lang_id_embeds, bos_embeds], dim=1)
        attention_mask = torch.cat(
            [audio_attention_mask, torch.ones(batch_size, 3, dtype=torch.long, device=target_device)], dim=1
        )
        return inputs_embeds, attention_mask, audio_embeds

    # Original: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/model.py#L141
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_values: torch.Tensor | None = None,
        language_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | OmniASRModelOutputWithPast:
        r"""
        language_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Index into the language embedding table for each audio input, as produced by
            [`OmniASRProcessor.__call__`] from its `language` argument. Defaults to the language-agnostic entry (0).
        """

        if input_values is None and input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify one of `input_values`, `input_ids` or `inputs_embeds`.")

        audio_embeds = None
        if input_values is not None:
            # First step: build full audio context (audio | lid_marker | lang_id | bos). This is the whole
            # decoder context, so `input_ids` / `inputs_embeds` are not read when `input_values` is given.
            inputs_embeds, attention_mask, audio_embeds = self._build_audio_context(
                input_values, attention_mask=attention_mask, language_ids=language_ids
            )

        elif inputs_embeds is None:
            # Subsequent decoding steps: the newly generated tokens are passed as `input_ids`.
            inputs_embeds = self.get_input_embeddings()(input_ids)

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
            Index into the language embedding table for each audio input, as produced by
            [`OmniASRProcessor.__call__`] from its `language` argument. Defaults to the language-agnostic entry (0).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the causal language modeling loss. They are shifted internally, so they must be
            aligned with the decoder sequence -- i.e. as long as `input_ids` / `inputs_embeds`, with `-100` on the
            positions that should not contribute to the loss (the audio context, and padding).
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

    # Bypasses Voxtral's override, which forwards `input_features` on the first decoding step: OmniASR has no such
    # input. The signature has to name `inputs_embeds` explicitly, since `generate` inspects it to decide whether
    # the model can be driven from embeddings.
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ):
        return GenerationMixin.prepare_inputs_for_generation(self, input_ids, inputs_embeds=inputs_embeds, **kwargs)

    # TODO avoid this override, and use `prepare_inputs_for_generation` instead?
    # The audio is the entire decoder context, so it is turned into `inputs_embeds` here and the rest of the
    # decoding loop -- including `prepare_inputs_for_generation` -- is the standard one.
    def generate(self, input_values=None, language_ids=None, attention_mask=None, **kwargs):
        """Generate token sequences from audio input."""
        if input_values is None:
            input_values = kwargs.pop("input_values", None)
        if language_ids is None:
            language_ids = kwargs.pop("language_ids", None)

        if input_values is not None:
            # The audio context is left-padded, so `bos` is the last position of every sequence and decoding
            # continues from there for the whole batch.
            inputs_embeds, attention_mask, _ = self.model._build_audio_context(
                input_values, attention_mask=attention_mask, language_ids=language_ids
            )
            return GenerationMixin.generate(self, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

        return GenerationMixin.generate(self, attention_mask=attention_mask, **kwargs)


__all__ = [
    "OmniASRForCTC",
    "OmniASRForConditionalGeneration",
    "OmniASRModel",
    "OmniASRSpeechEncoder",
    "OmniASRPreTrainedModel",
]

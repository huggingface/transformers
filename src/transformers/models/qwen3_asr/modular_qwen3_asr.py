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

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioPreTrainedModel
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    SinusoidsPositionEmbedding,
    _get_feat_extract_output_lengths,
)
from ..whisper.modeling_whisper import WhisperAttention, WhisperEncoderLayer


@auto_docstring(checkpoint="bezzam/Qwen3-ASR-1.7B")
@strict
class Qwen3ASREncoderConfig(Qwen2_5OmniAudioEncoderConfig):
    r"""
    max_source_positions (`int`, *optional*, defaults to 1500):
        The maximum sequence length that this model might ever be used with.
    n_window (`int`, *optional*, defaults to 50):
        Half the number of mel frames in one encoder chunk. Each chunk processed by the conv stack has
        ``2 * n_window`` mel frames (1 second of audio at 16 kHz with a 10 ms hop).
    n_window_infer (`int`, *optional*, defaults to 800):
        Number of mel frames worth of audio over which each attention window spans. Must be a multiple
        of ``n_window * 2`` so attention windows align with encoder chunks.
    downsample_hidden_size (`int`, *optional*, defaults to 480):
        Hidden size of the convolutional downsampling stack.
    output_dim (`int`, *optional*, defaults to 3584):
        Dimensionality of the output.
    """

    model_type = "qwen3_asr_audio_encoder"

    n_window: int = 50
    n_window_infer: int = 800
    downsample_hidden_size: int = 480
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024


@auto_docstring(checkpoint="bezzam/Qwen3-ASR-1.7B")
@strict
class Qwen3ASRConfig(PreTrainedConfig):
    r"""
    audio_token_id (`int`, *optional*, defaults to 151676):
        The audio token id to encode the audio prompt.

    Example:

    ```python
    >>> from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRConfig

    >>> # Initializing a Qwen3ASR style configuration
    >>> configuration = Qwen3ASRConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr"
    sub_configs = {"audio_config": AutoConfig, "text_config": AutoConfig}

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151676
    pad_token_id: int = 151645
    eos_token_id: list[int] | tuple[int, ...] | int = (151643, 151645)
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "qwen3_asr_audio_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["qwen3_asr_audio_encoder"](
                encoder_layers=24,
                encoder_attention_heads=16,
                encoder_ffn_dim=4096,
                d_model=1024,
                output_dim=2048,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"](
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=65536,
                tie_word_embeddings=True,
            )

        super().__post_init__(**kwargs)


@auto_docstring
class Qwen3ASRPreTrainedModel(Qwen2AudioPreTrainedModel):
    _no_split_modules = ["Qwen3ASREncoderLayer", "Qwen3DecoderLayer"]
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, SinusoidsPositionEmbedding):
            log_timescale_increment = np.log(module.max_timescale) / (module.channels // 2 - 1)
            inv_timescales = torch.exp(-log_timescale_increment * torch.arange(module.channels // 2).float())
            scaled_time = torch.arange(module.length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
            init.copy_(
                module.positional_embedding,
                torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            )


# NOTE (ebezzam): Whisper sets bias=False for self.k_proj, which differs from original Qwen3 ASR: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L472
# but does not make a difference since softmax is invariant to constant offsets in the logits
class Qwen3ASRAttention(WhisperAttention):
    pass


class Qwen3ASREncoderLayer(WhisperEncoderLayer):
    pass


@auto_docstring(
    custom_intro="""
    The audio model for Qwen3 ASR without any head or projection on top.
    """
)
class Qwen3ASREncoder(Qwen3OmniMoeAudioEncoder):
    config: Qwen3ASREncoderConfig
    _no_split_modules = ["Qwen3ASREncoderLayer"]
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": Qwen3ASREncoderLayer,
        "attentions": Qwen3ASRAttention,
    }

    def __init__(self, config: Qwen3ASREncoderConfig):
        super().__init__(config)
        del self.conv_chunksize
        self.layers = nn.ModuleList([Qwen3ASREncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def _post_cnn_length(lengths: torch.Tensor) -> torch.Tensor:
        """Length after three (k=3, s=2, p=1) convolutions; zero-length input stays zero."""
        for _ in range(3):
            lengths = torch.where(lengths > 0, (lengths - 1) // 2 + 1, torch.zeros_like(lengths))
        return lengths

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, padded_feature_length)`):
                Log-mel features. `padded_feature_length` must be a multiple of `self.n_window * 2`.
            input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`):
                1 for valid mel frames and 0 for padding.
        """
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        chunk_len = self.n_window * 2
        num_chunks = padded_feature_length // chunk_len

        chunked = (
            input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
        )

        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, time_steps = conv_out.size()
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, conv_channels * freq_bins)
        )
        conv_out = conv_out + self.positional_embedding.positional_embedding[:time_steps, :].to(conv_out.dtype)
        chunk_embeds = conv_out.view(batch_size, num_chunks, time_steps, -1)

        # Mask out post-cnn positions that came from zero-padded mel frames.
        chunk_mel_lens = input_features_mask.view(batch_size, num_chunks, chunk_len).sum(dim=-1)
        chunk_post_cnn_lens = self._post_cnn_length(chunk_mel_lens)
        post_cnn_positions = torch.arange(time_steps, device=input_features.device)
        valid_post_cnn_mask = post_cnn_positions[None, None, :] < chunk_post_cnn_lens[:, :, None]
        sequence_length = num_chunks * time_steps
        hidden_states = chunk_embeds.reshape(batch_size, sequence_length, -1)
        sequence_mask = valid_post_cnn_mask.reshape(batch_size, sequence_length).to(dtype=torch.long)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=sequence_mask,
        )

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask=attention_mask, **kwargs)
            hidden_states = hidden_states * sequence_mask.to(hidden_states.dtype).unsqueeze(-1)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class Qwen3ASRModel(Qwen3ASRPreTrainedModel):
    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.audio_tower = Qwen3ASREncoder(config.audio_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to get the audio embeddings from input features (a log mel spectrogram)."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`):
            1 for valid mel frames and 0 for padding.
        """
        audio_output = self.audio_tower(
            input_features=input_features,
            input_features_mask=input_features_mask,
            **kwargs,
        )
        audio_embeds = audio_output.last_hidden_state
        input_lengths = input_features_mask.sum(-1).to(torch.long)
        audio_token_lengths = _get_feat_extract_output_lengths(input_lengths, self.config.audio_config.n_window)
        valid_mask = (
            torch.arange(audio_embeds.shape[1], device=audio_embeds.device)[None, :] < audio_token_lengths[:, None]
        )
        audio_output.pooler_output = audio_embeds[valid_mask]
        return audio_output

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        input_features_mask (`torch.LongTensor` of shape `(batch_size, padded_feature_length)`):
            1 for valid mel frames and 0 for padding.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask, return_dict=True).pooler_output

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return outputs


@auto_docstring(
    custom_intro="""
    The Qwen3ASR model which consists of an audio encoder and a language model.
    """
)
class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.model = Qwen3ASRModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        """
        return self.model.get_audio_features(
            input_features=input_features,
            input_features_mask=input_features_mask,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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

    def prepare_inputs_for_generation(self, *args, is_first_iteration: bool = False, **kwargs):
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if is_first_iteration or not model_inputs.get("use_cache", False):
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask

        return model_inputs


@auto_docstring(checkpoint="bezzam/Qwen3-ForcedAligner-0.6B")
@strict
class Qwen3ForcedAlignerConfig(Qwen3ASRConfig):
    r"""
    num_timestamp_bins (`int`, *optional*, defaults to 5000):
        Number of discrete timestamp bins the model can predict. Each bin corresponds
        to a time offset of ``timestamp_segment_time`` milliseconds (set on the processor),
        so the maximum representable duration is ``num_timestamp_bins * timestamp_segment_time`` ms
        (e.g. 5000 * 80 ms = 400 s).
    timestamp_token_id (`int`, *optional*, defaults to 151705):
        Token ID of the ``<timestamp>`` marker in the tokenizer vocabulary. These markers
        delimit word boundaries in the forced-alignment input sequence.

    Example:

    ```python
    >>> from transformers import Qwen3ASRForForcedAlignment, Qwen3ForcedAlignerConfig

    >>> # Initializing a Qwen3ForcedAligner style configuration
    >>> configuration = Qwen3ForcedAlignerConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ASRForForcedAlignment(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_forced_aligner"

    num_timestamp_bins: int = 5000
    timestamp_token_id: int = 151705


@auto_docstring(
    custom_intro="""
    The Qwen3 Forced Aligner model which consists of an audio encoder, a language model backbone,
    and a token classification head for forced alignment.
    """
)
class Qwen3ASRForForcedAlignment(Qwen3ASRPreTrainedModel):
    config_class = Qwen3ForcedAlignerConfig

    def __init__(self, config: Qwen3ForcedAlignerConfig):
        super().__init__(config)
        self.num_timestamp_bins = config.num_timestamp_bins
        self.model = Qwen3ASRModel(config)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.num_timestamp_bins, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_audio_features(
            input_features=input_features,
            input_features_mask=input_features_mask,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the forced alignment loss. Indices should be in `[0, ..., config.num_timestamp_bins - 1]`.
        """

        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.num_timestamp_bins)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3ASREncoderConfig",
    "Qwen3ASRConfig",
    "Qwen3ASREncoder",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRModel",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ForcedAlignerConfig",
    "Qwen3ASRForForcedAlignment",
]

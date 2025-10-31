# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from typing import Optional

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...masking_utils import eager_mask, padding_mask_function
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder, Qwen2AudioPreTrainedModel
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration, VoxtralMultiModalProjector
from ..whisper.modeling_whisper import WhisperAttention, WhisperEncoderLayer
from .configuration_audioflamingo3 import AudioFlamingo3Config


logger = logging.get_logger(__name__)


class AudioFlamingo3Attention(WhisperAttention):
    pass


class AudioFlamingo3EncoderLayer(WhisperEncoderLayer):
    pass


class AudioFlamingo3PreTrainedModel(Qwen2AudioPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    The audio model from AudioFlamingo3 without any head or projection on top.
    """
)
class AudioFlamingo3Encoder(Qwen2AudioEncoder):
    """
    AudioFlamingo3 encoder: Whisper encoder, average pool (time/2), then LayerNorm.
    """

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Log-Mel features extracted from raw audio. Use the processor/feature extractor to compute and pad
                these features from waveform input.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, S, S)`, *optional*):
                Unlike Whisper, the attention_mask is used within the encoder layers. They represent
                pre-pool attention masks on the time axis, with `0` on valid positions and `-inf` on
                padded positions (added to attention logits). If `None`, full attention is used. Here `S` is the
                sequence length after the conv front-end (typically `ceil(T_mel/2)`).
        """

        # Conv front-end
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # Add positions, dropout
        embed_pos = self.embed_positions.weight
        hidden_states = (inputs_embeds + embed_pos).to(inputs_embeds.dtype)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Transformer stack
        for layer in self.layers:
            drop = self.training and torch.rand([]) < self.layerdrop
            if not drop:
                hidden_states = layer(hidden_states, attention_mask)[0]

        # AvgPool (time/2) + LayerNorm
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states).permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class AudioFlamingo3MultiModalProjector(VoxtralMultiModalProjector):
    """
    Audio adaptor (small MLP) that projects AudioFlamingo3Encoder features
    to the LLM embedding space so they can replace `<sound>` tokens.
    """

    def __init__(self, config: AudioFlamingo3Config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )


@auto_docstring(
    custom_intro="""
    The AudioFlamingo3 model which consists of a fine-tuned Whisper encoder, a multi-modal projector and a Qwen2 language model.
    """
)
class AudioFlamingo3ForConditionalGeneration(VoxtralForConditionalGeneration):
    def get_audio_features(
        self, input_features: torch.FloatTensor, input_features_mask: torch.Tensor
    ) -> torch.FloatTensor:
        """
        This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector.
        Args:
            input_features (`torch.FloatTensor`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
                Mask to avoid performing attention on padded feature indices.

        Returns:
            `torch.FloatTensor`:
                The audio embeddings.
        """
        # Prepare attention mask for transformer layers
        batch_size = input_features.shape[0]
        seq_len = (input_features.shape[-1] - 1) // 2 + 1  # After conv2 downsampling
        encoder_attention_mask = eager_mask(
            batch_size=batch_size,
            cache_position=torch.arange(seq_len, device=input_features.device),
            kv_length=seq_len,
            mask_function=padding_mask_function(input_features_mask),
            dtype=self.audio_tower.conv1.weight.dtype,
        )

        # Encode audio
        encoder_output = self.audio_tower(input_features, attention_mask=encoder_attention_mask)
        audio_embeds = self.multi_modal_projector(encoder_output.last_hidden_state)

        # Mask according to avg pooling (which is after attention blocks)
        post_lengths = (input_features_mask.sum(-1) - 2) // 2 + 1
        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_embeds = audio_embeds[valid_mask.to(audio_embeds.device)]
        return audio_embeds

    def get_audio_embeds(self):
        raise NotImplementedError("This method is not supported for AudioFlamingo3ForConditionalGeneration.")

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        >>> MODEL_ID = "nvidia/audio-flamingo-3"
        >>> processor = AutoProcessor.from_pretrained(MODEL_ID)
        >>> model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto").eval()

        >>> conversations = [
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {"type": "text", "text": "Transcribe the input speech."},
        >>>                 {"type": "audio", "path": "audio_1.wav"},
        >>>             ],
        >>>         }
        >>>     ],
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {"type": "text", "text": "Describe the song."},
        >>>                 {"type": "audio", "path": "audio_2.wav"},
        >>>             ],
        >>>         }
        >>>     ]
        >>> ]

        >>> batch = processor.apply_chat_template(
        >>>     conversations,
        >>>     tokenize=True,
        >>>     add_generation_prompt=True,
        >>>     sampling_rate=getattr(processor.feature_extractor, "sampling_rate", 16000),
        >>> ).to(model.device)

        >>> gen_ids = model.generate(**batch, max_new_tokens=512)

        >>> inp_len = batch["input_ids"].shape[1]
        >>> new_tokens = gen_ids[:, inp_len:]
        >>> texts = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(texts)
        ["Transcription of the input speech: Good morning everyone...", "The song is an orchestral piece..."]
        ```"""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask)

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        logits = outputs[0]

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage

        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask

        return model_inputs


__all__ = ["AudioFlamingo3ForConditionalGeneration", "AudioFlamingo3PreTrainedModel", "AudioFlamingo3Encoder"]

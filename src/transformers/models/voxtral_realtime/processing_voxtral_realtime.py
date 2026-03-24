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

from ...tokenization_mistral_common import MistralCommonBackend
from ...utils import auto_docstring, is_mistral_common_available, is_soundfile_available, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    pass

if is_soundfile_available():
    pass

if is_mistral_common_available():
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.chunk import RawAudio
    from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack


logger = logging.get_logger(__name__)


class VoxtralRealtimeProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
        },
    }


@auto_docstring
@requires(backends=("mistral-common",))
class VoxtralRealtimeProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(tokenizer, MistralCommonBackend):
            raise ValueError("`tokenizer` must be a `MistralCommonBackend` tokenizer.")

        super().__init__(feature_extractor, tokenizer)

        if feature_extractor.win_length != self.mistral_common_audio_config.encoding_config.window_size:
            raise ValueError(
                f"feature_extractor.win_length ({feature_extractor.win_length}) "
                f"and tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config.window_size "
                f"({self.mistral_common_audio_config.encoding_config.window_size}) must be equal"
            )

        if feature_extractor.hop_length != self.mistral_common_audio_config.encoding_config.hop_length:
            raise ValueError(
                f"feature_extractor.hop_length ({feature_extractor.hop_length}) "
                f"and tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config.hop_length "
                f"({self.mistral_common_audio_config.encoding_config.hop_length}) must be equal"
            )

        if feature_extractor.feature_size != self.mistral_common_audio_config.encoding_config.num_mel_bins:
            raise ValueError(
                f"feature_extractor.feature_size ({feature_extractor.feature_size}) "
                f"and tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config.num_mel_bins "
                f"({self.mistral_common_audio_config.encoding_config.num_mel_bins}) must be equal"
            )

        if feature_extractor.sampling_rate != self.mistral_common_audio_config.sampling_rate:
            raise ValueError(
                f"feature_extractor.sampling_rate ({feature_extractor.sampling_rate}) "
                f"and tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config.sampling_rate "
                f"({self.mistral_common_audio_config.sampling_rate}) must be equal"
            )

    @property
    def mistral_common_audio_config(self):
        return self.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.audio_config

    @property
    def num_delay_tokens(self):
        return self.mistral_common_audio_config.num_delay_tokens

    @property
    def num_right_pad_tokens(self):
        return self.mistral_common_audio_config.n_right_pad_tokens

    @property
    def audio_length_per_tok(self):
        return self.mistral_common_audio_config.audio_length_per_tok

    @property
    def raw_audio_length_per_tok(self):
        return self.mistral_common_audio_config.raw_audio_length_per_tok

    @property
    def num_mel_frames_first_audio_chunk(self):
        # it is actually num_left_pad_tokens + num_delay_tokens + 1
        # but the call to `encode_transcription` will add the left pad token
        num_prefill_tokens = self.num_delay_tokens + 1
        num_prefill_mel_frames = num_prefill_tokens * self.audio_length_per_tok
        return num_prefill_mel_frames

    @property
    def num_samples_first_audio_chunk(self) -> int:
        num_prefill_mel_frames = self.num_mel_frames_first_audio_chunk
        num_prefill_audio_samples = (
            num_prefill_mel_frames - 1
        ) * self.feature_extractor.hop_length + self.feature_extractor.win_length // 2

        return num_prefill_audio_samples

    @property
    def num_samples_per_audio_chunk(self) -> int:
        return self.audio_length_per_tok * self.feature_extractor.hop_length + self.feature_extractor.win_length

    def __call__(
        self,
        audio: AudioInput | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool | None = True,
        **kwargs: Unpack[VoxtralRealtimeProcessorKwargs],
    ):
        r"""
        Main method to prepare audio input for the Voxtral Realtime model. This method encodes the audio into
        a transcription request using `mistral_common`, tokenizes the resulting text, and extracts mel spectrogram
        features using the feature extractor. Supports both streaming and non-streaming modes.

        Args:
            audio (`AudioInput`, *optional*):
                Input audio or batch of audios as NumPy arrays or PyTorch tensors.
            is_streaming (`bool`, *optional*, defaults to `False`):
                Whether to process audio in streaming mode. When `True`, audio can be passed in chunks
                using `is_first_audio_chunk` to distinguish the first chunk from subsequent ones.
            is_first_audio_chunk (`bool`, *optional*, defaults to `True`):
                Whether the current audio is the first chunk in a streaming session. When `True`, the audio
                is encoded into a full transcription request with tokenized text. When `False`, only audio
                features are extracted (text encoding is skipped). Must be `True` when `is_streaming=False`.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to the model. Returned when `is_first_audio_chunk=True`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
              Returned when `is_first_audio_chunk=True`.
            - **input_features** -- Mel spectrogram features extracted from the audio input.
            - **num_delay_tokens** -- The number of delay tokens used for streaming.
        """
        output_kwargs = self._merge_kwargs(VoxtralRealtimeProcessorKwargs, **kwargs)

        if not is_streaming and not is_first_audio_chunk:
            raise ValueError("In non-streaming mode (`is_streaming=False`), `is_first_audio_chunk` must be `True`.")

        audio = make_list_of_audio(audio)
        input_ids, texts, audio_arrays = [], [], []
        if is_first_audio_chunk:
            for audio_el in audio:
                # NOTE: format here is used only for serialization and therefore we can use wav for any audio array
                audio = Audio(
                    audio_array=audio_el, sampling_rate=output_kwargs["audio_kwargs"]["sampling_rate"], format="wav"
                )
                transcription_request = TranscriptionRequest(
                    audio=RawAudio.from_audio(audio),
                    streaming=StreamingMode.ONLINE if is_streaming else StreamingMode.OFFLINE,
                    language=None,
                )
                tokenized_transcription_request = self.tokenizer.tokenizer.encode_transcription(transcription_request)

                input_ids.append(tokenized_transcription_request.tokens)
                texts.append(tokenized_transcription_request.text)
                audio_arrays.extend([el.audio_array for el in tokenized_transcription_request.audios])

                text_encoding = self.tokenizer(input_ids, **output_kwargs["text_kwargs"])
        else:
            # when not the first audio chunk, we only encode audio
            audio_arrays = audio
            text_encoding = {}

        audio_encoding = self.feature_extractor(
            audio_arrays,
            center=is_first_audio_chunk,
            **output_kwargs["audio_kwargs"],
        )

        encoding = {**text_encoding, **audio_encoding, "num_delay_tokens": self.num_delay_tokens}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return BatchFeature(data=encoding, tensor_type=return_tensors)


__all__ = ["VoxtralRealtimeProcessor"]

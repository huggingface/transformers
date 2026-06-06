# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


from tokenizers.decoders import DecodeStream

from ...audio_utils import AudioInput, make_list_of_audio
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class ParakeetProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "longest",
            "return_attention_mask": True,
            "subsampling_factor": 8,
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class ParakeetProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer, blank_token="<blank>", decoder_type=None):
        r"""
        blank_token (`str`, *optional*, defaults to `"<blank>"`):
            Blank token for transducer decoding.
        decoder_type (`str`, *optional*):
            Decoding/timestamp emission mode. Possible values:
            
            - `"ctc"`: Consecutive identical tokens are merged into one emission.
            - `"rnnt"`: Repeated tokens are kept; each token gets a 1-frame timestamp span.
            - `"tdt"`: Repeated tokens are kept; each token span is based on its predicted duration. Punctuation is attached to the preceding token.
            
            If `None` (older checkpoints) the decoder type is inferred automatically for backward compatibility.
        """ 
        self.blank_token = blank_token
        self.blank_token_id = tokenizer.convert_tokens_to_ids(blank_token)
        self.decoder_type = decoder_type
        super().__init__(feature_extractor, tokenizer)

    @property
    def _decoder_type(self):
        if self.decoder_type is not None:
            return self.decoder_type
        # BC: CTC and TDT checkpoints pushed to the hub before `decoder_type` existed, so it is unset for them.
        # If decoder_type is not specified, use TDT when there is no blank token, otherwise CTC; 
        # if it is specified, use the provided decoder type
        return "ctc" if self.blank_token not in self.tokenizer.get_vocab() else "tdt"

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        sampling_rate: int | None = None,
        **kwargs: Unpack[ParakeetProcessorKwargs],
    ):
        r"""
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. This should match the sampling rate expected by the feature
            extractor (defaults to 16000 Hz). If provided, it will be validated against the processor's expected
            sampling rate, and an error will be raised if they don't match. If not provided, a warning will be
            issued and the default sampling rate will be assumed.
        """
        audio = make_list_of_audio(audio)

        output_kwargs = self._merge_kwargs(
            ParakeetProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if sampling_rate is None:
            logger.warning_once(
                f"You've provided audio without specifying the sampling rate. It will be assumed to be {output_kwargs['audio_kwargs']['sampling_rate']}, which can result in silent errors."
            )
        elif sampling_rate != output_kwargs["audio_kwargs"]["sampling_rate"]:
            raise ValueError(
                f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate of the processor ({output_kwargs['audio_kwargs']['sampling_rate']}). Please provide resampled the audio to the expected sampling rate."
            )

        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        else:
            inputs["labels"] = encodings["input_ids"]
            # Prepend blank token to labels to form decoder_input_ids.
            # The TDT decoder expects [blank, label_0, ..., label_{U-1}] as input,
            if isinstance(text, str):
                text = [text]
            decoder_text = [self.blank_token + t for t in text]
            decoder_encodings = self.tokenizer(decoder_text, **output_kwargs["text_kwargs"])
            inputs["decoder_input_ids"] = decoder_encodings["input_ids"]
            return inputs

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["labels", "decoder_input_ids"]

    def batch_decode(self, *args, **kwargs):
        kwargs.setdefault("group_tokens", self._decoder_type == "ctc")
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, durations=None, **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.decode`] and post-process the timestamps (if provided for TDT) as
        in the NeMo library.
        """
        kwargs.setdefault("group_tokens", self._decoder_type == "ctc")
        decoded = self.tokenizer.decode(*args, **kwargs)

        if durations is not None:
            token_ids = args[0]
            # Derive per-step frame indices from cumulative sum of durations.
            timestamps = durations.cumsum(dim=-1) - durations

            output_kwargs = self._merge_kwargs(
                ParakeetProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            )
            frame_rate = (
                self.feature_extractor.hop_length
                / self.feature_extractor.sampling_rate
                * output_kwargs["audio_kwargs"]["subsampling_factor"]
            )
            # Filter padding/blank tokens and decode per sequence to keep track of token-level timestamps
            # See `compute_rnnt_timestamps` in NeMo:
            # https://github.com/NVIDIA-NeMo/NeMo/blob/1692a8fb97e1aadc883cfadd2a57c4e8a1b793aa/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L993
            skip_ids = {self.tokenizer.pad_token_id, self.blank_token_id}
            proc_timestamps = []
            for batch_ids, batch_timestamps, batch_durations in zip(token_ids, timestamps, durations):
                stream = DecodeStream(skip_special_tokens=True)
                timestamp_dict = []
                for i, token_id in enumerate(batch_ids):
                    if int(token_id) in skip_ids:
                        continue
                    chunk = stream.step(self.tokenizer._tokenizer, int(token_id))
                    if chunk is not None:
                        # TDT sizes a token by its predicted duration; RNN-T tokens each span a single frame
                        # (their per-step value is a 0/1 encoder advance, not a span).
                        token_span = int(batch_durations[i]) if self._decoder_type == "tdt" else 1
                        start = int(batch_timestamps[i])
                        timestamp_dict.append(
                            {
                                "token": chunk,
                                "start": start,
                                "end": start + token_span,
                            }
                        )
                proc_timestamps.append(self._refine_timestamps(timestamp_dict, frame_rate))

            return decoded, proc_timestamps
        return decoded

    def _refine_timestamps(
        self, char_offsets, frame_rate, supported_punctuation=["?", "'", "¡", "¿", "-", ":", ",", "%", "/", ".", "!"]
    ):
        for i, offset in enumerate(char_offsets):
            # Convert frame indices to seconds
            offset["start"] = offset["start"] * frame_rate
            offset["end"] = offset["end"] * frame_rate

            # If token is a punctuation mark, set its start and end offset as start and end of previous token.
            # This is part of the TDT timestamp post-processing; RNN-T mirrors NeMo's raw char-level timestamps,
            # which keep every token (punctuation included) at its own emitted frame, so it is skipped there.
            if self._decoder_type == "tdt" and offset["token"] in supported_punctuation and i > 0:
                offset["start"] = char_offsets[i - 1]["end"]
                offset["end"] = offset["start"]

        return char_offsets


__all__ = ["ParakeetProcessor"]

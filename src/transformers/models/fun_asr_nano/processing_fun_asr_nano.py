# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Processor for Fun-ASR-Nano."""

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class FunAsrNanoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
        },
        "text_kwargs": {
            "padding": True,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class FunAsrNanoProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "FunAsrNanoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|object_ref_start|>",
        audio_downsample_rate=1,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The token used as a placeholder for audio in the text.
        audio_downsample_rate (`int`, *optional*, defaults to 1):
            Downsampling ratio applied by the audio adaptor, used to expand the audio placeholder token to the right
            number of audio tokens.
        """
        if tokenizer is not None and tokenizer.convert_tokens_to_ids(audio_token) is None:
            raise ValueError(f"Audio token {audio_token!r} is not present in the tokenizer vocabulary.")

        self.audio_token = audio_token
        self.audio_downsample_rate = audio_downsample_rate
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        sampling_rate: int | None = None,
        **kwargs: Unpack[FunAsrNanoProcessorKwargs],
    ) -> BatchFeature:
        r"""
        sampling_rate (`int`, *optional*):
            Sampling rate of the input audio. Must be 16000 for Fun-ASR-Nano.
        """
        if text is None:
            raise ValueError("You need to specify `text` input to process.")

        output_kwargs = self._merge_kwargs(
            FunAsrNanoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]
        return_tensors = text_kwargs.pop("return_tensors", None)

        text = list(text) if isinstance(text, list) else [text]

        audio_features = None
        if audio is not None:
            audio = make_list_of_audio(audio)
            audio_features = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate or audio_kwargs.get("sampling_rate"),
                return_tensors=return_tensors,
            )

            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} "
                    f"in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}."
                )

            # Expand each audio placeholder into as many tokens as the (downsampled) audio feature length.
            audio_lengths = audio_features["feature_lengths"].tolist()
            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    num_tokens = (audio_length - 1) // self.audio_downsample_rate + 1
                    replace_str.append(self.audio_token * int(num_tokens))
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **text_kwargs)

        if audio_features is not None:
            return BatchFeature(data={**text_inputs, **audio_features})

        return BatchFeature(data=dict(text_inputs))

    # `decode` and `batch_decode` are inherited from `ProcessorMixin` and forward to the tokenizer; the base `decode`
    # already handles batches, so no custom override is needed here.

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(feature_extractor_input_names + tokenizer_input_names))


__all__ = ["FunAsrNanoProcessor"]

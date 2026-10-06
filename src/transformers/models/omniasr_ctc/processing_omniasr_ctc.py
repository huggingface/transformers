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

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring


class OmniASRCTCProcessorKwargs(ProcessingKwargs, total=False):  # trf-ignore: TRF019
    _defaults = {
        "audio_kwargs": {"sampling_rate": 16000},
        "text_kwargs": {"padding": True},
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class OmniASRCTCProcessor(ProcessorMixin):
    valid_processor_kwargs = OmniASRCTCProcessorKwargs

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[OmniASRCTCProcessorKwargs],
    ) -> BatchFeature:
        r"""
        text (`str`, `list[str]`, *optional*):
            The transcription(s), one per audio input, from which the CTC `labels` are built.

        Returns:
            [`BatchFeature`]: `input_values` and its `padding_mask`, and `labels` whenever `text` is given.
        """
        output_kwargs = self._merge_kwargs(
            OmniASRCTCProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            labels = encodings["input_ids"]
            # Mask padding positions with -100 so the CTC loss ignores them.
            if "attention_mask" in encodings:
                labels[encodings["attention_mask"] == 0] = -100
            inputs["labels"] = labels

        return inputs

    def decode(self, *args, **kwargs):
        # CTC decoding collapses runs of identical tokens.
        kwargs.setdefault("group_tokens", True)
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # The transcription is only tokenized into the CTC `labels`.
        return self.feature_extractor.model_input_names + ["labels"]


__all__ = ["OmniASRCTCProcessor"]

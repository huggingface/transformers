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

from ...audio_utils import AudioInput, make_list_of_audio
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires

if is_torch_available():
    import torch


LANGUAGES = {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}
_NO_SPACE_LANGS = {"ja", "zh"}


logger = logging.get_logger(__name__)


class CohereAsrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "longest",
            "return_attention_mask": True,
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
@requires(backends=("torch",))
class CohereAsrProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def get_decoder_prompt_ids(self, language: str, punctuation: bool = True) -> list[int]:
        """Build the decoder prompt token IDs for the given language and punctuation settings."""
        if language not in LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language!r}. Supported languages: {', '.join(sorted(LANGUAGES))}."
            )
        pnc_token = "<|pnc|>" if punctuation else "<|nopnc|>"
        tokens = [
            "▁",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            f"<|{language}|>",
            f"<|{language}|>",
            pnc_token,
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        language: str,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        punctuation: bool = True,
        sampling_rate: int | None = None,
        **kwargs: Unpack[CohereAsrProcessorKwargs],
    ):
        r"""
        language (`str`):
            Language code (e.g. `"en"`, `"es"`, `"fr"`) used to build the decoder prompt. The processor
            constructs the full decoder prompt and returns `decoder_input_ids` alongside the audio features.
        punctuation (`bool`, defaults to `True`):
            Whether to enable punctuation in the decoder prompt.
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. This should match the sampling rate expected by the feature
            extractor (defaults to 16000 Hz). If provided, it will be validated against the processor's expected
            sampling rate, and an error will be raised if they don't match. If not provided, a warning will be
            issued and the default sampling rate will be assumed.
        """
        audio = make_list_of_audio(audio)

        output_kwargs = self._merge_kwargs(
            CohereAsrProcessorKwargs,
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

        inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        prompt_ids = self.get_decoder_prompt_ids(language=language, punctuation=punctuation)
        batch_size = inputs["input_features"].shape[0]
        inputs["decoder_input_ids"] = torch.tensor([prompt_ids] * batch_size, dtype=torch.long)

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            inputs["labels"] = encodings["input_ids"]

        return inputs

    def decode(self, *args, audio_chunk_index=None, language=None, **kwargs):
        texts = self.tokenizer.decode(*args, **kwargs)
        if audio_chunk_index is None:
            return texts
        if language is None:
            raise ValueError("`language` must be provided when `audio_chunk_index` is given.")
        separator = "" if language in _NO_SPACE_LANGS else " "
        return self._reassemble_chunk_texts(texts, audio_chunk_index, separator)

    @staticmethod
    def _reassemble_chunk_texts(
        texts: list[str],
        audio_chunk_index: list[tuple[int, int | None]],
        separator: str = " ",
    ) -> list[str]:
        """Reassemble per-chunk transcription texts back into per-sample strings.

        When audio inputs are longer than the feature extractor's `max_audio_clip_s`, they are split into
        overlapping chunks before being fed to the model. This means a single original audio sample can
        produce multiple decoded text segments. This method reverses that chunking: it groups the decoded
        texts by their original sample index using `chunk_map`, orders the chunks, and joins them
        with `separator` to reconstruct one transcription string per input sample.

        Args:
            texts: Decoded text strings, one per model output (i.e. one per chunk).
            audio_chunk_index: List of `(sample_idx, chunk_idx)` tuples that map each entry in
                `texts` back to its original sample and chunk position. A `chunk_idx` of `None`
                indicates the sample was not chunked.
            separator: String used to join chunks belonging to the same sample. Defaults to a
                space; callers pass an empty string for languages that don't use spaces between
                words (e.g. Chinese, Japanese).

        Returns:
            A list of reassembled transcription strings, one per original input sample.
        """
        max_sample_idx = max(sample_idx for sample_idx, _ in audio_chunk_index)
        outputs = [""] * (max_sample_idx + 1)
        chunked = {}

        for (sample_idx, chunk_idx), text in zip(audio_chunk_index, texts):
            if chunk_idx is None:
                outputs[sample_idx] = text
            else:
                if sample_idx not in chunked:
                    chunked[sample_idx] = []
                chunked[sample_idx].append((chunk_idx, text))

        for sample_idx, chunk_items in chunked.items():
            chunk_items.sort(key=lambda item: item[0])
            non_empty = [t for _, t in chunk_items if t and t.strip()]
            parts = [non_empty[0].rstrip()] + [t.strip() for t in non_empty[1:]]
            outputs[sample_idx] = separator.join(parts)

        return outputs

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["labels"]


__all__ = ["CohereAsrProcessor"]

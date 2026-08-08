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


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

# `language_mapping` indexes the language embedding table from 1, leaving row 0 for the language-agnostic mode.
LANGUAGE_AGNOSTIC = "auto"
LANGUAGE_AGNOSTIC_ID = 0


class OmniASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class OmniASRProcessor(ProcessorMixin):
    valid_processor_kwargs = OmniASRProcessorKwargs

    def __init__(self, feature_extractor, tokenizer, language_mapping=None, group_tokens=None):
        r"""
        language_mapping (`dict[str, int]`, *optional*):
            Mapping from a language code (e.g. `"eng_Latn"`) to its index in the model's language embedding table.
            Only the LLM variant is language-conditioned, so only its checkpoints carry a mapping; when it is
            `None`, `language` cannot be resolved and no `language_ids` are produced.
        group_tokens (`bool`, *optional*):
            Whether [`~OmniASRProcessor.decode`] collapses runs of identical tokens. This is what CTC decoding
            requires, and what the autoregressive LLM variant must not do. Defaults to `True` for the CTC variant
            and `False` for the LLM variant.
        """
        super().__init__(feature_extractor, tokenizer)
        self.language_mapping = language_mapping
        if group_tokens is None:
            # Checkpoints converted before `group_tokens` was explicit are recognised by carrying a mapping.
            group_tokens = language_mapping is None
        self.group_tokens = group_tokens

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        language: str | list[str] = "auto",
        sampling_rate: int | None = None,
        **kwargs: Unpack[OmniASRProcessorKwargs],
    ):
        """
        Processes audio input and optionally text/language for OmniASR models.

        For the CTC variant, pass `audio` (and optionally `text` for training labels).
        For the LLM variant, pass `audio` and `language` (e.g., `["eng_Latn"]`).

        Args:
            audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, *optional*):
                The audio input, passed to the feature extractor.
            text (`str`, `list[str]`, *optional*):
                Text input, passed to the tokenizer (used for training labels).
            language (`str` or `list[str]`, *optional*, defaults to `"auto"`):
                Language code(s) for the LLM variant (e.g. `"eng_Latn"` or `["eng_Latn", "fra_Latn"]`), resolved
                into the `language_ids` model input via `language_mapping`. Either a single code applied to the
                whole batch, or one per audio. `"auto"` selects the model's language-agnostic mode; naming the
                language explicitly gives better transcription quality. Ignored by the CTC variant, which is not
                language-conditioned.
            sampling_rate (`int`, *optional*):
                The sampling rate of the audio input. Will warn if not provided.

        Returns:
            [`BatchFeature`]: A dictionary-like object with `input_values` and optionally
            `attention_mask`, `language_ids`, and `labels`.
        """
        audio = make_list_of_audio(audio)

        output_kwargs = self._merge_kwargs(
            OmniASRProcessorKwargs,
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

        # Only the LLM variant is language-conditioned, and only its checkpoints ship a mapping.
        if self.language_mapping is not None:
            inputs["language_ids"] = self._resolve_language_ids(language, len(audio))
        elif language != LANGUAGE_AGNOSTIC:
            logger.warning_once(
                f"`language={language!r}` is ignored: this processor has no `language_mapping`, so the model it "
                "belongs to is not language-conditioned."
            )

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            labels = encodings["input_ids"]
            # Mask padding positions with -100 so the CTC loss ignores them.
            # (pad_token_id=0 satisfies labels >= 0, which would otherwise inflate target_lengths.)
            if "attention_mask" in encodings:
                labels[encodings["attention_mask"] == 0] = -100
            inputs["labels"] = labels

        return inputs

    def decode(self, *args, **kwargs):
        # CTC decoding collapses runs of identical tokens; the autoregressive LLM variant must keep them.
        kwargs.setdefault("group_tokens", self.group_tokens)
        return self.tokenizer.decode(*args, **kwargs)

    def _resolve_language_ids(self, language: str | list[str], batch_size: int) -> "torch.LongTensor":
        if not is_torch_available():
            raise ImportError("Resolving `language` into `language_ids` requires PyTorch. Please install PyTorch.")
        if isinstance(language, str):
            language = [language] * batch_size
        if len(language) == 1 and batch_size > 1:
            # A single language code is broadcast to every audio sample in the batch.
            language = list(language) * batch_size
        if len(language) != batch_size:
            raise ValueError(f"Received {len(language)} `language` entries for {batch_size} audio input(s).")

        language_ids = []
        for lang in language:
            key = lang.lower()
            if key == LANGUAGE_AGNOSTIC:
                language_ids.append(LANGUAGE_AGNOSTIC_ID)
            elif key in self.language_mapping:
                language_ids.append(self.language_mapping[key])
            else:
                raise ValueError(
                    f"Unknown `language={lang!r}`. Pass {LANGUAGE_AGNOSTIC!r} for the language-agnostic mode, or one "
                    f"of the {len(self.language_mapping)} codes in `language_mapping`, e.g. "
                    f"{sorted(self.language_mapping)[:5]}."
                )
        return torch.tensor(language_ids, dtype=torch.long)

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        language_input_names = ["language_ids"] if self.language_mapping is not None else []
        return feature_extractor_input_names + language_input_names + ["labels"]


__all__ = ["OmniASRProcessor"]

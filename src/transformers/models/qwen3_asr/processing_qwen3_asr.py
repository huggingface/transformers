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

import re
import unicodedata

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import TextInput


class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
            "n_window": 50,  # should match config.n_window
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


def _get_feat_extract_output_lengths(input_lengths, n_window=50):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    chunk_len = n_window * 2
    input_lengths_leave = input_lengths % chunk_len
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // chunk_len) * 13
    return output_lengths


class Qwen3ASRProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen3ASR processor.
    [`Qwen3ASRProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen3ASRProcessor.__call__`] and [`~Qwen3ASRProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
        timestamp_segment_time (`int`, *optional*, defaults to 80):
            The segment time in milliseconds used for grouping timestamps during forced alignment. This should match the
            value used during training of the forced aligner model.
    """

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None, timestamp_segment_time: int = 80):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.timestamp_segment_time = timestamp_segment_time
        self.audio_token = self.tokenizer.audio_token
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.audio_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput,
        output_labels: bool | None = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare one or several text sequence(s) and audio waveform(s) for the model.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded.
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Must be as many ``text``
                inputs as ``audio`` inputs.
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training.
        """
        call_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]

        audio = make_list_of_audio(audio)
        if len(text) != len(audio):
            raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

        # Prepare audio
        data = self.feature_extractor(audio, **audio_kwargs)
        data["input_features_mask"] = data.pop("attention_mask")

        # Replace audio tokens in text
        audio_lengths = (
            _get_feat_extract_output_lengths(data["input_features_mask"].sum(-1), audio_kwargs["n_window"])
            .cpu()
            .numpy()
        )
        audio_token_pattern = re.compile(re.escape(self.audio_token))
        for sample_idx, num_tokens in enumerate(audio_lengths):
            text[sample_idx] = audio_token_pattern.sub(self.audio_token * int(num_tokens), text[sample_idx])

        # Prepare text
        text_inputs = self.tokenizer(text, **text_kwargs)
        data.update(text_inputs)

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @staticmethod
    def _normalize_audio(audio: AudioInput) -> list:
        """Normalize audio input(s) into a flat list."""
        if isinstance(audio, str):
            return [audio]
        if isinstance(audio, (list, tuple)) and audio and all(isinstance(a, str) for a in audio):
            return list(audio)
        return make_list_of_audio(audio)

    @staticmethod
    def _normalize_languages(
        language: str | list[str] | None, batch_size: int, allow_broadcast: bool = False
    ) -> list[str | None]:
        """Broadcast / validate a language argument to match batch_size."""
        if language is None:
            return [None] * batch_size
        if isinstance(language, str):
            return [language] * batch_size
        if isinstance(language, (list, tuple)):
            if allow_broadcast and len(language) == 1 and batch_size > 1:
                return list(language) * batch_size
            if len(language) != batch_size:
                raise ValueError(f"Got {len(language)} language(s) for {batch_size} sample(s); counts must match.")
            return list(language)
        raise TypeError("`language` must be a string, a list of strings, or `None`.")

    @staticmethod
    def _audio_content_item(audio_item) -> dict:
        """Build a chat-template content dict for a single audio item."""
        if isinstance(audio_item, str):
            return {"type": "audio", "path": audio_item}
        return {"type": "audio", "audio": audio_item}

    def apply_transcription_request(
        self,
        audio: AudioInput | list[AudioInput],
        language: str | list[str] | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the chat template.

        Args:
            audio (`AudioInput` or `list[AudioInput]`):
                Audio to transcribe. Can be a URL string, local path, numpy array, or a list of these.
            language (`str` or `list[str]`, *optional*):
                Language hint(s) to include in the system prompt (e.g. "English", "Chinese").
                A list must be the same length as the audio batch.
                When `None`, the model performs automatic language detection.
            **kwargs:
                Additional keyword arguments forwarded to
                [`~Qwen3ASRProcessor.apply_chat_template`].

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`Qwen3ASRForConditionalGeneration.generate`].
        """
        audio_items = self._normalize_audio(audio)
        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")
        languages = self._normalize_languages(language, batch_size)

        conversations = []
        for lang, audio_item in zip(languages, audio_items):
            messages = []
            if lang is not None:
                messages.append({"role": "system", "content": [{"type": "text", "text": lang}]})
            messages.append({"role": "user", "content": [self._audio_content_item(audio_item)]})
            conversations.append(messages)

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def decode(self, *args, return_format="raw", **kwargs):
        """
        Forward arguments to the tokenizer's decode and optionally parse the ASR output.

        Qwen3 ASR outputs transcription in the format: ``language <LANG><asr_text>transcribed text``

        Args:
            return_format (`str`, *optional*, defaults to `"raw"`):
                Options:

                - ``"raw"``: Return raw decoded strings from the tokenizer.
                - ``"parsed"``: Return a dict (or list of dicts) with ``"language"`` and ``"transcription"`` keys.
                - ``"transcription_only"``: Extract only the transcribed text (after ``<asr_text>``).

                ``skip_special_tokens`` is hard-set to ``True`` for ``"parsed"`` and ``"transcription_only"``.
        """
        valid_formats = ["raw", "parsed", "transcription_only"]
        if return_format not in valid_formats:
            raise ValueError(f"return_format must be one of {valid_formats}.")
        if return_format != "raw":
            kwargs["skip_special_tokens"] = True

        decoded = self.tokenizer.decode(*args, **kwargs)
        if return_format == "parsed":
            decoded = self.parse_output(decoded)
        elif return_format == "transcription_only":
            decoded = self.extract_transcription(decoded)
        return decoded

    @staticmethod
    def _parse_single_output(text: str) -> dict:
        """Parse a single decoded ASR string into language + transcription."""
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[-1]
        marker = "<asr_text>"
        if marker not in text:
            return {"language": None, "transcription": text}
        prefix, transcription = text.split(marker, 1)
        prefix = prefix.strip()
        language = None
        if prefix.startswith("language "):
            language = prefix[len("language ") :].strip()
        elif prefix:
            language = prefix
        return {"language": language, "transcription": transcription.strip()}

    @staticmethod
    def parse_output(text: str | list[str]) -> dict | list[dict]:
        """
        Parse Qwen3 ASR raw output into a structured dict.

        The model outputs ``language <LANG><asr_text>transcribed text``.
        This method returns a dict with ``"language"`` and ``"transcription"`` keys.

        Args:
            text (`str` or `list[str]`): Raw decoded output(s).

        Returns:
            `dict` or `list[dict]`: Parsed output(s). Each dict has keys
            ``"language"`` (str or None) and ``"transcription"`` (str).
            Returns the original string as the transcription if parsing fails.
        """
        if isinstance(text, str):
            return Qwen3ASRProcessor._parse_single_output(text)
        return [Qwen3ASRProcessor._parse_single_output(raw_text) for raw_text in text]

    @staticmethod
    def extract_transcription(text: str | list[str]) -> str | list[str]:
        """
        Extract transcription text from Qwen3 ASR raw output.

        The model outputs ``language <LANG><asr_text>transcribed text``.
        This method extracts the text after ``<asr_text>``.

        Args:
            text (`str` or `list[str]`): Raw decoded output(s).

        Returns:
            `str` or `list[str]`: Extracted transcription(s). Returns the
            original string if ``<asr_text>`` is not found.
        """
        if isinstance(text, str):
            return Qwen3ASRProcessor._parse_single_output(text)["transcription"]
        return [Qwen3ASRProcessor._parse_single_output(raw_text)["transcription"] for raw_text in text]

    @staticmethod
    def _is_cjk_char(char: str) -> bool:
        """
        Return True for CJK ideograph characters.
        Original: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L62
        """
        codepoint = ord(char)
        return (
            (0x4E00 <= codepoint <= 0x9FFF)
            or (0x3400 <= codepoint <= 0x4DBF)
            or (0x20000 <= codepoint <= 0x2A6DF)
            or (0x2A700 <= codepoint <= 0x2B73F)
            or (0x2B740 <= codepoint <= 0x2B81F)
            or (0x2B820 <= codepoint <= 0x2CEAF)
            or (0xF900 <= codepoint <= 0xFAFF)
            or (0x2F800 <= codepoint <= 0x2FA1F)
        )

    @staticmethod
    def _is_kept_char(char: str) -> bool:
        """Return True for characters kept during forced-alignment tokenisation."""
        if char == "'":
            return True
        category = unicodedata.category(char)
        return category.startswith("L") or category.startswith("N") or Qwen3ASRProcessor._is_cjk_char(char)

    @staticmethod
    def _clean_tokens(raw_tokens) -> list[str]:
        """Filter each raw token to kept characters, dropping empty results."""
        return [
            cleaned
            for token in raw_tokens
            if (cleaned := "".join(char for char in token if Qwen3ASRProcessor._is_kept_char(char)))
        ]

    @staticmethod
    def split_words_for_alignment(text: str | list[str], language: str | None = None) -> list[str]:
        """
        Split text into word-level tokens suitable for forced alignment.
        Original: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L101-L145

        The tokenization strategy depends on the language:

        - **Japanese**: Uses the ``nagisa`` library for morphological analysis
          (install with ``pip install nagisa``).
        - **Korean**: Uses the ``soynlp`` library for tokenization
          (install with ``pip install soynlp``).
        - **All other languages** (including Chinese): CJK characters are emitted
          individually; space-delimited scripts produce whole words. Punctuation
          is dropped.

        Args:
            text (`str`): Transcript text.
            language (`str` or `None`, *optional*):
                Language of the transcript (e.g. ``"Japanese"``, ``"Korean"``,
                ``"English"``, ``"Chinese"``).  When ``None``, falls back to the
                default CJK / space-based tokenizer.

        Returns:
            `list[str]`: Word-level tokens.
        """
        text = text.strip()
        lang = language.lower() if language else ""

        if lang == "japanese":
            try:
                import nagisa
            except ImportError:
                raise ImportError(
                    "Japanese forced alignment requires the `nagisa` package. Install it with: pip install nagisa"
                )
            return Qwen3ASRProcessor._clean_tokens(nagisa.tagging(text).words)

        if lang == "korean":
            try:
                from soynlp.tokenizer import LTokenizer
            except ImportError:
                raise ImportError(
                    "Korean forced alignment requires the `soynlp` package. Install it with: pip install soynlp"
                )
            return Qwen3ASRProcessor._clean_tokens(LTokenizer().tokenize(text))

        # Default: CJK characters individually, space-delimited words otherwise
        tokens: list[str] = []
        char_buffer: list[str] = []

        def flush_buffer():
            if char_buffer:
                word = "".join(char_buffer)
                if word:
                    tokens.append(word)
                char_buffer.clear()

        for char in text:
            if Qwen3ASRProcessor._is_cjk_char(char):
                flush_buffer()
                tokens.append(char)
            elif char.isspace():
                flush_buffer()
            elif Qwen3ASRProcessor._is_kept_char(char):
                char_buffer.append(char)
        flush_buffer()
        return tokens

    @staticmethod
    def _fix_timestamps(raw: np.ndarray) -> list[int]:
        """
        Monotonize predicted timestamps using longest increasing subsequence, then interpolate outliers.
        Original: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L147
        """
        data = raw.tolist()
        num_values = len(data)
        if num_values == 0:
            return []

        # Find longest increasing subsequence (LIS) via O(n²) DP
        dp = [1] * num_values
        parent = [-1] * num_values
        for current in range(1, num_values):
            for prev in range(current):
                if data[prev] <= data[current] and dp[prev] + 1 > dp[current]:
                    dp[current] = dp[prev] + 1
                    parent[current] = prev

        # Backtrack to get LIS indices
        is_normal = [False] * num_values
        trace_idx = dp.index(max(dp))
        while trace_idx != -1:
            is_normal[trace_idx] = True
            trace_idx = parent[trace_idx]

        # Interpolate non-LIS positions
        result = data.copy()
        block_start = 0
        while block_start < num_values:
            if is_normal[block_start]:
                block_start += 1
                continue
            # Find contiguous block of outlier values [block_start, block_end)
            block_end = block_start
            while block_end < num_values and not is_normal[block_end]:
                block_end += 1
            block_len = block_end - block_start
            left = next((result[pos] for pos in range(block_start - 1, -1, -1) if is_normal[pos]), None)
            right = next((result[pos] for pos in range(block_end, num_values) if is_normal[pos]), None)
            if block_len <= 2:
                for pos in range(block_start, block_end):
                    if left is None:
                        result[pos] = right
                    elif right is None:
                        result[pos] = left
                    else:
                        result[pos] = left if (pos - (block_start - 1)) <= (block_end - pos) else right
            else:
                fill = left if left is not None else right
                if left is not None and right is not None:
                    step = (right - left) / (block_len + 1)
                    for pos in range(block_start, block_end):
                        result[pos] = left + step * (pos - block_start + 1)
                elif fill is not None:
                    for pos in range(block_start, block_end):
                        result[pos] = fill
            block_start = block_end

        return [int(v) for v in result]

    def prepare_forced_aligner_inputs(
        self,
        audio: AudioInput,
        transcript: str | list[str],
        language: str | list[str] | None = None,
        **kwargs,
    ) -> tuple[BatchFeature, list[list[str]]]:
        """
        Prepare inputs for the forced aligner model.

        Args:
            audio (`AudioInput`):
                Audio input(s).  Accepts paths, URLs, numpy arrays, or a list of these.
            transcript (`str` or `list[str]`):
                Transcript(s) to align against the audio.
            language (`str`, `list[str]`, or `None`, *optional*):
                Language hint(s). Currently unused in tokenization but reserved for
                language-specific tokenizers (e.g. Japanese, Korean).
            **kwargs:
                Additional keyword arguments forwarded to
                [`~Qwen3ASRProcessor.apply_chat_template`].

        Returns:
            `tuple[BatchFeature, list[list[str]]]`:
                - ``inputs``: A [`BatchFeature`] with ``input_ids``, ``attention_mask``,
                  ``input_features``, and ``input_features_mask`` ready for the forced
                  aligner model.
                - ``word_lists``: A list (one per sample) of word-level token lists used
                  to build the input. Pass these to
                  [`~Qwen3ASRProcessor.decode_forced_alignment`] to pair timestamps
                  with words.
        """
        if isinstance(transcript, str):
            transcript = [transcript]

        audio_items = self._normalize_audio(audio)
        batch_size = len(audio_items)
        if len(transcript) != batch_size:
            raise ValueError(f"Got {len(transcript)} transcript(s) but {batch_size} audio(s); they must match 1:1.")

        languages = self._normalize_languages(language, batch_size, allow_broadcast=True)
        word_lists = [self.split_words_for_alignment(t, lang) for t, lang in zip(transcript, languages)]

        conversations = []
        for wl, audio_item in zip(word_lists, audio_items):
            content = [self._audio_content_item(audio_item)]
            content.extend({"type": "text", "text": word} for word in wl)
            conversations.append([{"role": "user", "content": content}])

        inputs = self.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            inputs["position_ids"] = position_ids

        return inputs, word_lists

    def decode_forced_alignment(
        self,
        logits,
        input_ids,
        word_lists: list[list[str]],
        timestamp_token_id: int,
        timestamp_segment_time: float | None = None,
    ) -> list[list[dict]]:
        """
        Decode forced aligner model outputs into word-level timestamps.

        Args:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, num_timestamp_bins)`):
                Classification logits from [`Qwen3ASRForForcedAlignment`].
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Input token IDs used for the forward pass.
            word_lists (`list[list[str]]`):
                Word-level token lists as returned by
                [`~Qwen3ASRProcessor.prepare_forced_aligner_inputs`].
            timestamp_token_id (`int`):
                Token ID of the ``<timestamp>`` marker (from
                ``model.config.timestamp_token_id``).
            timestamp_segment_time (`float`, *optional*):
                Milliseconds per timestamp class. If not provided, uses `self.timestamp_segment_time`.

        Returns:
            `list[list[dict]]`: One list per sample.  Each inner list contains dicts
            with keys ``"text"`` (`str`), ``"start_time"`` (`float`, seconds), and
            ``"end_time"`` (`float`, seconds).
        """
        if timestamp_segment_time is None:
            timestamp_segment_time = self.timestamp_segment_time
        pred_ids = logits.argmax(dim=-1)
        batch_results = []

        for sample_idx, word_list in enumerate(word_lists):
            mask = input_ids[sample_idx] == timestamp_token_id
            masked_pred = pred_ids[sample_idx][mask]
            raw_ms = (masked_pred.float() * timestamp_segment_time).cpu().numpy()
            fixed_ms = self._fix_timestamps(raw_ms)

            items = [
                {
                    "text": word,
                    "start_time": round(fixed_ms[word_idx * 2] / 1000.0, 3),
                    "end_time": round(fixed_ms[word_idx * 2 + 1] / 1000.0, 3),
                }
                for word_idx, word in enumerate(word_list)
            ]
            batch_results.append(items)

        return batch_results

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["input_features_mask"]))


__all__ = ["Qwen3ASRProcessor"]

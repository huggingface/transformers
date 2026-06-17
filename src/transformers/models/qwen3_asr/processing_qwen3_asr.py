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

import unicodedata

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring
from ...utils.import_utils import is_nagisa_available, is_soynlp_available


# fmt: off
# The ASR model was trained with these full names as system prompts.
LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

# The forced aligner supports a subset of the ASR languages.
FORCED_ALIGNER_LANGUAGES = {
    "Chinese", "English", "Cantonese", "French", "German",
    "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
}
# fmt: on

SUPPORTED_LANGUAGE_NAMES = set(LANGUAGE_CODE_TO_NAME.values())


def resolve_language(language: str | None) -> str | None:
    """Map a language code or name to the canonical full name, with validation.

    Accepts language codes (e.g. ``"zh"``, ``"en"``) or full names
    (e.g. ``"Chinese"``, ``"English"``). Returns the full name.
    Raises ``ValueError`` if the language is not recognized.
    ``None`` passes through unchanged (auto-detect).
    """
    if language is None:
        return None
    # Try code lookup first
    resolved = LANGUAGE_CODE_TO_NAME.get(language.lower())
    if resolved is not None:
        return resolved
    # Check if it's already a valid full name (case-insensitive)
    for name in SUPPORTED_LANGUAGE_NAMES:
        if language.lower() == name.lower():
            return name
    raise ValueError(
        f"Unsupported language: {language!r}. Use a language code "
        f"(e.g. 'en', 'zh') or full name (e.g. 'English', 'Chinese'). "
        f"Supported codes: {sorted(LANGUAGE_CODE_TO_NAME.keys())}. "
        f"Supported names: {sorted(SUPPORTED_LANGUAGE_NAMES)}."
    )


def _prepare_language_inputs(
    language: str | list[str] | None, batch_size: int, allow_broadcast: bool = False
) -> list[str | None]:
    """Broadcast / validate a language argument to match batch_size.

    Accepts language codes (e.g. ``"zh"``, ``"en"``) or full names
    (e.g. ``"Chinese"``, ``"English"``). Each value is resolved to the
    canonical full language name via :func:`resolve_language`.
    """
    if language is None:
        return [None] * batch_size
    if isinstance(language, str):
        return [resolve_language(language)] * batch_size
    if isinstance(language, (list, tuple)):
        if allow_broadcast and len(language) == 1 and batch_size > 1:
            return [resolve_language(language[0])] * batch_size
        if len(language) != batch_size:
            raise ValueError(f"Got {len(language)} language(s) for {batch_size} sample(s); counts must match.")
        return [resolve_language(lang) for lang in language]
    raise TypeError("`language` must be a string, a list of strings, or `None`.")


def _audio_content_item(audio_item) -> dict:
    """Build a chat-template content dict for a single audio item."""
    if isinstance(audio_item, str):
        return {"type": "audio", "path": audio_item}
    return {"type": "audio", "audio": audio_item}


def _is_cjk_char(char: str) -> bool:
    """
    Return True for Chinese-Japanese-Korean (CJK) ideograph characters.
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


def _is_kept_char(char: str) -> bool:
    """Return True for characters kept during forced-alignment tokenisation (letters, numbers, apostrophes, CJK)."""
    if char == "'":
        return True
    category = unicodedata.category(char)
    return category.startswith("L") or category.startswith("N") or _is_cjk_char(char)


def _clean_tokens(raw_tokens) -> list[str]:
    """Filter each raw token to kept characters, dropping empty results."""
    return [cleaned for token in raw_tokens if (cleaned := "".join(char for char in token if _is_kept_char(char)))]


def _parse_single_output(text: str) -> dict:
    """Parse a single decoded ASR string into language + transcription like the original implementation."""
    if text is None:
        return {"language": None, "transcription": ""}
    text = str(text).strip()
    if not text:
        return {"language": None, "transcription": ""}

    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1]

    # Apply repetition fix from original implementation
    text = _detect_and_fix_repetitions(text)

    marker = "<asr_text>"
    if marker not in text:
        # No tag — treat the whole string as plain transcription
        return {"language": None, "transcription": text.strip()}

    prefix, transcription = text.split(marker, 1)
    prefix = prefix.strip()

    # Empty-audio heuristic: "language None<asr_text>"
    if prefix.lower() == "language none":
        t = transcription.strip()
        return {"language": None, "transcription": t}

    language = None
    for line in prefix.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("language "):
            val = line[len("language ") :].strip()
            if val:
                language = val
        else:
            language = line
        break  # only inspect the first non-empty line, matching the original

    return {"language": language or None, "transcription": transcription.strip()}


def _fix_timestamps(raw: np.ndarray) -> list[int]:
    """
    Ensure predicted timestamps are monotonically increasing.

    The model may predict out-of-order timestamps. This method:
    1. Finds the longest increasing subsequence (LIS) — these are "good" timestamps.
    2. Marks everything else as an outlier.
    3. Fills outlier blocks by snapping short blocks (\u22642) to the nearest
       good neighbour, or linearly interpolating longer blocks between
       the surrounding good values.

    Original: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L147
    """
    data = raw.tolist()
    num_values = len(data)

    # --- Step 1: find the longest increasing subsequence (LIS) via O(n\u00b2) DP ---
    # dp[idx]     = length of the LIS ending at index idx
    # parent[idx] = previous index in that LIS (-1 = start of chain)
    dp = [1] * num_values
    parent = [-1] * num_values

    for current in range(1, num_values):
        for prev in range(current):
            if data[prev] <= data[current] and dp[prev] + 1 > dp[current]:
                dp[current] = dp[prev] + 1
                parent[current] = prev

    # --- Step 2: backtrack to recover LIS indices and mark them as "normal" ---
    max_length = max(dp)
    max_idx = dp.index(max_length)

    lis_indices = []
    idx = max_idx
    while idx != -1:
        lis_indices.append(idx)
        idx = parent[idx]
    lis_indices.reverse()

    is_normal = [False] * num_values
    for idx in lis_indices:
        is_normal[idx] = True

    # --- Step 3: replace outlier blocks with interpolated / snapped values ---
    result = data.copy()
    block_start = 0

    while block_start < num_values:
        if is_normal[block_start]:
            block_start += 1
            continue

        # Scan forward to find the end of this contiguous outlier block
        block_end = block_start
        while block_end < num_values and not is_normal[block_end]:
            block_end += 1

        anomaly_count = block_end - block_start

        if anomaly_count <= 2:
            # Short block: snap each position to the closer good neighbour
            left_val = None
            for scan in range(block_start - 1, -1, -1):
                if is_normal[scan]:
                    left_val = result[scan]
                    break

            right_val = None
            for scan in range(block_end, num_values):
                if is_normal[scan]:
                    right_val = result[scan]
                    break

            for pos in range(block_start, block_end):
                if left_val is None:
                    result[pos] = right_val
                elif right_val is None:
                    result[pos] = left_val
                else:
                    result[pos] = left_val if (pos - (block_start - 1)) <= (block_end - pos) else right_val

        else:
            # Long block: linearly interpolate between the surrounding good values
            left_val = None
            for scan in range(block_start - 1, -1, -1):
                if is_normal[scan]:
                    left_val = result[scan]
                    break

            right_val = None
            for scan in range(block_end, num_values):
                if is_normal[scan]:
                    right_val = result[scan]
                    break

            if left_val is not None and right_val is not None:
                step = (right_val - left_val) / (anomaly_count + 1)
                for pos in range(block_start, block_end):
                    result[pos] = left_val + step * (pos - block_start + 1)
            elif left_val is not None:
                for pos in range(block_start, block_end):
                    result[pos] = left_val
            elif right_val is not None:
                for pos in range(block_start, block_end):
                    result[pos] = right_val

        block_start = block_end

    return [int(val) for val in result]


def _detect_and_fix_repetitions(text, threshold=20):
    """
    Original implementation uses this post-processing to remove repeated characters and patterns in the ASR output
    https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/utils.py#L432
    """

    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i : i + count])
                i += count
        return "".join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break

                pattern = s[i : i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx : start_idx + k] != pattern:
                        valid = False
                        break

                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index : end_index + k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break

            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return "".join(result)

    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


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


@auto_docstring
class Qwen3ASRProcessor(ProcessorMixin):
    valid_processor_kwargs = Qwen3ASRProcessorKwargs

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None, timestamp_segment_time: float = 80):
        r"""
        timestamp_segment_time (`float`, *optional*):
            Milliseconds per timestamp class. Defaults to 80 ms.
        """
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.timestamp_segment_time = timestamp_segment_time
        self.audio_token = self.tokenizer.audio_token
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.audio_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_eos_token)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput,
        output_labels: bool | None = False,
        **kwargs: Unpack[Qwen3ASRProcessorKwargs],
    ) -> BatchFeature:
        r"""
        output_labels (bool, *optional*, default=False):
            Whether to return labels for training.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_features`, `input_features_mask`).
        """
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if output_labels:
            kwargs["return_mm_token_type_ids"] = True
        model_inputs = super().__call__(audio=audio, text=text, **kwargs)

        if output_labels:
            labels = model_inputs.pop("mm_token_type_ids")
            for token_id in [
                self.audio_token_id,
                self.tokenizer.pad_token_id,
                self.audio_bos_token_id,
                self.audio_eos_token_id,
            ]:
                labels[labels == token_id] = -100
            model_inputs["labels"] = labels

        return BatchFeature(data=model_inputs, tensor_type="pt")

    def validate_inputs(
        self,
        audio: AudioInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(audio=audio, text=text, **kwargs)

        if text is not None and audio is not None and len(text) != len(audio):
            raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

    def _get_audio_token_length(self, audio_lengths, n_window=50):
        chunk_len = n_window * 2
        remainder = audio_lengths % chunk_len  # mel frames in the final partial chunk
        feat_lengths = (remainder - 1) // 2 + 1  # after first conv (stride 2)
        per_chunk_tokens = (feat_lengths - 1) // 2 + 1  # after second conv (stride 2)
        token_lengths = (
            (per_chunk_tokens - 1) // 2 + 1 + (audio_lengths // chunk_len) * 13
        )  # after third conv + full chunks
        return token_lengths.cpu().numpy()

    def _process_audio(self, audio: AudioInput, **kwargs):
        n_window = kwargs.get("n_window", 50)
        audio_inputs = self.feature_extractor(audio, **kwargs)
        audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")

        audio_lengths = self._get_audio_token_length(audio_inputs["input_features_mask"].sum(-1), n_window)
        audio_inputs["num_audio_tokens"] = audio_lengths

        audio_replacements = [self.replace_audio_token(audio_inputs, idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        num_tokens = int(audio_inputs["num_audio_tokens"][audio_idx])
        return self.audio_token * num_tokens

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
                Language hint(s) to include in the system prompt. Accepts full names
                (e.g. ``"English"``, ``"Chinese"``) or ISO codes (e.g. ``"en"``, ``"zh"``).
                A list must be the same length as the audio batch.
                When ``None``, the model performs automatic language detection.
            **kwargs:
                Additional keyword arguments forwarded to
                [`~Qwen3ASRProcessor.apply_chat_template`].

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`Qwen3ASRForConditionalGeneration.generate`].
        """
        audio_items = make_list_of_audio_chat_template(audio)
        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")
        languages = _prepare_language_inputs(language, batch_size)

        conversations = []
        for lang, audio_item in zip(languages, audio_items):
            messages = []
            if lang is not None:
                messages.append({"role": "system", "content": [{"type": "text", "text": lang}]})
            messages.append({"role": "user", "content": [_audio_content_item(audio_item)]})
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

    def parse_output(self, text: str | list[str]) -> dict | list[dict]:
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
            return _parse_single_output(text)
        return [_parse_single_output(raw_text) for raw_text in text]

    def extract_transcription(self, text: str | list[str]) -> str | list[str]:
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
            return _parse_single_output(text)["transcription"]
        return [_parse_single_output(raw_text)["transcription"] for raw_text in text]

    def split_words_for_alignment(self, text: str | list[str], language: str | None = None) -> list[str]:
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
                Language of the transcript. Accepts full names (e.g. ``"Japanese"``,
                ``"English"``) or codes (e.g. ``"ja"``, ``"en"``).  When ``None``,
                falls back to the default CJK / space-based tokenizer.

        Returns:
            `list[str]`: Word-level tokens.
        """
        text = text.strip()
        lang = language.lower() if language else ""

        if lang == "japanese":
            if not is_nagisa_available():
                raise ImportError(
                    "Japanese forced alignment requires the `nagisa` package. Install it with: pip install nagisa"
                )
            import nagisa

            return _clean_tokens(nagisa.tagging(text).words)

        if lang == "korean":
            if not is_soynlp_available():
                raise ImportError(
                    "Korean forced alignment requires the `soynlp` package. Install it with: pip install soynlp"
                )
            from soynlp.tokenizer import LTokenizer

            return _clean_tokens(LTokenizer().tokenize(text))

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
            if _is_cjk_char(char):
                flush_buffer()
                tokens.append(char)
            elif char.isspace():
                flush_buffer()
            elif _is_kept_char(char):
                char_buffer.append(char)
        flush_buffer()
        return tokens

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

        audio_items = make_list_of_audio_chat_template(audio)
        batch_size = len(audio_items)
        if len(transcript) != batch_size:
            raise ValueError(f"Got {len(transcript)} transcript(s) but {batch_size} audio(s); they must match 1:1.")

        languages = _prepare_language_inputs(language, batch_size, allow_broadcast=True)

        # Validate that all languages are supported by the forced aligner
        for lang in languages:
            if lang is not None and lang not in FORCED_ALIGNER_LANGUAGES:
                aligner_codes = sorted(
                    code for code, name in LANGUAGE_CODE_TO_NAME.items() if name in FORCED_ALIGNER_LANGUAGES
                )
                raise ValueError(
                    f"Language {lang!r} is not supported by the forced aligner. "
                    f"Supported languages: {sorted(FORCED_ALIGNER_LANGUAGES)} "
                    f"(codes: {aligner_codes})."
                )

        word_lists = [self.split_words_for_alignment(t, lang) for t, lang in zip(transcript, languages)]

        conversations = []
        for wl, audio_item in zip(word_lists, audio_items):
            content = [_audio_content_item(audio_item)]
            content.extend({"type": "text", "text": word} for word in wl)
            conversations.append([{"role": "user", "content": content}])

        inputs = self.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )

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
            logits (`torch.Tensor` of shape `(batch_size, seq_len, num_labels)`):
                Classification logits from [`Qwen3ASRForTokenClassification`].
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
            fixed_ms = _fix_timestamps(raw_ms)

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
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["num_audio_tokens"]

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["input_features_mask"]))


__all__ = ["Qwen3ASRProcessor"]

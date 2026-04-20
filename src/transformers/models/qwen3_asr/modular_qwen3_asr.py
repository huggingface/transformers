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
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3ForConditionalGeneration
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioPreTrainedModel
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths


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

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "qwen3_audio_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["qwen3_audio_encoder"](
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
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


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
    """

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
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
        audio_lengths = _get_feat_extract_output_lengths(data["input_features_mask"].sum(-1)).cpu().numpy()
        audio_token_pattern = re.compile(re.escape(self.audio_token))
        for i, num_tokens in enumerate(audio_lengths):
            text[i] = audio_token_pattern.sub(self.audio_token * int(num_tokens), text[i])

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
        if isinstance(audio, str):
            audio_items: list = [audio]
        elif isinstance(audio, (list, tuple)) and audio and all(isinstance(a, str) for a in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if language is None:
            languages = [None] * batch_size
        elif isinstance(language, str):
            languages = [language] * batch_size
        elif isinstance(language, (list, tuple)):
            if len(language) != batch_size:
                raise ValueError(
                    f"Received {len(language)} language(s) for {batch_size} audio sample(s); counts must match."
                )
            languages = list(language)
        else:
            raise TypeError("`language` must be a string, a list of strings, or `None`.")

        conversations = []
        for lang, audio_item in zip(languages, audio_items):
            content = []
            if isinstance(audio_item, str):
                content.append({"type": "audio", "path": audio_item})
            else:
                content.append({"type": "audio", "audio": audio_item})

            messages = []
            if lang is not None:
                messages.append({"role": "system", "content": [{"type": "text", "text": lang}]})
            messages.append({"role": "user", "content": content})
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
    def _strip_chat_prefix(text: str) -> str:
        """Strip chat template prefixes like ``system\\n...\\nassistant\\n``."""
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[-1]
        return text

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
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        results = []
        for t in text:
            t = Qwen3ASRProcessor._strip_chat_prefix(t)
            marker = "<asr_text>"
            language = None
            transcription = t

            if marker in t:
                prefix, transcription = t.split(marker, 1)
                transcription = transcription.strip()
                # prefix is "language <LANG>"
                prefix = prefix.strip()
                if prefix.startswith("language "):
                    language = prefix[len("language ") :].strip()
                elif prefix:
                    language = prefix

            results.append({"language": language, "transcription": transcription})

        return results[0] if is_single else results

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
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        results = []
        for t in text:
            t = Qwen3ASRProcessor._strip_chat_prefix(t)
            marker = "<asr_text>"
            if marker in t:
                t = t.split(marker, 1)[-1].strip()
            results.append(t)

        return results[0] if is_single else results

    # ── Forced alignment helpers ──

    @staticmethod
    def _is_cjk_char(ch: str) -> bool:
        """
        Return True for CJK ideograph characters.
        Original: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L62
        """
        cp = ord(ch)
        return (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        )

    @staticmethod
    def _is_kept_char(ch: str) -> bool:
        """Return True for characters kept during forced-alignment tokenisation."""
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N") or Qwen3ASRProcessor._is_cjk_char(ch)

    @staticmethod
    def tokenize_for_alignment(text: str, language: str | None = None) -> list[str]:
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
            raw_tokens = nagisa.tagging(text)
            tokens = []
            for w in raw_tokens.words:
                cleaned = "".join(ch for ch in w if Qwen3ASRProcessor._is_kept_char(ch))
                if cleaned:
                    tokens.append(cleaned)
            return tokens

        if lang == "korean":
            try:
                from soynlp.tokenizer import LTokenizer
            except ImportError:
                raise ImportError(
                    "Korean forced alignment requires the `soynlp` package. Install it with: pip install soynlp"
                )
            ko_tokenizer = LTokenizer()
            raw_tokens = ko_tokenizer.tokenize(text)
            tokens = []
            for w in raw_tokens:
                cleaned = "".join(ch for ch in w if Qwen3ASRProcessor._is_kept_char(ch))
                if cleaned:
                    tokens.append(cleaned)
            return tokens

        # Default: CJK characters individually, space-delimited words otherwise
        tokens: list[str] = []
        buf: list[str] = []

        def flush():
            if buf:
                word = "".join(buf).strip()
                if word:
                    tokens.append(word)
                buf.clear()

        for ch in text:
            if Qwen3ASRProcessor._is_cjk_char(ch):
                flush()
                tokens.append(ch)
            elif ch.isspace():
                flush()
            elif Qwen3ASRProcessor._is_kept_char(ch):
                buf.append(ch)
        flush()
        return tokens

    @staticmethod
    def _fix_timestamps(raw: np.ndarray) -> list[int]:
        """
        Original: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L147
        """
        data = raw.tolist()
        n = len(data)
        if n == 0:
            return []

        dp = [1] * n
        parent = [-1] * n
        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_idx = dp.index(max(dp))
        lis_idx: list[int] = []
        idx = max_idx
        while idx != -1:
            lis_idx.append(idx)
            idx = parent[idx]
        lis_idx.reverse()

        is_normal = [False] * n
        for idx in lis_idx:
            is_normal[idx] = True

        result = data.copy()
        i = 0
        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]:
                    j += 1
                count = j - i
                left = next((result[k] for k in range(i - 1, -1, -1) if is_normal[k]), None)
                right = next((result[k] for k in range(j, n) if is_normal[k]), None)
                if count <= 2:
                    for k in range(i, j):
                        if left is None:
                            result[k] = right
                        elif right is None:
                            result[k] = left
                        else:
                            result[k] = left if (k - (i - 1)) <= (j - k) else right
                else:
                    if left is not None and right is not None:
                        step = (right - left) / (count + 1)
                        for k in range(i, j):
                            result[k] = left + step * (k - i + 1)
                    elif left is not None:
                        for k in range(i, j):
                            result[k] = left
                    elif right is not None:
                        for k in range(i, j):
                            result[k] = right
                i = j
            else:
                i += 1

        return [int(v) for v in result]

    def apply_forced_alignment_request(
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

        if isinstance(audio, str):
            audio_items: list = [audio]
        elif isinstance(audio, (list, tuple)) and audio and all(isinstance(a, str) for a in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))

        batch_size = len(audio_items)
        if len(transcript) != batch_size:
            raise ValueError(f"Got {len(transcript)} transcript(s) but {batch_size} audio(s); they must match 1:1.")

        if language is None:
            languages: list[str | None] = [None] * batch_size
        elif isinstance(language, str):
            languages = [language] * batch_size
        elif isinstance(language, (list, tuple)):
            if len(language) == 1 and batch_size > 1:
                languages = list(language) * batch_size
            elif len(language) != batch_size:
                raise ValueError(f"Got {len(language)} language(s) for {batch_size} audio(s); they must match 1:1.")
            else:
                languages = list(language)
        else:
            raise TypeError("`language` must be a string, a list of strings, or `None`.")

        word_lists = [self.tokenize_for_alignment(t, lang) for t, lang in zip(transcript, languages)]

        conversations = []
        for wl, audio_item in zip(word_lists, audio_items):
            content = []
            if isinstance(audio_item, str):
                content.append({"type": "audio", "path": audio_item})
            else:
                content.append({"type": "audio", "audio": audio_item})
            # Each word becomes a separate text item; the chat template joins them with <timestamp><timestamp> markers.
            for word in wl:
                content.append({"type": "text", "text": word})

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
        logits: torch.Tensor,
        input_ids: torch.LongTensor,
        word_lists: list[list[str]],
        timestamp_token_id: int,
        timestamp_segment_time: float,
    ) -> list[list[dict]]:
        """
        Decode forced aligner model outputs into word-level timestamps.

        Args:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, classify_num)`):
                Classification logits from [`Qwen3ForcedAlignerForTokenClassification`].
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Input token IDs used for the forward pass.
            word_lists (`list[list[str]]`):
                Word-level token lists as returned by
                [`~Qwen3ASRProcessor.apply_forced_alignment_request`].
            timestamp_token_id (`int`):
                Token ID of the ``<timestamp>`` marker (from
                ``model.config.timestamp_token_id``).
            timestamp_segment_time (`float`):
                Milliseconds per timestamp class (from
                ``model.config.timestamp_segment_time``).

        Returns:
            `list[list[dict]]`: One list per sample.  Each inner list contains dicts
            with keys ``"text"`` (`str`), ``"start_time"`` (`float`, seconds), and
            ``"end_time"`` (`float`, seconds).
        """
        pred_ids = logits.argmax(dim=-1)
        batch_results = []

        for i, word_list in enumerate(word_lists):
            mask = input_ids[i] == timestamp_token_id
            masked_pred = pred_ids[i][mask]
            raw_ms = (masked_pred.float() * timestamp_segment_time).cpu().numpy()
            fixed_ms = self._fix_timestamps(raw_ms)

            items = []
            for j, word in enumerate(word_list):
                start_ms = fixed_ms[j * 2]
                end_ms = fixed_ms[j * 2 + 1]
                items.append(
                    {
                        "text": word,
                        "start_time": round(start_ms / 1000.0, 3),
                        "end_time": round(end_ms / 1000.0, 3),
                    }
                )
            batch_results.append(items)

        return batch_results

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["input_features_mask"]))


class Qwen3ASRPreTrainedModel(Qwen2AudioPreTrainedModel):
    _no_split_modules = ["Qwen3OmniMoeAudioEncoderLayer", "Qwen3DecoderLayer"]
    _can_compile_fullgraph = False  # Audio encoder has data-dependent ops (same as Qwen3OmniMoe)
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Qwen3ASR model which consists of an audio encoder and a language model.
    """
)
class Qwen3ASRForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        del self.multi_modal_projector

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
            `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        """
        # Flatten batched features for the Qwen3OmniMoe audio encoder
        audio_feature_lengths = input_features_mask.sum(dim=1)
        input_features = input_features.permute(0, 2, 1)[input_features_mask.bool()].permute(1, 0)

        audio_output = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
            **kwargs,
        )
        audio_output.pooler_output = audio_output.last_hidden_state
        return audio_output

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
    ):
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

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            input_features=input_features,
            input_features_mask=input_features_mask,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


@auto_docstring(checkpoint="bezzam/Qwen3-ForcedAligner-0.6B")
@strict
class Qwen3ForcedAlignerConfig(Qwen3ASRConfig):
    r"""
    classify_num (`int`, *optional*, defaults to 5000):
        Number of classification labels for forced alignment.
    timestamp_token_id (`int`, *optional*, defaults to 151705):
        Token ID for timestamp markers in the alignment output.
    timestamp_segment_time (`int`, *optional*, defaults to 80):
        Time segment (in milliseconds) that each timestamp token represents.

    Example:

    ```python
    >>> from transformers import Qwen3ForcedAlignerForTokenClassification, Qwen3ForcedAlignerConfig

    >>> # Initializing a Qwen3ForcedAligner style configuration
    >>> configuration = Qwen3ForcedAlignerConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ForcedAlignerForTokenClassification(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_forced_aligner"

    classify_num: int = 5000
    timestamp_token_id: int = 151705
    timestamp_segment_time: int = 80


class Qwen3ForcedAlignerPreTrainedModel(Qwen3ASRPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    The Qwen3 Forced Aligner model which consists of an audio encoder, a language model backbone,
    and a token classification head for forced alignment.
    """
)
class Qwen3ForcedAlignerForTokenClassification(Qwen3ForcedAlignerPreTrainedModel):
    def __init__(self, config: Qwen3ForcedAlignerConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.classify_num = config.classify_num
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.model = AutoModel.from_config(config.text_config)
        self.classifier = nn.Linear(config.text_config.hidden_size, config.classify_num, bias=False)

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
        r"""
        input_features (`torch.FloatTensor`):
            Float values of mel features extracted from the raw speech waveform.
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padded feature indices.
        """
        # Flatten batched features for the Qwen3OmniMoe audio encoder
        audio_feature_lengths = input_features_mask.sum(dim=1)
        input_features = input_features.permute(0, 2, 1)[input_features_mask.bool()].permute(1, 0)

        audio_output = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
            **kwargs,
        )
        audio_output.pooler_output = audio_output.last_hidden_state
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
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.classify_num - 1]`.
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

        outputs = self.model(
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.classify_num)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRProcessor",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRPreTrainedModel",
    "Qwen3ForcedAlignerConfig",
    "Qwen3ForcedAlignerForTokenClassification",
    "Qwen3ForcedAlignerPreTrainedModel",
]

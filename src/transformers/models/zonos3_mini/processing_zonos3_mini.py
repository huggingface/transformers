# Copyright 2026 Zyphra and The HuggingFace Inc. team. All rights reserved.
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
#
# The prompt layout, condition buckets and take decoding are ported from the ZONOS3-Mini release
# (https://huggingface.co/Zyphra/ZONOS3-Mini-hf, Apache-2.0, Copyright Zyphra).

import bisect
import math
import re
from pathlib import Path

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, is_soundfile_available, is_torch_available, logging, requires_backends
from ...utils.import_utils import is_zyphra_textnorm_available, requires


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

if is_zyphra_textnorm_available():
    import zyphra_textnorm


logger = logging.get_logger(__name__)


class Zonos3MiniProcessorKwargs(ProcessingKwargs, total=False):
    """
    language (`str`, *optional*, defaults to `"en_us"`):
        Language code of the text (e.g. `"en_us"`), used to pick the text normalization grammar.
    text_normalization (`bool`, *optional*, defaults to `True`):
        Whether to normalize numbers, dates, currencies and units with `zyphra_textnorm` (when installed).
    condition_values (`dict[str, float]` or `list[float]`, *optional*):
        Target loudness (LUFS), fmax (Hz) and speaking rate (UTF-8 bytes per second), bucketed with the checkpoint's
        edges. `None` entries are unknown.
    condition_buckets (`dict[str, int]` or `list[int]`, *optional*):
        Condition buckets given directly (they take precedence over `condition_values`).
    silence_prefill_frames (`int`, *optional*, defaults to 3):
        Number of silence frames appended after the text, from which generation continues.
    speaker_mode (`str`, *optional*):
        How a voice clip conditions the speech: `"reference"` (its codec codes as a prefix to continue, the default) or
        `"embedding"` (its speaker embedding in a speaker row).
    speaker_embeddings (`torch.Tensor` of shape `(batch_size, 192)`, *optional*):
        Precomputed speaker embeddings, one per item (embedding mode).
    speaker_reference_codes (`list[torch.Tensor]`, *optional*):
        Precomputed reference codes, one `(num_frames, num_codebooks)` tensor per item (reference mode), e.g. the
        checkpoint's `voices/`.
    speaker_guidance (`bool`, *optional*, defaults to `False`):
        Whether to also return the unconditional prompts used by speaker guidance (`guidance_scale > 1`).
    """

    language: str | None
    text_normalization: bool | None
    condition_values: dict | list | None
    condition_buckets: dict | list | None
    silence_prefill_frames: int | None
    speaker_mode: str | None
    speaker_embeddings: object | None
    speaker_reference_codes: list | None
    speaker_guidance: bool | None


def collapse_line_breaks(text: str) -> str:
    """One plain space for every whitespace run that holds a line break, tab or other non-space whitespace."""
    return re.sub(r"\s*[^\S ]\s*", " ", text)


def shear_rows(codes: "torch.Tensor", pad: int, length: int) -> "torch.Tensor":
    """Delayed rows `[length, num_codebooks]` of codes `[num_frames, num_codebooks]`: row `r` holds codebook `j` of frame
    `r - j` (`pad` outside the frames)."""
    num_frames, num_codebooks = codes.shape
    rows = torch.full((length, num_codebooks), pad, dtype=torch.long)
    for j in range(num_codebooks):
        end = min(length, num_frames + j)
        if end > j:
            rows[j:end, j] = codes[: end - j, j]
    return rows


def undelay(rows: "torch.Tensor", start: int, end: int, pad: int) -> "torch.Tensor":
    """Undelayed frames `[start, end)` of the delayed rows `[num_rows, num_codebooks]`."""
    num_codebooks = rows.shape[1]
    rows = rows[start : min(end + num_codebooks - 1, len(rows))]
    frames = torch.full_like(rows, pad)
    for j in range(num_codebooks):
        if len(rows) > j:
            frames[: len(rows) - j, j] = rows[j:, j]
    return frames[: end - start]


@requires(backends=("torch",))
@auto_docstring
class Zonos3MiniProcessor(ProcessorMixin):
    text_kwargs = {"padding": True, "padding_side": "left"}
    audio_kwargs = {"sampling_rate": 44100}
    language = "en_us"
    text_normalization = True
    silence_prefill_frames = 3
    speaker_guidance = False
    valid_processor_kwargs = Zonos3MiniProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        codec_feature_extractor,
        audio_tokenizer=None,
        chat_template=None,
        languages: dict | None = None,
        markup_tags: list | None = None,
        condition_names: list | None = None,
        condition_edges: dict | None = None,
        condition_bucket_counts: list | None = None,
        silence_frame: list | None = None,
        max_silence_prefill_frames: int = 30,
        num_codebooks: int = 10,
        codebook_size: int = 4096,
        audio_eos_token_id: int = 4096,
        audio_pad_token_id: int = 4097,
        text_pad_token_id: int = 8196,
        reference_max_frames: int = 900,
        overlap_frames: int = 4,
        decode_context_frames: int = 30,
        headroom_db: float = 3.0,
        output_limit_dbfs: float = -1.0,
    ):
        r"""
        codec_feature_extractor ([`ZacFeatureExtractor`]):
            Prepares voice clips for the audio tokenizer, which encodes them into reference codes.
        audio_tokenizer ([`ZacModel`], *optional*):
            The ZAC-1 codec: it encodes voice clips into reference codes and decodes the generated codes into audio.
        languages (`dict[str, str | None]`, *optional*):
            The supported language codes and the text normalization grammar of each (`None`: no grammar).
        markup_tags (`list[str]`, *optional*):
            The nonverbal markup tags (e.g. `laugh`) the text normalization keeps.
        condition_names (`list[str]`, *optional*):
            The names of the conditions, in the order of the condition rows.
        condition_edges (`dict[str, list[float]]`, *optional*):
            The bucket edges of each condition, to bucket `condition_values`.
        condition_bucket_counts (`list[int]`, *optional*):
            The number of buckets of each condition; bucket `count` is unknown.
        silence_frame (`list[int]`, *optional*):
            The codec codes of a silence frame, for the silence prefill.
        max_silence_prefill_frames (`int`, *optional*, defaults to 30):
            The maximum number of silence frames.
        num_codebooks (`int`, *optional*, defaults to 10):
            The number of audio columns of each row.
        codebook_size (`int`, *optional*, defaults to 4096):
            The number of codes of each codebook.
        audio_eos_token_id (`int`, *optional*, defaults to 4096):
            The end-of-audio id of the audio columns.
        audio_pad_token_id (`int`, *optional*, defaults to 4097):
            The padding id of the audio columns.
        text_pad_token_id (`int`, *optional*, defaults to 8196):
            The padding id of the text column.
        reference_max_frames (`int`, *optional*, defaults to 900):
            Reference codes are cut to this many frames.
        overlap_frames (`int`, *optional*, defaults to 4):
            Number of silence prefill frames decoded before the speech.
        decode_context_frames (`int`, *optional*, defaults to 30):
            The default number of reference frames decoded before the speech when decoding with context.
        headroom_db (`float`, *optional*, defaults to 3.0):
            Attenuation applied to the decoded audio, in dB.
        output_limit_dbfs (`float`, *optional*, defaults to -1.0):
            Knee of the soft limiter applied to the decoded audio, in dBFS.
        """
        self.languages = languages if languages is not None else {"en_us": "en"}
        self.markup_tags = markup_tags if markup_tags is not None else []
        self.condition_names = condition_names if condition_names is not None else []
        self.condition_edges = condition_edges if condition_edges is not None else {}
        self.condition_bucket_counts = condition_bucket_counts if condition_bucket_counts is not None else []
        self.silence_frame = silence_frame if silence_frame is not None else [audio_pad_token_id] * num_codebooks
        self.max_silence_prefill_frames = max_silence_prefill_frames
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.text_pad_token_id = text_pad_token_id
        self.reference_max_frames = reference_max_frames
        self.overlap_frames = overlap_frames
        self.decode_context_frames = decode_context_frames
        self.headroom_db = headroom_db
        self.output_limit_dbfs = output_limit_dbfs
        self._normalizer = None
        super().__init__(
            feature_extractor,
            tokenizer,
            codec_feature_extractor,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template,
        )

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "condition_ids",
            "speaker_token_mask",
            "speaker_embeddings",
            "speaker_input_features",
            "speaker_input_features_mask",
            "uncond_input_ids",
            "uncond_attention_mask",
            "uncond_condition_ids",
            "labels",
        ]

    # -- text ----------------------------------------------------------------------------------------------------------

    def normalize_text(self, text: str, language: str = "en_us", text_normalization: bool = True) -> str:
        r"""
        Collapses line breaks, then reads numbers, dates, currencies and units with the language's grammar of
        `zyphra_textnorm` (when installed and the language has one).

        Args:
            text (`str`): The text.
            language (`str`, *optional*, defaults to `"en_us"`): The language code.
            text_normalization (`bool`, *optional*, defaults to `True`): Whether to apply the grammar.
        """
        if not isinstance(text, str):
            raise ValueError(f"The text must be a string, got {type(text)}.")
        if "\ue000" in text or "\ue001" in text:
            raise ValueError("This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups.")
        code = str(language or "").strip().lower().replace("-", "_")
        if code not in self.languages:
            raise ValueError(f"Unsupported language code: {language!r}. Supported: {', '.join(self.languages)}.")
        text = collapse_line_breaks(text)
        grammar = self.languages[code]
        if not text_normalization or grammar is None:
            return text
        if self._normalizer is None:
            if not is_zyphra_textnorm_available():
                logger.warning_once(
                    "zyphra_textnorm is not installed: text is not normalized (only line breaks are collapsed). "
                    "`pip install zyphra-textnorm` reads numbers, dates, currencies and units as the model was trained."
                )
                return text
            self._normalizer = zyphra_textnorm.Normalizer(list(self.markup_tags))
        return self._normalizer.normalize(text, grammar)

    def _condition_ids(self, buckets, values) -> list[int]:
        """The bucket of each condition; unknown conditions get bucket `count`."""
        names = list(self.condition_names)

        def as_map(raw, what):
            if raw is None:
                return {}
            if isinstance(raw, dict):
                unknown = [str(key) for key in raw if str(key) not in names]
                if unknown:
                    raise ValueError(f"{what}: unknown condition name(s) {unknown}; expected {names}.")
                return {str(key): value for key, value in raw.items()}
            raw = list(raw)
            if len(raw) != len(names):
                raise ValueError(f"{what}: expected {len(names)} entries in the order {names}.")
            return dict(zip(names, raw))

        buckets_by_name, values_by_name = as_map(buckets, "condition_buckets"), as_map(values, "condition_values")
        condition_ids = []
        for name, count in zip(names, self.condition_bucket_counts):
            if buckets_by_name.get(name) is not None:
                bucket = int(buckets_by_name[name])
                if not 0 <= bucket <= count:
                    raise ValueError(f"condition_buckets[{name}] must be in [0, {count - 1}] or {count} (unknown).")
            elif values_by_name.get(name) is not None:
                value = float(values_by_name[name])
                if not math.isfinite(value):
                    raise ValueError(f"condition_values[{name}] must be finite.")
                edges = self.condition_edges.get(name)
                if not edges:
                    raise ValueError(f"The processor has no edges for {name}: pass `condition_buckets` instead.")
                bucket = bisect.bisect_right(list(edges), value)
            else:
                bucket = count
            condition_ids.append(bucket)
        return condition_ids

    # -- speaker -------------------------------------------------------------------------------------------------------

    def _reference_codes(self, codes) -> "torch.Tensor":
        try:
            codes = torch.as_tensor(codes, dtype=torch.long)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError(
                "Each speaker_reference_codes item must be a non-empty [frames, codebooks] array."
            ) from None
        if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[1] != self.num_codebooks:
            raise ValueError("Each speaker_reference_codes item must be a non-empty [frames, codebooks] array.")
        if bool(((codes < 0) | (codes >= self.codebook_size)).any()):
            raise ValueError(f"speaker_reference_codes must be codec ids in [0, {self.codebook_size - 1}].")
        return codes[: self.reference_max_frames]

    def encode_reference(self, audio: AudioInput, sampling_rate: int | None = None) -> list["torch.Tensor"]:
        r"""
        Encodes voice clips into reference codes with the audio tokenizer.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, or a list of them): The voice clips, at 44.1 kHz.
            sampling_rate (`int`, *optional*): The sampling rate of the clips.

        Returns:
            `list[torch.Tensor]`: One `(num_frames, num_codebooks)` tensor of codes per clip, cut to
            `reference_max_frames`.
        """
        audio_tokenizer = getattr(self, "audio_tokenizer", None)
        if audio_tokenizer is None:
            raise ValueError(
                f"{self.__class__.__name__} has no audio tokenizer, so voice clips cannot be encoded: pass "
                "`speaker_reference_codes` (e.g. the checkpoint's `voices/`) or use `speaker_mode='embedding'`."
            )
        inputs = self.codec_feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = inputs.to(audio_tokenizer.device)
        with torch.no_grad():
            encoded = audio_tokenizer.encode(
                inputs["input_values"].to(audio_tokenizer.dtype),
                inputs["input_features"],
                padding_mask=inputs["padding_mask"],
                input_features_mask=inputs["input_features_mask"],
            )
        return [
            codes[:, mask.bool()].T.cpu()[: self.reference_max_frames]
            for codes, mask in zip(encoded.audio_codes, encoded.audio_codes_mask)
        ]

    # -- prompt --------------------------------------------------------------------------------------------------------

    def _blank_rows(self, count: int) -> "torch.Tensor":
        rows = torch.full((count, self.num_codebooks + 1), self.audio_pad_token_id, dtype=torch.long)
        rows[:, -1] = self.text_pad_token_id
        return rows

    def _delayed_rows(self, codes: "torch.Tensor", length: int) -> "torch.Tensor":
        rows = self._blank_rows(length)
        rows[:, : self.num_codebooks] = shear_rows(codes, self.audio_pad_token_id, length)
        return rows

    def _build_item(
        self,
        text_ids: list[int],
        reference_codes: "torch.Tensor | None",
        has_speaker_row: bool,
        condition_ids: list[int],
        silence_prefill_frames: int,
        guidance: bool,
        target_codes: "torch.Tensor | None",
    ) -> dict:
        """The rows of one item: `[reference prefix | speaker row][condition rows][BOS text EOS][silence][target]`."""
        text_rows = self._blank_rows(len(text_ids))
        text_rows[:, -1] = torch.tensor(text_ids, dtype=torch.long)
        silence = torch.tensor(self.silence_frame, dtype=torch.long)[None].repeat(silence_prefill_frames, 1)
        silence_rows = self._delayed_rows(silence, silence_prefill_frames)

        lead = []
        if reference_codes is not None:
            lead.append(self._delayed_rows(reference_codes, len(reference_codes) + self.num_codebooks - 1))
        if has_speaker_row:
            lead.append(self._blank_rows(1))
        lead = torch.cat(lead) if lead else self._blank_rows(0)
        conditions = self._blank_rows(len(condition_ids))
        target = (
            self._delayed_rows(target_codes, len(target_codes) + self.num_codebooks - 1)
            if target_codes is not None
            else self._blank_rows(0)
        )
        body = torch.cat([conditions, text_rows, silence_rows, target])

        offsets = [sum(count + 1 for count in self.condition_bucket_counts[:i]) for i in range(len(condition_ids))]

        def condition_column(ids, start):
            column = torch.full((start + len(body),), -1, dtype=torch.long)
            for i, (offset, bucket) in enumerate(zip(offsets, ids)):
                column[start + i] = offset + bucket
            return column

        item = {
            "input_ids": torch.cat([lead, body]),
            "condition_ids": condition_column(condition_ids, len(lead)),
            "speaker_token_mask": torch.zeros(len(lead) + len(body), dtype=torch.bool),
        }
        if has_speaker_row:
            item["speaker_token_mask"][len(lead) - 1] = True
        if target_codes is not None:
            labels = torch.full((len(lead) + len(body), self.num_codebooks), -100, dtype=torch.long)
            target_rows = target[:, : self.num_codebooks]
            labels[len(lead) + len(body) - len(target) :] = target_rows.masked_fill(
                target_rows == self.audio_pad_token_id, -100
            )
            item["labels"] = labels
        if guidance:
            unknown = list(self.condition_bucket_counts)
            item["uncond_input_ids"] = body.clone()
            item["uncond_condition_ids"] = condition_column(unknown, 0)
        return item

    @staticmethod
    def _left_pad(tensors: list["torch.Tensor"], length: int, value) -> "torch.Tensor":
        padded = []
        for tensor in tensors:
            pad = torch.full((length - len(tensor), *tensor.shape[1:]), value, dtype=tensor.dtype)
            padded.append(torch.cat([pad, tensor]))
        return torch.stack(padded)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        audio: AudioInput | None = None,
        output_labels: bool | None = None,
        **kwargs: Unpack[Zonos3MiniProcessorKwargs],
    ) -> BatchFeature:
        r"""
        text (`str` or `list[str]`):
            The text to speak, one string per item (the user turn rendered by
            [`~Zonos3MiniProcessor.apply_chat_template`]).
        audio (`np.ndarray`, `torch.Tensor`, or a list of them, *optional*):
            One voice clip per item, at 44.1 kHz (with `output_labels=True`, the voice clip if any, then the target,
            for each item).
        output_labels (`bool`, *optional*):
            Whether the last audio of each item is the target to train on: its codes are appended to the prompt and
            returned as `labels`.

        Returns:
            [`BatchFeature`] with `input_ids`, `attention_mask` and `condition_ids`; `speaker_token_mask` and
            `speaker_embeddings` or `speaker_input_features` / `speaker_input_features_mask` in embedding mode;
            `uncond_input_ids`, `uncond_attention_mask` and `uncond_condition_ids` with `speaker_guidance=True`;
            `labels` with `output_labels=True`.
        """
        output_kwargs = self._merge_kwargs(tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)
        text_kwargs, audio_kwargs = output_kwargs["text_kwargs"], output_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors not in (None, "pt"):
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")
        if text is None:
            raise ValueError("ZONOS3-Mini needs the text to speak.")
        texts = [text] if isinstance(text, str) else list(text)
        batch_size = len(texts)

        speaker_mode = output_kwargs.get("speaker_mode")
        speaker_embeddings = output_kwargs.get("speaker_embeddings")
        reference_codes = output_kwargs.get("speaker_reference_codes")
        guidance = bool(output_kwargs.get("speaker_guidance"))
        sampling_rate = audio_kwargs.get("sampling_rate")
        if speaker_mode not in (None, "reference", "embedding"):
            raise ValueError(f"speaker_mode={speaker_mode!r} is not supported: use 'reference' or 'embedding'.")

        clips = make_list_of_audio(audio) if audio is not None else []
        targets = None
        if output_labels:
            if len(clips) not in (batch_size, 2 * batch_size):
                raise ValueError("With `output_labels=True`, each item needs its target audio (after its voice clip).")
            has_voice = len(clips) == 2 * batch_size
            targets = clips[1::2] if has_voice else clips
            clips = clips[0::2] if has_voice else []
        if clips and len(clips) != batch_size:
            raise ValueError(f"Got {len(clips)} voice clips for {batch_size} texts: pass one voice clip per item.")
        if speaker_embeddings is not None and (reference_codes is not None or clips):
            raise ValueError(
                "Pass one speaker source: `speaker_embeddings`, `speaker_reference_codes` or a voice clip."
            )
        if reference_codes is not None and clips:
            raise ValueError("Pass one speaker source: `speaker_reference_codes` or a voice clip, not both.")
        if speaker_mode == "reference" and speaker_embeddings is not None:
            raise ValueError("speaker_mode='reference' conditions on reference codes: use speaker_mode='embedding'.")
        if speaker_mode == "embedding" and reference_codes is not None:
            raise ValueError("speaker_mode='embedding' conditions on a speaker embedding, not on reference codes.")

        outputs = {}
        item_reference_codes = [None] * batch_size
        has_speaker_row = speaker_embeddings is not None or (bool(clips) and speaker_mode == "embedding")
        if reference_codes is not None:
            if len(reference_codes) != batch_size:
                raise ValueError(f"Got {len(reference_codes)} speaker_reference_codes for {batch_size} texts.")
            item_reference_codes = [self._reference_codes(codes) for codes in reference_codes]
        elif clips and not has_speaker_row:
            item_reference_codes = self.encode_reference(clips, sampling_rate=sampling_rate)
        elif clips:
            features = self.feature_extractor(clips, sampling_rate=sampling_rate, return_tensors="pt")
            outputs["speaker_input_features"] = features["input_features"]
            outputs["speaker_input_features_mask"] = features["attention_mask"]
        elif speaker_embeddings is not None:
            speaker_embeddings = torch.as_tensor(speaker_embeddings, dtype=torch.float32).reshape(batch_size, -1)
            outputs["speaker_embeddings"] = speaker_embeddings
        if speaker_mode == "embedding" and not has_speaker_row:
            raise ValueError("speaker_mode='embedding' needs a voice clip or `speaker_embeddings` for every item.")
        has_speaker = has_speaker_row or item_reference_codes[0] is not None
        if guidance and not has_speaker:
            raise ValueError("Speaker guidance needs a speaker: a voice clip, reference codes or a speaker embedding.")
        target_codes = [None] * batch_size
        if targets is not None:
            target_codes = self.encode_reference(targets, sampling_rate=sampling_rate)

        silence_prefill_frames = output_kwargs.get("silence_prefill_frames")
        if not 0 <= int(silence_prefill_frames) <= self.max_silence_prefill_frames:
            raise ValueError(f"silence_prefill_frames must be in [0, {self.max_silence_prefill_frames}].")
        condition_ids = self._condition_ids(
            output_kwargs.get("condition_buckets"), output_kwargs.get("condition_values")
        )
        normalized = [
            self.normalize_text(t, output_kwargs.get("language"), output_kwargs.get("text_normalization"))
            for t in texts
        ]
        text_ids = self.tokenizer(normalized, add_special_tokens=True)["input_ids"]

        items = [
            self._build_item(ids, codes, has_speaker_row, condition_ids, int(silence_prefill_frames), guidance, target)
            for ids, codes, target in zip(text_ids, item_reference_codes, target_codes)
        ]
        lengths = [len(item["input_ids"]) for item in items]
        if guidance:
            lengths += [len(item["uncond_input_ids"]) for item in items]
        length = max(lengths)

        def attention_mask(rows):
            return self._left_pad([torch.ones(len(r), dtype=torch.long) for r in rows], length, 0)

        def pad_rows(rows):
            # padding rows are blank rows (audio and text padding)
            return torch.stack([torch.cat([self._blank_rows(length - len(r)), r]) for r in rows])

        rows = [item["input_ids"] for item in items]
        outputs["input_ids"] = pad_rows(rows)
        outputs["attention_mask"] = attention_mask(rows)
        outputs["condition_ids"] = self._left_pad([item["condition_ids"] for item in items], length, -1)
        if has_speaker_row:
            outputs["speaker_token_mask"] = self._left_pad(
                [item["speaker_token_mask"] for item in items], length, False
            )
        if targets is not None:
            outputs["labels"] = self._left_pad([item["labels"] for item in items], length, -100)
        if guidance:
            uncond = [item["uncond_input_ids"] for item in items]
            outputs["uncond_input_ids"] = pad_rows(uncond)
            outputs["uncond_attention_mask"] = attention_mask(uncond)
            outputs["uncond_condition_ids"] = self._left_pad(
                [item["uncond_condition_ids"] for item in items], length, -1
            )
        return BatchFeature(outputs, tensor_type=return_tensors)

    # -- decoding ------------------------------------------------------------------------------------------------------

    def _take_codes(self, rows: "torch.Tensor", silence_prefill_frames: int) -> tuple["torch.Tensor", int]:
        """The undelayed frames a whole-take decode covers: from the last `overlap_frames` silence frames up to the
        earliest end-of-audio, and the index of the first frame."""
        num_codebooks = self.num_codebooks
        buffer = rows[:, :num_codebooks]
        prefill = silence_prefill_frames
        eos = None
        for i in range(prefill, len(buffer)):
            columns = (buffer[i] == self.audio_eos_token_id).nonzero().flatten().tolist()
            if columns:
                candidate = max(0, i - max(columns))
                eos = candidate if eos is None else min(eos, candidate)
        target = len(buffer) - (num_codebooks - 1)
        if eos is not None:
            target = min(target, eos)
        start = prefill - min(self.overlap_frames, prefill)
        if target <= prefill:
            return torch.zeros((0, num_codebooks), dtype=torch.long), start
        return undelay(buffer, start, target, self.audio_pad_token_id), start

    def _output_stage(self, audio: "torch.Tensor", headroom_db: float, fade_out_ms: float) -> "torch.Tensor":
        """The headroom gain, the soft limiter (knee at `output_limit_dbfs`) and the cosine fade-out."""
        audio = audio.float() * 10.0 ** (-headroom_db / 20.0)
        knee = 10.0 ** (self.output_limit_dbfs / 20.0)
        over = audio.abs() > knee
        if bool(over.any()):
            audio = audio.clone()
            peak = audio[over]
            audio[over] = torch.sign(peak) * (knee + (1.0 - knee) * torch.tanh((peak.abs() - knee) / (1.0 - knee)))
        if fade_out_ms > 0 and len(audio):
            count = min(len(audio), int(self.feature_extractor.sampling_rate * fade_out_ms / 1000.0))
            if count > 0:
                audio = audio.clone()
                audio[-count:] *= 0.5 * (1.0 + torch.cos(torch.linspace(0.0, math.pi, count)))
        return audio

    def decode(
        self,
        sequences: "torch.Tensor",
        silence_prefill_frames: int | None = None,
        speaker_reference_codes: "list[torch.Tensor] | None" = None,
        decode_context_frames: int | None = None,
        headroom_db: float | None = None,
        fade_out_ms: float = 0.0,
    ) -> list["torch.Tensor"]:
        r"""
        Decodes the rows `generate` returns into waveforms: the generated frames (from the last `overlap_frames`
        silence frames to the end-of-audio) through the audio tokenizer, then the headroom gain, the soft limiter and
        an optional fade-out.

        Args:
            sequences (`torch.Tensor` of shape `(batch_size, num_rows, num_codebooks + 1)` or `(num_rows, num_codebooks + 1)`):
                The rows `generate` returns (prompt included).
            silence_prefill_frames (`int`, *optional*):
                The number of silence frames of the prompts, the processor's default if not given.
            speaker_reference_codes (`list[torch.Tensor]`, *optional*):
                The reference codes of each item: when given, the last `decode_context_frames` of them are decoded
                before the speech (and dropped), for a smoother start.
            decode_context_frames (`int`, *optional*):
                Number of reference frames decoded before the speech, the processor's `decode_context_frames` if not
                given.
            headroom_db (`float`, *optional*):
                Attenuation in dB, the processor's `headroom_db` if not given.
            fade_out_ms (`float`, *optional*, defaults to 0.0):
                Duration of the cosine fade-out, in milliseconds.

        Returns:
            `list[torch.Tensor]`: One float32 waveform at 44.1 kHz per item (empty when nothing precedes the
            end-of-audio).
        """
        audio_tokenizer = getattr(self, "audio_tokenizer", None)
        if audio_tokenizer is None:
            raise ValueError(f"{self.__class__.__name__} has no audio tokenizer, so codes cannot be decoded.")
        sequences = torch.as_tensor(sequences)
        if sequences.ndim == 2:
            sequences = sequences[None]
        if silence_prefill_frames is None:
            silence_prefill_frames = self.silence_prefill_frames
        headroom_db = self.headroom_db if headroom_db is None else headroom_db
        if decode_context_frames is None:
            decode_context_frames = self.decode_context_frames
        hop_length = audio_tokenizer.config.hop_length

        waveforms = []
        for i, rows in enumerate(sequences.cpu().long()):
            # the generated rows follow the last text row (EOS) and the silence prefill rows
            text_rows = (rows[:, -1] != self.text_pad_token_id).nonzero().flatten()
            buffer = rows[int(text_rows[-1]) + 1 :] if len(text_rows) else rows
            frames, start = self._take_codes(buffer, silence_prefill_frames)
            if not len(frames):
                waveforms.append(torch.zeros(0, dtype=torch.float32))
                continue
            context = None
            if decode_context_frames > 0 and speaker_reference_codes is not None and start == 0:
                reference = self._reference_codes(speaker_reference_codes[i])
                context = reference[len(reference) - min(decode_context_frames, len(reference)) :]
                frames = torch.cat([context, frames])
            with torch.no_grad():
                audio = audio_tokenizer.decode(frames.T[None].to(audio_tokenizer.device)).audio_values[0, 0]
            audio = audio.float().cpu()
            if context is not None:
                audio = audio[len(context) * hop_length :]
            waveforms.append(self._output_stage(audio, headroom_db, fade_out_ms))
        return waveforms

    def save_audio(
        self,
        audio: AudioInput,
        saving_path: str | Path | list[str | Path],
        sampling_rate: int | None = None,
    ):
        r"""
        Writes the decoded waveform(s) to WAV file(s).

        Args:
            audio (`torch.Tensor`, `np.ndarray` or a list of them):
                The waveform(s), as returned by [`~Zonos3MiniProcessor.decode`].
            saving_path (`str`, `Path` or a list of them):
                One path per waveform.
            sampling_rate (`int`, *optional*):
                Sampling rate of the waveforms, 44.1 kHz by default.
        """
        sampling_rate = self.feature_extractor.sampling_rate if sampling_rate is None else sampling_rate
        requires_backends(self, ["soundfile"])
        audio = make_list_of_audio(audio)
        if isinstance(saving_path, (str, Path)):
            saving_path = [saving_path]
        if len(audio) != len(saving_path):
            raise ValueError("The number of audio and saving paths must be the same.")
        for audio_value, path in zip(audio, saving_path):
            if isinstance(audio_value, torch.Tensor):
                audio_value = audio_value.cpu().float().numpy()
            sf.write(path, audio_value, sampling_rate)


__all__ = ["Zonos3MiniProcessor"]

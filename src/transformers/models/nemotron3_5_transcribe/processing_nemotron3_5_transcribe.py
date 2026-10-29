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
"""Processor class for Nemotron3_5Transcribe."""

import re
import unicodedata
from bisect import bisect_right

import numpy as np

from ...audio_utils import AudioInput, make_audio_chat_template_content, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import (
    auto_docstring,
    is_scipy_available,
    is_torch_available,
    is_torchaudio_available,
    requires_backends,
)
from ...utils.import_utils import requires


if is_torch_available():
    import torch

if is_torchaudio_available():
    from torchaudio.functional import forced_align, merge_tokens

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


# speaker-attributed transcripts open every speaker turn with a `<spk:N>` tag
SPEAKER_TAG_PATTERN = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)

# Word timestamps, ported from NeMo's `MultiSpeakerSOTWordTimestampAligner`. A "frame" is a frame of the word timestamp
# head (`subsampling_factor` spectrogram frames), a "tag" a `<spk:N>` speaker tag, a "speaker" a diarization speaker.
KEPT_TAG_PATTERN = re.compile(r"(?i:<spk:\d+>)|<(?:ev|sty):[^<>]*>")  # speaker, event and style tags
# the scripts that set the language of a transcript, by priority, as Unicode character name prefixes
SCRIPT_LANGUAGES = [
    ("ko", "HANGUL"),
    ("ja", "HIRAGANA"),
    ("ja", "KATAKANA"),
    ("zh", "CJK UNIFIED"),
    ("zh", "CJK COMPATIBILITY"),
    ("th", "THAI"),
]


def _lexical_characters(text: str) -> list[str]:
    """Letters, marks and numbers of `text`, NFC-normalized."""
    return [char for char in unicodedata.normalize("NFC", text) if unicodedata.category(char)[0] in "LMN"]


def _character_token_ids(tokenizer, character: str) -> list[int]:
    """A character is its own piece when the vocabulary has one, else its pieces without the bare word boundary."""
    token_ids = [tokenizer.convert_tokens_to_ids(character)]
    if token_ids == [tokenizer.unk_token_id]:
        word_boundary_id = tokenizer.convert_tokens_to_ids("▁")
        token_ids = [i for i in tokenizer.encode(" " + character, add_special_tokens=False) if i != word_boundary_id]
    if tokenizer.unk_token_id in token_ids:
        raise ValueError(f"Unrepresentable character: {character!r}")
    return token_ids


def _tokenize_transcript(tokenizer, text: str, language: str | None, blank_id: int) -> dict[int | None, list]:
    """
    `(word, token ids)` of the words of each tag of a transcript: the words before the first `<spk:N>` tag have tag
    `None`, a transcript without tags is tag 0. Words get a leading space, the SentencePiece prefix of the original
    tokenizer. Chinese and Japanese (`language`, detected from the script when `None`) are aligned character by character.
    """
    parts = SPEAKER_TAG_PATTERN.split(text.replace("—", " "))  # [text, tag, text, tag, text, ...]
    tags = [0 if len(parts) == 1 else None] + [int(tag) for tag in parts[1::2]]
    words = [(word, tag) for tag, part in zip(tags, parts[::2]) for word in part.split()]
    if language is None:
        names = [
            unicodedata.name(char, "") for word, _ in words if not KEPT_TAG_PATTERN.fullmatch(word) for char in word
        ]
        language = next((lang for lang, prefix in SCRIPT_LANGUAGES if any(n.startswith(prefix) for n in names)), None)
    if language == "th":
        raise ValueError("Word timestamps are not supported for Thai, which needs the `pythainlp` word segmenter.")

    streams = {}
    for word, tag in words:
        if language in ("zh", "ja") and not KEPT_TAG_PATTERN.fullmatch(word):
            units = [(char, _character_token_ids(tokenizer, char)) for char in _lexical_characters(word)]
        else:
            units = [(word, tokenizer.encode(" " + word, add_special_tokens=False))]
        if any(not token_ids or max(token_ids) >= blank_id for _, token_ids in units):
            raise ValueError("Tokenizer produced IDs outside the non-blank CTC vocabulary.")
        streams.setdefault(tag, []).extend(units)
    return streams


def _match_tags_to_speakers(
    log_probs: "torch.Tensor", streams: dict, speaker_probabilities: "torch.Tensor", threshold: float, blank_id: int
) -> dict[int | None, int | None]:
    """
    Diarization speaker of each tag, matched one to one on the tokens they share: a speaker collects the greedy tokens
    decoded while it is active, a tag the tokens of its words, and the IDF-weighted cosine similarity of their counts,
    penalized when their amounts of speech differ, is maximized by linear assignment. Unmatched tags get `None`.
    """
    vocab_size, num_speakers = log_probs.shape[1], speaker_probabilities.shape[1]
    targets = [[token for _, token_ids in words for token in token_ids] for words in streams.values()]
    # as the original, the greedy decoding chooses among the blank, first on ties, and the tokens of the transcript only
    classes = torch.tensor([blank_id] + sorted({token for target in targets for token in target}))
    greedy = classes[log_probs[:, classes].argmax(dim=1)]
    is_onset = (greedy != blank_id) & torch.cat([torch.tensor([True]), greedy[1:] != greedy[:-1]])
    tag_counts = torch.stack([torch.bincount(torch.tensor(target), minlength=vocab_size) for target in targets])
    speaker_counts = torch.stack(
        [
            torch.bincount(greedy[is_onset & (active >= threshold)], minlength=vocab_size)
            for active in speaker_probabilities.T
        ]
    )
    tag_counts, speaker_counts = tag_counts.double(), speaker_counts.double()

    idf = torch.log1p(num_speakers / (1 + (speaker_counts > 0).sum(dim=0).double()))
    tag_vectors, speaker_vectors = (tag_counts * idf).sqrt(), (speaker_counts * idf).sqrt()
    tag_norms, speaker_norms = (tag_vectors**2).sum(dim=1).sqrt(), (speaker_vectors**2).sum(dim=1).sqrt()
    similarity = tag_vectors @ speaker_vectors.T / (tag_norms[:, None] * speaker_norms[None, :]).clamp(min=1e-8)
    similarity -= 0.1 * ((tag_counts.sum(dim=1)[:, None] + 1) / (speaker_counts.sum(dim=1)[None, :] + 1)).log().abs()

    tag_to_speaker = dict.fromkeys(streams)
    for tag, speaker in zip(*linear_sum_assignment(-similarity.numpy())):
        tag_to_speaker[list(streams)[tag]] = int(speaker)
    return tag_to_speaker


def _align_words(
    log_probs: "torch.Tensor",
    words: list[tuple[str, list[int]]],
    probabilities: "torch.Tensor",
    blank_id: int,
    frame_duration: float,
    max_word_duration: float,
) -> list[dict] | None:
    """
    Start and end of each `(word, token ids)`, force-aligned on the frames where their speaker has a `probabilities`
    of at least 0.01, its log-probability added to the token log-probabilities. A word spans the frames of its tokens,
    for at most `max_word_duration`. `None` when the words do not fit on those frames.
    """
    target = [token for _, token_ids in words for token in token_ids]
    frames = (probabilities >= 0.01).nonzero().flatten()
    num_repeats = sum(token == next_token for token, next_token in zip(target, target[1:]))
    if len(frames) < len(target) + num_repeats:  # a blank must separate repeated tokens
        return None
    emissions = log_probs[frames] + probabilities[frames].clamp(min=1e-6).log()[:, None]
    emissions[:, blank_id] = log_probs[frames, blank_id]
    labels, scores = forced_align(emissions[None], torch.tensor([target]), blank=blank_id)
    spans = iter(merge_tokens(labels[0], scores[0], blank=blank_id))  # one span per token, in order

    timestamps = []
    for word, token_ids in words:
        word_spans = [next(spans) for _ in token_ids]
        start = frames[word_spans[0].start].item() * frame_duration
        end = min((frames[word_spans[-1].end - 1].item() + 1) * frame_duration, start + max_word_duration)
        timestamps.append({"word": word, "start": start, "end": end})
    return timestamps


def _refine_word_timestamps(
    words: list[dict], segments: list[tuple[float, float]], frame_duration: float, max_word_frames: int
) -> list[dict]:
    """
    Snaps the words of one speaker, in time order, to its diarization `segments`: the first word of a segment starts
    with it; a word ends where the next word of its segment starts, else with its segment; a word outside segments ends
    where the next word starts. Shifts and durations are capped at `max_word_frames` frames of at most 80 ms.
    """
    limit = max_word_frames * min(frame_duration, 0.08)
    segments = sorted(segments)
    starts = [start for start, _ in segments]
    # the segment containing the start of each word, `None` outside segments
    segment_of = []
    for word in words:
        index = bisect_right(starts, word["start"]) - 1
        segment_of.append(index if index >= 0 and word["start"] < segments[index][1] else None)

    for i, (word, segment) in enumerate(zip(words, segment_of)):
        is_first_of_segment = segment is not None and segment_of[i - 1 : i] != [segment]
        if is_first_of_segment and 0 < word["start"] - segments[segment][0] <= limit:
            word["start"] = segments[segment][0]
    for i, (word, segment) in enumerate(zip(words, segment_of)):
        if segment is not None and segment_of[i + 1 : i + 2] != [segment]:
            end = segments[segment][1]
        else:
            end = words[i + 1]["start"] if i + 1 < len(words) else word["end"]
        word["end"] = max(word["start"], min(end, word["start"] + limit))
    return words


def _resample_frames(probabilities: "torch.Tensor", num_frames: int) -> "torch.Tensor":
    """Linear interpolation of `probabilities` `(frames, speakers)` onto `num_frames` frames, first and last aligned."""
    if probabilities.shape[0] == num_frames:
        return probabilities
    positions = torch.linspace(0, probabilities.shape[0] - 1, num_frames)
    lower, upper = positions.floor().long(), positions.ceil().long()
    fraction = (positions - lower)[:, None]
    return probabilities[lower] * (1 - fraction) + probabilities[upper] * fraction


def _diarization_segments(is_active: "torch.Tensor", frame_duration: float, audio_duration: float) -> list[dict]:
    """Each run of activity of `is_active` `(speakers, frames)` as `{"speaker", "start", "end"}`, sorted by time."""
    transitions = torch.nn.functional.pad(is_active.to(torch.int8), (1, 1)).diff(dim=1)
    starts, ends = (transitions == 1).nonzero().tolist(), (transitions == -1).nonzero()[:, 1].tolist()
    segments = [
        {"speaker": speaker, "start": start * frame_duration, "end": min(end * frame_duration, audio_duration)}
        for (speaker, start), end in zip(starts, ends)
    ]
    segments = [segment for segment in segments if segment["end"] > segment["start"]]
    return sorted(segments, key=lambda segment: (segment["start"], segment["end"], segment["speaker"]))


class Nemotron3_5TranscribeProcessorKwargs(ProcessingKwargs, total=False):
    # call defaults (`padding`, `padding_side`) ship in the checkpoint's `tokenizer_config.json`; outputs are always
    # PyTorch tensors
    _defaults = {}


@requires(backends=("torch",))
@auto_docstring
class Nemotron3_5TranscribeProcessor(ProcessorMixin):
    valid_processor_kwargs = Nemotron3_5TranscribeProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        alignment_tokenizer,
        chat_template=None,
        audio_token="<|audio|>",
        default_transcription_prompt="Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.",
        subsampling_factor=8,
    ):
        r"""
        alignment_tokenizer (`PreTrainedTokenizerFast`):
            The vocabulary of the word timestamp head: 23,862 SentencePiece BPE pieces, the head's class 23,862 being the
            CTC blank. Only used by [`~Nemotron3_5TranscribeProcessor.align_word_timestamps`], saved in the
            `alignment_tokenizer/` subfolder. Prepend a space to reproduce the original SentencePiece ids of any text,
            including a text that starts with a tag: `alignment_tokenizer.encode(" " + text, add_special_tokens=False)`.
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            Special token used to represent audio inputs in the chat template.
        default_transcription_prompt (`str`, *optional*):
            Default prompt used by [`~Nemotron3_5TranscribeProcessor.apply_transcription_request`].
        subsampling_factor (`int`, *optional*, defaults to 8):
            Number of spectrogram frames per audio token, mirroring
            `Nemotron3_5TranscribeAsrAudioConfig.subsampling_factor`. The spectrogram is padded to a multiple of it.
        """
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.default_transcription_prompt = default_transcription_prompt
        self.subsampling_factor = subsampling_factor
        super().__init__(feature_extractor, tokenizer, alignment_tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[Nemotron3_5TranscribeProcessorKwargs],
    ) -> BatchFeature:
        r"""
        output_labels (bool, *optional*, default=False):
            Whether to return labels for training.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_features`, `input_features_mask`).
        """
        # Check only if passed explicitly as another value since by default we'll use `pt`
        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if output_labels:
            kwargs["return_mm_token_type_ids"] = True
        model_inputs = super().__call__(audio=audio, text=text, **kwargs)

        if output_labels:
            mm_token_type_ids = model_inputs.pop("mm_token_type_ids")
            labels = model_inputs["input_ids"].clone()
            labels[mm_token_type_ids != 0] = -100  # audio positions
            labels[labels == self.tokenizer.pad_token_id] = -100
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

    def _get_audio_token_length(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
        # one audio token per `subsampling_factor` spectrogram frames, the last group may be incomplete
        return (audio_lengths + self.subsampling_factor - 1) // self.subsampling_factor

    def _process_audio(self, audio, **kwargs):
        audio = self.feature_extractor.fetch_audio(audio)
        audio_inputs = self.feature_extractor(audio, **kwargs)
        # whole audio tokens: the spectrogram is padded to a multiple of `subsampling_factor` frames. The feature
        # extractor's `pad_to_multiple_of` cannot do it, it pads audio samples and the centered STFT adds a frame.
        padding = -audio_inputs["input_features"].shape[1] % self.subsampling_factor
        audio_inputs["input_features"] = torch.nn.functional.pad(audio_inputs["input_features"], (0, 0, 0, padding))
        audio_inputs["input_features_mask"] = torch.nn.functional.pad(audio_inputs.pop("attention_mask"), (0, padding))
        audio_inputs["num_audio_tokens"] = self._get_audio_token_length(audio_inputs["input_features_mask"].sum(-1))
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_audio_tokens = audio_inputs["num_audio_tokens"][audio_idx]
        return self.audio_token * num_audio_tokens

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["input_features_mask"]

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["num_audio_tokens"]

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[Nemotron3_5TranscribeProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Prepare inputs for automatic speech recognition without manually writing the default transcription prompt.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                each sample uses `default_transcription_prompt`.
            **kwargs:
                Additional keyword arguments forwarded to
                [`~Nemotron3_5TranscribeProcessor.apply_chat_template`] (for example `text_kwargs`, `audio_kwargs`, ...).
                Thinking is disabled unless `enable_thinking=True` is passed (the chat template enables it by default).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`Nemotron3_5TranscribeForConditionalGeneration.generate`].
        """
        audio_items: list[str | np.ndarray] = list(make_list_of_audio_chat_template(audio))
        audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if prompt is None:
            prompts = [self.default_transcription_prompt] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, list | tuple):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = []
            for item in prompt:
                if item is None:
                    prompts.append(self.default_transcription_prompt)
                elif isinstance(item, str):
                    prompts.append(item)
                else:
                    raise TypeError("Each prompt must be a string or `None`.")
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        make_audio_chat_template_content(audio_item),
                    ],
                }
            ]
            for prompt_text, audio_item in zip(prompts, audio_items)
        ]

        kwargs.setdefault("enable_thinking", False)
        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def extract_speaker_dict(self, text: str | list[str]) -> list[dict] | list[list[dict]]:
        """
        Splits a speaker-attributed transcript into speaker turns, in the format of
        [`VibeVoiceAsrProcessor.extract_speaker_dict`] without the timestamps, which this model does not predict.

        Args:
            text (`str` or `list[str]`):
                Single text or batch of texts, decoded with `skip_special_tokens=False`: the `<spk:N>` speaker tags are
                special tokens. Everything from the end-of-text token on is dropped.

        Returns:
            `list[dict]` for a single text, `list[list[dict]]` for a batch: one dict per speaker turn, with the
            `"Speaker"` index `N` of its `<spk:N>` tag and its `"Content"`. A text without tags is a single turn with
            `"Speaker": None`, as is the text before the first tag, if any.

        Example:

        ```python
        >>> processor.extract_speaker_dict("<spk:0> Hi, how are you? <spk:1> Fine, thanks.<|im_end|>")
        [{'Speaker': 0, 'Content': 'Hi, how are you?'}, {'Speaker': 1, 'Content': 'Fine, thanks.'}]
        ```
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        speaker_dicts = []
        for sample_text in texts:
            # in a batch, generation pads the finished sequences after their end-of-text token
            if self.tokenizer.eos_token is not None:
                sample_text = sample_text.split(self.tokenizer.eos_token)[0]
            # [text before the first tag, speaker, content, speaker, content, ...]
            parts = SPEAKER_TAG_PATTERN.split(sample_text)
            turns = [
                {"Speaker": int(speaker), "Content": content.strip()}
                for speaker, content in zip(parts[1::2], parts[2::2])
            ]
            if parts[0].strip() or not turns:
                turns.insert(0, {"Speaker": None, "Content": parts[0].strip()})
            speaker_dicts.append(turns)
        return speaker_dicts[0] if is_single else speaker_dicts

    def align_word_timestamps(
        self,
        transcriptions: str | list[str],
        word_timestamp_logits: "torch.Tensor",
        input_features_mask: "torch.Tensor",
        speaker_logits: "torch.Tensor | None" = None,
        language: str | list[str | None] | None = None,
        speaker_activity_threshold: float = 0.5,
        max_word_duration: float = 1.0,
        max_word_frames: int = 12,
    ) -> dict | list[dict]:
        """
        Times each word of generated transcripts and attributes it to a speaker: the words of each `<spk:N>` tag are
        force-aligned (`torchaudio.functional.forced_align`) on the frames of the word timestamp head where the
        diarization speaker matched to the tag is active, then snapped to that speaker's diarization segments. Port of
        NeMo's `MultiSpeakerSOTWordTimestampAligner`, with an exact search where it approximates long recordings.

        Args:
            transcriptions (`str` or `list[str]`):
                Generated text of each recording, decoded with `skip_special_tokens=False` to keep the `<spk:N>` tags.
            word_timestamp_logits (`torch.Tensor` of shape `(batch_size, num_frames * upsampling_factor, vocab_size)`):
                Logits of the word timestamp head (`word_timestamp_logits` of `generate(...,
                return_word_timestamp_logits=True)` or `model.get_word_timestamp_logits(...)`), the last class being the
                CTC blank. Their number of frames per encoder frame, the head's `upsampling_factor`, is read from their
                shape.
            input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
                Mask of the valid spectrogram frames of each recording, as returned by the processor: their number
                sets the duration of the recording, hence the time step of the head's frames.
            speaker_logits (`torch.Tensor` of shape `(batch_size, num_frames * subsampling_factor, num_speakers)`, *optional*):
                Logits of the diarization model (`speaker_logits` of `generate(..., return_word_timestamp_logits=True)`
                or of `get_audio_features`). Without them, each transcript is aligned as a single speaker and no
                diarization is returned.
            language (`str` or `list[str | None]`, *optional*):
                Language of the transcripts, one for all or one per transcript, detected from the script when `None`:
                `"zh"` and `"ja"` are aligned character by character, `"th"` is not supported.
            speaker_activity_threshold (`float`, *optional*, defaults to 0.5):
                Probability above which a diarization speaker is active.
            max_word_duration (`float`, *optional*, defaults to 1.0):
                Maximum duration of an aligned word, in seconds.
            max_word_frames (`int`, *optional*, defaults to 12):
                Maximum shift and duration of a word snapped to a diarization segment, in encoder frames of at most 80
                ms.

        Returns:
            `dict` or `list[dict]`: for the recording (or for each recording of a list of transcripts), as the word timestamps of the speech recognition pipeline, `"chunks"`
            (`{"text", "timestamp", "speaker"}` per word, `timestamp` = `(start, end)` in seconds, sorted by time;
            `speaker` is the `N` of the word's tag, `None` before the first tag and 0 in a transcript without tags),
            `"diarization"` (`{"speaker", "timestamp"}` activity segments of each diarization speaker),
            `"speaker_tag_to_diarization_speaker"` (`None` for an unmatched tag) and `"failed_speakers"` (the tags
            whose words could not be aligned, missing from `"chunks"`).

        Example:

        ```python
        >>> inputs = processor.apply_transcription_request(audio, prompt=speaker_attributed_prompt).to(model.device)
        >>> outputs = model.generate(**inputs, return_word_timestamp_logits=True, max_new_tokens=500)
        >>> transcriptions = processor.decode(
        ...     outputs.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=False
        ... )
        >>> timestamps = processor.align_word_timestamps(
        ...     transcriptions, outputs.word_timestamp_logits, inputs.input_features_mask, outputs.speaker_logits
        ... )
        >>> timestamps[0]["chunks"][0]  # {'text': ..., 'timestamp': (..., ...), 'speaker': 0}
        ```
        """
        requires_backends(self, ["scipy", "torchaudio"])
        is_single = isinstance(transcriptions, str)
        transcriptions = [transcriptions] if is_single else transcriptions
        languages = language if isinstance(language, list) else [language] * len(transcriptions)
        blank_id = word_timestamp_logits.shape[-1] - 1
        # the head's frames per encoder frame: the logits span the encoder frames of the padded spectrograms
        num_padded_frames = self._get_audio_token_length(input_features_mask.shape[-1])
        upsampling_factor, remainder = divmod(word_timestamp_logits.shape[1], num_padded_frames)
        if remainder or not upsampling_factor:
            raise ValueError(
                f"`word_timestamp_logits` has {word_timestamp_logits.shape[1]} frames, not a multiple of the "
                f"{num_padded_frames} encoder frames of `input_features_mask`."
            )
        hop_length, sampling_rate = self.feature_extractor.hop_length, self.feature_extractor.sampling_rate
        dropped_tokens = [
            token for token in self.tokenizer.all_special_tokens if not KEPT_TAG_PATTERN.fullmatch(token)
        ]

        results = []
        for i, (transcription, language, num_spectrogram_frames) in enumerate(
            zip(transcriptions, languages, input_features_mask.sum(-1).tolist(), strict=True)
        ):
            num_encoder_frames = self._get_audio_token_length(num_spectrogram_frames)
            num_frames = num_encoder_frames * upsampling_factor
            # the recording's duration to the spectrogram hop below, which NeMo takes from the audio samples
            audio_duration = num_spectrogram_frames * hop_length / sampling_rate
            frame_duration = audio_duration / num_frames
            for token in dropped_tokens:
                transcription = transcription.replace(token, " ")
            streams = _tokenize_transcript(self.alignment_tokenizer, transcription, language, blank_id)
            if not streams:
                raise ValueError(f"Transcription {i} is empty once its special tokens are removed.")
            log_probs = word_timestamp_logits[i, :num_frames].detach().log_softmax(-1).float().cpu()

            # the speaker of each tag, its probability per frame and its diarization segments
            speaker_probabilities, tag_to_speaker, diarization = None, dict.fromkeys(streams), []
            if speaker_logits is not None:
                logits = speaker_logits[i].detach().float().cpu()
                # as the model: the mean of the probabilities over the spectrogram frames of each frame
                frame_logits = logits[: num_encoder_frames * self.subsampling_factor].unflatten(
                    0, (num_encoder_frames, -1)
                )
                speaker_probabilities = _resample_frames(frame_logits.sigmoid().mean(1), num_frames)
                tag_to_speaker = _match_tags_to_speakers(
                    log_probs, streams, speaker_probabilities, speaker_activity_threshold, blank_id
                )
                is_active = logits[:num_spectrogram_frames].sigmoid() > speaker_activity_threshold
                diarization = _diarization_segments(is_active.T, hop_length / sampling_rate, audio_duration)

            words, failed_tags = [], []
            for tag, stream in streams.items():
                speaker = tag_to_speaker[tag]
                if speaker_probabilities is None:
                    probabilities = torch.ones(num_frames)
                elif speaker is None:
                    probabilities = torch.zeros(num_frames)  # an unmatched tag has no frame to align on
                else:
                    probabilities = speaker_probabilities[:, speaker]
                timestamps = _align_words(
                    log_probs, stream, probabilities, blank_id, frame_duration, max_word_duration
                )
                if timestamps is None:
                    failed_tags.append(tag)
                    continue
                segments = [(s["start"], s["end"]) for s in diarization if s["speaker"] == speaker]
                timestamps = _refine_word_timestamps(
                    timestamps, segments, frame_duration, max_word_frames * upsampling_factor
                )
                words += [{**word, "speaker": tag} for word in timestamps]

            words = sorted(words, key=lambda word: (word["start"], word["end"]))
            results.append(
                {
                    "chunks": [
                        {"text": word["word"], "timestamp": (word["start"], word["end"]), "speaker": word["speaker"]}
                        for word in words
                    ],
                    "diarization": [
                        {"speaker": segment["speaker"], "timestamp": (segment["start"], segment["end"])}
                        for segment in diarization
                    ],
                    "speaker_tag_to_diarization_speaker": tag_to_speaker,
                    "failed_speakers": failed_tags,
                }
            )
        return results[0] if is_single else results


__all__ = ["Nemotron3_5TranscribeProcessor"]

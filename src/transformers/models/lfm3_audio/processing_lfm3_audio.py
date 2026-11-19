# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from bisect import bisect_right
from pathlib import Path
from typing import Any

import torch

from ...audio_utils import AudioInput, load_audio, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AUDIO_TOKENIZER_KWARGS, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_soundfile_available
from ...utils.chat_template_utils import _get_template_variables
from ...utils.import_utils import requires


if is_soundfile_available():
    import soundfile as sf


@auto_docstring
class Lfm3AudioProcessor(ProcessorMixin):
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "LiquidCodecModel"

    text_kwargs = {"padding": True}

    # set by `__call__` while it expands the `audio_out_token` placeholders
    _audio_out_replacements: list[str] = []

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        codec_feature_extractor=None,
        audio_tokenizer=None,
        chat_template=None,
        audio_token: str = "<|audio|>",
        audio_out_token: str = "<|audio_out|>",
        audio_start_token: str = "<|audio_start|>",
        num_audio_tokens_per_chunk: int = 12,
        max_audio_chunks_per_context: int = 31,
        **kwargs,
    ):
        r"""
        feature_extractor ([`Lfm3AudioFeatureExtractor`]):
            The feature extractor computing the log-mel spectrograms of the input clips.
        tokenizer ([`PreTrainedTokenizerFast`]):
            The LFM3 tokenizer.
        codec_feature_extractor ([`LiquidCodecFeatureExtractor`]):
            The codec feature extractor, used to encode `audio_out` waveforms.
        audio_tokenizer ([`LiquidCodecModel`], *optional*):
            The codec, used to encode `audio_out` waveforms and to decode generated speech.
        chat_template (`str`, *optional*):
            A Jinja template to convert conversations into prompts.
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            The placeholder token of one speech encoder frame.
        audio_out_token (`str`, *optional*, defaults to `"<|audio_out|>"`):
            The placeholder token of one codec frame.
        audio_start_token (`str`, *optional*, defaults to `"<|audio_start|>"`):
            The token opening a speech segment.
        num_audio_tokens_per_chunk (`int`, *optional*, defaults to 12):
            The number of speech encoder frames in one chunk of a live transcription.
        max_audio_chunks_per_context (`int`, *optional*, defaults to 31):
            The maximum number of chunks in one live transcription context (30 seconds).
        """
        self.audio_token = getattr(tokenizer, "audio_token", audio_token)
        self.audio_out_token = getattr(tokenizer, "audio_out_token", audio_out_token)
        self.audio_start_token = getattr(tokenizer, "audio_start_token", audio_start_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_out_token_id = tokenizer.convert_tokens_to_ids(self.audio_out_token)
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids(self.audio_start_token)
        self.num_audio_tokens_per_chunk = num_audio_tokens_per_chunk
        self.max_audio_chunks_per_context = max_audio_chunks_per_context

        super().__init__(
            feature_extractor,
            tokenizer,
            codec_feature_extractor,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template,
        )

    @property
    def num_samples_per_audio_chunk(self) -> int:
        """Samples per chunk of a live transcription."""
        num_frames = self.num_audio_tokens_per_chunk * self.feature_extractor.encoder_stride
        return (num_frames - 1) * self.feature_extractor.hop_length + self.feature_extractor.n_fft

    @property
    def audio_chunk_hop_length(self) -> int:
        """Samples between the starts of two consecutive chunks."""
        num_frames = self.num_audio_tokens_per_chunk * self.feature_extractor.encoder_stride
        return num_frames * self.feature_extractor.hop_length

    @property
    def audio_eos_code(self) -> int:
        """Code of the end-of-speech frame."""
        return self.audio_tokenizer.config.codebook_size

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        audio_out: AudioInput | None = None,
        is_last_audio_chunk: bool | None = None,
        output_labels: bool | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        r"""
        audio_out (`AudioInput` or `torch.LongTensor`, or a list of them, *optional*):
            Speech segments, one per `audio_out_token` in the text: 24 kHz waveforms (encoded by the codec, end-of-speech
            frame appended) or codes `(num_codebooks, num_frames)`. Returned as `audio_input_ids`
            `(num_frames, num_codebooks)`.
        is_last_audio_chunk (`bool`, *optional*):
            Live transcription: whether `audio` is the last chunk of the stream. Returned as is.
        output_labels (`bool`, *optional*):
            Whether to return `labels` (`input_ids` with placeholders, `audio_start_token` and padding at `-100`) and,
            with `audio_out`, `audio_labels` (copy of `audio_input_ids`). Requires `return_tensors="pt"`.
        """
        audio_input_ids = None
        if audio_out is not None:
            if text is None:
                raise ValueError("`audio_out` requires `text`, which holds its placeholders.")

            audio_input_ids = self._prepare_audio_out(audio_out)
            self._audio_out_replacements = [self.audio_out_token * len(frames) for frames in audio_input_ids]

        try:
            outputs = super().__call__(text=text, audio=audio, **kwargs)
        finally:
            self._audio_out_replacements = []

        if audio_input_ids is not None:
            audio_input_ids = torch.cat(audio_input_ids)
            self._check_audio_out_not_truncated(outputs["input_ids"], num_frames=len(audio_input_ids))
            merged_kwargs = self._merge_kwargs(tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)
            return_tensors = merged_kwargs["text_kwargs"].get("return_tensors")
            outputs["audio_input_ids"] = BatchFeature({"ids": audio_input_ids}, tensor_type=return_tensors)["ids"]

        if is_last_audio_chunk is not None:
            outputs["is_last_audio_chunk"] = is_last_audio_chunk

        if output_labels:
            if not torch.is_tensor(outputs["input_ids"]):
                raise ValueError('`output_labels` requires `return_tensors="pt"`.')

            labels = outputs["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.audio_out_token_id] = -100
            labels[labels == self.audio_start_token_id] = -100
            labels[outputs["attention_mask"] == 0] = -100
            outputs["labels"] = labels

            if audio_input_ids is not None:
                outputs["audio_labels"] = outputs["audio_input_ids"].clone()

        return outputs

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]] | list[list[dict[str, Any]]],
        chat_template: str | None = None,
        generation_mode: str | list[str] | None = None,
        **kwargs,
    ):
        r"""
        [`~ProcessorMixin.apply_chat_template`] with the output mode of each answer.

        Args:
            generation_mode (`str` or `list[str]`, *optional*):
                Marker opening the assistant answer: `"text"` (none), `"audio"` (`<|audio_start|>`) or `"interleaved"`
                (`<|mixed_start|>`). One for all conversations, or one each. Requires `add_generation_prompt=True`.
        """
        if generation_mode is None:
            return super().apply_chat_template(conversation, chat_template=chat_template, **kwargs)

        if not kwargs.get("add_generation_prompt"):
            raise ValueError("`generation_mode` opens the assistant answer: it requires `add_generation_prompt=True`.")

        template = self._resolve_chat_template(chat_template)
        if "generation_mode" not in _get_template_variables(template):
            raise ValueError("This chat template does not support `generation_mode`.")

        is_batched = isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        conversations = conversation if is_batched else [conversation]
        if isinstance(generation_mode, str):
            generation_mode = [generation_mode] * len(conversations)
        elif not is_batched or len(generation_mode) != len(conversations):
            raise ValueError(
                f"Got {len(generation_mode)} generation modes for {len(conversations)} conversations: pass one mode, "
                "or one per conversation."
            )

        for mode in generation_mode:
            if mode not in ("text", "audio", "interleaved"):
                raise ValueError(f'`generation_mode` must be "text", "audio" or "interleaved", got {mode!r}.')

        # a list can not be a template kwarg: the mode goes on each conversation's last message
        conversations = [
            [*messages[:-1], {**messages[-1], "generation_mode": mode}]
            for messages, mode in zip((getattr(chat, "messages", chat) for chat in conversations), generation_mode)
        ]

        return super().apply_chat_template(
            conversations if is_batched else conversations[0], chat_template=template, **kwargs
        )

    def _check_audio_out_not_truncated(self, input_ids, num_frames: int):
        input_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        num_placeholders = sum(list(ids).count(self.audio_out_token_id) for ids in input_ids)
        if num_placeholders != num_frames:
            raise ValueError(
                f"Found {num_placeholders} {self.audio_out_token} tokens after tokenization for {num_frames} speech "
                "frames. Disable truncation or increase `max_length`."
            )

    def _prepare_audio_out(self, audio_out) -> list[torch.Tensor]:
        """Frames `(num_frames, num_codebooks)` of each `audio_out` entry."""
        entries = audio_out if isinstance(audio_out, (list, tuple)) else [audio_out]
        is_codes = [torch.is_tensor(entry) and not entry.is_floating_point() and entry.ndim == 2 for entry in entries]
        waveforms = [entry for entry, codes in zip(entries, is_codes) if not codes]
        encoded = iter(self._encode_audio_out(waveforms) if waveforms else [])
        return [entry.T.cpu() if codes else next(encoded) for entry, codes in zip(entries, is_codes)]

    @torch.no_grad()
    def _encode_audio_out(self, waveforms: list) -> list[torch.Tensor]:
        """Encodes 24 kHz waveforms with the codec, end-of-speech frame appended."""
        if getattr(self, "audio_tokenizer", None) is None:
            raise ValueError("Encoding `audio_out` waveforms requires the codec. Load it or pass codes instead.")

        sampling_rate = self.codec_feature_extractor.sampling_rate
        device = self.audio_tokenizer.device
        inputs = self.codec_feature_extractor(
            waveforms, sampling_rate=sampling_rate, return_tensors="pt", device=device
        )
        outputs = self.audio_tokenizer.encode(
            inputs["input_features"].to(device, self.audio_tokenizer.dtype),
            inputs["input_features_mask"].to(device),
        )

        frames = []
        for codes, mask in zip(outputs.audio_codes.cpu(), outputs.audio_codes_mask.cpu()):
            codes = codes[:, mask]
            eos_frame = torch.full_like(codes[:, :1], self.audio_eos_code)
            frames.append(torch.cat([codes, eos_frame], dim=-1).T)

        return frames

    def _multimodal_inputs_from_conversations(self, conversations, audio_loading_params, load_audio_from_video=False):
        """Also collects the `audio_out` blocks: `codes`, or a waveform (`audio`, `url`, `path`) at the codec rate."""
        batch_images, batch_videos, batch_audios, additional_inputs = super()._multimodal_inputs_from_conversations(
            conversations, audio_loading_params, load_audio_from_video
        )
        loading_params = {**audio_loading_params, "sampling_rate": self.codec_feature_extractor.sampling_rate}

        audio_out = []
        for conversation in conversations:
            for message in conversation:
                content = message.get("content") or []
                if isinstance(content, str):
                    continue
                for block in content:
                    if block["type"] != "audio_out":
                        continue
                    if "codes" in block:
                        audio_out.append(block["codes"])
                        continue
                    source = next((block[key] for key in ("audio", "url", "path") if key in block), None)
                    if source is not None:
                        audio_out.append(load_audio(source, **loading_params))

        if audio_out:
            additional_inputs["audio_out"] = audio_out

        return batch_images, batch_videos, batch_audios, additional_inputs

    # TODO @eustlb: let `ProcessorMixin.get_text_with_replacements` take extra placeholder groups, instead of a second
    # pass with `audio_token` swapped and the offsets of both passes merged
    def get_text_with_replacements(
        self, text: list[str], images_replacements=[], videos_replacements=[], audio_replacements=[]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        text, offsets = super().get_text_with_replacements(
            text, images_replacements, videos_replacements, audio_replacements
        )
        if not self._audio_out_replacements:
            return text, offsets

        # second pass: expand the `audio_out_token` placeholders
        audio_token, self.audio_token = self.audio_token, self.audio_out_token
        try:
            text, audio_out_offsets = super().get_text_with_replacements(
                text, audio_replacements=self._audio_out_replacements
            )
        finally:
            self.audio_token = audio_token

        # merge both passes' offsets: `span` into the original text, `new_span` into the final one
        merged_offsets = []
        for first, second in zip(offsets or [[] for _ in text], audio_out_offsets):
            first_ends = [entry["new_span"][1] for entry in first]
            first_gains = [0] + [entry["new_span"][1] - entry["span"][1] for entry in first]
            second_ends = [entry["span"][1] for entry in second]
            second_gains = [0] + [entry["new_span"][1] - entry["span"][1] for entry in second]

            merged = []
            for entry in first:
                gain = second_gains[bisect_right(second_ends, entry["new_span"][0])]
                new_span = (entry["new_span"][0] + gain, entry["new_span"][1] + gain)
                merged.append({**entry, "new_span": new_span})
            for entry in second:
                gain = first_gains[bisect_right(first_ends, entry["span"][0])]
                span = (entry["span"][0] - gain, entry["span"][1] - gain)
                merged.append({**entry, "type": "audio_out", "span": span})
            merged_offsets.append(sorted(merged, key=lambda entry: entry["span"][0]))

        return text, merged_offsets

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if text is None:
            return

        num_placeholders = sum(sample.count(self.audio_out_token) for sample in text)
        num_segments = len(self._audio_out_replacements)
        if num_placeholders != num_segments:
            raise ValueError(
                f"Found {num_placeholders} {self.audio_out_token} tokens in the text but {num_segments} entries in "
                "`audio_out`."
            )

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        # one token per `encoder_stride` frames, rounded up
        num_frames = int(audio_inputs["input_features_mask"][audio_idx].sum())
        num_tokens = -(num_frames // -self.feature_extractor.encoder_stride)
        return self.audio_token * num_tokens

    def decode(self, *args, audio_sequences: torch.LongTensor | None = None, **kwargs):
        """Decodes token ids to text, or generated speech (`audio_sequences`, `audio_sequences_mask`) to waveforms."""
        if audio_sequences is None and args and torch.is_tensor(args[0]) and args[0].ndim == 3:
            audio_sequences, *args = args
        if audio_sequences is not None:
            return self._decode_audio(audio_sequences, *args, **kwargs)
        return super().decode(*args, **kwargs)

    @torch.no_grad()
    def _decode_audio(
        self, audio_sequences: torch.LongTensor, audio_sequences_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Generated speech -> 24 kHz waveforms, one per speech segment, in batch order.

        Args:
            audio_sequences (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
                From `generate`.
            audio_sequences_mask (`torch.Tensor` of shape `(batch_size, num_frames)`):
                From `generate`: false on padding and end-of-speech frames.
        """
        if getattr(self, "audio_tokenizer", None) is None:
            raise ValueError("Decoding speech requires the codec. Load the processor with its `audio_tokenizer`.")

        # segments: runs of valid frames
        segments = []
        for codes, mask in zip(audio_sequences, audio_sequences_mask.bool().cpu()):
            padded_mask = torch.cat((mask.new_zeros(1), mask, mask.new_zeros(1))).int()
            edges = torch.diff(padded_mask).nonzero()[:, 0]
            segments.extend(codes[:, start:end] for start, end in edges.view(-1, 2).tolist())

        if not segments:
            return []

        num_frames = torch.tensor([segment.shape[-1] for segment in segments])
        codes = torch.nn.utils.rnn.pad_sequence([segment.T for segment in segments], batch_first=True).transpose(1, 2)
        mask = torch.arange(codes.shape[-1]) < num_frames[:, None]

        device = self.audio_tokenizer.device
        decoded = self.audio_tokenizer.decode(codes.to(device), mask.to(device))

        return [audio[keep] for audio, keep in zip(decoded.audio_values.cpu(), decoded.audio_values_mask.cpu())]

    @requires(backends=("soundfile",))
    def save_audio(self, audio: AudioInput, saving_path: str | Path | list[str | Path]):
        """Saves 24 kHz waveforms (e.g. from [`~Lfm3AudioProcessor.decode`]) to audio files."""
        audio = make_list_of_audio(audio)
        saving_path = [saving_path] if isinstance(saving_path, (str, Path)) else list(saving_path)
        if len(audio) != len(saving_path):
            raise ValueError(f"Got {len(audio)} waveforms but {len(saving_path)} saving paths.")

        sampling_rate = self.audio_tokenizer.config.sampling_rate
        for waveform, path in zip(audio, saving_path):
            if torch.is_tensor(waveform):
                waveform = waveform.cpu().float().numpy()
            sf.write(path, waveform, sampling_rate)

    # TODO: @eustlb, to be standardized in core
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, audio_tokenizer_kwargs: dict | None = None, **kwargs):
        r"""
        audio_tokenizer_kwargs (`dict`, *optional*):
            Kwargs passed only to the audio tokenizer's `from_pretrained`, e.g. `{"encoder_config": None}` to load
            the codec without its encoder.
        """
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if audio_tokenizer_kwargs:
            # loaded again: the base class passes the audio tokenizer its Hub access and placement kwargs only
            audio_tokenizer = processor.audio_tokenizer
            hub_kwargs = {key: kwargs[key] for key in AUDIO_TOKENIZER_KWARGS if key in kwargs}
            processor.audio_tokenizer = type(audio_tokenizer).from_pretrained(
                audio_tokenizer.name_or_path, **hub_kwargs, **audio_tokenizer_kwargs
            )
        return processor

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["audio_input_ids"]))


__all__ = ["Lfm3AudioProcessor"]

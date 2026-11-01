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
"""Processor class for LFM3-Audio."""

import math
import re
from typing import TypedDict

import numpy as np
import torch

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import TensorType, auto_docstring


class Lfm3AudioFeatureExtractorKwargs(TypedDict, total=False):
    """
    sampling_rate (`int`, *optional*):
        The sampling rate of the clips, which must be the extractor's own; it is checked, never resampled.
    return_tensors (`str` or [`~utils.TensorType`], *optional*):
        The tensor type of the returned features.
    load_audio_backend (`str`, *optional*):
        The decoding backend `apply_chat_template` loads audio with; the extractor itself ignores it.
    """

    # the extractor always right-pads to whole frames and returns the mask, so the padding, truncation and
    # attention-mask options of `AudioKwargs` are not offered rather than silently dropped
    sampling_rate: int | None
    return_tensors: str | TensorType | None
    load_audio_backend: str | None


class Lfm3AudioProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: Lfm3AudioFeatureExtractorKwargs
    _defaults = {}


@auto_docstring
class Lfm3AudioProcessor(ProcessorMixin):
    r"""
    Wraps the LFM3-Audio feature extractor, the LFM3 tokenizer and optionally the Liquid codec. The text carries one
    `audio_token` per input clip and one `audio_out_token` per output clip or speech frame of the prompt; the
    processor expands the clips to their encoder tokens (one per 8 spectrogram frames) and codec frames. The two
    placeholder tokens and the prompt format come from the tokenizer files: the model replaces the placeholders'
    embeddings with the speech features, so they never reach the model as text.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "Lfm3AudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "LiquidCodecModel"
    valid_processor_kwargs = Lfm3AudioProcessorKwargs

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        audio_tokenizer=None,
        chat_template=None,
        audio_token: str = "<|audio|>",
        audio_out_token: str = "<|audio_out|>",
        **kwargs,
    ):
        r"""
        feature_extractor ([`Lfm3AudioFeatureExtractor`]):
            The feature extractor computing the log-mel spectrograms.
        tokenizer ([`PreTrainedTokenizerFast`]):
            The LFM3 tokenizer.
        audio_tokenizer ([`LiquidCodecModel`], *optional*):
            The codec turning 24 kHz output clips into speech frames.
        chat_template (`str`, *optional*):
            A Jinja template to turn conversations into prompts.
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            The placeholder token of one encoded frame of an input clip.
        audio_out_token (`str`, *optional*, defaults to `"<|audio_out|>"`):
            The placeholder token of one speech frame of the prompt.
        """
        self.audio_token = getattr(tokenizer, "audio_token", audio_token)
        self.audio_out_token = getattr(tokenizer, "audio_out_token", audio_out_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_out_token_id = tokenizer.convert_tokens_to_ids(self.audio_out_token)
        super().__init__(feature_extractor, tokenizer, audio_tokenizer=audio_tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(self, text=None, audio: AudioInput | None = None, audio_out: AudioInput | None = None, **kwargs):
        r"""
        audio_out (`np.ndarray`, `torch.Tensor` or a list of them, *optional*):
            24 kHz clips, one per `audio_out_token` in the text, encoded with the codec into `audio_out_codes`; the
            placeholder is repeated once per frame plus once for the end-of-speech frame that closes every clip.
        """
        # the shared audio loader indexes the first clip before `validate_inputs` runs
        for name, clips in (("audio", audio), ("audio_out", audio_out)):
            if isinstance(clips, (list, tuple)) and len(clips) == 0:
                raise ValueError(f"`{name}` must hold at least one clip; pass `{name}=None` for prompts without one")
        if audio_out is None:
            return super().__call__(text=text, audio=audio, **kwargs)
        if text is None:
            raise ValueError("`audio_out` needs the text holding its placeholders")
        single = isinstance(text, str)
        prompts = [text] if single else list(text)
        expanded, codes, audio_out_offsets = self._expand_audio_out(prompts, audio_out)
        merged = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        return_tensors = merged["text_kwargs"].get("return_tensors")
        outputs = super().__call__(text=expanded[0] if single else expanded, audio=audio, **kwargs)
        self._check_audio_out_placeholders(outputs["input_ids"], codes.shape[0])
        if "text_replacement_offsets" in outputs:
            # the shared expansion saw the text after the speech placeholders were repeated
            outputs["text_replacement_offsets"] = self._compose_replacement_offsets(
                prompts, audio_out_offsets, outputs["text_replacement_offsets"]
            )
        outputs["audio_out_codes"] = self._as_return_tensors(codes, return_tensors)
        return outputs

    def _check_audio_out_placeholders(self, input_ids, num_frames: int) -> None:
        # truncation may have dropped placeholders the codes still cover
        rows = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        rows = [rows] if rows and isinstance(rows[0], int) else rows
        num_placeholders = sum(row.count(self.audio_out_token_id) for row in rows)
        if num_placeholders != num_frames:
            raise ValueError(
                f"Found {num_placeholders} {self.audio_out_token} tokens after tokenization for {num_frames} "
                "speech frames; disable truncation or shorten the prompt"
            )

    @staticmethod
    def _as_return_tensors(codes: torch.Tensor, return_tensors):
        return BatchFeature({"codes": codes}, tensor_type=return_tensors)["codes"]

    @staticmethod
    def _rewrite_with_offsets(prompt: str, pattern: str, replace, kind: str) -> tuple[str, list[dict]]:
        """Rewrites `prompt` like `ProcessorMixin.get_text_with_replacements`, with the same offset records."""
        pieces, row, last, offset = [], [], 0, 0
        for match in re.finditer(pattern, prompt):
            start, end = match.span()
            replacement = replace(match)
            pieces.extend((prompt[last:start], replacement))
            row.append(
                {
                    "type": kind,
                    "span": (start, end),
                    "new_span": (start + offset, start + offset + len(replacement)),
                    "text": match.group(),
                    "replacement": replacement,
                }
            )
            offset += len(replacement) - (end - start)
            last = end
        pieces.append(prompt[last:])
        return "".join(pieces), row

    @staticmethod
    def _compose_replacement_offsets(prompts: list[str], first: list[list[dict]], second) -> list[list[dict]]:
        """The records of two successive rewrites (the caller's text A to B, then B to C) as one rewrite of A: an entry
        of the second that covers entries of the first absorbs them, so every entry spans the caller's text."""
        second = second or [[] for _ in prompts]
        composed = []
        for prompt, row_first, row_second in zip(prompts, first, second):

            def to_first(position, row_first=row_first):
                return position - sum(
                    len(r["replacement"]) - (r["span"][1] - r["span"][0])
                    for r in row_first
                    if r["new_span"][1] <= position
                )

            mapped = []
            for entry in row_second:
                start, end = to_first(entry["span"][0]), to_first(entry["span"][1])
                mapped.append({**entry, "span": (start, end), "text": prompt[start:end]})
            kept = [
                r
                for r in row_first
                if not any(m["span"][0] <= r["span"][0] and r["span"][1] <= m["span"][1] for m in mapped)
            ]
            row, offset = [], 0
            for entry in sorted(kept + mapped, key=lambda r: r["span"]):
                start, end = entry["span"]
                row.append({**entry, "new_span": (start + offset, start + offset + len(entry["replacement"]))})
                offset += len(entry["replacement"]) - (end - start)
            composed.append(row)
        return composed

    def _expand_audio_out(
        self, prompts: list[str], audio_out: AudioInput
    ) -> tuple[list[str], torch.Tensor, list[list[dict]]]:
        if getattr(self, "audio_tokenizer", None) is None:
            raise ValueError("`audio_out` requires the processor to hold the codec as `audio_tokenizer`")
        clips = make_list_of_audio(audio_out)
        if sum(prompt.count(self.audio_out_token) for prompt in prompts) != len(clips):
            raise ValueError(f"The number of {self.audio_out_token} tokens in the text must match the number of clips")
        codes = [self.encode_audio(clip) for clip in clips]
        # the model ends speech with a frame of end-of-speech codes before the next text token; the prompt and the
        # training targets carry it the same way `generate(return_audio_eos=True)` does
        eos_frame = torch.full((1, codes[0].shape[1]), self.audio_eos_code, dtype=codes[0].dtype)
        codes = [torch.cat((frames, eos_frame)) for frames in codes]
        frame_counts = iter(code.shape[0] for code in codes)
        expanded, offsets = [], []
        for prompt in prompts:
            prompt, row = self._rewrite_with_offsets(
                prompt,
                re.escape(self.audio_out_token),
                lambda match: self.audio_out_token * next(frame_counts),
                "audio_out",
            )
            expanded.append(prompt)
            offsets.append(row)
        return expanded, torch.cat(codes), offsets

    @property
    def audio_eos_code(self) -> int:
        """The end-of-speech code: the first id after the codec's vocabulary."""
        return math.prod(self.audio_tokenizer.config.fsq_levels)

    @torch.no_grad()
    def encode_audio(self, audio) -> torch.Tensor:
        codec = self.audio_tokenizer
        waveform = torch.as_tensor(audio if torch.is_tensor(audio) else np.asarray(audio), dtype=torch.float32)
        # the quantizer rounds non-finite values to INT64_MIN, which would surface as codes
        if waveform.ndim != 1 or not torch.isfinite(waveform).all():
            raise ValueError("`audio_out` clips must be mono waveforms of finite samples")
        # an offloaded codec keeps some weights on `meta`: send the clip to a device that holds weights, and the hooks
        # move it on from there
        device = next(
            (param.device for param in codec.parameters() if param.device.type != "meta"), torch.device("cpu")
        )
        # the codec's return type follows its config's `return_dict`
        codes = codec.encode(waveform.to(device)[None], return_dict=True).audio_codes[0].T.cpu()
        # a compiled codec skips its latent check, and a non-finite latent rounds to INT64_MIN
        if codes.numel() and not 0 <= int(codes.min()) <= int(codes.max()) < self.audio_eos_code:
            raise ValueError("The audio overflows the codec; expected `audio_out` samples in [-1, 1]")
        return codes

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if text is None:
            return
        prompts = [text] if isinstance(text, str) else text
        num_placeholders = sum(prompt.count(self.audio_token) for prompt in prompts)
        num_clips = 0 if audio is None else len(make_list_of_audio(audio))
        if num_placeholders != num_clips:
            raise ValueError(
                f"Found {num_placeholders} {self.audio_token} tokens in the text but received {num_clips} clips"
            )

    def _process_audio(self, audio: AudioInput, **kwargs):
        # the backend only serves `apply_chat_template`'s audio loading
        kwargs.pop("load_audio_backend", None)
        audio_inputs = self.feature_extractor(audio, **kwargs)
        lengths = np.asarray(audio_inputs["feature_attention_mask"]).sum(-1).tolist()
        audio_inputs["num_audio_tokens"] = [
            self.feature_extractor.num_encoder_tokens(int(length)) for length in lengths
        ]
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(lengths))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * audio_inputs["num_audio_tokens"][audio_idx]

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_audio_tokens"]

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["audio_out_codes"]))


__all__ = ["Lfm3AudioProcessor"]

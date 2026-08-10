# Copyright 2026 The Dots Studio team and the HuggingFace Inc. team. All rights reserved.
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
"""Processor for Dots 3 Note Preview text, image, video, and audio inputs."""

from pathlib import Path

from ...image_utils import SizeDict
from ...processing_utils import ProcessingKwargs, ProcessorMixin, VideosKwargs
from ...utils import auto_docstring


_RELEASE_VISION_SIZE = SizeDict(shortest_edge=56 * 56, longest_edge=(36 * 28) ** 2)
_QWEN2_VL_IMAGE_DEFAULT_SIZE = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
_QWEN2_VL_VIDEO_DEFAULT_SIZE = {"shortest_edge": 128 * 28 * 28, "longest_edge": 28 * 28 * 768}


class Dots3NoteVideosKwargs(VideosKwargs, total=False):
    """
    seq (`int`, *optional*, defaults to 131072):
        Maximum sequence length used to budget video, audio, and output tokens.
    output_reserve (`int`, *optional*):
        Number of sequence tokens reserved for generation. Defaults to one quarter of `seq`.
    audio_cap (`float`, *optional*, defaults to 1.0):
        Maximum fraction of the input token budget that may be used by the video's audio track.
    audio_sr (`int`, *optional*, defaults to 16000):
        Sampling rate used when decoding the video's audio track.
    k_mode (`str`, *optional*, defaults to `"eval_ek"`):
        Strategy used to group sampled frames and interleave audio segments.
    max_new_tokens (`int`, *optional*, defaults to 0):
        Minimum number of sequence tokens reserved for generated output.
    video_question (`str`, *optional*):
        Question included in the deterministic preprocessing record key.
    jpeg_quality (`int`, *optional*, defaults to 85):
        JPEG quality used for the training-consistent frame round trip.
    """

    seq: int
    output_reserve: int | None
    audio_cap: float
    audio_sr: int
    k_mode: str
    max_new_tokens: int
    video_question: str
    jpeg_quality: int


class Dots3NoteProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Dots3NoteVideosKwargs
    _defaults = {}


@auto_docstring
class Dots3NoteProcessor(ProcessorMixin):
    valid_processor_kwargs = Dots3NoteProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        feature_extractor=None,
        chat_template=None,
    ):
        # Early checkpoints only shipped the legacy preprocessor config. Auto classes then
        # instantiate the Qwen2-VL defaults, whose pixel limits and temporal patch size do
        # not match Dots 3 Note Preview. Normalize fallback pixel limits while enforcing the vision
        # encoder's fixed temporal patch size; custom pixel limits remain untouched.
        if image_processor is not None:
            if dict(image_processor.size) == _QWEN2_VL_IMAGE_DEFAULT_SIZE:
                image_processor.size = SizeDict(**dict(_RELEASE_VISION_SIZE))
            if image_processor.temporal_patch_size == 2:
                image_processor.temporal_patch_size = 1
        if video_processor is not None:
            if dict(video_processor.size) == _QWEN2_VL_VIDEO_DEFAULT_SIZE:
                video_processor.size = SizeDict(**dict(_RELEASE_VISION_SIZE))
            if video_processor.temporal_patch_size == 2:
                video_processor.temporal_patch_size = 1

        self.image_token = "<|imgpad|>"
        self.image_start_token = "<|img|>"
        self.image_end_token = "<|endofimg|>"
        self.video_token = "<|video_pad|>"
        self.audio_token = "<|audio_comp_pad|>"
        self.audio_start_token = "<|audio_comp_start|>"
        self.audio_end_token = "<|audio_comp_end|>"
        self.image_token_id = self._single_token_id(tokenizer, self.image_token)
        self.video_token_id = self._single_token_id(tokenizer, self.video_token)
        self.audio_token_id = self._single_token_id(tokenizer, self.audio_token)
        self.audio_start_token_id = self._single_token_id(tokenizer, self.audio_start_token)
        self.audio_end_token_id = self._single_token_id(tokenizer, self.audio_end_token)
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            chat_template=chat_template,
        )

    @staticmethod
    def _single_token_id(tokenizer, token: str) -> int:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"{token!r} must encode to exactly one token, got {token_ids}")
        return int(token_ids[0])

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if videos is not None and (images is not None or audio is not None):
            raise ValueError(
                "Dots 3 Note Preview does not support mixing a native video with separate image/audio inputs"
            )
        if audio is None:
            return
        texts = [text] if isinstance(text, str) else text
        placeholder_count = 0 if texts is None else sum(sample.count(self.audio_token) for sample in texts)
        if placeholder_count != len(audio):
            raise ValueError(
                "audio/placeholder count mismatch: "
                f"received {len(audio)} audio item(s) but found {placeholder_count} "
                f"{self.audio_token!r} placeholder(s)"
            )

    def _process_audio(self, audio, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)
        replacements = [self.audio_token * int(token_length) for token_length in audio_inputs["audio_token_lengths"]]
        return audio_inputs, replacements

    def _process_videos(self, videos, **kwargs):
        if isinstance(videos, (str, bytes, Path)) or not isinstance(videos, (list, tuple)):
            videos = [videos]
        if len(videos) != 1:
            raise ValueError(
                f"Dots 3 Note Preview supports one request with exactly one native video, got {len(videos)}"
            )
        video = videos[0]
        if isinstance(video, (list, tuple)) and video and all(isinstance(item, (str, bytes, Path)) for item in video):
            if len(video) != 1:
                raise ValueError(
                    f"Dots 3 Note Preview supports exactly one native video per request, got {len(video)}"
                )
            video = video[0]

        audio_sample_rate = kwargs.pop("audio_sr", 16_000)
        if audio_sample_rate != self.feature_extractor.sampling_rate:
            raise ValueError(
                f"Dots 3 Note Preview audio preprocessing requires {self.feature_extractor.sampling_rate} Hz, "
                f"got {audio_sample_rate}"
            )
        video_kwargs = {
            "question": kwargs.pop("video_question", ""),
            "sequence_length": kwargs.pop("seq", 131_072),
            "output_reserve": kwargs.pop("output_reserve", None),
            "audio_cap": kwargs.pop("audio_cap", 1.0),
            "audio_sample_rate": audio_sample_rate,
            "k_mode": kwargs.pop("k_mode", "eval_ek"),
            "max_new_tokens": kwargs.pop("max_new_tokens", 0),
            "jpeg_quality": kwargs.pop("jpeg_quality", 85),
        }
        kwargs.pop("return_tensors", None)
        unsupported = {key: value for key, value in kwargs.items() if value is not None}
        if unsupported:
            raise ValueError(
                "Dots 3 Note Preview native video preprocessing uses the fixed SGLang-aligned transform; "
                f"unsupported video overrides: {sorted(unsupported)}"
            )
        parts = self.video_processor.preprocess_native(
            video,
            tokenizer=self.tokenizer,
            **video_kwargs,
        )
        images = [part.value for part in parts if part.kind == "image"]
        audios = [part.value for part in parts if part.kind == "audio"]
        if not images:
            raise ValueError("Dots 3 Note Preview video preprocessing produced no frames")

        image_inputs = self.image_processor(images, return_tensors="pt")
        audio_inputs = (
            self.feature_extractor(audios, sampling_rate=audio_sample_rate, return_tensors="pt") if audios else {}
        )
        image_grids = iter(image_inputs["image_grid_thw"])
        audio_lengths = iter(audio_inputs.get("audio_token_lengths", []))
        replacement = []
        for part in parts:
            if part.kind == "text":
                replacement.append(part.value)
            elif part.kind == "image":
                token_count = int(next(image_grids).prod()) // self.image_processor.merge_size**2
                replacement.append(self.image_start_token + self.image_token * token_count + self.image_end_token)
            else:
                token_count = int(next(audio_lengths))
                replacement.append(self.audio_start_token + self.audio_token * token_count + self.audio_end_token)

        processed = {**image_inputs, **audio_inputs}
        return processed, ["".join(replacement)]

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_tokens = int(image_inputs["image_grid_thw"][image_idx].prod()) // merge_length
        return self.image_token * num_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        raise RuntimeError(
            "Native Dots 3 Note Preview videos are expanded into timestamped image/audio blocks before tokenization"
        )

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * int(audio_inputs["audio_token_lengths"][audio_idx])

    @property
    def model_input_names(self) -> list[str]:
        names = [*super().model_input_names, "chunk_audio_indices"]
        return list(dict.fromkeys(names))


__all__ = ["Dots3NoteProcessor"]

# Copyright 2026 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.
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
"""Processor for Dots3-Note Omni text, image, video, and audio inputs."""

from __future__ import annotations

from ...image_utils import SizeDict
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


_RELEASE_VISION_SIZE = SizeDict(shortest_edge=56 * 56, longest_edge=(36 * 28) ** 2)
_QWEN2_VL_IMAGE_DEFAULT_SIZE = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
_QWEN2_VL_VIDEO_DEFAULT_SIZE = {"shortest_edge": 128 * 28 * 28, "longest_edge": 28 * 28 * 768}


class Dots3NoteOmniProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class Dots3NoteOmniProcessor(ProcessorMixin):
    valid_processor_kwargs = Dots3NoteOmniProcessorKwargs

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
        # not match Dots3-Note. Normalize fallback pixel limits while enforcing the vision
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

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_tokens = int(image_inputs["image_grid_thw"][image_idx].prod()) // merge_length
        return self.image_token * num_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        num_tokens = int(video_inputs["video_grid_thw"][video_idx].prod()) // merge_length
        return self.video_token * num_tokens

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * int(audio_inputs["audio_token_lengths"][audio_idx])

    @property
    def model_input_names(self) -> list[str]:
        names = [*super().model_input_names, "chunk_audio_indices"]
        return list(dict.fromkeys(names))


__all__ = ["Dots3NoteOmniProcessor"]

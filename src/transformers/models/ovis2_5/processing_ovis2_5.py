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
"""Processor class for Ovis2.5."""

from ...image_utils import make_flat_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring
from ...video_utils import make_batched_videos


class Ovis2_5ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class Ovis2_5Processor(ProcessorMixin):
    valid_processor_kwargs = Ovis2_5ProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        image_token="<image>",
        video_token="<video>",
        visual_atom_token="<ovis_visual_atom>",
        image_start_token="<ovis_image_start>",
        image_end_token="<ovis_image_end>",
        video_start_token="<ovis_video_start>",
        video_end_token="<ovis_video_end>",
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Text placeholder replaced by an expanded image sequence.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Text placeholder replaced by an expanded video sequence.
        visual_atom_token (`str`, *optional*, defaults to `"<ovis_visual_atom>"`):
            Token occupying each projected visual feature position.
        image_start_token (`str`, *optional*, defaults to `"<ovis_image_start>"`):
            Token placed before an image's visual atoms.
        image_end_token (`str`, *optional*, defaults to `"<ovis_image_end>"`):
            Token placed after an image's visual atoms.
        video_start_token (`str`, *optional*, defaults to `"<ovis_video_start>"`):
            Token placed before a video's visual atoms.
        video_end_token (`str`, *optional*, defaults to `"<ovis_video_end>"`):
            Token placed after a video's visual atoms.
        """
        self.image_token = image_token
        self.video_token = video_token
        # Raw placeholders are deliberately not model tokens. The model sees
        # only the five positive special tokens below.
        self.image_token_id = None
        self.video_token_id = None

        self.visual_atom_token = visual_atom_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.video_start_token = video_start_token
        self.video_end_token = video_end_token
        self.visual_atom_token_id = self._resolve_special_token_id(tokenizer, visual_atom_token)
        self.image_start_token_id = self._resolve_special_token_id(tokenizer, image_start_token)
        self.image_end_token_id = self._resolve_special_token_id(tokenizer, image_end_token)
        self.video_start_token_id = self._resolve_special_token_id(tokenizer, video_start_token)
        self.video_end_token_id = self._resolve_special_token_id(tokenizer, video_end_token)

        special_token_ids = {
            self.visual_atom_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.video_start_token_id,
            self.video_end_token_id,
        }
        if len(special_token_ids) != 5:
            raise ValueError("Ovis2.5's visual atom and four indicator tokens must have distinct token IDs.")

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    @staticmethod
    def _resolve_special_token_id(tokenizer, token: str) -> int:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer(token, add_special_tokens=False).input_ids
        if token_id is None or token_id < 0 or encoded != [token_id]:
            raise ValueError(
                f"Ovis2.5 requires `{token}` to be registered as one positive tokenizer token, "
                f"but got token_id={token_id} and encoded={encoded}."
            )
        return token_id

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if audio is not None:
            raise ValueError("Ovis2.5 does not support audio inputs.")

        image_count = 0
        if images is not None:
            if not (isinstance(images, (list, tuple)) and len(images) == 0):
                image_count = len(make_flat_list_of_images(images))

        video_count = 0
        if videos is not None:
            if not (isinstance(videos, (list, tuple)) and len(videos) == 0):
                video_count = len(make_batched_videos(videos))

        if image_count and video_count:
            raise ValueError(
                "Ovis2.5 supports only one visual modality at a time; provide images or one video, not both."
            )
        if video_count > 1:
            raise ValueError(f"Ovis2.5 supports exactly one video at a time, but received {video_count}.")

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_visual_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_start_token + self.visual_atom_token * num_visual_tokens + self.image_end_token

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        num_visual_tokens = video_inputs["video_grid_thw"][video_idx].prod() // merge_length
        return self.video_start_token + self.visual_atom_token * num_visual_tokens + self.video_end_token

    def _check_special_mm_tokens(self, text: list[str], text_inputs, modalities: list[str]):
        super()._check_special_mm_tokens(text, text_inputs, modalities)
        input_ids = text_inputs["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], int):
            input_ids = [input_ids]

        tokens_and_ids = (
            (self.visual_atom_token, self.visual_atom_token_id),
            (self.image_start_token, self.image_start_token_id),
            (self.image_end_token, self.image_end_token_id),
            (self.video_start_token, self.video_start_token_id),
            (self.video_end_token, self.video_end_token_id),
        )
        for token, token_id in tokens_and_ids:
            text_counts = [sample.count(token) for sample in text]
            input_counts = [sample_ids.count(token_id) for sample_ids in input_ids]
            if text_counts != input_counts:
                raise ValueError(
                    f"Mismatch in `{token}` count between expanded text and `input_ids`: "
                    f"got text={text_counts} and input_ids={input_counts}. "
                    "Visual tokens were likely truncated; disable truncation or increase `max_length`."
                )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """Compute multimodal token counts without materializing pixel values."""
        vision_data = {}

        if image_sizes is not None:
            images_kwargs = dict(Ovis2_5ProcessorKwargs._defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(height, width, images_kwargs)
                for height, width in image_sizes
            ]
            vision_data["num_image_patches"] = num_image_patches
            vision_data["num_image_tokens"] = [num_patches // merge_size**2 for num_patches in num_image_patches]

        if video_sizes is not None:
            videos_kwargs = dict(Ovis2_5ProcessorKwargs._defaults.get("videos_kwargs", {}))
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size") or self.video_processor.merge_size
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(num_frames, height, width, videos_kwargs)
                for num_frames, height, width in video_sizes
            ]
            vision_data["num_video_tokens"] = [num_patches // merge_size**2 for num_patches in num_video_patches]

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["Ovis2_5Processor"]

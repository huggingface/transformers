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
        **kwargs,
    ):
        self._image_placeholder = "<image>"
        self._video_placeholder = "<video>"

        self.image_token = (
            getattr(tokenizer, "image_token", None) or getattr(tokenizer, "video_token", None) or "<ovis_visual_atom>"
        )
        self.video_token = self.image_token
        self.image_start_token = getattr(tokenizer, "image_start_token", None) or "<ovis_image_start>"
        self.image_end_token = getattr(tokenizer, "image_end_token", None) or "<ovis_image_end>"
        self.video_start_token = getattr(tokenizer, "video_start_token", None) or "<ovis_video_start>"
        self.video_end_token = getattr(tokenizer, "video_end_token", None) or "<ovis_video_end>"

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = self.image_token_id
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_start_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)
        self.video_start_token_id = tokenizer.convert_tokens_to_ids(self.video_start_token)
        self.video_end_token_id = tokenizer.convert_tokens_to_ids(self.video_end_token)

        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        image_processor, tokenizer, video_processor = super()._get_arguments_from_pretrained(
            pretrained_model_name_or_path, processor_dict, **kwargs
        )
        processor_dict = processor_dict or {}

        # Released checkpoints only contain legacy SigLIP metadata, so use native defaults when no composite config exists.
        if "image_processor" not in processor_dict:
            from .image_processing_ovis2_5 import Ovis2_5ImageProcessor

            image_processor = Ovis2_5ImageProcessor()
        if "video_processor" not in processor_dict:
            from .video_processing_ovis2_5 import Ovis2_5VideoProcessor

            video_processor = Ovis2_5VideoProcessor()

        return [image_processor, tokenizer, video_processor]

    def prepare_inputs_layout(self, images=None, text=None, videos=None, **kwargs):
        images, text, videos, _ = super().prepare_inputs_layout(images=images, text=text, videos=videos, **kwargs)
        if text is not None:
            text = [
                sample.replace(self._image_placeholder, self.image_token).replace(
                    self._video_placeholder, self.video_token
                )
                for sample in text
            ]
        if images is not None:
            images = make_flat_list_of_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)
        return images, text, videos, None

    def validate_inputs(self, images=None, text=None, videos=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, **kwargs)

        image_count = len(images) if images is not None else 0
        video_count = len(videos) if videos is not None else 0

        if image_count and video_count:
            raise ValueError(
                "Ovis2.5 supports only one visual modality at a time; provide images or one video, not both."
            )
        if video_count > 1:
            raise ValueError(f"Ovis2.5 supports exactly one video at a time, but received {video_count}.")

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_visual_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_start_token + self.image_token * num_visual_tokens + self.image_end_token

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        num_visual_tokens = video_inputs["video_grid_thw"][video_idx].prod() // merge_length
        return self.video_start_token + self.video_token * num_visual_tokens + self.video_end_token

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


__all__ = ["Ovis2_5Processor"]

# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

import numpy as np

from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import BatchFeature, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput, make_batched_videos


logger = logging.get_logger(__name__)


class MiniCPMV4_6ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "common_kwargs": {
            "return_tensors": "pt",
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "return_mm_token_type_ids": False,
            "return_text_replacement_offsets": False,
        },
    }


@auto_docstring
class MiniCPMV4_6Processor(ProcessorMixin):
    valid_processor_kwargs = MiniCPMV4_6ProcessorKwargs
    unused_input_names = ["num_patches_per_image", "grids"]

    def __init__(self, image_processor=None, video_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.slice_mode = self.image_processor.slice_mode
        self.video_slice_mode = self.video_processor.slice_mode
        self.default_use_image_id = self.image_processor.use_image_id
        self.image_token_divisor = 4 if self.image_processor.downsample_mode == "4x" else 16
        self.video_token_divisor = 4 if self.video_processor.downsample_mode == "4x" else 16

        self.image_token = tokenizer.image_token
        self.video_token = tokenizer.video_token
        self.image_token_id = tokenizer.image_token_id
        self.video_token_id = tokenizer.video_token_id

        self.image_start_token = tokenizer.image_start_token
        self.image_end_token = tokenizer.image_end_token
        self.slice_start_token = tokenizer.slice_start_token
        self.slice_end_token = tokenizer.slice_end_token
        self.image_id_start_token = tokenizer.image_id_start_token
        self.image_id_end_token = tokenizer.image_id_end_token

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        images, text, videos, _ = self.prepare_inputs_layout(images=images, text=text, videos=videos, **kwargs)
        self.validate_inputs(images=images, text=text, videos=videos, **kwargs)

        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )
        use_image_id = merged_kwargs["images_kwargs"].pop("use_image_id", None)
        use_image_id = use_image_id if use_image_id is not None else self.default_use_image_id

        processed_images = processed_videos = {}
        images_replacements = videos_replacements = []
        if images is not None:
            processed_images, images_replacements = self._process_images(images, **merged_kwargs["images_kwargs"])
        if videos is not None:
            processed_videos, videos_replacements = self._process_videos(videos, **merged_kwargs["videos_kwargs"])

        text_inputs = {}
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if text is not None:
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
            return_text_replacement_offsets = merged_kwargs["text_kwargs"].pop(
                "return_text_replacement_offsets", False
            )

            if images_replacements and use_image_id:
                images_replacements = self._prepend_local_ids(text, images_replacements, self.image_token)

            if videos_replacements and use_image_id:
                videos_replacements = self._prepend_local_ids(text, videos_replacements, self.video_token)

            text, text_replacement_offsets = self.get_text_with_replacements(
                text,
                images_replacements,
                videos_replacements,
            )
            text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        # Pop unused keys from the inputs, e.g. inputs used only to compute number of image tokens
        data = {**text_inputs, **processed_images, **processed_videos}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if text is None:
            raise ValueError("You have to specify `text` input to process.")
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)

    def _process_images(self, images, **kwargs):
        img_downsample = kwargs.get("downsample_mode", self.image_processor.downsample_mode)
        image_token_divisor = 4 if img_downsample == "4x" else 16
        processed_images = self.image_processor(images, **kwargs)

        image_replacements = []
        images = make_flat_list_of_images(images)
        for idx in range(len(images)):
            replacement_text = self.replace_image_token(
                processed_images, image_idx=idx, image_token_divisor=image_token_divisor
            )
            image_replacements.append(replacement_text)
        return processed_images, image_replacements

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        image_grids = image_inputs["grids"]
        num_patches_per_image = image_inputs["num_patches_per_image"]
        target_sizes = image_inputs["target_sizes"]

        cum_patches = np.cumsum(num_patches_per_image)
        start_idx = cum_patches[image_idx - 1] if image_idx > 0 else 0
        end_idx = cum_patches[image_idx]
        img_target_sizes = target_sizes[start_idx:end_idx]
        num_tokens_per_patch = img_target_sizes.prod(-1) // kwargs["image_token_divisor"]
        num_rows, num_cols = image_grids[image_idx]

        # Build replacement WITHOUT image_id prefix; local ID added in get_text_with_replacements
        image_placeholder = (
            self.image_start_token + self.image_token * int(num_tokens_per_patch[0]) + self.image_end_token
        )

        if self.slice_mode and num_rows > 0 and num_cols > 0:
            per_slice_tokens = int(num_tokens_per_patch[1]) if len(num_tokens_per_patch) > 1 else 0
            slice_placeholder = self.slice_start_token + self.image_token * per_slice_tokens + self.slice_end_token
            slices = [slice_placeholder * num_cols for _ in range(num_rows)]
            image_placeholder += "\n".join(slices)

        return image_placeholder

    def _process_videos(self, videos, **kwargs):
        vid_downsample = kwargs.get("downsample_mode", self.video_processor.downsample_mode)
        video_token_divisor = 4 if vid_downsample == "4x" else 16
        processed_videos = self.video_processor(videos, **kwargs)

        video_replacements = []
        videos = make_batched_videos(videos)
        for idx in range(len(videos)):
            replacement_text = self.replace_video_token(
                processed_videos, video_idx=idx, video_token_divisor=video_token_divisor
            )
            video_replacements.append(replacement_text)
        return processed_videos, video_replacements

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        video_target_sizes = video_inputs["target_sizes_videos"]  # (total_num_frames * num_patches_per_frame, ...)
        num_frames_per_video = video_inputs["num_frames_per_video"]  # (total_num_videos, ...)
        video_grids = video_inputs["grids_videos"]  # (total_num_frames, ...)
        num_patches_per_frame = video_grids.prod(-1) + 1

        num_frames = num_frames_per_video[video_idx]
        cum_patches_per_frame = np.cumsum(num_patches_per_frame)
        num_past_frames = np.cumsum(num_frames_per_video)[video_idx] - num_frames

        video_placeholder = ""
        for frame_idx in range(num_frames):
            # we need cumulative frame idx, because inputs are shaped as `(total_num_frames, ...)`
            frame_start_idx = num_past_frames + frame_idx

            start_idx = cum_patches_per_frame[frame_start_idx - 1] if frame_start_idx > 0 else 0
            end_idx = cum_patches_per_frame[frame_start_idx]

            frame_ts = video_target_sizes[start_idx:end_idx]
            frame_tokens = frame_ts.prod(-1) // kwargs["video_token_divisor"]
            grid_rows, grid_cols = video_grids[frame_start_idx]

            if len(frame_tokens) == 0:
                continue

            frame_placeholder = self.image_start_token + self.video_token * int(frame_tokens[0]) + self.image_end_token
            if self.video_slice_mode and grid_rows > 0 and grid_cols > 0:
                per_slice_tokens = int(frame_tokens[1]) if len(frame_tokens) > 1 else 0
                slice_placeholder = self.slice_start_token + self.video_token * per_slice_tokens + self.slice_end_token
                slices = [slice_placeholder * grid_cols for _ in range(grid_rows)]
                frame_placeholder += "\n".join(slices)
            video_placeholder += frame_placeholder
        return video_placeholder

    def _prepend_local_ids(self, text, replacements, token):
        """Prepend local (per-sample) image/video ID tokens to each replacement string."""
        new_replacements = []
        global_idx = 0
        for sample in text:
            n_tokens = sample.count(token)
            for local_idx in range(n_tokens):
                prefix = f"{self.image_id_start_token}{local_idx}{self.image_id_end_token}"
                new_replacements.append(prefix + replacements[global_idx])
                global_idx += 1
        return new_replacements

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        texts = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)
        return [t.strip() for t in texts]


__all__ = ["MiniCPMV4_6Processor"]

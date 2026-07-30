# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from ...processing_utils import MultiModalData, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
class OnyxProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<|patch|>"
        self.video_token = "<|video|>"
        self.video_sep_token = "<|vid_frame_separator|>"
        self.video_start_token = "<|vid_start|>"
        self.video_end_token = "<|vid_end|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

        super().__init__(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Used by vLLM to size the image placeholders without running the image processor. ``image_sizes``
        are ``(height, width)`` pixel pairs; the returned ``num_image_tokens`` is the count of scattered
        ``<|patch|>`` tokens per image and ``num_image_patches`` the number of pixel-value patch rows.
        """
        vision_data = {}
        if image_sizes is not None:
            merge_size = self.image_processor.merge_size
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(height, width, kwargs)
                for height, width in image_sizes
            ]
            num_image_tokens = [patches // merge_size**2 for patches in num_image_patches]
            vision_data.update(num_image_tokens=num_image_tokens, num_image_patches=num_image_patches)
        return MultiModalData(**vision_data)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        # Reference format is [bos] + <|patch|>*n + text (no image start/end wrappers), matched here.
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        merge_length = self.video_processor.merge_size**2
        grid_thw = video_inputs["video_grid_thw"][video_idx]
        n_frames = int(grid_thw[0])
        tokens_per_group = int(grid_thw[1:].prod() // merge_length)

        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.fps is None:
            logger.warning_once(
                "Onyx requires frame timestamps to construct prompts, but the `fps` of the "
                "input video could not be inferred. Defaulting to `fps=24`. Please provide "
                "`video_metadata` for more accurate results."
            )
            metadata.fps = 24

        pt = self.video_processor.patch_temporal
        # one timestamp per temporal group -- stride by patch_temporal, pad by repeating last
        timestamps = list(metadata.timestamps[::pt])[:n_frames]
        while len(timestamps) < n_frames:
            timestamps.append(timestamps[-1] if timestamps else 0.0)

        replacement_str = self.video_start_token
        for g, ts in enumerate(timestamps):
            replacement_str += f"Time: {ts:.1f}s" + self.video_token * tokens_per_group
            replacement_str += self.video_sep_token if g < n_frames - 1 else self.video_end_token
        return replacement_str


__all__ = ["OnyxProcessor"]

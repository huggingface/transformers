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


from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, VideosKwargs
from ...utils import auto_docstring, logging


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

    def __init__(self, image_processor=None, video_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.slice_mode = self.image_processor.slice_mode
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

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if text is None:
            raise ValueError("You have to specify `text` input to process.")
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)

    def _process_images(self, images, **kwargs):
        use_image_id = kwargs.pop("use_image_id", None)
        use_image_id = use_image_id if use_image_id is not None else self.default_use_image_id
        # Store for get_text_with_replacements to apply local (per-sample) indexing
        self._images_use_image_id = use_image_id

        img_downsample = kwargs.get("downsample_mode", self.image_processor.downsample_mode)
        image_token_divisor = 4 if img_downsample == "4x" else 16

        processed_images = self.image_processor(images, **kwargs)

        image_grids = processed_images.pop("grids")
        num_patches_per_image = processed_images.pop("num_patches_per_image")
        target_sizes = processed_images["target_sizes"]

        image_replacements = []
        flat_index = 0
        for image_idx in range(len(image_grids)):
            n_patches = num_patches_per_image[image_idx]
            img_target_sizes = target_sizes[flat_index : flat_index + n_patches]
            num_tokens_per_patch = img_target_sizes.prod(-1) // image_token_divisor
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

            image_replacements.append(image_placeholder)
            flat_index += n_patches

        return processed_images, image_replacements

    def _process_videos(self, videos, **kwargs):
        use_image_id = kwargs.pop("use_image_id", None)
        use_image_id = use_image_id if use_image_id is not None else self.default_use_image_id
        # Store for get_text_with_replacements to apply local (per-sample) indexing
        self._videos_use_image_id = use_image_id

        vid_downsample = kwargs.get("downsample_mode", self.video_processor.downsample_mode)
        video_token_divisor = 4 if vid_downsample == "4x" else 16

        processed_videos = self.video_processor(videos, **kwargs)

        video_target_sizes = processed_videos["target_sizes_videos"]
        num_frames_per_video = processed_videos.pop("num_frames_per_video")
        video_grids = processed_videos.pop("grids_videos")
        num_patches_per_frame = processed_videos.pop("num_patches_per_frame")

        video_replacements = []
        flat_index = 0
        frame_offset = 0
        for video_idx in range(len(num_frames_per_video)):
            num_frames = num_frames_per_video[video_idx]
            video_placeholder = ""
            for f in range(num_frames):
                gf = frame_offset + f
                n_patches = num_patches_per_frame[gf]
                frame_ts = video_target_sizes[flat_index : flat_index + n_patches]
                frame_tokens = frame_ts.prod(-1) // video_token_divisor
                grid_rows, grid_cols = video_grids[gf]

                if len(frame_tokens) == 0:
                    flat_index += n_patches
                    continue

                frame_placeholder = (
                    self.image_start_token + self.video_token * int(frame_tokens[0]) + self.image_end_token
                )
                if self.slice_mode and grid_rows > 0 and grid_cols > 0:
                    per_slice_tokens = int(frame_tokens[1]) if len(frame_tokens) > 1 else 0
                    slice_placeholder = (
                        self.slice_start_token + self.video_token * per_slice_tokens + self.slice_end_token
                    )
                    slices = [slice_placeholder * grid_cols for _ in range(grid_rows)]
                    frame_placeholder += "\n".join(slices)

                video_placeholder += frame_placeholder
                flat_index += n_patches

            frame_offset += num_frames
            # NOTE: video_id prefix added in get_text_with_replacements with local indexing
            video_replacements.append(video_placeholder)

        return processed_videos, video_replacements

    def get_text_with_replacements(
        self,
        text,
        images_replacements=None,
        videos_replacements=None,
        audio_replacements=None,
    ):
        use_image_id_imgs = getattr(self, "_images_use_image_id", False)
        use_image_id_vids = getattr(self, "_videos_use_image_id", False)

        if images_replacements and use_image_id_imgs:
            images_replacements = self._prepend_local_ids(text, images_replacements, self.image_token)

        if videos_replacements and use_image_id_vids:
            videos_replacements = self._prepend_local_ids(text, videos_replacements, self.video_token)

        return super().get_text_with_replacements(
            text,
            images_replacements=images_replacements or [],
            videos_replacements=videos_replacements or [],
            audio_replacements=audio_replacements or [],
        )

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

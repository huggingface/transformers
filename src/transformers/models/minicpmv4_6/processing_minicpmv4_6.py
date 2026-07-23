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


from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class MiniCPMV4_6ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "common_kwargs": {
            "return_tensors": "pt",
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
    }


@auto_docstring
class MiniCPMV4_6Processor(ProcessorMixin):
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

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[MiniCPMV4_6ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- Token ids to be fed to a model.
            - **attention_mask** -- Mask indicating which tokens should be attended to.
            - **pixel_values** -- Processed image patches to be fed to a model.
            - **target_sizes** -- Patch grid sizes for the vision encoder.
        """

        if text is None:
            raise ValueError("You have to specify `text` input to process.")
        if isinstance(text, str):
            text = [text]
        text = text.copy()

        output_kwargs = self._merge_kwargs(
            MiniCPMV4_6ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        use_image_id = output_kwargs["images_kwargs"].pop("use_image_id", None)
        use_image_id = use_image_id if use_image_id is not None else self.default_use_image_id

        img_downsample = output_kwargs["images_kwargs"].get("downsample_mode", self.image_processor.downsample_mode)
        image_token_divisor = 4 if img_downsample == "4x" else 16

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            image_grids = image_inputs.pop("grids")
            num_patches_per_image = image_inputs.pop("num_patches_per_image")
            target_sizes = image_inputs["target_sizes"]

            flat_index = 0
            global_image_index = 0
            for i in range(len(text)):
                local_image_index = 0
                while self.image_token in text[i]:
                    n_patches = num_patches_per_image[global_image_index]
                    img_target_sizes = target_sizes[flat_index : flat_index + n_patches]
                    num_tokens_per_patch = img_target_sizes.prod(-1) // image_token_divisor
                    num_rows, num_cols = image_grids[global_image_index]

                    image_placeholder = (
                        self.image_start_token
                        + "<|placeholder|>" * int(num_tokens_per_patch[0])
                        + self.image_end_token
                    )
                    if use_image_id:
                        image_placeholder = (
                            f"{self.image_id_start_token}{local_image_index}{self.image_id_end_token}"
                            + image_placeholder
                        )

                    if self.slice_mode and num_rows > 0 and num_cols > 0:
                        per_slice_tokens = int(num_tokens_per_patch[1]) if len(num_tokens_per_patch) > 1 else 0
                        slice_placeholder = (
                            self.slice_start_token + "<|placeholder|>" * per_slice_tokens + self.slice_end_token
                        )
                        slices = [slice_placeholder * num_cols for _ in range(num_rows)]
                        image_placeholder += "\n".join(slices)

                    text[i] = text[i].replace(self.image_token, image_placeholder, 1)
                    flat_index += n_patches
                    global_image_index += 1
                    local_image_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        vid_downsample = output_kwargs["videos_kwargs"].get("downsample_mode", self.video_processor.downsample_mode)
        video_token_divisor = 4 if vid_downsample == "4x" else 16

        video_inputs = {}
        if videos is not None:
            video_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])

            video_target_sizes = video_inputs["target_sizes_videos"]
            num_frames_per_video = video_inputs.pop("num_frames_per_video")
            video_grids = video_inputs.pop("grids_videos")
            num_patches_per_frame = video_inputs.pop("num_patches_per_frame")

            flat_index = 0
            frame_offset = 0
            global_video_index = 0
            for i in range(len(text)):
                local_video_index = 0
                while self.video_token in text[i]:
                    num_frames = num_frames_per_video[global_video_index]
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
                            self.image_start_token + "<|placeholder|>" * int(frame_tokens[0]) + self.image_end_token
                        )
                        if self.slice_mode and grid_rows > 0 and grid_cols > 0:
                            per_slice_tokens = int(frame_tokens[1]) if len(frame_tokens) > 1 else 0
                            slice_placeholder = (
                                self.slice_start_token + "<|placeholder|>" * per_slice_tokens + self.slice_end_token
                            )
                            slices = [slice_placeholder * grid_cols for _ in range(grid_rows)]
                            frame_placeholder += "\n".join(slices)

                        video_placeholder += frame_placeholder
                        flat_index += n_patches

                    frame_offset += num_frames

                    if use_image_id:
                        video_placeholder = (
                            f"{self.image_id_start_token}{local_video_index}{self.image_id_end_token}"
                            + video_placeholder
                        )

                    text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    global_video_index += 1
                    local_video_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        texts = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)
        return [t.strip() for t in texts]


__all__ = ["MiniCPMV4_6Processor"]

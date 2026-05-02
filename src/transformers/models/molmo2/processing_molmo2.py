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

"""
Processor class for Molmo2.
"""

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
IMAGE_PATCH_TOKEN = "<im_patch>"  # Where to insert high-res tokens
IMAGE_LOW_RES_TOKEN = "<im_low>"  # Where to insert low-res tokens
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
FRAME_START_TOKEN = "<frame_start>"
IM_END_TOKEN = "<im_end>"
FRAME_END_TOKEN = "<frame_end>"
IM_COL_TOKEN = "<im_col>"
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"

IMAGE_TOKENS = [
    IMAGE_PATCH_TOKEN,
    IM_COL_TOKEN,
    IM_START_TOKEN,
    LOW_RES_IMAGE_START_TOKEN,
    FRAME_START_TOKEN,
    IM_END_TOKEN,
    FRAME_END_TOKEN,
    IMAGE_LOW_RES_TOKEN,
]


class Molmo2ImagesKwargs(ImagesKwargs, total=False):
    """
    max_crops (`int`, *optional*):
        Maximum number of image crops produced by the image processor.
    overlap_margins (`list[int]`, *optional*):
        Pixel margins `[left_right, top_bottom]` to overlap between neighboring crops.
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to patch features in the vision adapter.
    """

    max_crops: int | None
    overlap_margins: list[int] | None
    patch_size: int | None
    pooling_size: list[int] | None


class Molmo2VideosKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch for video frames.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to video patch features.
    max_fps (`int`, *optional*):
        Maximum sampling rate in frames per second for short videos.
    """

    patch_size: int | None
    pooling_size: list[int] | None
    max_fps: int | None


class Molmo2ProcessorKwargs(ProcessingKwargs, total=False):
    """Molmo2 processor kwargs"""

    images_kwargs: Molmo2ImagesKwargs
    videos_kwargs: Molmo2VideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class Molmo2Processor(ProcessorMixin):
    @property
    def model_input_names(self):
        return super().model_input_names + ["token_type_ids"]

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template: str | None = None,
        image_use_col_tokens: bool | None = True,
        use_single_crop_col_tokens: bool | None = None,
        use_single_crop_start_token: bool | None = True,
        video_use_col_tokens: bool | None = False,
        use_frame_special_tokens: bool | None = True,
        **kwargs,
    ) -> None:
        r"""
        image_use_col_tokens (`bool`, *optional*, defaults to `True`):
            Whether to append column-separator tokens (`<im_col>`) after each patch row of the high-resolution image
            view.
        use_single_crop_col_tokens (`bool`, *optional*):
            Whether to append column-separator tokens after each patch row of the low-resolution (single-crop) image
            view. If `None`, falls back to `image_use_col_tokens`.
        use_single_crop_start_token (`bool`, *optional*, defaults to `True`):
            Whether to start the low-resolution image view with `<low_res_im_start>` instead of the regular
            `<im_start>`.
        video_use_col_tokens (`bool`, *optional*, defaults to `False`):
            Whether to append column-separator tokens after each patch row of video frames.
        use_frame_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether to wrap each video frame with `<frame_start>` / `<frame_end>` tokens. If `False`, falls back to
            `<im_start>` / `<im_end>`.
        """
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template)

        self.image_token = IMAGE_PROMPT
        self.video_token = VIDEO_PROMPT
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.image_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in IMAGE_TOKENS]
        self.image_ids = self.image_token_ids
        self.image_use_col_tokens = image_use_col_tokens
        self.use_single_crop_col_tokens = use_single_crop_col_tokens
        self.use_single_crop_start_token = use_single_crop_start_token
        self.video_use_col_tokens = video_use_col_tokens
        self.use_frame_special_tokens = use_frame_special_tokens

    def get_image_tokens(self, image_grid: np.ndarray):
        resized_h, resized_w, height, width = image_grid
        per_row = np.full(width, IMAGE_PATCH_TOKEN)
        if self.image_use_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [IM_START_TOKEN],
            np.tile(per_row, [height]),
            [IM_END_TOKEN],
        ]
        per_row = np.full(resized_w, IMAGE_PATCH_TOKEN)
        use_single_crop_col_tokens = (
            self.image_use_col_tokens if self.use_single_crop_col_tokens is None else self.use_single_crop_col_tokens
        )
        image_start_token = LOW_RES_IMAGE_START_TOKEN if self.use_single_crop_start_token else IM_START_TOKEN
        if use_single_crop_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [image_start_token],
            np.tile(per_row, [resized_h]),
            [IM_END_TOKEN],
        ] + joint

        return np.concatenate(joint)

    def get_video_string(
        self,
        video_grid: np.ndarray,
        timestamps: np.ndarray,
    ):
        if self.use_frame_special_tokens:
            start_token_id = FRAME_START_TOKEN
            end_token_id = FRAME_END_TOKEN
        else:
            start_token_id = IM_START_TOKEN
            end_token_id = IM_END_TOKEN

        num_frames, h, w = video_grid
        video_string: str = ""
        for frame_idx, frame_time in enumerate(timestamps):
            # `per-frame-compact` time mode
            prev_space = " " if frame_idx > 0 else ""
            frame_prefix = prev_space + f"{frame_time:.1f} "  # explicit whitespace before/after image tokens

            video_string += frame_prefix
            per_row = np.full(w, IMAGE_PATCH_TOKEN)
            if self.video_use_col_tokens:
                per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
            extra_tokens = np.tile(per_row, [h])
            video_tokens = [
                [start_token_id],
                extra_tokens,
                [end_token_id],
            ]
            video_string += "".join(np.concatenate(video_tokens, 0))

        return video_string

    def insert_bos(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ):
        """
        Args:
            input_ids: [B, S] array with left padding
            attention_mask: [B, S] array (0 for pad, 1 for valid)
            bos_token_id: int
            pad_token_id: int
        Returns:
            input_ids_out: [B, S] or [B, S+1] array with bos inserted if needed
            attention_mask_out: same shape as input_ids_out
        """

        need_to_expand = len(input_ids.shape) == 1
        if need_to_expand:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]

        B, S = input_ids.shape

        # Handle zero-length sequence
        if S == 0:
            new_input_ids = np.full((B, 1), bos_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.ones((B, 1), dtype=attention_mask.dtype)
            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]
            return new_input_ids, new_attention_mask

        first_valid_index = (attention_mask == 1).argmax(axis=-1)  # [B]
        bos_already_present = np.all(input_ids[np.arange(B), first_valid_index] == bos_token_id)

        if bos_already_present:
            if need_to_expand:
                input_ids = input_ids[0]
                attention_mask = attention_mask[0]
            return input_ids, attention_mask
        else:
            new_input_ids = np.full((B, S + 1), pad_token_id, dtype=input_ids.dtype)
            new_attention_mask = np.zeros((B, S + 1), dtype=attention_mask.dtype)

            src_idx = np.tile(np.arange(S), (B, 1))  # [B, S]
            valid_mask = src_idx >= first_valid_index[:, None]  # [B, S]
            tgt_idx = src_idx + 1  # shift right
            batch_idx = np.tile(np.arange(B)[:, None], (1, S))  # [B, S]

            # flatten valid_positions
            flat_vals = input_ids[valid_mask]
            flat_batch = batch_idx[valid_mask]
            flat_tgt = tgt_idx[valid_mask]

            new_input_ids[flat_batch, flat_tgt] = flat_vals
            new_attention_mask[flat_batch, flat_tgt] = 1

            insert_pos = first_valid_index
            new_input_ids[np.arange(B), insert_pos] = bos_token_id
            new_attention_mask[np.arange(B), insert_pos] = 1

            if need_to_expand:
                new_input_ids = new_input_ids[0]
                new_attention_mask = new_attention_mask[0]

            return new_input_ids, new_attention_mask

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Molmo2ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_token_pooling** -- Indices of the patches in `image_grids` to pool for each token.
              Returned when `images` is not `None`.
            - **image_grids** -- Grids of images. Returned when `images` is not `None`.
            - **image_num_crops** -- Number of crops for each image. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos. Returned when `videos` is not `None`.
            - **video_token_pooling** -- Indices of the patches in `video_grids` to pool for each token.
              Returned when `videos` is not `None`.
            - **video_grids** -- Grids of videos. Returned when `videos` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            Molmo2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            image_grids = image_inputs["image_grids"]
        else:
            image_inputs = {}
            image_grids = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grids = videos_inputs["video_grids"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grids = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if image_grids is not None:
            index = 0
            for i in range(len(text)):
                num_images = text[i].count(self.image_token)
                image_grids_i = image_grids[index : index + num_images]
                for image_grid in image_grids_i:
                    image_tokens = self.get_image_tokens(image_grid)
                    image_string = "".join(image_tokens)
                    text[i] = text[i].replace(self.image_token, image_string, 1)
                index += num_images

        if video_grids is not None:
            index = 0
            for i in range(len(text)):
                num_videos = text[i].count(self.video_token)
                if num_videos > 1:
                    raise ValueError("At most one video is supported per sample.")
                video_grids_i = video_grids[index : index + num_videos]
                metadata_i = video_metadata[index : index + num_videos]
                for video_grid, metadata in zip(video_grids_i, metadata_i):
                    video_string = self.get_video_string(
                        video_grid,
                        metadata.timestamps,
                    )
                    text[i] = text[i].replace(self.video_token, video_string, 1)
                index += num_videos

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)

        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        input_ids, attention_mask = self.insert_bos(input_ids, attention_mask, bos, self.tokenizer.pad_token_id)

        if return_mm_token_type_ids:
            text_inputs["token_type_ids"] = self.create_mm_token_type_ids(input_ids.tolist())

        text_inputs["input_ids"] = input_ids.tolist()
        text_inputs["attention_mask"] = attention_mask.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )


__all__ = ["Molmo2Processor"]

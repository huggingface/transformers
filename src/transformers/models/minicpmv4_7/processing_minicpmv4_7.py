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
import torch

from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import BatchFeature, ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput, make_batched_videos
from ..minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor, MiniCPMV4_6ProcessorKwargs


logger = logging.get_logger(__name__)


class MiniCPMV4_7ProcessorKwargs(MiniCPMV4_6ProcessorKwargs, total=False):
    _defaults = {
        "common_kwargs": {
            "return_tensors": "pt",
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            # Always emit mm_token_type_ids (Qwen-compatible contract for M-RoPE).
            "return_mm_token_type_ids": True,
            "return_text_replacement_offsets": False,
        },
    }


@auto_docstring
class MiniCPMV4_7Processor(MiniCPMV4_6Processor):
    valid_processor_kwargs = MiniCPMV4_7ProcessorKwargs

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        # MiniCPM needs to override `__call__` due to `_prepend_local_ids`, i.e. we add local image id inside text
        # Current `replace_image_tokens` API assumes that each image-placeholder doesn't depend on the other!
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
        # Per-visual patch grids (not recoverable from config alone), one flat entry per visual
        # input, in the same order as the per-modality replacement strings.
        images_mrope_grids: list[list[list[int]]] = []
        videos_mrope_grids: list[list[list[int]]] = []
        if images is not None:
            processed_images, images_replacements, images_mrope_grids = self._process_images(
                images,
                **merged_kwargs["images_kwargs"],
            )
        if videos is not None:
            processed_videos, videos_replacements, videos_mrope_grids = self._process_videos(
                videos,
                **merged_kwargs["videos_kwargs"],
            )

        text_inputs = {}
        text_replacement_offsets = []
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if text is not None:
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop("return_mm_token_type_ids", True)
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

        # `target_sizes_mrope` must follow the order of the visual spans inside `input_ids`, which is
        # the order the placeholders appear in `text` — not the order the modalities are processed in.
        # A sample may interleave modalities (e.g. "<video>...</video><image>...</image>"), so the
        # per-modality grids collected above are re-sequenced through the text replacement offsets.
        mrope_inputs = {}
        if offsets_per_sample := text_replacement_offsets:
            mrope_tgt_sizes_per_sample = self._assemble_mrope_target_sizes(
                offsets_per_sample, images_mrope_grids, videos_mrope_grids
            )
            target_sizes_mrope = []
            for sample_grids in mrope_tgt_sizes_per_sample:
                target_sizes_mrope.append(
                    torch.tensor(sample_grids, dtype=torch.int32)
                    if sample_grids
                    else torch.zeros(0, 2, dtype=torch.int32)
                )
            # Do not return special_token_ids (available on model config) or image_bounds
            # (model recomputes bounds on compact/unpadded ids for left-padding safety).
            mrope_inputs = {"target_sizes_mrope": target_sizes_mrope}

        data = {**text_inputs, **processed_images, **processed_videos, **mrope_inputs}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def _process_images(self, images, **kwargs):
        processed_images, image_replacements = MiniCPMV4_6Processor._process_images(self, images, **kwargs)
        images = make_flat_list_of_images(images)
        # One flat list of patch grids per image, aligned with `image_replacements`.
        mrope_grids = []
        for idx in range(len(images)):
            mrope_grids.append(self._image_target_sizes(processed_images, idx).tolist())
        return processed_images, image_replacements, mrope_grids

    @staticmethod
    def _image_target_sizes(image_inputs: dict, image_idx: int):
        """Return the patch target sizes belonging to one image of the flattened batch."""
        cum_patches = np.cumsum(image_inputs["num_patches_per_image"])
        start_idx = cum_patches[image_idx - 1] if image_idx > 0 else 0
        end_idx = cum_patches[image_idx]
        return image_inputs["target_sizes"][start_idx:end_idx]

    def _process_videos(self, videos, **kwargs):
        processed_videos, video_replacements = MiniCPMV4_6Processor._process_videos(self, videos, **kwargs)
        videos = make_batched_videos(videos)
        # One flat list of patch grids per video, aligned with `video_replacements`. Frames are
        # concatenated in order, and a frame may itself span several patches.
        mrope_grids = []
        for idx in range(len(videos)):
            video_grids = []
            for frame_ts, _, _ in self._iter_video_frames(processed_videos, idx):
                video_grids.extend(frame_ts.tolist())
            mrope_grids.append(video_grids)
        return processed_videos, video_replacements, mrope_grids

    @staticmethod
    def _assemble_mrope_target_sizes(
        offsets_per_sample: list[list[dict]],
        images_mrope_grids: list[list[list[int]]],
        videos_mrope_grids: list[list[list[int]]],
    ) -> list[list[list[int]]]:
        """Flatten per-visual patch grids into per-sample lists, following the text order of the visuals.

        `get_text_with_replacements` walks each sample left to right and yields one offset entry per
        placeholder occurrence, tagged with its modality. Consuming the per-modality grid lists in that
        same order is what makes `target_sizes_mrope[b]` line up with the visual spans inside
        `input_ids[b]` even when a sample interleaves image and video placeholders.

        Grids are counted per visual input, not per frame: one video placeholder expands to all the
        frames of that video. For a batch of a single modality this reproduces the plain per-modality
        concatenation in use before, so single-modality behaviour is unchanged.
        """
        available = {
            "image": len(images_mrope_grids),
            "video": len(videos_mrope_grids),
        }
        num_placeholders = dict.fromkeys(available, 0)
        for sample_offsets in offsets_per_sample:
            for offset in sample_offsets:
                num_placeholders[offset["type"]] += 1

        for modality, expected in available.items():
            if num_placeholders[modality] != expected:
                raise ValueError(
                    f"Number of `{modality}` placeholders does not match the number of `{modality}` inputs: "
                    f"found {num_placeholders[modality]} placeholder(s) in `text` but received {expected} input(s). "
                    "Every placeholder must have a matching input."
                )

        image_iter = iter(images_mrope_grids)
        video_iter = iter(videos_mrope_grids)
        grids_per_sample = []
        for sample_offsets in offsets_per_sample:
            sample_grids = []
            for offset in sample_offsets:
                # One video input expands to one grid per frame, so all of its grids are appended here
                # in frame order; they then line up with that video's tokens from left to right.
                for grid in next(image_iter if offset["type"] == "image" else video_iter):
                    sample_grids.append(grid)
            grids_per_sample.append(sample_grids)
        return grids_per_sample

    def _iter_video_frames(self, video_inputs: dict, video_idx: int):
        """Yield `(frame_target_sizes, grid_rows, grid_cols)` per frame of one video, in text order."""
        video_target_sizes = video_inputs["target_sizes_videos"]
        num_frames_per_video = video_inputs["num_frames_per_video"]
        video_grids = video_inputs["grids_videos"]
        num_patches_per_frame = video_grids.prod(-1) + 1

        num_frames = num_frames_per_video[video_idx]
        cum_patches_per_frame = np.cumsum(num_patches_per_frame)
        num_past_frames = np.cumsum(num_frames_per_video)[video_idx] - num_frames

        for frame_idx in range(num_frames):
            frame_start_idx = num_past_frames + frame_idx

            start_idx = cum_patches_per_frame[frame_start_idx - 1] if frame_start_idx > 0 else 0
            end_idx = cum_patches_per_frame[frame_start_idx]

            grid_rows, grid_cols = video_grids[frame_start_idx]
            yield video_target_sizes[start_idx:end_idx], grid_rows, grid_cols


__all__ = ["MiniCPMV4_7Processor"]

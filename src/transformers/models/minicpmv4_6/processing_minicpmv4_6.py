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

import re

import numpy as np
import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Placeholder generation
# ---------------------------------------------------------------------------


def _get_grid_placeholder(
    grid: list[int],
    patch_visual_tokens: int,
    slice_start_token: str,
    slice_end_token: str,
    pad_token: str,
) -> str:
    if grid is None:
        return ""
    cols, rows = grid[0], grid[1]
    if cols == 0 or rows == 0:
        return ""
    slice_placeholder = slice_start_token + pad_token * patch_visual_tokens + slice_end_token
    slices = []
    for _i in range(rows):
        slices.append(slice_placeholder * cols)
    return "\n".join(slices)


def _get_slice_image_placeholder(
    grid: list[int],
    image_idx: int,
    source_image_visual_tokens: int,
    patch_visual_tokens: int,
    use_image_id: bool,
    slice_mode: bool,
    im_start_token: str,
    im_end_token: str,
    pad_token: str,
    im_id_start: str,
    im_id_end: str,
    slice_start_token: str,
    slice_end_token: str,
) -> str:
    image_placeholder = im_start_token + pad_token * source_image_visual_tokens + im_end_token
    if use_image_id:
        final_placeholder = f"{im_id_start}{image_idx}{im_id_end}" + image_placeholder
    else:
        final_placeholder = image_placeholder

    if slice_mode:
        final_placeholder += _get_grid_placeholder(
            grid=grid,
            patch_visual_tokens=patch_visual_tokens,
            slice_start_token=slice_start_token,
            slice_end_token=slice_end_token,
            pad_token=pad_token,
        )
    return final_placeholder


# ---------------------------------------------------------------------------
# ProcessingKwargs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


@auto_docstring
class MiniCPMV4_6Processor(ProcessorMixin):
    def __init__(self, image_processor=None, video_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, video_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.slice_mode = self.image_processor.slice_mode
        self.default_use_image_id = self.image_processor.use_image_id

        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
        self.video_start_token = "<video>"
        self.video_end_token = "</video>"
        self.slice_start_token = "<slice>"
        self.slice_end_token = "</slice>"
        self.image_id_start_token = "<image_id>"
        self.image_id_end_token = "</image_id>"
        self.image_pad_token = "<|image_pad|>"

        special_tokens = [
            self.image_start_token,
            self.image_end_token,
            self.video_start_token,
            self.video_end_token,
            self.slice_start_token,
            self.slice_end_token,
            self.image_id_start_token,
            self.image_id_end_token,
            self.image_pad_token,
        ]
        tokens_to_add = [
            AddedToken(tok, normalized=False, special=True)
            for tok in special_tokens
            if tok not in self.tokenizer.get_vocab()
        ]
        if tokens_to_add:
            self.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

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
        output_kwargs = self._merge_kwargs(
            MiniCPMV4_6ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        images_kwargs = output_kwargs["images_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]
        videos_kwargs = output_kwargs.get("videos_kwargs", {})

        use_image_id = images_kwargs.pop("use_image_id", None)

        image_inputs = None
        if images is not None:
            image_inputs = self.image_processor(images, **images_kwargs)

        video_inputs = None
        video_frame_counts = None
        if videos is not None:
            video_frames_per_video = self._extract_video_frames(videos, **videos_kwargs)
            video_frame_counts = [len(frames) for frames in video_frames_per_video]
            all_video_frames = [frame for frames in video_frames_per_video for frame in frames]
            if all_video_frames:
                video_inputs = self.image_processor(all_video_frames, **images_kwargs)

        return self._convert_images_texts_to_inputs(
            image_inputs,
            video_inputs,
            text,
            use_image_id=use_image_id,
            video_frame_counts=video_frame_counts,
            **text_kwargs,
        )

    def _extract_video_frames(self, videos, **kwargs) -> list[list]:
        """Extract frames from each video, returning a list of frame lists."""
        if isinstance(videos, str):
            videos = [videos]

        max_frames = kwargs.pop("max_frames", None)
        stack_frames = kwargs.pop("stack_frames", None)
        use_ffmpeg = kwargs.pop("use_ffmpeg", None)

        all_frames = []
        for video in videos:
            if isinstance(video, (list, tuple)):
                all_frames.append(list(video))
                continue
            main_frames, stacked = self.video_processor.extract_frames(
                video,
                max_frames=max_frames,
                stack_frames=stack_frames,
                use_ffmpeg=use_ffmpeg,
            )
            frames = []
            for i, frame in enumerate(main_frames):
                frames.append(frame)
                if stacked is not None and i < len(stacked) and stacked[i] is not None:
                    frames.append(stacked[i])
            all_frames.append(frames)
        return all_frames

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        texts = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)
        return [t.strip() for t in texts]

    def _convert_images_texts_to_inputs(
        self,
        image_inputs,
        video_inputs,
        texts: str | list[str],
        use_image_id=None,
        video_frame_counts: list[int] | None = None,
        return_tensors=None,
        max_length=None,
        truncation=None,
        **text_kwargs,
    ):
        has_images = image_inputs is not None and len(image_inputs["pixel_values"])
        has_videos = video_inputs is not None and len(video_inputs["pixel_values"])

        if not has_images and not has_videos:
            model_inputs = self.tokenizer(
                texts,
                return_tensors=return_tensors,
                truncation=truncation,
                max_length=max_length,
                **text_kwargs,
            )
            return BatchFeature(data={**model_inputs})

        def _to_tensor_list(inputs):
            return [
                [torch.from_numpy(pv) if isinstance(pv, np.ndarray) else pv for pv in patches]
                for patches in inputs["pixel_values"]
            ]

        image_pattern = f"({self.image_start_token}./{self.image_end_token})"
        video_pattern = f"({self.video_start_token}./{self.video_end_token})"

        if has_images:
            per_img_pv = _to_tensor_list(image_inputs)
            per_img_ts = image_inputs["target_sizes"]
            img_grids = image_inputs["grids"]
            img_src_vt = image_inputs["source_image_visual_tokens"]
            img_patch_vt = image_inputs["patch_visual_tokens"]
        else:
            per_img_pv = per_img_ts = img_grids = img_src_vt = img_patch_vt = []

        if has_videos:
            per_vframe_pv = _to_tensor_list(video_inputs)
            per_vframe_ts = video_inputs["target_sizes"]
            vframe_grids = video_inputs["grids"]
            vframe_src_vt = video_inputs["source_image_visual_tokens"]
            vframe_patch_vt = video_inputs["patch_visual_tokens"]
        else:
            per_vframe_pv = per_vframe_ts = vframe_grids = vframe_src_vt = vframe_patch_vt = []
            video_frame_counts = video_frame_counts or []

        if isinstance(texts, str):
            texts = [texts]

        default_use_image_id = use_image_id if use_image_id is not None else self.default_use_image_id

        final_texts = []
        per_sample_pixel_values = []
        per_sample_target_sizes = []

        image_index = 0
        video_index = 0
        vframe_index = 0
        for text in texts:
            combined_pattern = f"{re.escape(image_pattern)}|{re.escape(video_pattern)}"
            placeholders = re.findall(combined_pattern, text)

            parts = re.split(combined_pattern, text)
            final_text = parts[0]

            sample_pv = []
            sample_ts = []
            img_counter_in_sample = 0

            for k, placeholder in enumerate(placeholders):
                if placeholder == image_pattern:
                    idx = image_index
                    image_index += 1
                    if idx >= len(per_img_pv):
                        raise ValueError(f"Not enough images: need index {idx} but only {len(per_img_pv)} provided.")
                    expanded = _get_slice_image_placeholder(
                        grid=img_grids[idx],
                        image_idx=img_counter_in_sample,
                        source_image_visual_tokens=img_src_vt[idx],
                        patch_visual_tokens=img_patch_vt[idx],
                        use_image_id=default_use_image_id,
                        slice_mode=self.slice_mode,
                        im_start_token=self.image_start_token,
                        im_end_token=self.image_end_token,
                        pad_token=self.image_pad_token,
                        im_id_start=self.image_id_start_token,
                        im_id_end=self.image_id_end_token,
                        slice_start_token=self.slice_start_token,
                        slice_end_token=self.slice_end_token,
                    )
                    final_text += expanded
                    sample_pv.extend(per_img_pv[idx])
                    sample_ts.extend(per_img_ts[idx])
                    img_counter_in_sample += 1

                elif placeholder == video_pattern:
                    if video_index >= len(video_frame_counts):
                        raise ValueError(
                            f"Not enough videos: need index {video_index} but only {len(video_frame_counts)} provided."
                        )
                    n_frames = video_frame_counts[video_index]
                    video_index += 1
                    for f in range(n_frames):
                        fidx = vframe_index
                        vframe_index += 1
                        expanded = _get_slice_image_placeholder(
                            grid=vframe_grids[fidx],
                            image_idx=0,
                            source_image_visual_tokens=vframe_src_vt[fidx],
                            patch_visual_tokens=vframe_patch_vt[fidx],
                            use_image_id=False,
                            slice_mode=self.slice_mode,
                            im_start_token=self.image_start_token,
                            im_end_token=self.image_end_token,
                            pad_token=self.image_pad_token,
                            im_id_start=self.image_id_start_token,
                            im_id_end=self.image_id_end_token,
                            slice_start_token=self.slice_start_token,
                            slice_end_token=self.slice_end_token,
                        )
                        final_text += expanded
                        sample_pv.extend(per_vframe_pv[fidx])
                        sample_ts.extend(per_vframe_ts[fidx])

                final_text += parts[k + 1]

            final_texts.append(final_text)
            per_sample_pixel_values.append(sample_pv)
            per_sample_target_sizes.append(
                torch.tensor(sample_ts, dtype=torch.int32) if sample_ts else torch.zeros(0, 2, dtype=torch.int32)
            )

        tokenized = self.tokenizer(
            final_texts,
            return_tensors=return_tensors,
            truncation=truncation,
            max_length=max_length,
            **text_kwargs,
        )

        return BatchFeature(
            data={
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "pixel_values": per_sample_pixel_values,
                "target_sizes": per_sample_target_sizes,
            }
        )


__all__ = ["MiniCPMV4_6Processor"]

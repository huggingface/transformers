# Copyright 2026 the MiniMax AI Team and HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Processor for MiniMax M3 VL."""

import re

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack


class MiniMaxM3VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "videos_kwargs": {"do_resize": False, "return_metadata": True},
    }


class MiniMaxM3VLProcessor(ProcessorMixin):
    """Combines tokenizer + image_processor + video_processor for MiniMax M3 VL.

    Expands ``IMAGE_TOKEN`` / ``VIDEO_TOKEN`` markers in the prompt into the
    matching number of placeholder tokens (one per merged patch), wrapped in
    ``VISION_START_TOKEN`` / ``VISION_END_TOKEN`` brackets.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    tokenizer_class = "AutoTokenizer"

    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, **kwargs):
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) if tokenizer else None
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN) if tokenizer else None
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.VISION_START_TOKEN) if tokenizer else None
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.VISION_END_TOKEN) if tokenizer else None
        super().__init__(image_processor, tokenizer, video_processor)

    def _prune_video_tokens(self, input_text: str, video_segments: list[int], video_token: str) -> str:
        """Drop every other video-token according to ``temporal_patch_size`` (2:1 sampling)."""
        if not video_segments or self.video_processor.temporal_patch_size <= 1:
            return input_text
        pattern = re.escape(video_token)
        parts = re.split(f"({pattern})", input_text)

        def is_timestamp(text: str) -> bool:
            return text.endswith("seconds[>[") or text.endswith("seconds [>[ ")

        def extract_timestamp(text: str) -> str:
            start = text.rfind("]<]")
            if start == -1:
                raise ValueError(f"Failed to extract timestamp: {text}")
            return text[start:]

        final_parts: list[str] = []
        cur_seg = 0
        frame_in_seg = 0
        last_ts_len = 0
        for part in parts:
            if part == video_token:
                if cur_seg < len(video_segments):
                    if frame_in_seg % self.video_processor.temporal_patch_size == 0:
                        final_parts.append(part)
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[cur_seg]:
                            cur_seg += 1
                            frame_in_seg = 0
                        last_ts_len = 0
                    else:
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[cur_seg]:
                            cur_seg += 1
                            frame_in_seg = 0
                        if last_ts_len > 0 and final_parts:
                            final_parts[-1] = final_parts[-1][:-last_ts_len]
                            last_ts_len = 0
                else:
                    final_parts.append(part)
                    last_ts_len = 0
            else:
                final_parts.append(part)
                last_ts_len = len(extract_timestamp(part)) if is_timestamp(part) else 0
        return "".join(final_parts)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs: Unpack[MiniMaxM3VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            MiniMaxM3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        video_inputs = {}
        video_grid_thw = None
        video_metadata = None
        if videos is not None:
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = video_inputs.pop("video_metadata", None)
            else:
                video_metadata = video_inputs.get("video_metadata")

        if not isinstance(text, list):
            text = [text]
        text = list(text)

        placeholder = "]<]placeholder[>["

        if image_grid_thw is not None:
            merge_len = self.image_processor.merge_size**2
            idx = 0
            for i in range(len(text)):
                while self.IMAGE_TOKEN in text[i]:
                    n_tok = int(image_grid_thw[idx].prod() // merge_len)
                    text[i] = text[i].replace(
                        self.IMAGE_TOKEN,
                        self.VISION_START_TOKEN + placeholder * n_tok + self.VISION_END_TOKEN,
                        1,
                    )
                    idx += 1
                text[i] = text[i].replace(placeholder, self.IMAGE_TOKEN)

        if video_grid_thw is not None:
            merge_len = self.image_processor.merge_size**2
            idx = 0
            for i in range(len(text)):
                while self.VIDEO_TOKEN in text[i]:
                    grid_t = int(video_grid_thw[idx][0])
                    frame_seqlen = int(video_grid_thw[idx][1:].prod() // merge_len)
                    metadata = video_metadata[idx] if video_metadata is not None else None
                    chunk = ""
                    for f in range(grid_t):
                        if (
                            metadata is not None
                            and getattr(metadata, "fps", None) is not None
                            and getattr(metadata, "frames_indices", None) is not None
                        ):
                            ts = (
                                metadata.frames_indices[
                                    min(f * self.video_processor.temporal_patch_size, len(metadata.frames_indices) - 1)
                                ]
                                / metadata.fps
                            )
                            chunk += f"]<]{ts:.1f} seconds[>["
                        chunk += self.VISION_START_TOKEN + placeholder * frame_seqlen + self.VISION_END_TOKEN
                    text[i] = text[i].replace(self.VIDEO_TOKEN, chunk, 1)
                    idx += 1
                text[i] = text[i].replace(placeholder, self.VIDEO_TOKEN)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )


__all__ = ["MiniMaxM3VLProcessor"]

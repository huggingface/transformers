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
"""HuggingFace processor for the Onyx multimodal model (text + image + video).

Reproduces the token layout the model expects:

    image -> <|image_start|> + <|patch|> * N        + <|image_end|>
    video -> <|vid_start|> ( "Time: X.Xs" + <|video|> * P [+ <|vid_frame_separator|>] )* + <|vid_end|>

where N is chosen by ``OnyxImageProcessor.compute_image_size`` and P by
``OnyxVideoProcessor.compute_video_frame_size`` (both mirror the encoder's grid
logic).

The chat template emits one sentinel per media item (``<|image|>`` / ``<|video|>``),
which ``OnyxProcessor.__call__`` expands into the spans above.
"""

from __future__ import annotations

import torch
from PIL import Image

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin


IMAGE_SENTINEL = "<|image|>"
VIDEO_SENTINEL = "<|video|>"


ONYX_MM_CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- macro render_parts(content) -%}"
    "{%- if content is string -%}{{- content -}}"
    "{%- else -%}"
    "{%- for part in content -%}"
    "{%- if part['type'] == 'image' -%}{{- '<|image|>' -}}"
    "{%- elif part['type'] == 'video' -%}{{- '<|video|>' -}}"
    "{%- elif part['type'] == 'text' -%}{{- part['text'] -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{%- endmacro -%}"
    "{%- set ns = namespace(has_system=false) -%}"
    "{%- for m in messages -%}{%- if m['role'] == 'system' -%}{%- set ns.has_system = true -%}{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt and not ns.has_system -%}"
    "{{- '<|start|>system<|message|>You are a helpful assistant.<|eot|>' -}}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "{%- set role = message['role'] -%}"
    "{%- if role == 'assistant' -%}"
    "{%- set recipient = message.get('recipient') -%}"
    "{%- set end_turn = message.get('end_turn') -%}"
    "{%- if end_turn is none -%}"
    "{%- set end_turn = not (recipient and recipient != 'user') -%}"
    "{%- endif -%}"
    "{{- '<|start|>assistant' -}}"
    "{%- if recipient -%}{{- ' to=' + recipient -}}{%- endif -%}"
    "{{- '<|message|>' -}}{{- render_parts(message['content']) -}}"
    "{{- ('<|eot|>' if end_turn else '<|eom|>') -}}"
    "{%- elif role == 'tool' -%}"
    "{%- set name = message.get('name', '') -%}"
    "{{- '<|start|>tool ' + name + '<|message|>' -}}{{- render_parts(message['content']) -}}"
    "{{- '<|eot|>' -}}"
    "{%- else -%}"
    "{%- set header = role -%}"
    "{%- if message.get('name') -%}{%- set header = role + ' ' + message['name'] -%}{%- endif -%}"
    "{{- '<|start|>' + header + '<|message|>' -}}{{- render_parts(message['content']) -}}"
    "{{- '<|eot|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}"
)


class OnyxProcessor(ProcessorMixin):
    """Bundle ``OnyxImageProcessor`` + ``OnyxVideoProcessor`` + tokenizer.

    Images go through ``image_processor`` and videos through ``video_processor``;
    ``__call__`` expands per-media sentinels emitted by the chat template into the
    ``<|image_start|>...<|image_end|>`` / ``<|vid_start|>...<|vid_end|>`` spans.
    """

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template or ONYX_MM_CHAT_TEMPLATE,
            **kwargs,
        )

    def _sid(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    def _image_block(self, n_tokens: int) -> list[int]:
        return [self._sid("<|image_start|>")] + [self._sid("<|patch|>")] * n_tokens + [self._sid("<|image_end|>")]

    def _video_block(
        self,
        n_groups: int,
        tokens_per_group: int,
        timestamps: list[float] | None = None,
    ) -> list[int]:
        """Per-group ``Time: X.Xs`` + <|video|>*P, separated/terminated."""
        vid = self._sid("<|video|>")
        sep = self._sid("<|vid_frame_separator|>")
        pt = self.video_processor.patch_temporal
        fps = self.video_processor.video_sampling_fps
        block = [self._sid("<|vid_start|>")]
        for g in range(n_groups):
            ts = timestamps[g] if timestamps is not None else g * pt / fps
            block += self.tokenizer.encode(f"Time: {ts:.1f}s", add_special_tokens=False)
            block += [vid] * tokens_per_group
            block.append(sep if g < n_groups - 1 else self._sid("<|vid_end|>"))
        return block

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[list[Image.Image] | str] | None = None,
        video_timestamps: list[list[float]] | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("`text` is required (use apply_chat_template to build it).")
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError("OnyxProcessor supports a single text sample per call.")
            text = text[0]

        images = list(images or [])
        videos = list(videos or [])
        image_sentinel = self._sid(IMAGE_SENTINEL)
        video_sentinel = self._sid(VIDEO_SENTINEL)

        prepped_images = [self.image_processor.preprocess_image(im) for im in images]
        prepped_videos = [
            self.video_processor.preprocess_one(v, video_timestamps[i] if video_timestamps else None)
            for i, v in enumerate(videos)
        ]

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        n_img = sum(1 for t in ids if t == image_sentinel)
        n_vid = sum(1 for t in ids if t == video_sentinel)
        if n_img != len(prepped_images):
            raise ValueError(f"{n_img} image sentinel(s) in text but {len(prepped_images)} image(s) given.")
        if n_vid != len(prepped_videos):
            raise ValueError(f"{n_vid} video sentinel(s) in text but {len(prepped_videos)} video(s) given.")

        out_ids: list[int] = []
        pixel_values: list[torch.Tensor] = []
        img_i = vid_i = 0
        for tid in ids:
            if tid == image_sentinel:
                tensor, n_tokens = prepped_images[img_i]
                img_i += 1
                out_ids += self._image_block(n_tokens)
                pixel_values.append(tensor)
            elif tid == video_sentinel:
                groups, n_groups, tokens_per_group, ts = prepped_videos[vid_i]
                vid_i += 1
                out_ids += self._video_block(n_groups, tokens_per_group, ts or None)
                pixel_values += groups
            else:
                out_ids.append(tid)

        data: dict = {
            "input_ids": [out_ids],
            "attention_mask": [[1] * len(out_ids)],
        }
        batch = BatchFeature(data=data, tensor_type=return_tensors)
        if pixel_values:
            batch["pixel_values"] = pixel_values
        return batch


__all__ = ["OnyxProcessor", "ONYX_MM_CHAT_TEMPLATE"]

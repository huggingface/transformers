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
"""LLaVA-OneVision-2 multimodal processor."""

import re

import torch

from ...image_processing_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring
from ...vision_utils import get_vision_position_ids


@auto_docstring
class LlavaOnevision2Processor(ProcessorMixin):
    skip_tensor_conversion = ["video_metadata"]

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>"
        self.video_token = "<|vision_start|><|video_pad|><|vision_end|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    @auto_docstring
    def __call__(self, images=None, text=None, videos=None, **kwargs):
        # The encoder consumes a single patch stream, in the same order as visual placeholders.
        prompts = [text] if isinstance(text, str) else text
        outputs = super().__call__(images=images, text=text, videos=videos, **kwargs)
        if images is None and videos is None:
            return outputs
        is_codec = outputs.pop("is_codec", False)
        visual = {}
        for modality, pixel_key, grid_key, position_key in (
            ("image", "pixel_values", "image_grid_thw", "patch_positions"),
            ("video", "pixel_values_videos", "video_grid_thw", "patch_positions_videos"),
        ):
            if pixel_key not in outputs:
                visual[modality] = iter(())
                continue
            grids = torch.as_tensor(outputs.pop(grid_key))
            pixels = torch.as_tensor(outputs.pop(pixel_key))
            positions = torch.as_tensor(outputs.pop(position_key))
            chunks, offset = [], 0
            for grid in grids:
                count = int(grid.prod())
                rows = grid[None]
                if modality == "video" and not is_codec:
                    rows = grid[None].repeat(int(grid[0]), 1)
                    rows[:, 0] = 1
                chunks.append((pixels[offset : offset + count], rows, positions[offset : offset + count]))
                offset += count
            visual[modality] = iter(chunks)
        if prompts is None:
            ordered = [*visual["image"], *visual["video"]]
        else:
            ordered = []
            pattern = f"(?P<image>{re.escape(self.image_token)})|(?P<video>{re.escape(self.video_token)})"
            for prompt in prompts:
                for match in re.finditer(pattern, prompt):
                    item = next(visual[match.lastgroup], None)
                    if item is None:
                        raise ValueError(f"More {match.lastgroup} placeholders than visual inputs")
                    ordered.append(item)
            if any(next(iterator, None) is not None for iterator in visual.values()):
                raise ValueError("Each visual input must have a placeholder in text")
        for index, key in enumerate(("pixel_values", "image_grid_thw", "patch_positions")):
            outputs[key] = torch.cat([item[index] for item in ordered])
        return BatchFeature(
            dict(outputs), tensor_type=kwargs.get("return_tensors"), skip_tensor_conversion=self.skip_tensor_conversion
        )

    def _process_images(self, images, **kwargs):
        for name in ("patch_size", "merge_size", "temporal_patch_size"):
            if name in kwargs and kwargs[name] != getattr(self.image_processor, name):
                raise ValueError(
                    f"{name} must match the image processor configuration ({getattr(self.image_processor, name)})"
                )
        outputs, replacements = super()._process_images(images, **kwargs)
        outputs["patch_positions"] = get_vision_position_ids(
            torch.as_tensor(outputs["image_grid_thw"]), self.image_processor.merge_size, include_temporal=True
        )
        return outputs, replacements

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "patch_positions"]

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        count = (
            int(torch.as_tensor(image_inputs["image_grid_thw"][image_idx]).prod())
            // self.image_processor.merge_size**2
        )
        return self.image_token * count

    def _process_videos(self, videos, **kwargs):
        outputs = self.video_processor(videos, **kwargs)
        grids = torch.as_tensor(outputs["video_grid_thw"])
        coordinates = torch.as_tensor(outputs.pop("patch_positions"))
        outputs["patch_positions_videos"] = coordinates
        is_codec = outputs.get("is_codec", False)
        separator = "\n" if is_codec else ""
        replacements, offset = [], 0
        for grid, times in zip(grids, outputs.pop("frame_timestamps")):
            time, height, width = grid.tolist()
            if is_codec:
                values, counts = coordinates[offset : offset + int(grid.prod()), 0].unique_consecutive(
                    return_counts=True
                )
                runs = [
                    (value.item() / times, count.item() // self.video_processor.merge_size**2)
                    for value, count in zip(values, counts)
                    if value >= 0
                ]
            else:
                runs = [(timestamp, height * width // self.video_processor.merge_size**2) for timestamp in times]
            replacement = separator.join(
                f"<{timestamp:.1f} seconds><|vision_start|>{self.image_token * count}<|vision_end|>"
                for timestamp, count in runs
            )
            replacements.append(replacement + separator)
            offset += int(grid.prod())
        return outputs, replacements


__all__ = ["LlavaOnevision2Processor"]

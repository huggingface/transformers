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
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)

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


class OnyxProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"add_special_tokens": False},
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class OnyxProcessor(ProcessorMixin):
    valid_processor_kwargs = OnyxProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<|image|>"
        self.image_start_token = "<|image_start|>"
        self.image_end_token = "<|image_end|>"
        self.video_token = "<|video|>"
        self.video_sep_token = "<|vid_frame_separator|>"
        self.video_start_token = "<|vid_start|>"
        self.video_end_token = "<|vid_end|>"

        super().__init__(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template or ONYX_MM_CHAT_TEMPLATE,
            **kwargs,
        )

    # maybe chat template should add start-end tokens?
    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        merge_length = self.image_processor.downsample_factor**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_start_token + self.image_token * num_image_tokens + self.image_end_token

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        merge_length = self.video_processor.downsample_factor**2
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

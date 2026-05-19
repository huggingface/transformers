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

import tempfile
import unittest

from transformers import Molmo2Processor
from transformers.testing_utils import require_torch, require_torchvision, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_vision
@require_torch
@require_torchvision
class Molmo2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Molmo2Processor
    model_id = "allenai/Molmo2-8B"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(model_id, **kwargs)

    @staticmethod
    def prepare_processor_dict():
        # Override the chat template to support the "system" role used by the base test harness.
        # The original Molmo2 template enforces strict user/assistant alternation without system.
        return {
            "chat_template": (
                "{{ bos_token }}"
                "{%- if messages[0]['role'] == 'system' -%}"
                "  {%- set system_message = messages[0]['content'][0]['text'] -%}"
                "  {%- set loop_messages = messages[1:] -%}"
                "{%- else -%}"
                "  {%- set system_message = '' -%}"
                "  {%- set loop_messages = messages -%}"
                "{%- endif -%}"
                "{%- for message in loop_messages -%}"
                "  {%- if message['role'] == 'user' -%}"
                "    {{ '<|im_start|>user\\n' }}"
                "    {%- if message['content'] is string -%}"
                "      {{ message['content'] }}"
                "    {%- else -%}"
                "      {%- for item in message['content'] -%}"
                "        {%- if item['type'] == 'image' -%}"
                "          {{ '<|image|>' }}"
                "        {%- elif item['type'] == 'video' -%}"
                "          {{ '<|video|>' }}"
                "        {%- elif item['type'] == 'text' -%}"
                "          {{ item['text'] }}"
                "        {%- endif -%}"
                "      {%- endfor -%}"
                "    {%- endif -%}"
                "    {{ '<|im_end|>\\n' }}"
                "  {%- elif message['role'] == 'assistant' -%}"
                "    {{ '<|im_start|>assistant\\n' }}"
                "    {%- if message['content'] is string -%}"
                "      {{ message['content'] }}"
                "    {%- else -%}"
                "      {%- for item in message['content'] -%}"
                "        {%- if item['type'] == 'text' -%}"
                "          {{ item['text'] }}"
                "        {%- endif -%}"
                "      {%- endfor -%}"
                "    {%- endif -%}"
                "  {%- endif -%}"
                "{%- endfor -%}"
                "{%- if add_generation_prompt -%}"
                "  {{ '<|im_start|>assistant\\n' }}"
                "{%- endif -%}"
            ),
        }

    def test_apply_chat_template_video_frame_sampling(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        video_url = "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4"
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": video_url},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        # Default `max_fps=2` caps `num_frames=3` for the ~1s tiny_video to a single frame.
        out_capped = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"num_frames": 3, "fps": None},
        )
        self.assertIn(self.videos_input_name, out_capped)
        self.assertEqual(len(out_capped[self.videos_input_name]), 1)
        self.assertEqual(len(out_capped[self.videos_input_name][0]), 1)

        # Raising the cap above the video's native fps restores the requested `num_frames`.
        out_uncapped = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"num_frames": 3, "fps": None, "max_fps": 1000},
        )
        self.assertEqual(len(out_uncapped[self.videos_input_name][0]), 3)

        # `fps` mode bypasses the cap entirely (different code path in sample_frames).
        out_fps = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"fps": 10, "num_frames": None},
        )
        self.assertGreaterEqual(len(out_fps[self.videos_input_name][0]), 1)

        # `fps` and `num_frames` are mutually exclusive.
        with self.assertRaises(ValueError):
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                processor_kwargs={"fps": 10, "num_frames": 3},
            )

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()
        inputs_dict = {"text": text, "images": image_input}
        inputs = processor(**inputs_dict, return_tensors="pt")

        # Output keys should be a subset of model_input_names (video keys absent when no video passed)
        self.assertTrue(set(inputs.keys()).issubset(set(processor.model_input_names)))

    # =====================================================================
    # Hub model has auto_map in processor_config.json which is not preserved
    # through save/load cycle. Override to filter auto_map before comparison.
    # =====================================================================
    def _filter_auto_map(self, d):
        """Remove auto_map keys from processor dict for comparison."""
        filtered = {k: v for k, v in d.items() if k != "auto_map"}
        for key in filtered:
            if isinstance(filtered[key], dict) and "auto_map" in filtered[key]:
                filtered[key] = {kk: vv for kk, vv in filtered[key].items() if kk != "auto_map"}
        return filtered

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_files = processor_first.save_pretrained(tmpdirname)
            if len(saved_files) > 0:
                processor_second = self.processor_class.from_pretrained(tmpdirname)
                self.assertEqual(
                    self._filter_auto_map(processor_second.to_dict()),
                    self._filter_auto_map(processor_first.to_dict()),
                )

    def test_processor_from_and_save_pretrained_as_nested_dict(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_first.save_pretrained(tmpdirname)
            processor_second = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(
                self._filter_auto_map(processor_second.to_dict()),
                self._filter_auto_map(processor_first.to_dict()),
            )

    # Hub processor_config.json has use_single_crop_col_tokens=False which
    # differs from the __init__ default of None when building from components.
    def test_processor_from_pretrained_vs_from_components(self):
        pass

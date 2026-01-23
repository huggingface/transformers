# Copyright 2025 HuggingFace Inc.
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

from parameterized import parameterized

from transformers import VibeVoiceProcessor
from transformers.testing_utils import require_librosa, require_torch
from transformers.utils import is_torch_available

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


if is_torch_available():
    pass


@require_torch
class VibeVoiceProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VibeVoiceProcessor
    audio_input_name = "input_values"

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "bezzam/VibeVoice-1.5B"
        processor = VibeVoiceProcessor.from_pretrained(cls.checkpoint)
        cls.audio_bos_token = processor.audio_bos_token
        cls.audio_bos_token_id = processor.audio_bos_token_id
        cls.audio_eos_token = processor.audio_eos_token
        cls.audio_eos_token_id = processor.audio_eos_token_id
        cls.audio_diffusion_token = processor.audio_diffusion_token
        cls.audio_diffusion_token_id = processor.audio_diffusion_token_id
        cls.pad_token_id = processor.tokenizer.pad_token_id
        cls.bos_token_id = processor.tokenizer.bos_token_id
        cls.tmpdirname = tempfile.mkdtemp()
        processor.save_pretrained(cls.tmpdirname)

    def prepare_processor_dict(self):
        return {
            "chat_template": """{%- set system_prompt = system_prompt | default(" Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n") -%}
{{ system_prompt -}}
{%- set audio_bos_token = audio_bos_token | default("<|vision_start|>") %}
{%- set audio_eos_token = audio_eos_token | default("<|vision_end|>") %}
{%- set audio_diffusion_token = audio_diffusion_token | default("<|vision_pad|>") %}
{%- set ns = namespace(speakers_with_audio="") %}
{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set content = message['content'] %}
    {%- set has_audio = content | selectattr('type', 'equalto', 'audio') | list | length > 0 %}
    {%- if has_audio and role not in ns.speakers_with_audio %}
        {%- set ns.speakers_with_audio = ns.speakers_with_audio + role + "," %}
    {%- endif %}
{%- endfor %}

{%- if ns.speakers_with_audio %}
{{ " Voice input:\n" }}
{%- for speaker in ns.speakers_with_audio.rstrip(',').split(',') %}
{%- if speaker %}
 Speaker {{ speaker }}:{{ audio_bos_token }}{{ audio_diffusion_token }}{{ audio_eos_token }}{{ "\n" }}
{%- endif %}
{%- endfor %}
{%- endif %}
 Text input:{{ "\n" }}

{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list %}
    {%- for item in text_items %}
 Speaker {{ role }}: {{ item['text'] }}{{ "\n" }}
    {%- endfor %}
{%- endfor %}
 Speech output:{{ "\n" }}{{ audio_bos_token }}"""
        }

    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("VibeVoice only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

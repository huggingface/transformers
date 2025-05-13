# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import unittest

from transformers import AutoProcessor, GPT2TokenizerFast, Phi4MultimodalFeatureExtractor, Phi4MultimodalProcessor
from transformers.testing_utils import require_torch, require_torchaudio
from transformers.utils import is_torchvision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torchvision_available():
    from transformers import Phi4MultimodalImageProcessorFast


@require_torch
@require_torchaudio
class Phi4MultimodalProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Phi4MultimodalProcessor
    images_input_name = "image_pixel_values"
    audios_input_name = "audio_input_features"

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "microsoft/Phi-4-multimodal-instruct"
        cls.tmpdirname = "/home/raushan/llavas/dummy-phi"

        tokenizer = GPT2TokenizerFast.from_pretrained(
            cls.checkpoint, extra_special_tokens={"image_token": "<|image|>", "audio_token": "<|audio|>"}
        )
        image_processor = Phi4MultimodalImageProcessorFast.from_pretrained(cls.checkpoint, num_image_tokens=4)
        audio_processor = Phi4MultimodalFeatureExtractor.from_pretrained(cls.checkpoint)
        processor_kwargs = cls.prepare_processor_dict()
        processor = Phi4MultimodalProcessor(
            tokenizer=tokenizer, image_processor=image_processor, audio_processor=audio_processor, **processor_kwargs
        )
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token
        cls.audio_token = processor.audio_token

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_audio_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).audio_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{% for message in messages %}{{ '<|' + message['role'] + '|>' }}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<|image|>' }}{% elif content['type'] == 'audio' %}{{ '<|audio|>' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% endif %}{{ '<|end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}"
        }

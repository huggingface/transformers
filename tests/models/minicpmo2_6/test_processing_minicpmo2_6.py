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

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        MiniCPMO2_6Processor,
        MiniCPMV2_5ImageProcessor,
        PreTrainedTokenizerFast,
        WhisperFeatureExtractor,
    )


@require_vision
@require_torch
@require_torchvision
class MiniCPMO2_6ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniCPMO2_6Processor

    @staticmethod
    def prepare_processor_dict():
        # Renders images and audios like the converted checkpoint's chat template
        chat_template = (
            "{%- set ns = namespace(image_count=0) -%}"
            "{%- for message in messages -%}"
            "{%- if message['content'] is string -%}{%- set content = message['content'] -%}"
            "{%- else -%}{%- set parts = namespace(items=[]) -%}"
            "{%- for item in message['content'] -%}"
            "{%- if item['type'] == 'image' -%}"
            "{%- set parts.items = parts.items + ['<image_id>' ~ ns.image_count ~ '</image_id><unk>'] -%}"
            "{%- set ns.image_count = ns.image_count + 1 -%}"
            "{%- elif item['type'] == 'audio' -%}{%- set parts.items = parts.items + ['<|audio|>'] -%}"
            "{%- elif item['type'] == 'text' -%}{%- set parts.items = parts.items + [item['text']] -%}{%- endif -%}"
            "{%- endfor -%}{%- set content = parts.items | join('\n') -%}{%- endif -%}"
            "{{- '<|im_start|>' + message['role'] + '\n' + content + '<|im_end|>\n' -}}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
        )
        return {"chat_template": chat_template, "num_image_tokens": 4}

    @classmethod
    def _setup_tokenizer(cls):
        return PreTrainedTokenizerFast.from_pretrained("openbmb/MiniCPM-o-2_6")

    @classmethod
    def _setup_image_processor(cls):
        # One overview image and no slices keeps the tests fast.
        return MiniCPMV2_5ImageProcessor(scale_resolution=64, max_slice_nums=1)

    @classmethod
    def _setup_feature_extractor(cls):
        return WhisperFeatureExtractor()

    @unittest.skip(reason="The feature extractor's `attention_mask` is returned as `input_features_mask`")
    def test_subprocessor_defaults_3_audio(self):
        pass

    def test_audio_placeholder(self):
        processor = self.get_processor()
        # 2.5 s and 31 s: the second is split into a 30 s and a 1 s window
        audios = [np.zeros(40000, dtype=np.float32), np.zeros(496000, dtype=np.float32)]
        inputs = processor(
            text=[processor.audio_token, processor.audio_token], audio=audios, padding=True, return_tensors="pt"
        )

        self.assertListEqual(list(inputs["input_features"].shape), [3, 80, 3000])
        self.assertListEqual(inputs["input_features_mask"].sum(-1).tolist(), [250, 3000, 100])
        # 25 tokens per second, each second in its own <|audio_start|>...<|audio_end|>
        one_second = f"<|audio_start|>{processor.audio_token * 25}<|audio_end|>"
        half_second = f"<|audio_start|>{processor.audio_token * 12}<|audio_end|>"
        for ids, text in zip(inputs["input_ids"], [one_second * 2 + half_second, one_second * 31]):
            expected = processor.tokenizer(text)["input_ids"]
            self.assertListEqual(ids[ids != processor.tokenizer.pad_token_id].tolist(), expected)

        num_tokens = processor._get_num_multimodal_tokens(audio_lengths=[len(audio) for audio in audios])
        self.assertListEqual(num_tokens["num_audio_tokens"], [62, 775])

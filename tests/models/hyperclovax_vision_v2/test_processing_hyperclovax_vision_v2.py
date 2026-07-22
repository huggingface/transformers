# Copyright 2026 NAVER Corp. and The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import GPT2TokenizerFast, HyperCLOVAXVisionV2Processor, Qwen2VLVideoProcessor


@require_vision
@require_torch
@require_torchvision
class HyperCLOVAXVisionV2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HyperCLOVAXVisionV2Processor

    @classmethod
    def _setup_tokenizer(cls):
        # HyperCLOVAX Vision V2 uses a GPT-2 tokenizer; add the placeholder tokens the processor
        # relies on (`<|image_pad|>` / `<|video_pad|>`) plus a pad token so multimodal token counts match.
        tokenizer = GPT2TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|image_pad|>", "<|video_pad|>"]})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def _setup_video_processor(cls):
        # HyperCLOVAX Vision V2 reuses `Qwen2VLVideoProcessor` but with the checkpoint's pixel budget
        # (`min_pixels=3136`, `max_pixels=12845056`); the class default `min_pixels` (128*28*28) would
        # upscale the tiny test videos and overflow the shared max-length assertions.
        return Qwen2VLVideoProcessor(min_pixels=3136, max_pixels=12845056)

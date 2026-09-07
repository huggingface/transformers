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
"""Tests for loading the original `OpenGVLab/InternVL2-*` checkpoints natively."""

import unittest

from transformers import AutoConfig, AutoTokenizer, InternVL2Config, InternVLForConditionalGeneration
from transformers.testing_utils import cleanup, require_torch, slow, torch_device
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@slow
@require_torch
class InternVL2OriginalCheckpointTest(unittest.TestCase):
    """The original `OpenGVLab/InternVL2-*` checkpoints use the bespoke `internvl_chat`
    remote-code layout. They should load into the native implementation without
    `trust_remote_code`, on CPU, with every weight mapped."""

    checkpoint = "OpenGVLab/InternVL2-1B"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_original_internvl2_checkpoint_loads_natively(self):
        config = AutoConfig.from_pretrained(self.checkpoint, trust_remote_code=False)
        self.assertIsInstance(config, InternVL2Config)
        self.assertEqual(config.text_config.model_type, "qwen2")
        self.assertGreater(config.text_config.num_attention_heads, 0)
        self.assertGreater(config.vision_config.num_attention_heads, 0)

        model, loading_info = InternVLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            dtype=torch.bfloat16,
            trust_remote_code=False,
            output_loading_info=True,
        )
        self.assertIsInstance(model, InternVLForConditionalGeneration)
        self.assertEqual(len(loading_info["missing_keys"]), 0)
        self.assertEqual(len(loading_info["unexpected_keys"]), 0)
        self.assertEqual(len(loading_info["mismatched_keys"]), 0)

    def test_image_token_id_matches_tokenizer(self):
        """`config.image_token_id` must equal the checkpoint tokenizer's `<IMG_CONTEXT>` id.

        The model masks image positions with `config.image_token_id` while the processor takes
        the id from the tokenizer. If the two disagree no image features are ever spliced in and
        the model silently answers as if no image was given, without raising.
        """
        for checkpoint, text_model_type in [
            ("OpenGVLab/InternVL2-1B", "qwen2"),
            ("OpenGVLab/InternVL2-2B", "llama"),
        ]:
            with self.subTest(checkpoint=checkpoint):
                config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=False)
                # InternLM2 checkpoints ship a remote-code tokenizer, hence trust_remote_code here.
                tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
                self.assertEqual(config.text_config.model_type, text_model_type)
                self.assertEqual(config.image_token_id, tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))

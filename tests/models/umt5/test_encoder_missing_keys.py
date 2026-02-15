# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import UMT5Config, UMT5EncoderModel
from transformers.testing_utils import require_torch


@require_torch
class UMT5EncoderMissingKeysTest(unittest.TestCase):
    def test_encoder_embed_tokens_missing_key_is_ignored_on_load(self):
        config = UMT5Config(
            vocab_size=99,
            d_model=32,
            d_ff=37,
            d_kv=8,
            num_layers=2,
            num_heads=4,
            relative_attention_num_buckets=8,
        )
        model = UMT5EncoderModel(config).eval()
        state_dict = model.state_dict()
        state_dict.pop("encoder.embed_tokens.weight")

        _, loading_info = UMT5EncoderModel.from_pretrained(
            None,
            config=config,
            state_dict=state_dict,
            output_loading_info=True,
        )

        self.assertNotIn("encoder.embed_tokens.weight", loading_info["missing_keys"])

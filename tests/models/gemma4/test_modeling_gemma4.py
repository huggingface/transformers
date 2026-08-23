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
"""Testing suite for the PyTorch Gemma4 model."""

aaa = 1




import tempfile
import unittest
from contextlib import contextmanager

import pytest
from parameterized import parameterized

from transformers import logging
from transformers import (
    AutoTokenizer,
    Gemma4Config,
    Gemma4TextConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import ModelOutput

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        Gemma4ForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Model,
        Gemma4TextModel,
    )
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_vision_model


logger = logging.get_logger(__name__)

GEMMA4_RANDOM_MOE_FA2_SKIP_REASON = (
    "Randomly initialized Gemma4 MoE routers are too sensitive to tiny eager/FA2 input differences"
)


@slow
@require_torch_accelerator
class Gemma4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "google/gemma-4-E2B-it"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    # Note: we do not test FA2 as the head dim is 512 on some layers, which is not compatible with the kernels
    @parameterized.expand([("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. Outputs for every attention functions
        should be coherent and identical.
        """

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding="left")
        input_text = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in input_text
        ]
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = Gemma4ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=torch_device,
            attn_implementation=attn_implementation,
        )

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.get_text_config().sliding_window)

        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, cache_implementation="static")
        output_text = tokenizer.batch_decode(out[:, input_size:])

        EXPECTED_COMPLETIONS = Expectations(
            {
                ("cuda", 8): [
                    "That sounds lovely! It seems like you're really enjoying the place you'",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
                ("xpu", 5): [
                    "That sounds lovely! It seems like you're really enjoying the place you'",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
            }
        )
        logger.warning(f"[{self.id()}] output_text={output_text}")
        self.assertEqual(output_text, EXPECTED_COMPLETIONS.get_expectation())

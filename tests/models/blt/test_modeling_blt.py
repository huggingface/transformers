# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch BLT model."""

import unittest

from packaging import version

from transformers import AutoTokenizer, StaticCache, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        BLTConfig,
        BLTForCausalLM,
        BLTModel,
        BLTTokenizer
    )
    from transformers.models.blt.modeling_blt import BLTRotaryEmbedding

# import os
# import gc
# gc.collect()
# torch.cuda.empty_cache()


# os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"


class BLTModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = BLTConfig
        base_model_class = BLTModel
        causal_lm_class = BLTForCausalLM


@require_torch
class BLTModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            BLTModel,
            BLTForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": BLTModel,
            "text-generation": BLTForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = BLTModelTester
    rotary_embedding_layer = BLTRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = BLTForCausalLM if is_torch_available() else None


# @require_torch_accelerator
class BLTIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_read_token
    def test_blt(self):
        prompt = "my name is"

        EXPECTED_TEXT = " alex and i am a student at the university of michigan. i am a senior majoring in computer science and minoring in mathematics. i am also a member of the michigan math club and the michigan computer s"

        model = BLTForCausalLM.from_pretrained(
            "itazap/blt-1b", device_map="auto" #, torch_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b")

        input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)]).to(torch_device)

        output_ids = model.generate(
            input_ids, 
            max_new_tokens=200
        )

        generated_ids = output_ids[0][len(input_ids[0]):]
        output_text = tokenizer.decode(generated_ids.tolist())
        
        print(f'Prompt: "{prompt}"')
        print(f'Completion: "{output_text}"')
        print('here')

        self.assertEqual(output_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    def test_model_logits(self):
        input_ids = [1, 42, 21, 12, 43, 23, 1, 4]

        model = BLTForCausalLM.from_pretrained(
            "itazap/blt-1b", device_map="auto"
        )

        with torch.no_grad():
            output = model(torch.tensor([input_ids]).to(torch_device))[0] 

        EXPECTED_OUTPUT = torch.tensor([[-10.4948, -10.7065,  -6.1813, -10.5545, -10.3428,  -9.1493,  -8.4937,
          -8.6382,  -9.2159,  -9.5907,  -9.3679,  -8.4184,  -9.0655,  -3.4436,
           2.9616, -10.3157,  -6.3723,  -6.0133,  -9.7100,  -9.2128,  -8.8064,
          -9.8179,  -9.7516,  -9.4681,  -9.7715,  -9.4897,  -9.0491,  -9.8098,
          -9.4648,  -9.3294],
        [-13.3010, -13.1910,  -5.7230, -13.2895, -13.4864,  -8.7140,  -7.0275,
          -7.0182, -10.1362, -10.3762,  -9.9086,  -7.8049,  -8.8660,  -5.2711,
          -3.5778, -12.5346,  -9.1609,  -6.7925, -10.3717,  -9.2650, -10.6393,
         -11.4807, -11.2128, -10.9615, -10.5806, -10.8873, -11.0651, -11.3471,
         -10.5437,  -9.9688]]).to(torch_device) 
        
        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-4, atol=1e-4)

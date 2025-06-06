# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from parameterized import parameterized

from transformers import AutoTokenizer, MambaConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        MambaForCausalLM,
        MambaModel,
    )


class MambaModelTester(CausalLMModelTester):
    config_class = MambaConfig
    if is_torch_available():
        base_model_class = MambaModel
        causal_lm_class = MambaForCausalLM


@require_torch
class MambaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (MambaModel, MambaForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": MambaModel, "text-generation": MambaForCausalLM} if is_torch_available() else {}
    )
    model_tester_class = MambaModelTester

    @unittest.skip("The `input_embeds` when fed don't produce the same results.")
    def test_beam_sample_generate(self):
        pass


@require_torch
class MambaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "state-spaces/mamba-2.8b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @parameterized.expand([(torch_device,), ("cpu",)])
    def test_simple_generate(self, device):
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        tokenizer.pad_token = tokenizer.eos_token

        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", torch_dtype=torch.float32)
        model.to(device)
        input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"].to(device)

        out = model.generate(input_ids, do_sample=False, use_cache=True, max_new_tokens=10)
        output_sentence = tokenizer.decode(out[0, :])
        self.assertEqual(output_sentence, "Hey how are you doing?\n\nI'm so glad you're here.")

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits

        EXPECTED_LOGITS_NO_GRAD = torch.tensor(
            [
                -55.6909, -69.7903, -49.8981, -51.7581, -57.6544, -57.9368, -56.9591,
                -57.9033, -54.6787, -55.9261, -55.3011, -58.0765, -60.5642, -47.0176,
                -52.0344, -49.7836, -55.9463, -57.8957, -56.7627, -57.1080, -57.3434,
                -58.3015, -57.7875, -58.7760, -59.6037, -59.0665, -58.7087, -52.9293,
                -53.4654, -57.3466, -56.9294, -55.7314, -53.3141, -55.8171, -56.9879,
                -56.9121, -56.2139, -54.7198, -56.4134, -57.4825
            ])  # fmt: skip

        torch.testing.assert_close(logits[0, 0, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)

    @parameterized.expand([(torch_device,), ("cpu",)])
    def test_simple_generate_cuda_kernels_tiny(self, device):
        expected_output = "Hello my name is John and I am a newbie to the world"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(device)
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", torch_dtype=torch.float16).to(device)

        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    @parameterized.expand([(torch_device,), ("cpu",)])
    @slow
    def test_simple_generate_cuda_kernels_small(self, device):
        expected_output = "Hello my name is\n\nI am a\n\nI am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(device)
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-790m-hf", torch_dtype=torch.float16).to(device)

        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    @parameterized.expand([(torch_device,), ("cpu",)])
    @slow
    def test_simple_generate_cuda_kernels_mid(self, device):
        expected_output = "Hello my name is John and I am a\n\nI am a single father of a beautiful daughter. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(device)
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf", torch_dtype=torch.float16).to(device)

        output = model.generate(input_ids, max_new_tokens=20)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    @parameterized.expand([(torch_device,), ("cpu",)])
    @slow
    def test_simple_generate_cuda_kernels_big(self, device):
        expected_output = "Hello my name is John and I am a new member of this forum. I am a retired Marine and I am a member of the Marine Corps League. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(device)
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf", torch_dtype=torch.float16).to(device)

        output = model.generate(input_ids, max_new_tokens=30)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    @slow
    def test_compile_mamba_cache(self):
        expected_output = "Hello my name is John and I am a\n\nI am a single father of a beautiful daughter. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="pt").input_ids.to(torch_device)
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf", torch_dtype=torch.float16).to(
            torch_device
        )

        output = model.generate(input_ids, max_new_tokens=20, cache_implementation="mamba")
        output_sentence = self.tokenizer.decode(output[0].tolist())
        self.assertEqual(output_sentence, expected_output)

        model.forward = torch.compile(model.forward, fullgraph=True, mode="reduce-overhead")
        output = model.generate(input_ids, max_new_tokens=20, cache_implementation="mamba")
        output_sentence = self.tokenizer.decode(output[0].tolist())
        self.assertEqual(output_sentence, expected_output)

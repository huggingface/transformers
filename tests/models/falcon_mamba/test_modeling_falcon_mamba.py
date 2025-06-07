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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, FalconMambaConfig, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        FalconMambaForCausalLM,
        FalconMambaModel,
    )


# Copied from transformers.tests.models.mamba.MambaModelTester with Mamba->FalconMamba,mamba->falcon_mamba
class FalconMambaModelTester(CausalLMModelTester):
    config_class = FalconMambaConfig
    if is_torch_available():
        base_model_class = FalconMambaModel
        causal_lm_class = FalconMambaForCausalLM


@require_torch
# Copied from transformers.tests.models.mamba.MambaModelTest with Mamba->Falcon,mamba->falcon_mamba,FalconMambaCache->MambaCache
class FalconMambaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (FalconMambaModel, FalconMambaForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": FalconMambaModel, "text-generation": FalconMambaForCausalLM}
        if is_torch_available()
        else {}
    )
    model_tester_class = FalconMambaModelTester


@require_torch
@require_torch_accelerator
@slow
class FalconMambaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "tiiuae/falcon-mamba-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.text = "Hello today"

    def test_generation_bf16(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")

        inputs = self.tokenizer(self.text, return_tensors="pt").to(torch_device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        self.assertEqual(
            self.tokenizer.batch_decode(out, skip_special_tokens=False)[0],
            "Hello today I am going to show you how to make a simple and easy to make paper plane.\nStep",
        )

    @require_bitsandbytes
    def test_generation_4bit(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=quantization_config)

        inputs = self.tokenizer(self.text, return_tensors="pt").to(torch_device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        self.assertEqual(
            self.tokenizer.batch_decode(out, skip_special_tokens=False)[0],
            """Hello today I'm going to talk about the "C" in the "C-I-""",
        )

    def test_generation_torch_compile(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
        model = torch.compile(model)

        inputs = self.tokenizer(self.text, return_tensors="pt").to(torch_device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        self.assertEqual(
            self.tokenizer.batch_decode(out, skip_special_tokens=False)[0],
            "Hello today I am going to show you how to make a simple and easy to make paper plane.\nStep",
        )

    def test_batched_generation(self):
        model_id = "tiiuae/falcon-mamba-7b"
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.pad_token_id = tok.eos_token_id

        texts = ["Hello today", "Hello my name is Younes and today"]

        EXPECTED_OUTPUT = [
            "Hello today I'm going to show you how to make a 3D model of a house.\n",
            "Hello my name is Younes and today I will be talking about the topic of “The importance of the internet in our life”.\n",
        ]

        inputs = tok(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(torch_device)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=0, torch_dtype=torch.bfloat16)

        out = model.generate(**inputs, max_new_tokens=20)
        out = tok.batch_decode(out, skip_special_tokens=True)

        self.assertListEqual(out, EXPECTED_OUTPUT)

        # We test the same generations with inputs_embeds
        with torch.no_grad():
            inputs_embeds = model.get_input_embeddings()(inputs.pop("input_ids"))

        inputs["inputs_embeds"] = inputs_embeds
        out = model.generate(**inputs, max_new_tokens=20)
        out = tok.batch_decode(out, skip_special_tokens=True)

        self.assertListEqual(out, EXPECTED_OUTPUT)

    @require_torch_multi_accelerator
    def test_training_kernel(self):
        model_id = "tiiuae/falcon-mamba-7b"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        text = "Hello today"

        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            logits = torch.argmax(model(**inputs).logits, dim=-1)

        out_no_training = tokenizer.batch_decode(logits)

        model.train()
        lm_logits = model(**inputs).logits
        next_token = torch.argmax(lm_logits, dim=-1)

        out_training = tokenizer.batch_decode(next_token)

        # Just verify backward works
        loss = (1 - lm_logits).mean()
        loss.backward()

        self.assertEqual(out_training, out_no_training)

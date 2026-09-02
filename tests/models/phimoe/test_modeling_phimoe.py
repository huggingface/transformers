# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""Testing suite for the PyTorch PhiMoE model."""

import tempfile
import unittest

from parameterized import parameterized

from transformers import StaticCache, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        PhimoeForCausalLM,
        PhimoeModel,
    )

    end_of_text_token = 32000

    class PhimoeMiniWithStaticCache(torch.nn.Module):
        def __init__(self, model: PhimoeForCausalLM, batch_size: int, max_seq_len: int):
            super().__init__()
            self.model = model
            self.cache = StaticCache(config=model.config, max_cache_len=max_seq_len)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
        ) -> torch.FloatTensor:
            return self.model.forward(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                past_key_values=self.cache,
            ).logits

        @torch.no_grad()
        @staticmethod
        def generate(model: PhimoeForCausalLM, prompt_tokens: torch.LongTensor, max_seq_len: int) -> list[int]:
            model = PhimoeMiniWithStaticCache(model, 1, max_seq_len + prompt_tokens.shape[-1])

            response_tokens = []

            for input_pos in range(prompt_tokens.shape[-1]):
                result = model.forward(
                    input_ids=prompt_tokens[:, input_pos : input_pos + 1],
                )
                response_tokens.append(prompt_tokens[0][input_pos].item())

            current_token = torch.argmax(result[:, -1, :], dim=-1).item()
            response_tokens.append(current_token)

            while current_token != end_of_text_token and len(response_tokens) < max_seq_len:
                result = model.forward(
                    input_ids=torch.tensor([[current_token]], dtype=torch.long),
                )
                current_token = torch.argmax(result[:, -1, :], dim=-1).item()
                response_tokens.append(current_token)

            return response_tokens


class PhimoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = PhimoeModel


@require_torch
class PhimoeModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = PhimoeModelTester

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79292/workflows/fa2ba644-8953-44a6-8f67-ccd69ca6a476/jobs/1012905
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True

    @unittest.skip("PhiMoE's RoPE has custom parameterization")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("PhiMoE's RoPE has custom parameterization")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass


@slow
@require_torch
class PhimoeIntegrationTest(unittest.TestCase):
    model = None
    offload_dir = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.offload_dir = tempfile.TemporaryDirectory()
            # This checkpoint is ~78 GiB in bf16, which very nearly fills the accelerators of a CI
            # runner, so `device_map="auto"` must be given an explicit budget that leaves headroom on
            # every device. `get_balanced_memory` only reserves its 10% margin when a *single*
            # accelerator is visible (`max_memory[key] *= 0.9` sits inside `if num_devices == 1`);
            # with several it hands every device its full capacity, the whole model lands on the
            # accelerators with nothing spilled, and loading then OOMs while merging the experts'
            # `w1`/`w3` into `gate_up_proj` — that `torch.cat` needs a ~1.7 GiB temporary per layer.
            # Inference has no room for activations either. That is also why only the multi-gpu
            # variant fails: single-gpu does get the 0.9 margin and spills to CPU.
            #
            # So budget 80% of each accelerator and let the surplus go to CPU (and to
            # `offload_folder` beyond that). #48290's 60 GiB x num_accelerators CPU budget is
            # preserved, but passed as `max_memory` instead of by capping what psutil reports:
            # `patch_psutil_cpu_memory` takes `min(mem.total, limit)`, so a 240 GiB "cap" was inert
            # on this runner, and a CPU-side budget cannot create accelerator headroom anyway.
            # Read each device directly rather than via
            # `get_accelerator_total_memory_gib()`: that helper returns 0.0 whenever
            # `torch_device` is not exactly "cuda"/"xpu", and a budget guarded on
            # `> 0` then falls through to `max_memory=None` — which is no budget at
            # all. Measured on this runner: that revision verified `not_fixed` with
            # 20.24 GiB allocated on GPU 0, i.e. above the cap it thought it had set,
            # failing on the same `Concatenate` for `layers.15.mlp.experts.gate_up_proj`.
            # A fallback that silently disables the fix is how the original bug
            # survived `cap_psutil_cpu_memory` in the first place.
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                raise unittest.SkipTest("phimoe integration test needs an accelerator")
            n = torch.cuda.device_count()
            per_device = int(min(torch.cuda.get_device_properties(i).total_memory for i in range(n)) * 0.85 / 1024**3)
            max_memory = dict.fromkeys(range(n), f"{per_device}GiB")
            max_memory["cpu"] = "60GiB"
            cls.model = PhimoeForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-MoE-instruct",
                experts_implementation="eager",
                dtype="auto",
                device_map="auto",
                max_memory=max_memory,
                offload_folder=cls.offload_dir.name,
            )
        return cls.model

    @classmethod
    def tearDownClass(cls):
        del cls.model
        if cls.offload_dir is not None:
            cls.offload_dir.cleanup()
            cls.offload_dir = None
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_phimoe_instruct_logits(self):
        input_ids = {"input_ids": torch.tensor([[1212, 318, 281, 1672]], dtype=torch.long, device=torch_device)}

        model = self.get_model()
        model.eval()

        with torch.no_grad():
            output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor(
            [
                [-3.5625, -2.4375, -1.3672, 0.3438, -0.7539, -0.4590, 0.6133, -0.4531, 0.2188, -1.2422],
                [-0.9688, 0.3633, -0.4902, 2.3281, 0.6250, 3.1094, 0.3828, 0.1670, 0.5781, -2.1094],
            ]
        ).to(device=torch_device, dtype=output.dtype)  # fmt: skip

        torch.testing.assert_close(output[0, :2, :10], EXPECTED_OUTPUT, rtol=1e-4, atol=1e-4)

    def test_phimoe_instruct_generation(self):
        model = self.get_model()
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        outputs = model.generate(**inputs, max_new_tokens=30)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits are both delicious and nutritious fruits that can be combined in various ways to create",
        ]
        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    def test_phimoe_instruct_with_static_cache(self):
        model = self.get_model()
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            torch_device
        )

        response_tokens = PhimoeMiniWithStaticCache.generate(model, inputs["input_ids"], max_seq_len=30)
        output_text = tokenizer.batch_decode(torch.tensor([response_tokens], dtype=torch.long, device=torch_device))

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> C"
        ]
        self.assertListEqual(output_text, EXPECTED_OUTPUT)

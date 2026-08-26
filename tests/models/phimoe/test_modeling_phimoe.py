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

import logging
import tempfile
import unittest

import psutil

from transformers import StaticCache, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

logger = logging.getLogger(__name__)








if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        PhimoeForCausalLM,
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


@slow
@require_torch
class PhimoeIntegrationTest(unittest.TestCase):
    model = None
    offload_dir = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.offload_dir = tempfile.TemporaryDirectory()
            mem = psutil.virtual_memory()
            logger.warning(
                "psutil.virtual_memory() before from_pretrained: total=%.2f GiB, available=%.2f GiB",
                mem.total / 1024**3,
                mem.available / 1024**3,
            )
            cls.model = PhimoeForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-MoE-instruct",
                experts_implementation="eager",
                dtype="auto",
                device_map="auto",
                offload_folder=cls.offload_dir.name,
                max_memory={"cpu": "60GiB"},
            )
            logger.warning("device_map=%s", cls.model.hf_device_map)
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

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

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)








if is_torch_available():
    import torch

    from transformers import (
        PhimoeForCausalLM,
    )

    end_of_text_token = 32000


@slow
@require_torch
class PhimoeIntegrationTest(unittest.TestCase):
    model = None
    offload_dir = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.offload_dir = tempfile.TemporaryDirectory()
            cls.model = PhimoeForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-MoE-instruct",
                experts_implementation="eager",
                dtype="auto",
                device_map="auto",
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

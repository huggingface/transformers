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

from transformers import LlamaConfig
from transformers.testing_utils import require_accelerate, require_bitsandbytes, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import BitsAndBytesConfig, LlamaForCausalLM
    from transformers.integrations.accelerate import _get_device_map, compute_module_sizes
    from transformers.quantizers.auto import AutoHfQuantizer


NUM_FAKE_GPUS = 8
CPU_BUDGET = 10 * 1024**3


@require_torch
@require_accelerate
class DeviceMapTest(unittest.TestCase):
    """Tests for the `device_map='auto'` placement computed by `_get_device_map`.

    Everything runs on the meta device: placement is pure size arithmetic, so no accelerator
    is needed and `max_memory` can reference fake GPU indices.
    """

    def _build_meta_model(self):
        # Tiny model whose embedding/lm_head are much bigger than a decoder layer, as in most real
        # models: the embeddings are the biggest un-splittable atoms the placement has to handle.
        from accelerate import init_empty_weights

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=4096,
            tie_word_embeddings=False,
        )
        with init_empty_weights():
            model = LlamaForCausalLM(config)
        return model

    def _generous_max_memory(self, model, hf_quantizer=None):
        # Per-device budget of 3x an even split, and at least 2x the biggest un-splittable module
        # (the embeddings, which quantization does not shrink): the whole model fits on a few of
        # the 8 devices and every single module fits on any of them.
        module_sizes, _ = compute_module_sizes(model, hf_quantizer)
        largest_atom = max(module_sizes["model.embed_tokens"], module_sizes["lm_head"])
        budget_per_gpu = int(max(module_sizes[""] * 3 / NUM_FAKE_GPUS, 2 * largest_atom))
        max_memory = dict.fromkeys(range(NUM_FAKE_GPUS), budget_per_gpu)
        max_memory["cpu"] = CPU_BUDGET
        return max_memory

    def test_auto_device_map_respects_large_leaf_modules(self):
        """Regression test for #46823 (structural part).

        `get_balanced_memory` used to size its per-device buffer only from `_no_split_modules`,
        while the packing in `infer_auto_device_map` treats any childless module (e.g. embeddings)
        as an atom. A leaf bigger than the resulting caps skipped every capped device for good, and
        the model collapsed onto the last device and then spilled to cpu despite an ample budget.
        """
        model = self._build_meta_model()
        device_map = _get_device_map(model, "auto", self._generous_max_memory(model), hf_quantizer=None)

        placements = set(device_map.values())
        self.assertNotIn("cpu", placements)
        self.assertNotIn("disk", placements)
        # The budgets force a multi-device placement; a single used device would mean the caps
        # collapsed the placement again.
        self.assertGreaterEqual(len({d for d in placements if isinstance(d, int)}), 2)

    @require_bitsandbytes
    def test_bnb_4bit_auto_device_map_with_explicit_max_memory(self):
        """Regression test for #46823: 4-bit + device_map='auto' + explicit ample max_memory used to
        dispatch modules to cpu, which `Bnb4BitHfQuantizer.validate_environment` rejects with
        'Some modules are dispatched on the CPU or the disk...'."""
        model = self._build_meta_model()
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
        hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=False)
        hf_quantizer.preprocess_model(model=model, device_map="auto", dtype=torch.bfloat16)

        max_memory = self._generous_max_memory(model, hf_quantizer)
        # Must not raise: the model fits the requested budget with a wide margin.
        device_map = _get_device_map(model, "auto", max_memory, hf_quantizer)

        placements = set(device_map.values())
        self.assertNotIn("cpu", placements)
        self.assertNotIn("disk", placements)

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

"""Real NF4 weights retain shared LoRA gradients and independent execution caches."""

import copy
import re
import tempfile
import unittest
from pathlib import Path

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_bitsandbytes, require_peft, require_torch_gpu


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_flatten

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LayerExecutionCache, set_layer_execution_plan

    from . import test_layer_execution as execution_tests


@require_torch_gpu
@require_bitsandbytes
@require_peft
class LayerExecutionQuantizationTest(unittest.TestCase):
    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_qlora_gradients_cache_and_adapter_reload(self, family):
        import bitsandbytes as bnb
        from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

        order = (0, 1, 0, 1, 2)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            adapter = Path(directory) / "adapter"
            native = execution_tests.LayerExecutionModelTest().make_model(family)
            native.set_layer_execution_plan(order)
            native.save_pretrained(base)
            load_kwargs = {
                "local_files_only": True,
                "dtype": torch.bfloat16,
                "device_map": {"": torch.cuda.current_device()},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_storage=torch.bfloat16,
                ),
            }
            model = AutoModelForCausalLM.from_pretrained(base, **load_kwargs)
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
            model = get_peft_model(
                model, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["down_proj"])
            )
            self.assertTrue(any(isinstance(module, bnb.nn.Linear4bit) for module in model.modules()))
            parameter_ids = tuple(id(parameter) for parameter in model.parameters())
            keys = tuple(model.state_dict())
            set_layer_execution_plan(model, order)
            self.assertEqual(tuple(id(parameter) for parameter in model.parameters()), parameter_ids)
            self.assertEqual(tuple(model.state_dict()), keys)
            frozen = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if not parameter.requires_grad
            }

            reference = copy.deepcopy(model)
            native_reference = reference.get_base_model()
            source_layers = list(native_reference.get_decoder().layers)
            native_reference.set_layer_execution_plan(None)
            decoder = native_reference.get_decoder()
            decoder.layers = torch.nn.ModuleList([copy.deepcopy(source_layers[index]) for index in order])
            decoder.config.num_hidden_layers = len(order)
            if getattr(decoder.config, "layer_types", None) is not None:
                decoder.config.layer_types = [decoder.config.layer_types[index] for index in order]
            for index, layer in enumerate(decoder.layers):
                for module in layer.modules():
                    if hasattr(module, "layer_idx"):
                        module.layer_idx = index
            model.train()
            reference.train()
            inputs = torch.tensor([[1, 2, 3, 4]], device=model.device)
            actual = model(inputs, labels=inputs, use_cache=False)
            expected = reference(inputs, labels=inputs, use_cache=False)
            torch.testing.assert_close(actual.loss, expected.loss, atol=1e-6, rtol=1e-5)
            actual.loss.backward()
            expected.loss.backward()
            gradients = {}
            for name, parameter in reference.named_parameters():
                if parameter.grad is not None:
                    source_name = re.sub(r"\.layers\.(\d+)\.", lambda match: f".layers.{order[int(match[1])]}.", name)
                    gradients[source_name] = gradients.get(source_name, 0) + parameter.grad
            nonzero = 0
            trainable = []
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIn("lora_", name)
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-5, rtol=3e-4, msg=name)
                    nonzero += int(torch.count_nonzero(parameter.grad).item() > 0)
                    trainable.append(parameter)
            self.assertGreater(nonzero, 0)
            optimizer = torch.optim.AdamW(trainable, lr=0.01)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in model.named_parameters():
                if name in frozen:
                    self.assertTrue(
                        torch.equal(
                            parameter.detach().contiguous().view(torch.uint8),
                            frozen[name].contiguous().view(torch.uint8),
                        )
                    )

            model.eval()
            with torch.no_grad():
                logits = model(inputs, use_cache=False).logits
                cache = model(inputs, use_cache=True).past_key_values
                self.assertIsInstance(cache, LayerExecutionCache)
                self.assertEqual(len(cache.layers), len(order))
                pointers = {}
                for index, layer in enumerate(cache.layers):
                    state = [
                        getattr(layer, key, None) for key in ("keys", "values", "conv_states", "recurrent_states")
                    ]
                    state.append(cache.layer_states[index])
                    for tensor in tree_flatten(state)[0]:
                        if isinstance(tensor, torch.Tensor) and tensor.numel():
                            address = (str(tensor.device), tensor.data_ptr())
                            self.assertEqual(pointers.get(address, index), index)
                            pointers[address] = index
                torch.testing.assert_close(
                    model.generate(inputs, max_new_tokens=3, do_sample=False),
                    model.generate(inputs, max_new_tokens=3, do_sample=False, use_cache=False),
                    atol=0,
                    rtol=0,
                )
            model.save_pretrained(adapter)
            restored = AutoPeftModelForCausalLM.from_pretrained(adapter, **load_kwargs)
            # Reapply the k-bit preparation used for training, including FP32 non-quantized weights.
            restored = prepare_model_for_kbit_training(restored, use_gradient_checkpointing=False).eval()
            self.assertEqual(restored.get_layer_execution_plan().layer_order, order)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs, use_cache=False).logits, logits, atol=0, rtol=0)

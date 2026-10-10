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

"""CUDA kernels, mixed precision, and compiled execution against independent logical layers."""

import tempfile
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch_gpu
from transformers.utils import is_hqq_available, is_optimum_quanto_available


if is_torch_available():
    import torch

    from transformers import LayerExecutionCache

    from . import test_layer_execution as execution_tests
    from .test_layer_execution_distributed import _source_name


@require_torch_gpu
class LayerExecutionAcceleratorTest(unittest.TestCase):
    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_bfloat16_trainer_checkpoint_resume(self, family):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("The CUDA device does not support bfloat16.")
        from transformers import Trainer, TrainingArguments

        model = execution_tests.LayerExecutionModelTest().make_model(family).cuda()
        order = (0, 1, 0, 1, 2)
        model.set_layer_execution_plan(order)
        # Initialize CUDA linalg before DataParallel starts its replica threads (pytorch/pytorch#90613).
        # Qwen3.5's reference recurrent kernel also triggers this race without an execution plan.
        if family == "qwen3_5" and torch.cuda.device_count() > 1:
            with torch.no_grad():
                model(torch.tensor([[1, 2, 3, 4]], device="cuda"), use_cache=False)
        dataset = [{"input_ids": [1, index + 2, 3, 4], "labels": [1, index + 2, 3, 4]} for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                max_steps=2,
                gradient_checkpointing=True,
                bf16=True,
                report_to="none",
                save_strategy="steps",
                save_steps=1,
                logging_strategy="no",
                disable_tqdm=True,
                optim="adamw_torch",
            )
            trainer = Trainer(model=model, train_dataset=dataset, args=arguments)
            self.assertTrue(torch.isfinite(torch.tensor(trainer.train().training_loss)))
            restored = type(model).from_pretrained(directory + "/checkpoint-1").cuda()
            self.assertEqual(restored.get_layer_execution_plan().layer_order, order)
            resumed = Trainer(model=restored, train_dataset=dataset, args=arguments)
            resumed.train(resume_from_checkpoint=directory + "/checkpoint-1")
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, restored.get_parameter(name), atol=0, rtol=0, msg=name)

    @parameterized.expand([(family, dtype) for family in ("llama", "qwen3_5") for dtype in ("float32", "bfloat16")])
    def test_checkpointed_shared_gradients_and_generation(self, family, dtype):
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            self.skipTest("The CUDA device does not support bfloat16.")
        helper = execution_tests.LayerExecutionModelTest()
        model = helper.make_model(family)
        order = (0, 1, 0, 1, 2)
        reference = helper.expanded_reference(model, order).to(device="cuda", dtype=getattr(torch, dtype)).train()
        model.to(device="cuda", dtype=getattr(torch, dtype)).train()
        model.set_layer_execution_plan(order)
        model.gradient_checkpointing_enable()
        reference.gradient_checkpointing_enable()
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
        actual = model(inputs, labels=inputs, use_cache=False)
        expected = reference(inputs, labels=inputs, use_cache=False)
        tolerances = {"atol": 0.003, "rtol": 0.03} if dtype == "bfloat16" else {"atol": 3e-5, "rtol": 3e-4}
        torch.testing.assert_close(actual.logits, expected.logits, **tolerances)
        torch.testing.assert_close(actual.loss, expected.loss, **tolerances)
        actual.loss.backward()
        expected.loss.backward()
        gradients = {}
        for name, parameter in reference.named_parameters():
            if parameter.grad is not None:
                name = _source_name(name, order)
                gradients[name] = gradients.get(name, 0) + parameter.grad
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients.get(name), msg=name, **tolerances)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model.generate(inputs, max_new_tokens=3), model.generate(inputs, max_new_tokens=3, use_cache=False)
            )

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_inductor_static_decode(self, family):
        model = execution_tests.LayerExecutionModelTest().make_model(family).cuda()
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        cache = LayerExecutionCache(model.config, cache_implementation="static", max_cache_len=16)
        prefix = torch.tensor([[1, 2, 3]], device="cuda")
        self.addCleanup(torch._dynamo.reset)
        with torch.no_grad():
            model(prefix, past_key_values=cache)
            compiled = torch.compile(model, backend="inductor", fullgraph=True)
            for token in (4, 5):
                query = torch.tensor([[token]], device="cuda")
                actual = compiled(query, past_key_values=cache).logits
                prefix = torch.cat((prefix, query), dim=1)
                torch.testing.assert_close(actual, model(prefix, use_cache=False).logits[:, -1:], atol=3e-5, rtol=3e-4)

    @parameterized.expand(
        [(family, backend) for family in ("llama", "qwen3_5") for backend in ("torch", "quanto", "hqq")]
    )
    def test_quantized_cache_decode_reorder_crop_and_reset(self, family, backend):
        if backend == "quanto" and not is_optimum_quanto_available():
            self.skipTest("optimum-quanto is not installed.")
        if backend == "hqq" and not is_hqq_available():
            self.skipTest("hqq is not installed.")
        source = execution_tests.LayerExecutionModelTest().make_model(family)
        source.config.hidden_size = 128
        source.config.intermediate_size = 256
        source.config.head_dim = 64
        model = type(source)(source.config).cuda().eval()
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        options = {"backend": backend, "nbits": 4, "q_group_size": 64, "residual_length": 2}
        cache = LayerExecutionCache(model.config, cache_implementation="quantized", cache_config=options)
        cache.batch_repeat_interleave(2)
        prefix = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        with torch.no_grad():
            initial = model(prefix, past_key_values=cache).logits
            torch.testing.assert_close(initial, model(prefix, use_cache=False).logits)
            query = torch.tensor([[7], [8]], device="cuda")
            actual = model(query, past_key_values=cache).logits
            prefix = torch.cat((prefix, query), dim=1)
            torch.testing.assert_close(actual, model(prefix, use_cache=False).logits[:, -1:], atol=0.04, rtol=0.2)
            cache.reorder_cache(torch.tensor([1, 0, 1], device="cuda"))
            prefix = prefix[[1, 0, 1]]
            query = torch.tensor([[9], [10], [11]], device="cuda")
            actual = model(query, past_key_values=cache).logits
            prefix = torch.cat((prefix, query), dim=1)
            torch.testing.assert_close(actual, model(prefix, use_cache=False).logits[:, -1:], atol=0.04, rtol=0.2)
            # Pure attention histories can be truncated directly; hybrid recurrent histories need snapshots.
            if family == "llama":
                cache.crop(3)
                query = torch.tensor([[12], [13], [14]], device="cuda")
                actual = model(query, past_key_values=cache).logits
                expected = model(torch.cat((prefix[:, :3], query), dim=1), use_cache=False).logits[:, -1:]
                torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.2)
            generated = model.generate(
                prefix, num_beams=2, max_new_tokens=3, cache_implementation="quantized", cache_config=options
            )
            self.assertEqual(generated.shape, (3, 8))
            cache.reset()
            torch.testing.assert_close(
                model(prefix[:, :3], past_key_values=cache).logits, model(prefix[:, :3], use_cache=False).logits
            )

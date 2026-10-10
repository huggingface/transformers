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

"""Real two-rank looped pipeline forward, backward, optimizer, cache, and tied-weight checks."""

import copy
import tempfile
import unittest

from parameterized import parameterized

from transformers.testing_utils import require_torch, require_torch_multi_gpu


def _pipeline_worker(rank, store, family, tied, checkpointing, device_type="cpu", local_rank=None, init_method=None):
    import os
    from unittest.mock import patch

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from transformers import DistributedConfig
    from transformers.distributed.pipeline_parallel import apply_pipeline_parallelism

    from . import test_layer_execution as helpers
    from .test_layer_execution_distributed import _initialize_worker, _source_name

    device = _initialize_worker(rank, store, device_type, local_rank, init_method)
    try:
        helper = helpers.LayerExecutionModelTest()
        model = helper.make_model(family).to(device)
        if tied:
            model.config.tie_word_embeddings = True
            model.tie_weights()
        source = copy.deepcopy(model)
        order = (0, 1, 2, 0, 1, 2)
        reference = helper.expanded_reference(source, order).to(device).train()
        model.set_layer_execution_plan(order)
        apply_pipeline_parallelism(model, init_device_mesh(device_type, (2,)))
        model.train()
        if checkpointing:
            model.gradient_checkpointing_enable()
            reference.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
        for microbatch in (inputs, inputs + 1):
            options = {"output_hidden_states": True, "output_attentions": True, "use_cache": False}
            actual = model(microbatch, labels=microbatch, **options)
            expected = reference(microbatch, labels=microbatch, **options)
            torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual.loss, expected.loss)
            torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual.attentions, expected.attentions, atol=2e-5, rtol=2e-4)

            def objective(output):
                states = output.hidden_states + output.attentions
                return output.loss + sum(value.float().square().mean() for value in states) * 0.05

            (objective(actual) / 2).backward()
            (objective(expected) / 2).backward()
        actual = model(inputs, use_cache=False, output_hidden_states=True)
        expected = reference(inputs, use_cache=False, output_hidden_states=True)
        (actual.hidden_states[1].float().square().mean() * 0.01).backward()
        (expected.hidden_states[1].float().square().mean() * 0.01).backward()
        tuple_output = model(inputs, labels=inputs, use_cache=False, return_dict=False)
        torch.testing.assert_close(tuple_output[0], reference(inputs, labels=inputs).loss)
        gradients = {}
        for name, parameter in reference.named_parameters():
            if parameter.grad is not None:
                key = _source_name(name, order)
                gradients[key] = gradients.get(key, 0) + parameter.grad
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                gradients[name],
                atol=3e-5,
                rtol=3e-4,
                msg=lambda message: f"rank {rank}, {name}: {message}",
            )
        optimizer.step()
        # Every owner has applied the same update as a single-process shared-parameter reference.
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.001)
        for name, parameter in source.named_parameters():
            parameter.grad = gradients[name]
        source_optimizer.step()
        source.set_layer_execution_plan(order)
        source.eval()
        model.eval()
        with torch.no_grad():
            backends = (
                ("dynamic", "static", "offloaded", "offloaded_static")
                if device_type == "cuda"
                else ("dynamic", "static")
            )
            for backend in backends:
                torch.testing.assert_close(
                    model.generate(inputs, max_new_tokens=3, cache_implementation=backend, disable_compile=True),
                    source.generate(inputs, max_new_tokens=3),
                )
            changed = (0, 1, 1, 2)
            model.set_layer_execution_plan(changed)
            source.set_layer_execution_plan(changed)
            torch.testing.assert_close(model(inputs).logits, source(inputs).logits, atol=2e-5, rtol=2e-4)
            selected = {"output_hidden_states": [0, 2]}
            torch.testing.assert_close(
                model(inputs, **selected).hidden_states, source(inputs, **selected).hidden_states
            )
        model.save_pretrained(store + "-model")
        restored = type(source).from_pretrained(store + "-model").to(device).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs).logits, source(inputs).logits, atol=2e-5, rtol=2e-4)
        with (
            patch("torch._C._get_accelerator", return_value=device),
            patch.dict(os.environ, {"LOCAL_RANK": str(device.index) if device_type == "cuda" else str(rank)}),
        ):
            restored_pipeline, info = type(source).from_pretrained(
                store + "-model", distributed_config=DistributedConfig(pp_size=2), output_loading_info=True
            )
        assert not info["missing_keys"] and not info["unexpected_keys"], info
        with torch.no_grad():
            torch.testing.assert_close(restored_pipeline(inputs).logits, source(inputs).logits, atol=2e-5, rtol=2e-4)
    finally:
        dist.destroy_process_group()


def _adapter_pipeline_worker(rank, store, family, device_type="cpu", local_rank=None, init_method=None):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from transformers import LayerExecutionPlan
    from transformers.distributed.pipeline_parallel import apply_pipeline_parallelism

    from . import test_layer_execution_adapters as adapter_tests
    from .test_layer_execution_distributed import _initialize_worker
    from .test_layer_execution_integration import make_stateful_model

    device = _initialize_worker(rank, store, device_type, local_rank, init_method)
    try:
        model = (
            adapter_tests.LayerExecutionAdapterTest().make_model("opt")
            if family == "opt"
            else make_stateful_model(family)
        ).to(device)
        order = tuple(range(model.config.num_hidden_layers)) * 2
        plan = LayerExecutionPlan(order, kv_sharing="native" if family == "gemma3n" else "independent")
        model.set_layer_execution_plan(plan)
        reference = copy.deepcopy(model).train()
        apply_pipeline_parallelism(model, init_device_mesh(device_type, (2,)))
        model.train()
        model.gradient_checkpointing_enable()
        reference.gradient_checkpointing_enable()
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
        options = {"use_cache": False, "output_hidden_states": True, "output_attentions": True}
        actual = model(inputs, labels=inputs, **options)
        expected = reference(inputs, labels=inputs, **options)
        torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual.attentions, expected.attentions, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
        actual.loss.backward()
        expected.loss.backward()
        parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            if parameters[name].grad is not None:
                torch.testing.assert_close(parameter.grad, parameters[name].grad, atol=3e-5, rtol=3e-4, msg=name)
        model.eval()
        reference.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model.generate(inputs, max_new_tokens=3), reference.generate(inputs, max_new_tokens=3)
            )
    finally:
        dist.destroy_process_group()


def _continuous_worker(rank, store, parallelism, device_type="cpu", local_rank=None, init_method=None):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from transformers.distributed.pipeline_parallel import apply_pipeline_parallelism

    from . import test_layer_execution as helpers
    from .test_layer_execution_distributed import _initialize_worker, _shard_tensor_parallel

    device = _initialize_worker(rank, store, device_type, local_rank, init_method)
    try:
        for family in ("llama", "qwen3_5"):
            model = helpers.LayerExecutionModelTest().make_model(family)
            if parallelism == "tp":
                config = copy.deepcopy(model.config)
                config.num_key_value_heads = 2
                model = type(model)(config).eval()
            model = model.to(device)
            model.set_layer_execution_plan((0, 1, 0, 1, 2))
            reference = copy.deepcopy(model)
            prompts = [[1, 2, 3], [4, 5, 6], [7, 8]]
            with torch.no_grad():
                expected = [
                    reference.generate(torch.tensor([prompt], device=device), max_new_tokens=3)[
                        0, len(prompt) :
                    ].tolist()
                    for prompt in prompts
                ]
            if parallelism == "pp":
                apply_pipeline_parallelism(model, init_device_mesh(device_type, (2,)))
            else:
                _shard_tensor_parallel(model, device_type)
            manager = model.init_continuous_batching()
            manager.start()
            if manager.is_tp_driver:
                identifiers = manager.add_requests(prompts, max_new_tokens=3)
                results = {}
                for _ in identifiers:
                    output = manager.get_result(timeout=15)
                    if output is None:
                        raise AssertionError(f"Missing distributed continuous output: {manager._fatal_error}")
                    if output.error:
                        raise AssertionError(output.error)
                    results[output.request_id] = output.generated_tokens
                for identifier, tokens in zip(identifiers, expected):
                    if results[identifier] != tokens:
                        raise AssertionError((parallelism, family, results[identifier], tokens))
                manager.stop(timeout=15)
            else:
                manager._generation_thread.join(timeout=20)
                if manager.is_running():
                    raise AssertionError("Continuous batching worker did not stop with its driver.")
                if manager._fatal_error:
                    raise manager._fatal_error
            manager.destroy()
    finally:
        dist.destroy_process_group()


def _trainer_pipeline_worker(rank, store, device_type="cpu", mixed_precision=False, local_rank=None, init_method=None):
    import os

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from transformers import Trainer, TrainingArguments
    from transformers.distributed.pipeline_parallel import apply_pipeline_parallelism

    from . import test_layer_execution as helpers
    from .test_layer_execution_distributed import _initialize_worker

    if init_method is None:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE="2",
            LOCAL_RANK=str(rank),
            LOCAL_WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )
    device = _initialize_worker(rank, store, device_type, local_rank, init_method)
    try:
        model = helpers.LayerExecutionModelTest().make_model("llama").to(device)
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        reference = copy.deepcopy(model).train()
        apply_pipeline_parallelism(model, init_device_mesh(device_type, (2,)))
        dataset = [{"input_ids": [1, index + 2, 3, 4], "labels": [1, index + 2, 3, 4]} for index in range(8)]
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=store + "-training",
                use_cpu=device_type == "cpu",
                bf16=mixed_precision,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                max_steps=2,
                learning_rate=0.001,
                lr_scheduler_type="constant",
                train_sampling_strategy="sequential",
                optim="adamw_torch",
                gradient_checkpointing=True,
                max_grad_norm=0.01,
                save_steps=1,
                save_strategy="steps",
                report_to=[],
                disable_tqdm=True,
            ),
            train_dataset=dataset,
        )
        if trainer.get_total_train_batch_size(trainer.args) != 2:
            raise AssertionError("PP ranks must count as one logical training replica.")
        batches = list(trainer.get_train_dataloader())
        if len(batches) != len(dataset):
            raise AssertionError("Pipeline data loader dropped examples.")
        trainer.train()
        optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0)
        for start in (0, 2):
            optimizer.zero_grad()
            for entry in dataset[start : start + 2]:
                inputs = torch.tensor([entry["input_ids"]], device=device)
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=mixed_precision):
                    loss = reference(inputs, labels=inputs, use_cache=False).loss / 2
                loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.01)
            optimizer.step()
        parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, parameters[name], atol=1e-5, rtol=5e-4, msg=name)
        uninterrupted = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        prediction = trainer.predict(dataset)
        if prediction.predictions.shape[0] != len(dataset):
            raise AssertionError("Pipeline evaluation duplicated or dropped examples.")
        checkpoint = store + "-training/checkpoint-1"
        trainer.train(resume_from_checkpoint=checkpoint)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, uninterrupted[name], atol=0, rtol=0, msg=name)
    finally:
        dist.destroy_process_group()


@require_torch
class LayerExecutionPipelineTest(unittest.TestCase):
    @parameterized.expand([("llama",), ("qwen3_5",)])
    @require_torch_multi_gpu
    def test_cuda_pipeline_training_generation_and_checkpoint(self, family):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_pipeline_worker, args=(f"{directory}/store", family, True, True, "cuda"), nprocs=2, join=True)

    @parameterized.expand([(False,), (True,)])
    @require_torch_multi_gpu
    def test_cuda_trainer_clipping_checkpoint_and_resume(self, mixed_precision):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _trainer_pipeline_worker,
                args=(f"{directory}/store", "cuda", mixed_precision),
                nprocs=2,
                join=True,
            )

    @parameterized.expand([("pp",), ("tp",)])
    @require_torch_multi_gpu
    def test_cuda_continuous_driver_synchronizes_workers(self, parallelism):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_continuous_worker, args=(f"{directory}/store", parallelism, "cuda"), nprocs=2, join=True)

    @parameterized.expand([("gemma3n",), ("recurrent_gemma",), ("opt",)])
    @require_torch_multi_gpu
    def test_cuda_pipeline_dependencies_streams_and_projected_adapter(self, family):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_adapter_pipeline_worker, args=(f"{directory}/store", family, "cuda"), nprocs=2, join=True)

    def test_trainer_batches_global_clipping_checkpoint_and_resume(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_trainer_pipeline_worker, args=(f"{directory}/store",), nprocs=2, join=True)

    @parameterized.expand([("pp",), ("tp",)])
    def test_continuous_driver_synchronizes_workers(self, parallelism):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_continuous_worker, args=(f"{directory}/store", parallelism), nprocs=2, join=True)

    @parameterized.expand([("gemma3n",), ("recurrent_gemma",), ("opt",)])
    def test_pipeline_dependencies_streams_and_projected_adapter(self, family):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_adapter_pipeline_worker, args=(f"{directory}/store", family), nprocs=2, join=True)

    @parameterized.expand(
        [
            (family, tied, checkpointing)
            for family, tied in (("llama", False), ("llama", True), ("qwen3_5", False))
            for checkpointing in (False, True)
        ]
    )
    def test_pipeline_shared_gradients_and_cached_generation(self, family, tied, checkpointing):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_pipeline_worker, args=(f"{directory}/store", family, tied, checkpointing), nprocs=2, join=True)

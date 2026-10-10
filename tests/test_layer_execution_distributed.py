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

import copy
import math
import os
import pickle
import re
import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_multi_gpu
from transformers.utils import is_torch_greater_or_equal


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel

    from transformers import LayerExecutionCache, get_layer_execution_plan, set_layer_execution_plan

    from . import test_layer_execution as execution_tests
    from . import test_layer_execution_adapters as adapter_tests


_MODES = (
    "ddp",
    "ddp_checkpoint",
    "ddp_reentrant_static",
    "ddp_unused",
    "ddp_change_plan",
    "ddp_pickled",
    "ddp_adapter_checkpoint",
    "trainer_checkpoint",
    "trainer_unused_checkpoint",
    "fsdp",
    "fsdp_checkpoint",
    "fsdp_adapter_checkpoint",
    "tp",
    "tp_checkpoint",
)


def _source_name(name, order):
    return re.sub(r"\.layers\.(\d+)\.", lambda match: f".layers.{order[int(match[1])]}.", name)


def _shard_tensor_parallel(model, device_type):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, distribute_tensor

    from transformers.distributed.tensor_parallel import apply_tensor_parallelism

    state = {name: value.clone() for name, value in model.state_dict().items()}
    mesh = init_device_mesh(device_type, (2,))
    apply_tensor_parallelism(model, mesh)
    # The loading path normally fills the DTensor placeholders from a checkpoint. Use the same placements locally.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = state[name]
            if isinstance(parameter, DTensor):
                value = distribute_tensor(value, mesh, parameter.placements)
            parameter.copy_(value)


def _full_tensor(tensor):
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def _initialize_worker(rank, store, device_type, local_rank=None, init_method=None):
    """Support both local spawn and independent torchrun agents with distinct local/global ranks."""
    torch.set_num_threads(1)
    device_index = rank if local_rank is None else local_rank
    device = torch.device(device_type, device_index) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=init_method or f"file://{store}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    return device


def _train_with_trainer(model, rank, store, mode, device_type, supports_resume=True, external_launch=False):
    from transformers import Trainer, TrainingArguments

    if not external_launch:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE="2",
            LOCAL_RANK=str(rank),
            LOCAL_WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )
    order = (0, 0, 2) if "unused" in mode else (0, 1, 0, 1, 2)
    model.set_layer_execution_plan(order)
    original = {name: _full_tensor(value).clone() for name, value in model.state_dict().items()}
    dataset = [{"input_ids": [1, index + 2, 3, 4], "labels": [1, index + 2, 3, 4]} for index in range(8)]
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=f"{store}-{model.config.model_type}",
            use_cpu=device_type == "cpu",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=2,
            gradient_checkpointing=True,
            report_to="none",
            save_strategy="steps",
            save_steps=1,
            save_only_model=not supports_resume,
            logging_strategy="no",
            disable_tqdm=True,
            optim="adamw_torch",
        ),
    )
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise AssertionError("Trainer produced a non-finite training loss.")
    if model.get_layer_execution_plan().layer_order != order:
        raise AssertionError("Trainer changed the layer execution plan.")
    updated = False
    for name, parameter in model.named_parameters():
        value = _full_tensor(parameter.detach())
        copies = [torch.empty_like(value) for _ in range(2)]
        dist.all_gather(copies, value)
        torch.testing.assert_close(copies[0], copies[1], atol=0, rtol=0)
        if "unused" in mode and ".layers.1." in name:
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        updated |= not torch.equal(value, original[name])
    if not updated:
        raise AssertionError("Trainer did not update the shared parameters.")
    uninterrupted = {name: _full_tensor(parameter.detach()).clone() for name, parameter in model.named_parameters()}
    if supports_resume:
        trainer.train(resume_from_checkpoint=f"{store}-{model.config.model_type}/checkpoint-1")
        restored = model
    else:
        # Native load-time TP/FSDP2 currently supports model checkpoints, but explicitly rejects optimizer resume.
        restored = type(model).from_pretrained(
            f"{store}-{model.config.model_type}/checkpoint-2",
            distributed_config=copy.deepcopy(model.config.distributed_config),
        )
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(_full_tensor(parameter), uninterrupted[name], atol=0, rtol=0, msg=name)
    if restored.get_layer_execution_plan().layer_order != order:
        raise AssertionError("Restoring the model changed the execution plan.")
    model.eval()
    inputs = torch.tensor([[1, 2, 3]], device=model.device)
    with torch.no_grad():
        torch.testing.assert_close(
            model.generate(inputs, max_new_tokens=2, do_sample=False),
            model.generate(inputs, use_cache=False, max_new_tokens=2, do_sample=False),
        )


def _distributed_worker(rank, store, mode, device_type, serialized_models, local_rank=None, init_method=None):
    device = _initialize_worker(rank, store, device_type, local_rank, init_method)
    try:
        helper = execution_tests.LayerExecutionModelTest()
        factory = adapter_tests.LayerExecutionAdapterTest() if "adapter" in mode else helper
        families = ("opt", "mamba") if "adapter" in mode else ("llama", "qwen3_5")
        for family_index, family in enumerate(families):
            model = pickle.loads(serialized_models[family_index]) if serialized_models else factory.make_model(family)
            if mode.startswith("tp"):
                config = copy.deepcopy(model.config)
                config.num_key_value_heads = 2  # Each TP rank needs at least one complete KV head.
                model = type(model)(config)
            model = model.to(device).train()
            if mode.startswith("trainer"):
                _train_with_trainer(model, rank, store, mode, device_type, external_launch=init_method == "env://")
                continue
            source_reference = copy.deepcopy(model)
            order = (0, 0, 2) if mode == "ddp_unused" else (0, 1, 0, 1, 2)
            if mode.startswith("ddp"):
                model.set_layer_execution_plan(order)
                if "checkpoint" in mode:
                    model.gradient_checkpointing_enable()
                if "reentrant" in mode:
                    model.gradient_checkpointing_enable({"use_reentrant": True})
                wrapped = DistributedDataParallel(
                    model,
                    device_ids=[device.index] if device_type == "cuda" else None,
                    find_unused_parameters=mode == "ddp_unused",
                    static_graph="static" in mode,
                )
                if set_layer_execution_plan(wrapped, order) is not wrapped:
                    raise AssertionError("The configuration API must retain the distributed wrapper.")
                if get_layer_execution_plan(wrapped).layer_order != order:
                    raise AssertionError("The configuration API must resolve the wrapped text decoder.")
            elif mode.startswith("fsdp"):
                from torch.distributed.device_mesh import init_device_mesh
                from torch.distributed.fsdp import fully_shard

                mesh = init_device_mesh(device_type, (2,))
                for layer in model.get_decoder().layers:
                    fully_shard(layer, mesh=mesh, reshard_after_forward=True)
                fully_shard(model, mesh=mesh)
                model.set_layer_execution_plan(order)
                wrapped = model
            else:
                _shard_tensor_parallel(model, device_type)
                model.set_layer_execution_plan(order)
                wrapped = model
            if "checkpoint" in mode and not mode.startswith("ddp"):
                model.gradient_checkpointing_enable()

            # An epsilon above tiny random-model gradients avoids amplifying reduction-order rounding into updates.
            # TP has both ordinary tensors and DTensors; they cannot share a CUDA foreach kernel.
            optimizer = torch.optim.AdamW(wrapped.parameters(), lr=0.003, eps=1e-6, foreach=False)
            reference_optimizer = torch.optim.AdamW(source_reference.parameters(), lr=0.003, eps=1e-6, foreach=False)
            for step in range(2):
                if mode == "ddp_change_plan" and step == 1:
                    order = (0, 1, 2, 1, 2)
                    set_layer_execution_plan(wrapped, order)
                # Independent logical modules are an oracle for both execution order and the sum of shared gradients.
                reference = helper.expanded_reference(source_reference, order).to(device).train()
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                microbatches = 2 if mode.startswith("ddp") and "static" not in mode else 1
                for microbatch in range(microbatches):
                    inputs = torch.tensor(
                        [[1, 2 + step, 3 + microbatch, 4], [5, 6 + microbatch, 7 + step, 8]], device=device
                    )
                    local_inputs = inputs if mode.startswith("tp") else inputs[rank : rank + 1]
                    context = wrapped.no_sync() if microbatch < microbatches - 1 else nullcontext()
                    with context:
                        (wrapped(local_inputs, labels=local_inputs, use_cache=False).loss / microbatches).backward()
                    (reference(inputs, labels=inputs, use_cache=False).loss / microbatches).backward()

                gradients = {}
                for name, parameter in reference.named_parameters():
                    if parameter.grad is not None:
                        source_name = _source_name(name, order)
                        gradients[source_name] = gradients.get(source_name, 0) + parameter.grad
                for name, parameter in model.named_parameters():
                    expected = gradients.get(name)
                    if expected is None:
                        if parameter.grad is not None:
                            raise AssertionError(f"An omitted source layer received a gradient: {name}")
                    else:
                        torch.testing.assert_close(
                            _full_tensor(parameter.grad),
                            expected,
                            atol=3e-5,
                            rtol=3e-4,
                            msg=lambda detail: f"{family}, step {step}, {name}: {detail}",
                        )
                        source_reference.get_parameter(name).grad = expected
                optimizer.step()
                reference_optimizer.step()
                for name, parameter in model.named_parameters():
                    # TP changes reduction order; AdamW can amplify rounding on tiny random-model gradients.
                    torch.testing.assert_close(
                        _full_tensor(parameter),
                        source_reference.get_parameter(name),
                        atol=1e-5,
                        rtol=5e-4,
                        msg=lambda detail: f"{family}, step {step}, {name}: {detail}",
                    )

            model.eval()
            # Optimizer equivalence was checked above. Cache comparisons need exactly the same updated weights.
            source_reference.load_state_dict({name: _full_tensor(value) for name, value in model.state_dict().items()})
            reference = helper.expanded_reference(source_reference, order).to(device).eval()
            with torch.no_grad():
                next_inputs = inputs[:, :1]
                cache_name = model.get_decoder()._layer_execution_adapter.cache_name
                expected = reference(torch.cat([inputs, next_inputs], dim=1), use_cache=False)
                prefill = reference(inputs, use_cache=False).logits
                generated = reference.generate(inputs, use_cache=False, max_new_tokens=2, do_sample=False)
                backends = (
                    ("dynamic", "static", "offloaded", "offloaded_static") if device_type == "cuda" else ("dynamic",)
                )
                for backend in backends:
                    cache = LayerExecutionCache(model.config, cache_implementation=backend, max_cache_len=8)
                    actual = model(inputs, **{cache_name: cache}, use_cache=True)
                    torch.testing.assert_close(actual.logits, prefill, atol=1e-6, rtol=1e-5)
                    actual = model(next_inputs, **{cache_name: getattr(actual, cache_name)}, use_cache=True)
                    torch.testing.assert_close(actual.logits, expected.logits[:, -1:], atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(
                        model.generate(
                            inputs,
                            max_new_tokens=2,
                            do_sample=False,
                            cache_implementation=backend,
                            disable_compile=True,
                        ),
                        generated,
                    )
    finally:
        dist.destroy_process_group()


def _run_distributed(mode, device_type):
    serialized_models = None
    if mode == "ddp_pickled":
        helper = execution_tests.LayerExecutionModelTest()
        serialized_models = tuple(
            pickle.dumps(helper.make_model(family).set_layer_execution_plan([0, 1, 0, 1, 2]))
            for family in ("llama", "qwen3_5")
        )
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _distributed_worker,
            args=(os.path.join(directory, "store"), mode, device_type, serialized_models),
            nprocs=2,
            join=True,
        )


def _native_trainer_worker(rank, store, parallelism, local_rank=None, init_method=None):
    from transformers import DistributedConfig

    if init_method is None:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE="2",
            LOCAL_RANK=str(rank),
            LOCAL_WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )
    _initialize_worker(rank, store, "cuda", local_rank, init_method)
    try:
        for family in ("llama", "qwen3_5"):
            source = execution_tests.LayerExecutionModelTest().make_model(family)
            if parallelism == "tp":
                config = copy.deepcopy(source.config)
                config.num_key_value_heads = 2
                source = type(source)(config)
            source.set_layer_execution_plan((0, 1, 0, 1, 2))
            checkpoint = f"{store}-{family}-source"
            if rank == 0:
                source.save_pretrained(checkpoint)
            dist.barrier()
            model, info = type(source).from_pretrained(
                checkpoint,
                distributed_config=DistributedConfig(**{parallelism + "_size": 2}),
                output_loading_info=True,
            )
            if info["missing_keys"] or info["unexpected_keys"]:
                raise AssertionError(info)
            _train_with_trainer(
                model,
                rank,
                store,
                "trainer_checkpoint",
                "cuda",
                supports_resume=False,
                external_launch=init_method == "env://",
            )
    finally:
        dist.destroy_process_group()


@require_torch
@unittest.skipUnless(is_torch_available() and torch.distributed.is_gloo_available(), "Requires Gloo")
class LayerExecutionDistributedCPUTest(unittest.TestCase):
    @parameterized.expand([(mode,) for mode in _MODES])
    def test_training_and_cached_inference(self, mode):
        if mode.startswith("fsdp") and not is_torch_greater_or_equal("2.14"):
            self.skipTest("CPU FSDP2 validation requires PyTorch 2.14")
        _run_distributed(mode, "cpu")


@require_torch_multi_gpu
class LayerExecutionDistributedCUDATest(unittest.TestCase):
    @parameterized.expand([(mode,) for mode in _MODES])
    def test_training_and_cached_inference(self, mode):
        _run_distributed(mode, "cuda")

    @parameterized.expand([("tp",), ("fsdp",)])
    def test_native_loading_trainer_model_checkpoint(self, parallelism):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_native_trainer_worker, args=(f"{directory}/store", parallelism), nprocs=2, join=True)


@require_torch_multi_gpu
class LayerExecutionDataParallelCUDATest(unittest.TestCase):
    @parameterized.expand(
        [(family, checkpointing) for family in ("llama", "qwen3_5") for checkpointing in (False, True)]
    )
    def test_replica_binding_and_shared_gradients(self, family, checkpointing):
        helper = execution_tests.LayerExecutionModelTest()
        model = helper.make_model(family).to("cuda:0").train()
        order = (0, 1, 0, 1, 2)
        reference = helper.expanded_reference(model, order).to("cuda:0").train()
        parameters = tuple(id(parameter) for parameter in model.parameters())
        keys = tuple(model.state_dict())
        wrapped = torch.nn.DataParallel(model, device_ids=[0, 1])
        if set_layer_execution_plan(wrapped, order) is not wrapped:
            raise AssertionError("The plan API must retain the DataParallel wrapper.")
        if checkpointing:
            model.gradient_checkpointing_enable()
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 4, 5], [6, 7, 8, 9]], device="cuda:0")
        actual = wrapped(inputs, labels=inputs, use_cache=False)
        expected = reference(inputs, labels=inputs, use_cache=False)
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(actual.loss.mean(), expected.loss, atol=1e-6, rtol=1e-5)
        actual.loss.mean().backward()
        expected.loss.backward()
        gradients = {}
        for name, parameter in reference.named_parameters():
            if parameter.grad is not None:
                source_name = _source_name(name, order)
                gradients[source_name] = gradients.get(source_name, 0) + parameter.grad
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-5, rtol=3e-4, msg=name)
        if tuple(id(parameter) for parameter in model.parameters()) != parameters or tuple(model.state_dict()) != keys:
            raise AssertionError("DataParallel changed the original source parameters or checkpoint keys.")

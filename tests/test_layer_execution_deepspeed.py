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

"""ZeRO keeps one parameter set while repeated module calls accumulate shared gradients."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_deepspeed, require_peft, require_torch, require_torch_multi_gpu


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.utils._pytree import tree_flatten

    from transformers import (
        AutoModelForCausalLM,
        LayerExecutionCache,
        PreTrainedConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        get_layer_execution_plan,
        set_layer_execution_plan,
    )
    from transformers.layer_execution.deepspeed import configure_deepspeed_layer_execution

    from . import test_layer_execution as execution_tests
    from .test_layer_execution_distributed import _initialize_worker, _source_name


def _zero_worker(rank, store, stage, checkpointing):
    import deepspeed
    from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
    device = _initialize_worker(rank, store, "cuda")
    try:
        for family in ("llama", "qwen3_5"):
            helper = execution_tests.LayerExecutionModelTest()
            model = helper.make_model(family).to(device).train()
            order = (0, 1, 0, 1, 2)
            reference = helper.expanded_reference(model, order).to(device).train()
            updated_reference = helper.make_model(family).to(device).train()
            updated_reference.load_state_dict(model.state_dict())
            parameters = tuple(id(parameter) for parameter in model.parameters())
            keys = tuple(model.state_dict())
            model.set_layer_execution_plan(order)
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            config = {
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": 0,
                "zero_optimization": {
                    "stage": stage,
                    "overlap_comm": False,
                    "reduce_bucket_size": 1024,
                    "allgather_bucket_size": 1024,
                    "stage3_param_persistence_threshold": 0,
                },
            }
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.AdamW(updated_reference.parameters(), lr=0.01)
            engine, _, _, _ = deepspeed.initialize(
                model=model, optimizer=optimizer, config=config, dist_init_required=False
            )
            configure_deepspeed_layer_execution(engine)
            inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
            expected = reference(inputs, labels=inputs, use_cache=False)
            expected.loss.backward()
            gradients = {}
            for name, parameter in reference.named_parameters():
                if parameter.grad is not None:
                    source_name = _source_name(name, order)
                    gradients[source_name] = gradients.get(source_name, 0) + parameter.grad
            actual = engine(inputs[rank : rank + 1], labels=inputs[rank : rank + 1], use_cache=False)
            engine.backward(actual.loss)
            for name, parameter in model.named_parameters():
                gradient = safe_get_full_grad(parameter)
                if gradient is None or not torch.isfinite(gradient).all():
                    raise AssertionError(f"Missing or non-finite ZeRO gradient: {name}")
                torch.testing.assert_close(gradient, gradients[name], atol=3e-5, rtol=3e-4, msg=name)
            for name, parameter in updated_reference.named_parameters():
                parameter.grad = gradients[name].clone()
            reference_optimizer.step()
            engine.step()
            updated_parameters = dict(updated_reference.named_parameters())
            for name, parameter in model.named_parameters():
                full = safe_get_full_fp32_param(parameter)
                torch.testing.assert_close(full, updated_parameters[name], atol=3e-5, rtol=3e-4, msg=name)
            if (
                tuple(id(parameter) for parameter in model.parameters()) != parameters
                or tuple(model.state_dict()) != keys
            ):
                raise AssertionError("ZeRO changed the registered source parameters or checkpoint keys.")

            new_order = (0, 1, 2, 0, 1, 2)
            if set_layer_execution_plan(engine, new_order) is not engine:
                raise AssertionError("Setting the plan must preserve the DeepSpeed engine.")
            if get_layer_execution_plan(engine).layer_order != new_order:
                raise AssertionError("The DeepSpeed wrapper lost the execution plan.")
            updated_reference.set_layer_execution_plan(new_order)
            model.eval()
            updated_reference.eval()
            prompt = inputs[:1, :3]
            with torch.no_grad():
                output = model(prompt, use_cache=True)
                expected = updated_reference(prompt, use_cache=False).logits
                torch.testing.assert_close(output.logits, expected, atol=2e-5, rtol=2e-4)
                if not isinstance(output.past_key_values, LayerExecutionCache):
                    raise AssertionError("ZeRO did not return an execution cache.")
                cache = output.past_key_values
                if len(cache.layers) != len(new_order):
                    raise AssertionError("Cache state count differs from logical execution count.")
                pointers = {}
                for index, layer in enumerate(cache.layers):
                    state = [
                        getattr(layer, key, None) for key in ("keys", "values", "conv_states", "recurrent_states")
                    ]
                    state.append(cache.layer_states[index])
                    for tensor in tree_flatten(state)[0]:
                        if isinstance(tensor, torch.Tensor) and tensor.numel():
                            address = (str(tensor.device), tensor.data_ptr())
                            if pointers.get(address, index) != index:
                                raise AssertionError("ZeRO logical cache buffers alias.")
                            pointers[address] = index
                cached = model.generate(prompt, max_new_tokens=3, do_sample=False, synced_gpus=True)
                uncached = model.generate(prompt, max_new_tokens=3, do_sample=False, use_cache=False, synced_gpus=True)
                torch.testing.assert_close(cached, uncached, atol=0, rtol=0)
            checkpoint = f"{store}-{family}-checkpoint"
            with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None):
                if rank == 0:
                    model.save_pretrained(checkpoint)
                dist.barrier()
            restored = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
            if restored.get_layer_execution_plan().layer_order != new_order:
                raise AssertionError("Saving the ZeRO model lost its execution plan.")
            with torch.no_grad():
                torch.testing.assert_close(restored(prompt, use_cache=False).logits, output.logits, atol=0, rtol=0)
            set_layer_execution_plan(engine, None)
            if get_layer_execution_plan(engine) is not None or hasattr(
                model, "_layer_execution_deepspeed_coordinator"
            ):
                raise AssertionError("Disabling the plan did not remove the shared-gradient integration.")
            engine.destroy()
    finally:
        dist.destroy_process_group()


def _shared_adapter_worker(rank, store, stage, checkpointing, overlap):
    import deepspeed
    from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad
    from peft import LoraConfig, get_peft_model

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
    device = _initialize_worker(rank, store, "cuda")
    try:
        for family in ("llama", "qwen3_5"):
            native = execution_tests.LayerExecutionModelTest().make_model(family).to(device).train()
            config = LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["down_proj"])
            model = get_peft_model(native, config)
            model.add_adapter("reference", config)
            model.set_adapter("default")
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(std=0.01)
            expected_source = copy.deepcopy(model)
            expected_optimizer = torch.optim.AdamW(
                [parameter for parameter in expected_source.parameters() if parameter.requires_grad], lr=0.01
            )
            order = (0, 1, 0, 1, 2)
            model.set_layer_execution_plan(order)
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            source_ids = tuple(id(parameter) for parameter in model.parameters())
            keys = tuple(model.state_dict())
            engine, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=torch.optim.AdamW(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
                ),
                config={
                    "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 2,
                    "gradient_clipping": 0,
                    "zero_optimization": {
                        "stage": stage,
                        "overlap_comm": overlap,
                        "reduce_bucket_size": 100_000,
                        "allgather_bucket_size": 100_000,
                        "stage3_param_persistence_threshold": 0,
                    },
                },
                dist_init_required=False,
            )
            if configure_deepspeed_layer_execution(engine) is not engine:
                raise AssertionError("Configuring shared gradients must preserve the engine.")
            # A second configuration must not install duplicate hooks.
            configure_deepspeed_layer_execution(engine)
            for update in range(2):
                expanded_config = copy.deepcopy(expected_source.config)
                expanded_config.num_hidden_layers = len(order)
                if getattr(expanded_config, "layer_types", None) is not None:
                    expanded_config.layer_types = [expected_source.config.layer_types[index] for index in order]
                reference = get_peft_model(type(native)(expanded_config).to(device).train(), config)
                reference.add_adapter("reference", config)
                reference.set_adapter("default")
                source_state = expected_source.state_dict()
                reference.load_state_dict(
                    {name: source_state[_source_name(name, order)] for name in reference.state_dict()}
                )
                inputs = torch.tensor([[1, 2 + update, 3, 4], [5, 6, 7, 8], [2, 3, 4, 5], [6, 7, 8, 9]], device=device)
                reference(inputs, labels=inputs, use_cache=False).loss.backward()
                gradients = {}
                for name, parameter in reference.named_parameters():
                    if parameter.grad is not None:
                        name = _source_name(name, order)
                        gradients[name] = gradients.get(name, 0) + parameter.grad
                for micro_step in range(2):
                    batch = inputs[rank * 2 + micro_step : rank * 2 + micro_step + 1]
                    actual = engine(batch, labels=batch, use_cache=False)
                    # DPO switches adapters between policy forward and backward, under no_grad.
                    with torch.no_grad():
                        model.set_adapter("reference")
                        model(batch, use_cache=False)
                        model.set_adapter("default")
                    engine.backward(actual.loss)
                    if micro_step == 1:
                        for name, parameter in model.named_parameters():
                            if parameter.requires_grad:
                                torch.testing.assert_close(
                                    safe_get_full_grad(parameter), gradients[name], atol=3e-5, rtol=3e-4, msg=name
                                )
                    engine.step()
                    if any(hasattr(parameter, "ds_grad_is_ready") for parameter in model.parameters()):
                        raise AssertionError("The shared-gradient integration leaked temporary parameter flags.")
                expected_optimizer.zero_grad()
                for name, parameter in expected_source.named_parameters():
                    if parameter.requires_grad:
                        parameter.grad = gradients[name].clone()
                expected_optimizer.step()
                expected_parameters = dict(expected_source.named_parameters())
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(
                            safe_get_full_fp32_param(parameter),
                            expected_parameters[name],
                            atol=3e-5,
                            rtol=3e-4,
                            msg=name,
                        )
                if (
                    tuple(id(parameter) for parameter in model.parameters()) != source_ids
                    or tuple(model.state_dict()) != keys
                ):
                    raise AssertionError("Shared-gradient integration changed parameters or checkpoint keys.")
            engine.destroy()
    finally:
        dist.destroy_process_group()


def _peft_checkpoint_worker(rank, directory, stage, phase, loop):
    import deepspeed
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    )
    device = _initialize_worker(rank, f"{directory}/{phase}-store", "cuda")
    try:
        base_path = f"{directory}/base"
        if phase == "first":
            native = execution_tests.LayerExecutionModelTest().make_model("llama")
            if loop:
                native.set_layer_execution_plan((0, 1, 1, 2))
            if rank == 0:
                native.save_pretrained(base_path)
            dist.barrier()
        native = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16).to(device)
        config = LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["down_proj"])
        model = get_peft_model(native, config)
        model.add_adapter("reference", config)
        model.set_adapter("default")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".reference." in name:
                    parameter.fill_(0.25 if phase == "first" else -0.5)
        source_ids = tuple(id(parameter) for parameter in model.parameters())
        plan = get_layer_execution_plan(model)

        def adapter_values():
            with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None):
                values = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                    if "lora_" in name
                }
                for name, parameter in model.named_parameters():
                    if ".reference." in name:
                        if parameter.requires_grad or parameter.grad is not None:
                            raise AssertionError("The reference adapter must remain frozen.")
            return values

        class RestoreAudit(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                self.initial = adapter_values()
                if phase == "resume":
                    expected = torch.load(f"{directory}/first-adapters-{rank}.pt", weights_only=True)
                    torch.testing.assert_close(self.initial, expected, atol=0, rtol=0)

        audit = RestoreAudit()
        trainer = Trainer(
            model=model,
            train_dataset=[{"input_ids": [1, index + 2, 3, 4], "labels": [1, index + 2, 3, 4]} for index in range(8)],
            callbacks=[audit],
            args=TrainingArguments(
                output_dir=f"{directory}/training",
                max_steps=1 if phase == "first" else 2,
                per_device_train_batch_size=2,
                bf16=True,
                learning_rate=0.01,
                lr_scheduler_type="constant",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                save_strategy="steps" if phase == "first" else "no",
                save_steps=1,
                report_to="none",
                disable_tqdm=True,
                deepspeed={
                    "train_batch_size": "auto",
                    "train_micro_batch_size_per_gpu": "auto",
                    "gradient_accumulation_steps": "auto",
                    "gradient_clipping": "auto",
                    "bf16": {"enabled": True},
                    "zero_optimization": {
                        "stage": stage,
                        "overlap_comm": False,
                        "reduce_bucket_size": 1024,
                        "allgather_bucket_size": 1024,
                        "stage3_param_persistence_threshold": 0,
                        "stage3_gather_16bit_weights_on_model_save": True,
                    },
                },
            ),
        )
        checkpoint = f"{directory}/training/checkpoint-1"
        trainer.train(resume_from_checkpoint=checkpoint if phase == "resume" else None)
        values = adapter_values()
        for name in values:
            if ".reference." in name:
                torch.testing.assert_close(values[name], audit.initial[name], atol=0, rtol=0)
        if not any(
            not torch.equal(value, audit.initial[name]) for name, value in values.items() if ".default." in name
        ):
            raise AssertionError("The policy adapter did not update.")
        if phase == "first":
            torch.save(values, f"{directory}/first-adapters-{rank}.pt")
            exported_reference = load_file(f"{checkpoint}/reference/adapter_model.safetensors")
            if not exported_reference or not all(tensor.numel() for tensor in exported_reference.values()):
                raise AssertionError("The exported frozen reference adapter is empty.")
        if tuple(id(parameter) for parameter in model.parameters()) != source_ids:
            raise AssertionError("Saving or restoring the checkpoint replaced source parameters.")
        if get_layer_execution_plan(model) != plan:
            raise AssertionError("Saving or restoring the checkpoint changed the execution plan.")
        trainer.model_wrapped.destroy()
    finally:
        dist.destroy_process_group()


@require_torch
class LayerExecutionDeepSpeedConfigurationTest(unittest.TestCase):
    def test_nested_decoder_plan_does_not_require_parent_config_alias(self):
        class Engine(torch.nn.DataParallel):
            def __init__(self, model):
                super().__init__(model)
                self.optimizer = type("Optimizer", (), {"reduce_ready_partitions_and_remove_grads": lambda *_: None})()

            def zero_optimization_stage(self):
                return 3

        native = execution_tests.LayerExecutionModelTest().make_model("llama")
        model = execution_tests._NestedCausalLM(native.config)
        model.config = PreTrainedConfig()
        engine = Engine(model)
        set_layer_execution_plan(engine, (0, 1, 1, 2))
        self.assertIsNone(model.config._get_layer_execution_config())
        coordinator = model._layer_execution_deepspeed_coordinator
        self.assertIs(configure_deepspeed_layer_execution(engine), engine)
        self.assertIs(model._layer_execution_deepspeed_coordinator, coordinator)
        set_layer_execution_plan(engine, None)
        self.assertFalse(hasattr(model, "_layer_execution_deepspeed_coordinator"))

    def test_native_models_do_not_resolve_decoder_or_optimizer(self):
        class Engine(torch.nn.DataParallel):
            def __init__(self, model, stage):
                super().__init__(model)
                self.stage = stage

            def zero_optimization_stage(self):
                return self.stage

            @property
            def optimizer(self):
                raise AssertionError("Native execution must not require the shared-gradient API.")

        for stage in (2, 3):
            model = execution_tests.LayerExecutionModelTest().make_model("llama")
            engine = Engine(model, stage)
            hooks = (dict(model._forward_pre_hooks), dict(model._forward_hooks))
            with patch.object(model, "get_decoder", side_effect=AssertionError("Native decoder must stay untouched.")):
                self.assertIs(configure_deepspeed_layer_execution(engine), engine)
            self.assertEqual((dict(model._forward_pre_hooks), dict(model._forward_hooks)), hooks)
            self.assertFalse(hasattr(model, "_layer_execution_deepspeed_coordinator"))
            plain_engine = Engine(torch.nn.Linear(2, 2), stage)
            self.assertIs(configure_deepspeed_layer_execution(plain_engine), plain_engine)


@require_torch_multi_gpu
@require_deepspeed
class LayerExecutionDeepSpeedTest(unittest.TestCase):
    @parameterized.expand([(stage, checkpoint) for stage in (2, 3) for checkpoint in (False, True)])
    def test_shared_gradients_updates_cache_and_save(self, stage, checkpoint):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_zero_worker, args=(f"{directory}/store", stage, checkpoint), nprocs=2, join=True)

    @parameterized.expand(
        [(stage, checkpoint, False) for stage in (2, 3) for checkpoint in (False, True)]
        + [(3, checkpoint, True) for checkpoint in (False, True)]
    )
    @require_peft
    def test_shared_adapter_gradients_accumulation_and_updates(self, stage, checkpoint, overlap):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _shared_adapter_worker, args=(f"{directory}/store", stage, checkpoint, overlap), nprocs=2, join=True
            )

    @parameterized.expand([(stage, loop) for stage in (2, 3) for loop in (False, True)])
    @require_peft
    def test_frozen_reference_adapter_survives_cold_trainer_resume(self, stage, loop):
        with tempfile.TemporaryDirectory() as directory:
            for phase in ("first", "resume"):
                mp.spawn(_peft_checkpoint_worker, args=(directory, stage, phase, loop), nprocs=2, join=True)

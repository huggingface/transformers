# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""Offline SST train/evaluate/save/resume/generate checks for distributed Trainer."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import SST, Adafactor, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments


class Tokens(Dataset):
    def __init__(self):
        self.tokens = torch.randint(0, 64, (32, 16), generator=torch.Generator().manual_seed(17))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        return {"input_ids": tokens, "labels": tokens.clone()}


def assert_same_state(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same_state(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_same_state(left, right)
    else:
        assert first == second


def inner_optimizer(optimizer):
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    return optimizer


def optimizer_state(optimizer):
    # ZeRO-3 temporarily empties the base optimizer's groups between updates, so its state_dict() cannot map keys.
    return {
        id(parameter) if isinstance(parameter, torch.Tensor) else parameter: copy.deepcopy(state)
        for parameter, state in optimizer.state.items()
    }


class ObservedSST(SST):
    def __init__(self, *args, **kwargs):
        self.projections = 0
        self.maximum_leakage = 0.0
        self.maximum_replica_difference = 0.0
        self.maximum_movement = 0.0
        self.initial_weights = {}
        super().__init__(*args, **kwargs)
        if self.is_initialized:
            for name, parameter in self._parameters():
                if name in self._basis:
                    self.initial_weights[name] = self._read_matrix(parameter).detach().cpu().clone()

    def project(self):
        base = inner_optimizer(self.optimizer)
        before = optimizer_state(base)
        super().project()
        assert_same_state(before, optimizer_state(base))
        self.projections += 1
        for name, parameter in self._parameters():
            if name not in self._basis:
                continue
            basis = self._basis[name]
            matrix = self._read_matrix(parameter).detach().float()
            left, right = basis.left.to(matrix.device), basis.right.to(matrix.device)
            delta = matrix - (left * basis.singular_values.to(matrix.device).unsqueeze(0)) @ right.mH
            leakage = max((left.mH @ delta).abs().max().item(), (delta @ right).abs().max().item())
            self.maximum_leakage = max(self.maximum_leakage, leakage)
            if name in self.initial_weights:
                movement = (matrix.cpu() - self.initial_weights[name]).abs().max().item()
                self.maximum_movement = max(self.maximum_movement, movement)
            if dist.is_initialized():
                replicas = [torch.empty_like(matrix) for _ in range(dist.get_world_size())]
                dist.all_gather(replicas, matrix)
                self.maximum_replica_difference = max(
                    self.maximum_replica_difference,
                    max((matrix - replica).abs().max().item() for replica in replicas),
                )


def arguments(options, output_dir):
    extra = {}
    if options.dp_replicate_size > 1:
        from accelerate import ParallelismConfig

        world_size = int(os.environ["WORLD_SIZE"])
        if options.backend != "fsdp2" or world_size % options.dp_replicate_size:
            raise ValueError("Hybrid replication requires FSDP2 and a divisible world size.")
        extra["parallelism_config"] = ParallelismConfig(
            dp_replicate_size=options.dp_replicate_size,
            dp_shard_size=world_size // options.dp_replicate_size,
        )
    if options.backend.startswith("fsdp"):
        version = int(options.backend[-1])
        extra["fsdp"] = True
        extra["fsdp_config"] = {
            "version": version,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "use_orig_params": not options.flat_parameters,
            "reshard_after_forward": options.sharding if version == 1 else options.sharding == "full_shard",
            "state_dict_type": "SHARDED_STATE_DICT",
            "cpu_ram_efficient_loading": False,
            "cpu_offload": options.offload,
        }
    elif options.backend.startswith("zero"):
        stage = int(options.backend[-1])
        extra["deepspeed"] = {
            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "bf16": {"enabled": options.precision == "bf16"},
            "fp16": {"enabled": options.precision == "fp16", "initial_scale_power": 8},
            "zero_optimization": {
                "stage": stage,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "zero_allow_untested_optimizer": True,
        }
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=4,
        optim="sgd" if options.optim == "sgd" else "adamw_torch",
        learning_rate=0.02 if options.optim == "sgd" else 0.01,
        lr_scheduler_type="constant",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        save_steps=2,
        save_strategy="steps",
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataloader_pin_memory=False,
        use_cpu=not torch.cuda.is_available(),
        ddp_backend="cpu:gloo,cuda:nccl" if options.backend == "fsdp2" and options.offload else None,
        bf16=options.precision == "bf16",
        fp16=options.precision == "fp16",
        seed=42,
        data_seed=42,
        **extra,
    )


def make_trainer(options, output_dir):
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    custom = {}
    if options.optim == "rmsprop":
        custom["optimizer_cls_and_kwargs"] = (torch.optim.RMSprop, {"lr": 0.01, "momentum": 0.9})
    elif options.optim == "adafactor":
        custom["optimizer_cls_and_kwargs"] = (
            Adafactor,
            {"lr": 0.01, "relative_step": False, "scale_parameter": False},
        )
    elif options.optim == "torch_adafactor":
        custom["optimizer_cls_and_kwargs"] = (torch.optim.Adafactor, {"lr": 0.01})
    elif options.optim == "fused_adamw":
        custom["optimizer_cls_and_kwargs"] = (torch.optim.AdamW, {"lr": 0.01, "fused": True})
    return Trainer(
        model=LlamaForCausalLM(config),
        args=arguments(options, output_dir),
        train_dataset=Tokens(),
        eval_dataset=Tokens(),
        optimizer_wrapper=(
            ObservedSST,
            {"rank_fraction": 0.99, "target_modules": [r".*self_attn.*", r".*mlp.*"]},
        ),
        **custom,
    )


def weights(trainer):
    state = trainer.accelerator.get_state_dict(trainer.model_wrapped)
    if state is None:
        return {}
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["ddp", "fsdp1", "fsdp2", "zero1", "zero2", "zero3"], default="ddp")
    parser.add_argument(
        "--optim",
        choices=["sgd", "adamw", "fused_adamw", "rmsprop", "adafactor", "torch_adafactor"],
        default="adamw",
    )
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--sharding", choices=["full_shard", "shard_grad_op"], default="full_shard")
    parser.add_argument("--flat_parameters", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--dp_replicate_size", type=int, default=1)
    options = parser.parse_args()
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    trainer = make_trainer(options, options.output_dir / "full")
    trainer.train()
    assert trainer.args.world_size == int(os.environ.get("WORLD_SIZE", "1"))
    assert isinstance(trainer.optimizer, SST)
    assert len(trainer.optimizer.sst_state_dict()["protected_weights"]) == 14
    assert trainer.optimizer.projections == 4
    assert trainer.optimizer.maximum_leakage < 2e-6, trainer.optimizer.maximum_leakage
    assert trainer.optimizer.maximum_replica_difference == 0.0
    assert trainer.optimizer.maximum_movement > 0.0
    final_weights = weights(trainer)
    basis = copy.deepcopy(trainer.optimizer.sst_state_dict())
    result = {
        "backend": options.backend,
        "optim": options.optim,
        "precision": options.precision,
        "sharding": options.sharding,
        "flat_parameters": options.flat_parameters,
        "offload": options.offload,
        "dp_replicate_size": options.dp_replicate_size,
        "world_size": trainer.args.world_size,
        "torch": torch.__version__,
        "maximum_protected_leakage": trainer.optimizer.maximum_leakage,
        "maximum_replica_difference": trainer.optimizer.maximum_replica_difference,
        "maximum_protected_weight_movement": trainer.optimizer.maximum_movement,
        "optimizer_state_unchanged_by_projection": True,
        "evaluation_loss": trainer.evaluate()["eval_loss"],
    }
    checkpoint = options.output_dir / "full" / "checkpoint-2"
    is_main = trainer.is_world_process_zero()
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    resumed = make_trainer(options, options.output_dir / "resumed")
    with patch("torch.linalg.svd", side_effect=AssertionError("SST resume must restore, not refresh, its basis")):
        resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 4
    assert resumed.optimizer.projections == 2
    assert_same_state(basis, resumed.optimizer.sst_state_dict())
    resumed_weights = weights(resumed)
    assert final_weights.keys() == resumed_weights.keys()
    differences = {
        name: (tensor.float() - resumed_weights[name].float()).abs().max().item()
        for name, tensor in final_weights.items()
    }
    difference = max(differences.values(), default=0.0)
    assert difference < 3e-6, sorted(differences.items(), key=lambda item: item[1], reverse=True)[:5]
    result["maximum_resume_difference"] = difference
    result["fixed_basis_restored_without_svd"] = True
    final_dir = options.output_dir / "final-model"
    if resumed.is_fsdp_enabled:
        resumed.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    resumed.save_model(str(final_dir))
    if dist.is_initialized():
        dist.barrier()
    if is_main:
        if resumed.is_deepspeed_enabled:
            from transformers.integrations.deepspeed import unset_hf_deepspeed_config

            unset_hf_deepspeed_config()
        reloaded = LlamaForCausalLM.from_pretrained(final_dir, local_files_only=True)
        generated = reloaded.generate(torch.tensor([[1, 4, 5]]), max_new_tokens=2, do_sample=False)
        assert generated.shape[1] >= 4
        result["saved_model_generation"] = True
        options.output_dir.mkdir(parents=True, exist_ok=True)
        (options.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

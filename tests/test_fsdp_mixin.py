# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""FSDP tester mixin for model tests."""

import json
import logging
import os
import socket
import sys
import tempfile
import time
import traceback
import unittest.mock
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, is_torch_available
from transformers.testing_utils import (
    backend_device_count,
    backend_empty_cache,
    backend_torch_accelerator_module,
    init_test_logger,
    is_fsdp_test,
    require_fsdp,
)
from transformers.trainer_utils import set_seed


logger = logging.getLogger("transformers.training_test")


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
    from torch.nn.parallel import DistributedDataParallel as DDP

    from transformers.distributed import DistributedConfig
    from transformers.distributed.fsdp import apply_fully_shard_data_parallel, initialize_fsdp
    from transformers.distributed.tensor_parallel import replace_layer_number_by_wildcard
    from transformers.distributed.utils import (
        gather_full_state_dict,
        load_optimizer_distributed,
        save_optimizer_distributed,
    )


# =============================================================================
# Constants
# =============================================================================

BATCH_SIZE = 2
SEQ_LEN = 64
NUM_STEPS = 20
LR = 3e-4
SEED = 42
FSDP_TOP_MODEL_NAMES = {
    # FSDP coverage is gated on models declaring `base_model_fsdp_plan` (config) +
    # class-level `_fsdp_plan` (head class). Listed here are models whose test
    # class extends `CausalLMModelTest` (so they pick up the FSDP mixin) and
    # which use the standard embed_tokens / layers.* / norm naming.
    # Dense
    "llama",
    "mistral",
    "qwen3",
    "phi",
    "olmo3",
    "gemma2",
    # MoE
    "mixtral",
    "qwen3_moe",
    "qwen2_moe",
    "qwen3_5_moe",
    "deepseek_v2",
    "gpt_oss",
    "glm4_moe_lite",
}


# =============================================================================
# Distributed helpers (top-level for pickling by mp.spawn)
# =============================================================================


def _get_distributed_device_type():
    device_type = torch._C._get_accelerator().type
    return "cpu" if device_type == "mps" else device_type


def _get_distributed_backend():
    backend_map = {"cpu": "gloo", "cuda": "nccl", "xpu": "xccl", "hpu": "hccl"}
    return backend_map.get(_get_distributed_device_type(), "gloo")


def _get_rank_device(rank):
    device_type = _get_distributed_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, rank)


def _set_rank_device(rank):
    accelerator_module = backend_torch_accelerator_module(_get_distributed_device_type())
    if accelerator_module is not None and hasattr(accelerator_module, "set_device"):
        accelerator_module.set_device(rank)


def _get_accelerator_rng_state():
    accelerator_module = backend_torch_accelerator_module(_get_distributed_device_type())
    if accelerator_module is None or not hasattr(accelerator_module, "get_rng_state"):
        return None
    return accelerator_module.get_rng_state()


def _set_accelerator_rng_state(rng_state):
    accelerator_module = backend_torch_accelerator_module(_get_distributed_device_type())
    if rng_state is not None and accelerator_module is not None and hasattr(accelerator_module, "set_rng_state"):
        accelerator_module.set_rng_state(rng_state)


def _get_available_fsdp_workers():
    if _get_distributed_device_type() == "cpu":
        return os.cpu_count() or 1
    return backend_device_count(_get_distributed_device_type())


def _fsdp_global_wrapper(rank, test_name, func, func_args, func_kwargs, world_size, port, results_file):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    _set_determinism(SEED)

    dist.init_process_group(backend=_get_distributed_backend(), rank=rank, world_size=world_size)
    _set_rank_device(rank)

    if rank == 0:
        start_time = time.perf_counter()
        print(f"[FSDP] Starting test: {test_name}", flush=True)

    error = None
    try:
        func(rank, *func_args, **func_kwargs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    error_flag = torch.tensor([1 if error else 0], device=_get_rank_device(rank))
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    any_failed = error_flag.item() > 0

    if rank == 0:
        elapsed = time.perf_counter() - start_time
        status = "FAIL" if any_failed else "PASS"
        output_stream = sys.stderr if any_failed else sys.stdout
        print(f"[FSDP] {status} test: {test_name} ({elapsed:.1f}s)", file=output_stream, flush=True)
        with open(results_file, "w") as f:
            json.dump({"error": error or ("Failed on another rank" if any_failed else None)}, f)

    backend_empty_cache(_get_distributed_device_type())
    dist.barrier()
    dist.destroy_process_group()


def _set_determinism(seed):
    torch.use_deterministic_algorithms(True)
    if _get_distributed_device_type() == "cuda" and torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    set_seed(seed)


# =============================================================================
# Training & comparison helpers (top-level for pickling)
# =============================================================================


def _build_repeated_training_batches(config, device, num_steps):
    """Create one deterministic batch and reuse it across steps."""
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device, generator=generator)
    labels = input_ids.clone()
    return [(input_ids, labels)] * num_steps


def _create_shared_tmpdir(rank):
    if rank == 0:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        tmpdir_list = [tmpdir]
    else:
        tmpdir_obj = None
        tmpdir_list = [None]
    dist.broadcast_object_list(tmpdir_list, src=0)
    return tmpdir_list[0], tmpdir_obj


def _gather_ddp_state_dict(model):
    # Only rank 0 returns data to match gather_full_state_dict semantics, so the
    # downstream DDP-vs-FSDP comparison runs once on rank 0 instead of N times.
    if dist.get_rank() != 0:
        return {}
    return {k: v.clone().detach().cpu() for k, v in model.module.state_dict().items()}


def _resolve_fsdp_plan_paths(model):
    """Expand model._fsdp_plan into (paths, strategy) entries.

    Wildcard keys are expanded via ``replace_layer_number_by_wildcard``. When
    weights are tied, the standalone embed_tokens entry is skipped and any
    ``"lm_head"`` keep entry is rewritten to the tied source path (the keep
    group will wrap the shared parameter once).
    """
    plan = model._fsdp_plan

    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )
    tied_source = None
    if weights_tied:
        for name, mod in model.named_modules():
            if mod is input_embed:
                tied_source = name
                break

    name_to_module = dict(model.named_modules())
    entries: list[tuple[list[str], str]] = []
    for key, strategy in plan.items():
        if weights_tied and key == tied_source:
            continue
        if weights_tied and key == "lm_head" and strategy == "keep_full_weight":
            entries.append(([tied_source], strategy))
            continue

        if key in name_to_module:
            entries.append(([key], strategy))
            continue
        matched = [name for name in name_to_module if replace_layer_number_by_wildcard(name) == key]
        if matched:
            entries.append((matched, strategy))
    return entries


def _build_manual_fsdp_plan(config, device):
    """Build a manual FSDP2 plan by expanding model._fsdp_plan."""
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device)

    module_plan: dict[str, list[str]] = {}
    for paths, strategy in _resolve_fsdp_plan_paths(model):
        for path in paths:
            module_plan[path] = [strategy]

    del model
    return {"modules": module_plan}


def _save_init_pretrained(rank, config, dtype):
    """Save a deterministic initial model to a shared tmpdir for from_pretrained loading."""
    tmpdir, tmpdir_obj = _create_shared_tmpdir(rank)
    if rank == 0:
        set_seed(SEED)
        model = AutoModelForCausalLM.from_config(config).to(dtype)
        model.save_pretrained(tmpdir)
        del model
    dist.barrier()
    return tmpdir, tmpdir_obj


def _save_training_state(model, optimizer, training_state_dir):
    """Save optimizer (canonical DCP path) plus per-rank RNG for resume."""
    save_optimizer_distributed(model, optimizer, os.path.join(training_state_dir, "optim"))
    rng = {"cpu": torch.get_rng_state()}
    accel = _get_accelerator_rng_state()
    if accel is not None:
        rng["accel"] = accel
    torch.save(rng, os.path.join(training_state_dir, f"rng_rank{dist.get_rank()}.pt"))


def _load_training_state(model, optimizer, training_state_dir):
    """Inverse of `_save_training_state`."""
    load_optimizer_distributed(model, optimizer, os.path.join(training_state_dir, "optim"))
    rng = torch.load(os.path.join(training_state_dir, f"rng_rank{dist.get_rank()}.pt"), weights_only=False)
    torch.set_rng_state(rng["cpu"])
    if "accel" in rng:
        _set_accelerator_rng_state(rng["accel"])


def train_ddp(rank, batches, lr, device, dtype, init_model_dir):
    _set_determinism(SEED)
    model = AutoModelForCausalLM.from_pretrained(init_model_dir, torch_dtype=dtype, attn_implementation="eager").to(
        device
    )
    # MoE/conditional-routing variants) may not use all params on
    # every step, and DDP would otherwise fail. Specifying find_unused_parameters=True allows running backward on a subgraph of the model.
    ddp_kwargs = {"find_unused_parameters": True}
    if device.type != "cpu":
        ddp_kwargs["device_ids"] = [rank]
    ddp_model = DDP(model, **ddp_kwargs)
    ddp_model.train()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

    losses, grad_norms = [], []
    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = ddp_model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = output.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=float("inf"))
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    state_dict = _gather_ddp_state_dict(ddp_model)

    del optimizer, ddp_model, model
    backend_empty_cache(_get_distributed_device_type())
    dist.barrier()

    return losses, grad_norms, state_dict


def train_fsdp2(
    rank,
    batches,
    lr,
    dtype,
    init_model_dir,
    checkpoint_step,
    fsdp_plan,
):
    # -- Phase 1: Pre-checkpoint run -- train only the first `checkpoint_step` steps, then save
    _set_determinism(SEED)
    distributed_config = DistributedConfig(fsdp_size=dist.get_world_size(), fsdp_plan=fsdp_plan)
    pre_ckpt_model = AutoModelForCausalLM.from_pretrained(
        init_model_dir,
        torch_dtype=dtype,
        distributed_config=distributed_config,
        attn_implementation="eager",
    )
    pre_ckpt_model.train()
    pre_ckpt_optimizer = torch.optim.Adam(pre_ckpt_model.parameters(), lr=lr)

    pre_ckpt_losses, pre_ckpt_grad_norms = [], []
    for step in range(0, checkpoint_step):
        input_ids, labels = batches[step]
        pre_ckpt_optimizer.zero_grad()
        output = pre_ckpt_model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = output.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(pre_ckpt_model.parameters(), max_norm=float("inf"))
        pre_ckpt_optimizer.step()

        pre_ckpt_losses.append(loss.detach().item())
        pre_ckpt_grad_norms.append(grad_norm)

    # -- Phase 2: Save checkpoint, then load into a fresh model
    #   tmpdir/
    #     model/           <- HF safetensors via save_pretrained (DCP + consolidation)
    #     training_state/  <- distcp (optimizer + RNG)
    tmpdir, tmpdir_obj = _create_shared_tmpdir(rank)
    try:
        model_dir = os.path.join(tmpdir, "model")
        training_state_dir = os.path.join(tmpdir, "training_state")

        pre_ckpt_model.save_pretrained(model_dir, is_main_process=(rank == 0))
        _save_training_state(pre_ckpt_model, pre_ckpt_optimizer, training_state_dir)
        dist.barrier()

        # Intentionally scramble RNG to prove checkpoint restore works
        _set_determinism(SEED + 1234)
        resumed_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            distributed_config=distributed_config,
            attn_implementation="eager",
        )
        resumed_model.train()
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=lr)

        _load_training_state(resumed_model, resumed_optimizer, training_state_dir)
        dist.barrier()
    finally:
        if rank == 0:
            tmpdir_obj.cleanup()

    # -- Phase 3: Post-checkpoint run -- continue training the remaining steps from the resumed model
    post_ckpt_losses, post_ckpt_grad_norms = [], []
    for step in range(checkpoint_step, len(batches)):
        input_ids, labels = batches[step]
        resumed_optimizer.zero_grad()
        output = resumed_model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = output.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(resumed_model.parameters(), max_norm=float("inf"))
        resumed_optimizer.step()

        post_ckpt_losses.append(loss.detach().item())
        post_ckpt_grad_norms.append(grad_norm)

    combined_losses = pre_ckpt_losses + post_ckpt_losses
    combined_grad_norms = pre_ckpt_grad_norms + post_ckpt_grad_norms
    combined_state_dict = gather_full_state_dict(resumed_model)

    return combined_losses, combined_grad_norms, combined_state_dict


# =============================================================================
# Distributed test implementations (top-level for pickling by mp.spawn)
# =============================================================================
def _test_fsdp2_save_load_impl(rank, config_class, config_dict):
    """Train FSDP2 model, save via save_pretrained, load via from_pretrained, compare state dicts."""
    init_test_logger()

    device = _get_rank_device(rank)
    config = config_class.from_dict(config_dict)

    batches = _build_repeated_training_batches(config, device, 3)

    distributed_config = DistributedConfig(fsdp_size=dist.get_world_size())

    init_tmpdir, init_tmpdir_obj = _save_init_pretrained(rank, config, torch.float32)
    try:
        _set_determinism(SEED)
        model = AutoModelForCausalLM.from_pretrained(
            init_tmpdir,
            distributed_config=distributed_config,
            attn_implementation="eager",
        )
        dist.barrier()
    finally:
        if rank == 0 and init_tmpdir_obj is not None:
            init_tmpdir_obj.cleanup()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        output.loss.backward()
        optimizer.step()

    state_dict_before = gather_full_state_dict(model)

    tmpdir, tmpdir_obj = _create_shared_tmpdir(rank)
    try:
        model.save_pretrained(tmpdir, is_main_process=(rank == 0))
        dist.barrier()

        new_model = AutoModelForCausalLM.from_pretrained(
            tmpdir,
            distributed_config=distributed_config,
            attn_implementation="eager",
        )
        dist.barrier()
    finally:
        if rank == 0:
            tmpdir_obj.cleanup()

    state_dict_after = gather_full_state_dict(new_model)

    for key in state_dict_before:
        assert key in state_dict_after, f"Key {key} missing after load"
        torch.testing.assert_close(
            state_dict_before[key],
            state_dict_after[key],
            rtol=0,
            atol=0,
            msg=f"Weight mismatch for {key} after save/load",
        )

    if rank == 0:
        logger.debug(f"FSDP2 save/load test passed: all {len(state_dict_before)} parameters match exactly.")


def _test_fsdp2_sharding_structure_impl(rank, config_class, config_dict, tie_word_embeddings):
    """
    Verify that apply_fully_shard_data_parallel(fsdp_plan=None) wraps exactly the right modules.

    Expected FSDP targets:
    UNTIED                              TIED
    ──────                              ────
    1. embed_tokens  (reshard=True)     1. (skip — embed goes to step 3)
    2. layers[i]     (reshard=True)     2. layers[i]     (reshard=True)
    3. [norm, lm_head] (reshard=False)  3. [norm, embed_tokens] (reshard=False)
    4. root                             4. root
    """
    init_test_logger()

    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings

    auto_plan = None
    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan={})

    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)

    # Expected FSDP targets come from model._fsdp_plan: every resolved path gets a
    # fully_shard call (keep_full_weight entries are bundled into one group, but each
    # member still appears as an FSDP-wrapped module in named_modules), plus the root.
    expected_targets = {""}
    for paths, _strategy in _resolve_fsdp_plan_paths(model):
        expected_targets.update(paths)

    model = apply_fully_shard_data_parallel(model, device_mesh, fsdp_plan=auto_plan)

    actual_targets = {name for name, module in model.named_modules() if type(module).__name__.startswith("FSDP")}

    if rank == 0:
        logger.debug(f"  Weights tied: {config.tie_word_embeddings}")
        logger.debug(f"  Expected FSDP targets: {sorted(expected_targets)}")
        logger.debug(f"  Actual FSDP targets:   {sorted(actual_targets)}")

    missing = expected_targets - actual_targets
    extra = actual_targets - expected_targets
    assert not missing and not extra, (
        f"FSDP target mismatch.\n"
        f"  Missing (expected but not wrapped): {sorted(missing)}\n"
        f"  Extra (wrapped but not expected):   {sorted(extra)}"
    )

    if rank == 0:
        logger.debug(f"  FSDP sharding structure OK ({len(actual_targets)} targets)")


def _test_fsdp2_plan_vs_ddp_impl(rank, config_class, config_dict, tie_word_embeddings, plan_mode, dtype=None):
    """Validate DDP-vs-FSDP2 trace matching for either auto or manual plan mode."""
    init_test_logger()

    if dtype is None:
        dtype = torch.float32

    device = _get_rank_device(rank)
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings

    if plan_mode == "auto":
        fsdp_plan = None
        test_label = "FSDP2(auto)"
    elif plan_mode == "manual":
        fsdp_plan = _build_manual_fsdp_plan(config, device)
        test_label = "FSDP2(manual)"
    else:
        raise ValueError(f"Unsupported plan_mode '{plan_mode}'. Expected 'auto' or 'manual'.")

    checkpoint_step = NUM_STEPS // 2
    init_model_dir, init_tmpdir_obj = _save_init_pretrained(rank, config, dtype)
    batches = _build_repeated_training_batches(config, device, NUM_STEPS)
    try:
        ddp_losses, ddp_grad_norms, ddp_state_dict = train_ddp(rank, batches, LR, device, dtype, init_model_dir)

        fsdp_losses, fsdp_grad_norms, fsdp_state_dict = train_fsdp2(
            rank,
            batches,
            LR,
            dtype,
            init_model_dir=init_model_dir,
            checkpoint_step=checkpoint_step,
            fsdp_plan=fsdp_plan,
        )
    finally:
        if rank == 0 and init_tmpdir_obj is not None:
            init_tmpdir_obj.cleanup()

    for step in range(len(ddp_losses)):
        torch.testing.assert_close(
            torch.tensor(ddp_losses[step]),
            torch.tensor(fsdp_losses[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Loss mismatch at step {step}: DDP={ddp_losses[step]}, {test_label}={fsdp_losses[step]}",
        )
        torch.testing.assert_close(
            torch.tensor(ddp_grad_norms[step]),
            torch.tensor(fsdp_grad_norms[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Grad norm mismatch at step {step}: DDP={ddp_grad_norms[step]}, {test_label}={fsdp_grad_norms[step]}",
        )

    for key in ddp_state_dict:
        assert key in fsdp_state_dict, f"Key {key} missing from {test_label} state dict"
        torch.testing.assert_close(
            ddp_state_dict[key],
            fsdp_state_dict[key],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Weight mismatch for {key}: DDP vs {test_label}",
        )

    if rank == 0:
        logger.debug(f"DDP and {test_label} comparison checks passed.")


# =============================================================================
# Mixin class
# =============================================================================


class FSDPTesterMixin(ABC):
    fsdp_nproc_per_node: int = 2
    skip_fsdp_tests: bool = False

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    def _skip_if_fsdp_disabled(self):
        if self.skip_fsdp_tests:
            self.skipTest("FSDP tests disabled for this model (skip_fsdp_tests=True)")

    def _skip_if_insufficient_devices(self):
        self._skip_if_fsdp_disabled()
        available_workers = _get_available_fsdp_workers()
        if available_workers < self.fsdp_nproc_per_node:
            self.skipTest(f"Need at least {self.fsdp_nproc_per_node} FSDP workers, have {available_workers}")

    def _get_fsdp_model_name(self):
        module_parts = self.__class__.__module__.split(".")
        if len(module_parts) >= 3 and module_parts[0] == "tests" and module_parts[1] == "models":
            return module_parts[2]
        return None

    def _skip_if_fsdp_model_not_selected(self):
        model_name = self._get_fsdp_model_name()
        if model_name not in FSDP_TOP_MODEL_NAMES:
            model_label = model_name or self.__class__.__module__
            self.skipTest(
                "FSDP mixin coverage is currently limited to the top-10 dense and top-10 MoE model suites "
                f"(skipping {model_label})."
            )

    def _create_model_on_meta(self, config):
        """Instantiate a model on the meta device (no memory allocated)."""
        auto_classes = [AutoModelForCausalLM, AutoModelForSeq2SeqLM]
        for auto_cls in auto_classes:
            try:
                with torch.device("meta"):
                    return auto_cls.from_config(config)
            except Exception:
                continue
        self.skipTest(f"Cannot instantiate model with any Auto class for config {type(config).__name__}")

    def _get_tiny_config(self):
        """Get config class and serialized dict for passing to spawned processes."""
        config = self.model_tester.get_config()
        config.vocab_size = 256
        config.hidden_size = 64
        config.intermediate_size = 128
        if hasattr(config, "ffn_config"):
            # Keep nested FFN projections consistent with resized hidden size.
            if hasattr(config.ffn_config, "ffn_hidden_size"):
                config.ffn_config.ffn_hidden_size = config.hidden_size
            if hasattr(config.ffn_config, "hidden_size"):
                config.ffn_config.hidden_size = config.intermediate_size
        if hasattr(config, "num_attention_heads"):
            config.num_attention_heads = 4
        if hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = 4
        if hasattr(config, "vocab_size_per_layer_input"):
            config.vocab_size_per_layer_input = config.vocab_size
        # `to_diff_dict()` avoids nested config pollution (e.g. DBRX ffn_config receiving
        # generic PretrainedConfig keys that its constructor rejects).
        config_dict = config.to_diff_dict()
        return type(config), config_dict

    def _run_fsdp2_distributed_test(self, test_name, test_impl, *test_args, **test_kwargs):
        self._skip_if_fsdp_model_not_selected()
        self._skip_if_insufficient_devices()

        config_class, config_dict = self._get_tiny_config()
        func_args = (config_class, config_dict, *test_args)

        results_file = tempfile.mktemp(suffix=".json")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        try:
            mp.spawn(
                _fsdp_global_wrapper,
                args=(test_name, test_impl, func_args, test_kwargs, self.fsdp_nproc_per_node, port, results_file),
                nprocs=self.fsdp_nproc_per_node,
            )

            with open(results_file) as f:
                result = json.load(f)
        finally:
            if os.path.exists(results_file):
                os.unlink(results_file)

        if result["error"] is not None:
            self.fail(f"FSDP test '{test_name}' failed:\n{result['error']}")

    # =========================================================================
    # Test: get_transformer_block_classes (CPU, meta device)
    # =========================================================================

    @is_fsdp_test
    def test_fsdp_plan_declared(self):
        """The model exposes a non-empty `_fsdp_plan` derived from config + class-level overrides."""
        self._skip_if_fsdp_disabled()
        self._skip_if_fsdp_model_not_selected()
        start_time = time.perf_counter()
        logger.info("[FSDP] Starting test: test_fsdp_plan_declared")
        status = "FAIL"
        try:
            config = self.model_tester.get_config()
            model = self._create_model_on_meta(config)

            plan = getattr(model, "_fsdp_plan", None)
            self.assertTrue(plan, f"No _fsdp_plan declared for {type(model).__name__}")
            status = "PASS"
        finally:
            logger.info("[FSDP] %s test: test_fsdp_plan_declared (%.1fs)", status, time.perf_counter() - start_time)

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_sharding_structure_untied(self):
        self._run_fsdp2_distributed_test(
            "test_fsdp2_sharding_structure_untied", _test_fsdp2_sharding_structure_impl, False
        )

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_sharding_structure_tied(self):
        self._run_fsdp2_distributed_test(
            "test_fsdp2_sharding_structure_tied", _test_fsdp2_sharding_structure_impl, True
        )

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_save_load(self):
        self._run_fsdp2_distributed_test("test_fsdp2_save_load", _test_fsdp2_save_load_impl)

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_auto_plan_vs_ddp_untied(self):
        self._run_fsdp2_distributed_test(
            "test_fsdp2_auto_plan_vs_ddp_untied", _test_fsdp2_plan_vs_ddp_impl, False, "auto"
        )

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_auto_plan_vs_ddp_tied(self):
        self._run_fsdp2_distributed_test(
            "test_fsdp2_auto_plan_vs_ddp_tied", _test_fsdp2_plan_vs_ddp_impl, True, "auto"
        )

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_manual_plan_vs_ddp_untied(self):
        self._run_fsdp2_distributed_test(
            "test_fsdp2_manual_plan_vs_ddp_untied", _test_fsdp2_plan_vs_ddp_impl, False, "manual"
        )

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_manual_plan_vs_ddp_tied(self):
        self._run_fsdp2_distributed_test(
            "test_fsdp2_manual_plan_vs_ddp_tied", _test_fsdp2_plan_vs_ddp_impl, True, "manual"
        )

    # =========================================================================
    # Policy propagation tests (CPU-only, patch fully_shard, no distributed)
    # =========================================================================

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_policies_propagate_auto(self):
        """Auto plan with cpu_offload+mixed_precision flags routes both policies into every fully_shard call."""
        self._skip_if_fsdp_disabled()
        self._skip_if_fsdp_model_not_selected()
        config_class, config_dict = self._get_tiny_config()
        config = config_class.from_dict(config_dict)
        set_seed(SEED)
        model = AutoModelForCausalLM.from_config(config)

        with unittest.mock.patch("transformers.distributed.fsdp.fully_shard") as mock_shard:
            apply_fully_shard_data_parallel(
                model,
                unittest.mock.MagicMock(),
                fsdp_plan={"cpu_offload": True, "mixed_precision": True},
            )

        self.assertGreater(mock_shard.call_count, 0, "fully_shard was never called")
        for call in mock_shard.call_args_list:
            self.assertIsInstance(call.kwargs.get("offload_policy"), CPUOffloadPolicy)
            mp_policy = call.kwargs.get("mp_policy")
            self.assertIsInstance(mp_policy, MixedPrecisionPolicy)
            self.assertEqual(mp_policy.param_dtype, torch.bfloat16)
            self.assertEqual(mp_policy.reduce_dtype, torch.float32)

    @is_fsdp_test
    @require_fsdp
    def test_fsdp2_policies_propagate_manual(self):
        """Manual plan: per-module policy tokens land on that module's fully_shard call, others stay clean."""
        self._skip_if_fsdp_disabled()
        self._skip_if_fsdp_model_not_selected()
        config_class, config_dict = self._get_tiny_config()
        config = config_class.from_dict(config_dict)
        set_seed(SEED)
        model = AutoModelForCausalLM.from_config(config)

        mlp = model.get_submodule("model.layers.0.mlp")
        norm = model.get_submodule("model.norm")
        with unittest.mock.patch("transformers.distributed.fsdp.fully_shard") as mock_shard:
            apply_fully_shard_data_parallel(
                model,
                unittest.mock.MagicMock(),
                fsdp_plan={
                    "modules": {
                        "model.layers.0.mlp": ["free_full_weight", "cpu_offload", "mixed_precision"],
                        "model.norm": ["keep_full_weight"],
                    }
                },
            )

        mlp_call = next(c for c in mock_shard.call_args_list if c.args and c.args[0] is mlp)
        norm_call = next(c for c in mock_shard.call_args_list if c.args and c.args[0] is norm)

        self.assertIsInstance(mlp_call.kwargs["offload_policy"], CPUOffloadPolicy)
        self.assertIsInstance(mlp_call.kwargs["mp_policy"], MixedPrecisionPolicy)
        self.assertTrue(mlp_call.kwargs["reshard_after_forward"])

        self.assertNotIn("offload_policy", norm_call.kwargs)
        self.assertNotIn("mp_policy", norm_call.kwargs)
        self.assertFalse(norm_call.kwargs["reshard_after_forward"])

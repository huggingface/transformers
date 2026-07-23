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

"""FSDP tester mixin for model tests."""

import json
import logging
import os
import socket
import sys
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager

from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, is_torch_available
from transformers.testing_utils import (
    backend_device_count,
    backend_empty_cache,
    backend_torch_accelerator_module,
    init_test_logger,
    is_fsdp_test,
    require_torch_greater_or_equal,
)
from transformers.trainer_utils import set_seed


logger = logging.getLogger("transformers.training_test")


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP

    from transformers.distributed import DistributedConfig
    from transformers.distributed.fsdp import _resolve_tied_embed_lm_head_plan, expand_fsdp_plan
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
DDP_FSDP_RTOL = 1e-5
DDP_FSDP_ATOL = 1e-5

# Set to None to run distributed FSDP tests for every model with a plan.
FSDP_DISTRIBUTED_TEST_MODEL_TYPES = {"cohere2_moe"}


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


def _get_available_fsdp_workers():
    if _get_distributed_device_type() == "cpu":
        return os.cpu_count() or 1
    return backend_device_count(_get_distributed_device_type())


def _set_determinism(seed):
    torch.use_deterministic_algorithms(True)
    if _get_distributed_device_type() == "cuda" and torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    set_seed(seed)


@contextmanager
def _distributed_tmpdir(rank):
    if rank == 0:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir_list = [tmpdir_obj.name]
    else:
        tmpdir_obj = None
        tmpdir_list = [None]
    dist.broadcast_object_list(tmpdir_list, src=0)
    try:
        yield tmpdir_list[0]
    finally:
        if rank == 0 and tmpdir_obj is not None:
            tmpdir_obj.cleanup()


@contextmanager
def _deterministic_init_model_dir(rank, config, dtype):
    with _distributed_tmpdir(rank) as model_dir:
        if rank == 0:
            set_seed(SEED)
            model = AutoModelForCausalLM.from_config(config).to(dtype)
            model.save_pretrained(model_dir)
            del model
        dist.barrier()
        yield model_dir


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


# =============================================================================
# Training helpers (top-level for pickling)
# =============================================================================


def _build_repeated_training_batches(config, device, num_steps):
    """Create one deterministic batch and reuse it across steps."""
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device, generator=generator)
    labels = input_ids.clone()
    return [(input_ids, labels)] * num_steps


def _run_training_steps(model, optimizer, batches, *, track_grad_norms=True):
    """Forward/backward/step over batches. Returns (losses, grad_norms)."""
    losses, grad_norms = [], []
    for input_ids, labels in batches:
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
        loss.backward()
        if track_grad_norms:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            grad_norms.append(grad_norm)
        optimizer.step()
        losses.append(loss.detach().item())
    return losses, grad_norms


def _save_training_state(model, optimizer, training_state_dir):
    """Save optimizer (canonical DCP path) plus per-rank RNG for resume."""
    save_optimizer_distributed(model, optimizer, os.path.join(training_state_dir, "optim"))
    rng = {"cpu": torch.get_rng_state()}
    accelerator_module = backend_torch_accelerator_module(_get_distributed_device_type())
    if accelerator_module is not None and hasattr(accelerator_module, "get_rng_state"):
        accel_rng = accelerator_module.get_rng_state()
        if accel_rng is not None:
            rng["accel"] = accel_rng
    torch.save(rng, os.path.join(training_state_dir, f"rng_rank{dist.get_rank()}.pt"))


def _load_training_state(model, optimizer, training_state_dir):
    """Inverse of `_save_training_state`."""
    load_optimizer_distributed(model, optimizer, os.path.join(training_state_dir, "optim"))
    rng = torch.load(os.path.join(training_state_dir, f"rng_rank{dist.get_rank()}.pt"), weights_only=False)
    torch.set_rng_state(rng["cpu"])
    if "accel" in rng:
        accelerator_module = backend_torch_accelerator_module(_get_distributed_device_type())
        if accelerator_module is not None and hasattr(accelerator_module, "set_rng_state"):
            accelerator_module.set_rng_state(rng["accel"])


def _checkpoint_and_resume(pre_model, pre_optimizer, dtype, distributed_config, lr):
    """Save model+optimizer, scramble RNG, reload and restore training state."""
    rank = dist.get_rank()
    with _distributed_tmpdir(rank) as tmpdir:
        model_dir = os.path.join(tmpdir, "model")
        training_state_dir = os.path.join(tmpdir, "training_state")

        pre_model.save_pretrained(model_dir, is_main_process=(rank == 0))
        _save_training_state(pre_model, pre_optimizer, training_state_dir)
        dist.barrier()

        # Intentionally scramble RNG to prove checkpoint restore works
        _set_determinism(SEED + 1234)
        resumed_model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, distributed_config=distributed_config
        )
        resumed_model.train()
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=lr)
        _load_training_state(resumed_model, resumed_optimizer, training_state_dir)
        dist.barrier()
        return resumed_model, resumed_optimizer


def train_ddp(rank, batches, lr, device, dtype, init_model_dir):
    _set_determinism(SEED)
    model = AutoModelForCausalLM.from_pretrained(init_model_dir, torch_dtype=dtype).to(device)
    # MoE/conditional-routing variants may not use all params on every step, and DDP would otherwise fail.
    ddp_kwargs = {"find_unused_parameters": True}
    if device.type != "cpu":
        ddp_kwargs["device_ids"] = [rank]
    ddp_model = DDP(model, **ddp_kwargs)
    ddp_model.train()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

    losses, grad_norms = _run_training_steps(ddp_model, optimizer, batches)
    if dist.get_rank() != 0:
        state_dict = {}
    else:
        # Only rank 0 returns data to match gather_full_state_dict semantics.
        state_dict = {k: v.clone().detach().cpu() for k, v in ddp_model.module.state_dict().items()}

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
):
    distributed_config = DistributedConfig(fsdp_size=dist.get_world_size())

    # Phase 1: Pre-checkpoint run
    _set_determinism(SEED)
    pre_ckpt_model = AutoModelForCausalLM.from_pretrained(
        init_model_dir, torch_dtype=dtype, distributed_config=distributed_config
    )
    pre_ckpt_model.train()
    pre_ckpt_optimizer = torch.optim.Adam(pre_ckpt_model.parameters(), lr=lr)
    pre_ckpt_losses, pre_ckpt_grad_norms = _run_training_steps(
        pre_ckpt_model, pre_ckpt_optimizer, batches[:checkpoint_step]
    )

    # Phase 2: Save checkpoint, then load into a fresh model
    resumed_model, resumed_optimizer = _checkpoint_and_resume(
        pre_ckpt_model, pre_ckpt_optimizer, dtype, distributed_config, lr
    )

    # Phase 3: Post-checkpoint run
    post_ckpt_losses, post_ckpt_grad_norms = _run_training_steps(
        resumed_model, resumed_optimizer, batches[checkpoint_step:]
    )

    return (
        pre_ckpt_losses + post_ckpt_losses,
        pre_ckpt_grad_norms + post_ckpt_grad_norms,
        gather_full_state_dict(resumed_model),
    )


# =============================================================================
# Distributed test implementations (top-level for pickling by mp.spawn)
# =============================================================================


def _test_fsdp2_save_load_impl(rank, config_class, config_dict):
    """Save FSDP2 model via save_pretrained, load via from_pretrained, compare state dicts."""
    init_test_logger()

    config = config_class.from_dict(config_dict)
    distributed_config = DistributedConfig(fsdp_size=dist.get_world_size())

    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        _set_determinism(SEED)
        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)
        dist.barrier()

        state_dict_before = gather_full_state_dict(model)

        with _distributed_tmpdir(rank) as tmpdir:
            model.save_pretrained(tmpdir, is_main_process=(rank == 0))
            dist.barrier()
            new_model = AutoModelForCausalLM.from_pretrained(tmpdir, distributed_config=distributed_config)
            dist.barrier()

        state_dict_after = gather_full_state_dict(new_model)
        for key in state_dict_before:
            assert key in state_dict_after, f"After save/load: Key {key} missing after load"
            torch.testing.assert_close(
                state_dict_before[key],
                state_dict_after[key],
                rtol=0,
                atol=0,
                msg=f"After save/load: Weight mismatch for {key}",
            )

        if rank == 0:
            logger.debug(f"FSDP2 save/load test passed: all {len(state_dict_before)} parameters match exactly.")


def _test_fsdp2_save_load_dcp_impl(rank, config_class, config_dict):
    """Save FSDP2 model via save_pretrained(distributed_checkpoint=True), reload, compare state dicts."""
    init_test_logger()

    config = config_class.from_dict(config_dict)
    distributed_config = DistributedConfig(fsdp_size=dist.get_world_size())

    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        _set_determinism(SEED)
        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)
        dist.barrier()

        state_dict_before = gather_full_state_dict(model)

        with _distributed_tmpdir(rank) as tmpdir:
            model.save_pretrained(tmpdir, is_main_process=(rank == 0), distributed_checkpoint=True)
            dist.barrier()
            new_model = AutoModelForCausalLM.from_pretrained(tmpdir, distributed_config=distributed_config)
            dist.barrier()

        state_dict_after = gather_full_state_dict(new_model)
        for key in state_dict_before:
            assert key in state_dict_after, f"After DCP save/load: Key {key} missing after load"
            torch.testing.assert_close(
                state_dict_before[key],
                state_dict_after[key],
                rtol=0,
                atol=0,
                msg=f"After DCP save/load: Weight mismatch for {key}",
            )

        if rank == 0:
            logger.debug(f"FSDP2 DCP save/load test passed: all {len(state_dict_before)} parameters match exactly.")


def _test_fsdp2_sharding_structure_impl(rank, config_class, config_dict, tie_word_embeddings):
    """Verify that apply_fully_sharded_data_parallel wraps exactly the right modules."""
    init_test_logger()

    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings
    distributed_config = DistributedConfig(fsdp_size=dist.get_world_size())

    with _deterministic_init_model_dir(rank, config, torch.float32) as init_dir:
        _set_determinism(SEED)
        model = AutoModelForCausalLM.from_pretrained(init_dir, distributed_config=distributed_config)
        dist.barrier()

        adapted_fsdp_plan = _resolve_tied_embed_lm_head_plan(model._fsdp_plan, model)
        reshard_targets, no_reshard_targets = expand_fsdp_plan(model, adapted_fsdp_plan)
        expected_targets = {""} | {name for name, _ in reshard_targets + no_reshard_targets}
        actual_targets = {name for name, module in model.named_modules() if type(module).__name__.startswith("FSDP")}

        if rank == 0:
            logger.debug(f"  Weights tied: {tie_word_embeddings}")
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


def _test_fsdp2_plan_vs_ddp_impl(rank, config_class, config_dict, tie_word_embeddings, dtype=None):
    """Validate DDP-vs-FSDP2 trace matching using the model's declared FSDP plan."""
    init_test_logger()

    if dtype is None:
        dtype = torch.float32

    device = _get_rank_device(rank)
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings

    checkpoint_step = NUM_STEPS // 2
    batches = _build_repeated_training_batches(config, device, NUM_STEPS)

    with _deterministic_init_model_dir(rank, config, dtype) as init_model_dir:
        ddp_losses, ddp_grad_norms, ddp_state_dict = train_ddp(rank, batches, LR, device, dtype, init_model_dir)
        fsdp_losses, fsdp_grad_norms, fsdp_state_dict = train_fsdp2(
            rank,
            batches,
            LR,
            dtype,
            init_model_dir=init_model_dir,
            checkpoint_step=checkpoint_step,
        )

    for step in range(len(ddp_losses)):
        torch.testing.assert_close(
            torch.tensor(ddp_losses[step]),
            torch.tensor(fsdp_losses[step]),
            rtol=DDP_FSDP_RTOL,
            atol=DDP_FSDP_ATOL,
            msg=f"Loss mismatch at step {step}: DDP={ddp_losses[step]}, FSDP2={fsdp_losses[step]}",
        )
        torch.testing.assert_close(
            torch.tensor(ddp_grad_norms[step]),
            torch.tensor(fsdp_grad_norms[step]),
            rtol=DDP_FSDP_RTOL,
            atol=DDP_FSDP_ATOL,
            msg=f"Grad norm mismatch at step {step}: DDP={ddp_grad_norms[step]}, FSDP2={fsdp_grad_norms[step]}",
        )

    for key in ddp_state_dict:
        assert key in fsdp_state_dict, f"Key {key} missing from FSDP2 state dict"
        torch.testing.assert_close(
            ddp_state_dict[key],
            fsdp_state_dict[key],
            rtol=DDP_FSDP_RTOL,
            atol=DDP_FSDP_ATOL,
            msg=f"Weight mismatch for {key}: DDP vs FSDP2",
        )

    if rank == 0:
        logger.debug("DDP and FSDP2 comparison checks passed.")


# =============================================================================
# Mixin class
# =============================================================================


class FSDPTesterMixin(ABC):
    fsdp_nproc_per_node: int = 2
    # TODO(3outeille): do we put the CONSTANTS in the mixin class ?

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    def _skip_if_insufficient_devices(self):
        available_workers = _get_available_fsdp_workers()
        if available_workers < self.fsdp_nproc_per_node:
            self.skipTest(f"Need at least {self.fsdp_nproc_per_node} FSDP workers, have {available_workers}")

    def _has_fsdp_plan(self) -> bool:
        config = self.model_tester.get_config()
        return hasattr(config, "base_model_fsdp_plan") and config.base_model_fsdp_plan is not None

    def _skip_if_fsdp_distributed_not_enabled(self):
        if not self._has_fsdp_plan():
            self.skipTest("Model does not have an FSDP plan (base_model_fsdp_plan)")

        config = self.model_tester.get_config()
        # Only top-N models are tested, set FSDP_DISTRIBUTED_TEST_MODEL_TYPES = None to run all tests.
        if (
            FSDP_DISTRIBUTED_TEST_MODEL_TYPES is not None
            and config.model_type not in FSDP_DISTRIBUTED_TEST_MODEL_TYPES
        ):
            self.skipTest(
                f"FSDP distributed tests are not enabled for model_type={config.model_type!r} "
                f"(enabled: {sorted(FSDP_DISTRIBUTED_TEST_MODEL_TYPES)}). Set FSDP_DISTRIBUTED_TEST_MODEL_TYPES = None to run all tests."
            )

    def _get_tiny_config(self):
        """Get config class and serialized dict for passing to spawned processes."""
        config = self.model_tester.get_config()
        config.vocab_size = 256
        config.hidden_size = 64
        config.intermediate_size = 128
        if hasattr(config, "ffn_config"):
            if hasattr(config.ffn_config, "ffn_hidden_size"):
                config.ffn_config.ffn_hidden_size = config.hidden_size
            if hasattr(config.ffn_config, "hidden_size"):
                config.ffn_config.hidden_size = config.intermediate_size
        if hasattr(config, "num_attention_heads"):
            config.num_attention_heads = 4
        if hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = 4
        if hasattr(config, "moe_intermediate_size"):
            config.moe_intermediate_size = 32
        if hasattr(config, "vocab_size_per_layer_input"):
            config.vocab_size_per_layer_input = config.vocab_size
        return type(config), config.to_diff_dict()

    def _run_fsdp2_distributed_test(self, test_name, test_impl, *test_args, **test_kwargs):
        self._skip_if_insufficient_devices()
        self._skip_if_fsdp_distributed_not_enabled()

        config_class, config_dict = self._get_tiny_config()
        func_args = (config_class, config_dict, *test_args)

        results_file = tempfile.mktemp(suffix=".json")
        # port binding
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

    @is_fsdp_test
    def test_fsdp_plan_declared(self):
        """The model exposes a non-empty `_fsdp_plan` derived from config + class-level overrides."""
        if not self._has_fsdp_plan():
            self.skipTest("Model does not have an FSDP plan (base_model_fsdp_plan)")

        config = self.model_tester.get_config()
        auto_classes = [AutoModelForCausalLM, AutoModelForSeq2SeqLM]  # TODO(3outeille): why AutoModelForSeq2SeqLM ?
        for auto_cls in auto_classes:
            try:
                with torch.device("meta"):
                    model = auto_cls.from_config(config)
                    break
            except Exception:
                continue
        else:
            self.skipTest(f"Cannot instantiate model with any Auto class for config {type(config).__name__}")
        self.assertTrue(model._fsdp_plan, f"No _fsdp_plan declared for {type(model).__name__}")

    @parameterized.expand(["untied", "tied"])
    @require_torch_greater_or_equal("2.7")
    @is_fsdp_test
    def test_fsdp2_sharding_structure(self, label):
        self._run_fsdp2_distributed_test(
            f"test_fsdp2_sharding_structure_{label}",
            _test_fsdp2_sharding_structure_impl,
            label == "tied",
        )

    @require_torch_greater_or_equal("2.7")
    @is_fsdp_test
    def test_fsdp2_save_load(self):
        self._run_fsdp2_distributed_test("test_fsdp2_save_load", _test_fsdp2_save_load_impl)

    @require_torch_greater_or_equal("2.7")
    @is_fsdp_test
    def test_fsdp2_save_load_dcp(self):
        self._run_fsdp2_distributed_test("test_fsdp2_save_load_dcp", _test_fsdp2_save_load_dcp_impl)

    @parameterized.expand(["untied", "tied"])
    @require_torch_greater_or_equal("2.7")
    @is_fsdp_test
    def test_fsdp2_plan_vs_ddp(self, label):
        self._run_fsdp2_distributed_test(
            f"test_fsdp2_plan_vs_ddp_{label}",
            _test_fsdp2_plan_vs_ddp_impl,
            label == "tied",
        )

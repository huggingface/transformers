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

import logging
import os
import socket
import tempfile
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, is_torch_available
from transformers.testing_utils import (
    Colors,
    backend_device_count,
    init_test_logger,
    require_fsdp,
    require_torch_multi_accelerator,
    torch_device,
)
from transformers.trainer_utils import set_seed


logger = logging.getLogger("transformers.training_test")


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.multiprocessing as mp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.tensor import DTensor
    from torch.nn.parallel import DistributedDataParallel as DDP

    from transformers.integrations.fsdp import _find_final_norm, apply_fsdp2, get_transformer_block_classes, initialize_fsdp


# =============================================================================
# Constants
# =============================================================================

BATCH_SIZE = 2
SEQ_LEN = 64
NUM_STEPS = 20
LR = 3e-4
SEED = 42


# =============================================================================
# Distributed helpers (top-level for pickling by mp.spawn)
# =============================================================================


def _fsdp_global_wrapper(rank, func, world_size, port, func_args, func_kwargs):
    """Set up distributed environment and run the test function."""
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def _get_free_port():
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _fsdp_init_distributed(world_size):
    """Decorator to run function in distributed mode using mp.spawn."""

    def _init_distributed(func):
        def wrapper(*args, **kwargs):
            port = _get_free_port()
            spawn_args = (func, world_size, port, args, kwargs)
            mp.spawn(_fsdp_global_wrapper, args=spawn_args, nprocs=world_size)

        return wrapper

    return _init_distributed


# =============================================================================
# Training & comparison helpers (top-level for pickling)
# =============================================================================


def _create_deterministic_data(batch_size, seq_len, vocab_size, device, seed):
    """Create deterministic random training data using torch.randint."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = input_ids.clone()
    return [(input_ids, labels)]


def _gather_fsdp2_state_dict(model):
    """Gather FSDP2 sharded parameters into full tensors via DTensor.full_tensor()."""
    state_dict = {}
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, DTensor):
            state_dict[name] = tensor.full_tensor().clone().detach()
        else:
            state_dict[name] = tensor.clone().detach()
    return state_dict


def _compute_grad_norm(model):
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            total_norm_sq += grad.data.float().norm(2).item() ** 2
    return total_norm_sq ** 0.5


def _log_comparison_table(title, ddp_vals, fsdp_vals):
    """Log a side-by-side comparison table for DDP vs FSDP2 values."""
    C = Colors
    SEP = f"{C.DIM}|{C.RESET}"
    ROW = f"  {C.DIM}{'─' * 52}{C.RESET}"

    logger.info(f"  {C.BOLD}{title}{C.RESET}")
    logger.info(ROW)
    logger.info(
        f"  {C.DIM}{'step':>4}{C.RESET}  "
        f"{SEP}  {C.BLUE}{C.BOLD}{'DDP':^14}{C.RESET}  "
        f"{SEP}  {C.MAGENTA}{C.BOLD}{'FSDP2':^14}{C.RESET}  "
        f"{SEP}  {C.DIM}{'diff':^10}{C.RESET}"
    )
    logger.info(ROW)
    for step in range(len(ddp_vals)):
        diff = abs(ddp_vals[step] - fsdp_vals[step])
        match = f"{C.GREEN}={C.RESET}" if diff < 1e-6 else f"{C.YELLOW}{diff:.1e}{C.RESET}"
        logger.info(
            f"  {C.DIM}{step + 1:>4}{C.RESET}  "
            f"{SEP}  {C.BLUE}{ddp_vals[step]:>14.6f}{C.RESET}  "
            f"{SEP}  {C.MAGENTA}{fsdp_vals[step]:>14.6f}{C.RESET}  "
            f"{SEP}  {match:^10}"
        )
    logger.info(ROW)


def _train_ddp(rank, config, batches, lr, device, dtype):
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    ddp_model = DDP(model, device_ids=[rank]).to(dtype)
    ddp_model.train()

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = ddp_model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        grad_norm = _compute_grad_norm(ddp_model)
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    state_dict = {k: v.clone().detach() for k, v in ddp_model.module.state_dict().items()}
    return losses, grad_norms, state_dict


def _train_fsdp2(rank, config, batches, lr, device_map, device_mesh, dtype, fsdp_plan):
    """Run an FSDP2 training loop with Adam."""
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map).to(dtype)
    model = apply_fsdp2(model, device_mesh, fsdp_plan=fsdp_plan)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        grad_norm = _compute_grad_norm(model)
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    state_dict = _gather_fsdp2_state_dict(model)
    return losses, grad_norms, state_dict


def _assert_ddp_fsdp_match(ddp_losses, ddp_grad_norms, ddp_state_dict, fsdp_losses, fsdp_grad_norms, fsdp_state_dict, label="FSDP2"):
    """Assert that DDP and FSDP2 training produced identical results."""
    for step in range(len(ddp_losses)):
        torch.testing.assert_close(
            torch.tensor(ddp_losses[step]),
            torch.tensor(fsdp_losses[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Step {step} loss mismatch: DDP={ddp_losses[step]}, {label}={fsdp_losses[step]}",
        )

    for step in range(len(ddp_grad_norms)):
        torch.testing.assert_close(
            torch.tensor(ddp_grad_norms[step]),
            torch.tensor(fsdp_grad_norms[step]),
            rtol=1e-5,
            atol=1e-5,
            msg=f"Step {step} grad norm mismatch: DDP={ddp_grad_norms[step]}, {label}={fsdp_grad_norms[step]}",
        )

    for key in ddp_state_dict:
        assert key in fsdp_state_dict, f"Key {key} missing from {label} state dict"
        torch.testing.assert_close(
            ddp_state_dict[key],
            fsdp_state_dict[key],
            rtol=1e-5,
            atol=1e-5,
            msg=f"Weight mismatch for {key}",
        )


# =============================================================================
# Distributed test implementations (top-level for pickling by mp.spawn)
# =============================================================================


def _test_fsdp2_sharding_structure_impl(rank, config_class, config_dict, tie_word_embeddings):
    """
    Verify that apply_fsdp2(fsdp_plan="auto") wraps exactly the right modules.

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

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")

    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)

    block_classes = get_transformer_block_classes(model)
    assert block_classes, "get_transformer_block_classes found no block classes"

    decoder_layer_names = {
        name for name, module in model.named_modules() if type(module) in block_classes
    }
    assert len(decoder_layer_names) == config.num_hidden_layers

    id_to_name = {id(module): name for name, module in model.named_modules()}

    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    final_norm = _find_final_norm(model, decoder_layer_names)
    weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )

    embed_name = id_to_name.get(id(input_embed))
    output_name = id_to_name.get(id(output_embed))
    norm_name = id_to_name.get(id(final_norm))

    expected_targets = (
        {""}
        | decoder_layer_names
        | {embed_name}
        | {norm_name}
    )
    if not weights_tied:
        expected_targets |= {output_name}

    model = apply_fsdp2(model, device_mesh, fsdp_plan="auto")

    actual_targets = {
        name for name, module in model.named_modules()
        if type(module).__name__.startswith("FSDP")
    }

    if rank == 0:
        logger.info(f"  Weights tied: {weights_tied}")
        logger.info(f"  Expected FSDP targets: {sorted(expected_targets)}")
        logger.info(f"  Actual FSDP targets:   {sorted(actual_targets)}")

    missing = expected_targets - actual_targets
    extra = actual_targets - expected_targets
    assert not missing and not extra, (
        f"FSDP target mismatch.\n"
        f"  Missing (expected but not wrapped): {sorted(missing)}\n"
        f"  Extra (wrapped but not expected):   {sorted(extra)}"
    )

    if rank == 0:
        logger.info(f"  FSDP sharding structure OK ({len(actual_targets)} targets)")


def _test_fsdp2_auto_plan_vs_ddp_impl(rank, config_class, config_dict, dtype, tie_word_embeddings):
    """Compare losses, grad norms, and final weights between DDP and FSDP2."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    ddp_losses, ddp_grad_norms, ddp_state_dict = _train_ddp(rank, config, batches, LR, device, dtype)

    dist.barrier()

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = _train_fsdp2(
        rank, config, batches, LR, device_map, device_mesh, dtype, fsdp_plan="auto",
    )

    dist.barrier()

    if rank == 0:
        logger.info("")
        _log_comparison_table("Loss per step", ddp_losses, fsdp_losses)
        logger.info("")
        _log_comparison_table("Gradient norm per step", ddp_grad_norms, fsdp_grad_norms)
        logger.info("")

    _assert_ddp_fsdp_match(ddp_losses, ddp_grad_norms, ddp_state_dict, fsdp_losses, fsdp_grad_norms, fsdp_state_dict)


def _test_fsdp2_manual_plan_vs_ddp_impl(rank, config_class, config_dict, dtype, tie_word_embeddings):
    """Compare DDP vs FSDP2 with a per-sublayer manual plan (self_attn + mlp buckets)."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    ddp_losses, ddp_grad_norms, ddp_state_dict = _train_ddp(rank, config, batches, LR, device, dtype)

    dist.barrier()

    # Build manual plan by discovering the sub-modules of each decoder layer.
    set_seed(SEED)
    probe_model = AutoModelForCausalLM.from_config(config).to(device)
    block_classes = get_transformer_block_classes(probe_model)

    fsdp_plan = {}
    for name, module in probe_model.named_modules():
        # Embed tokens
        if module is probe_model.get_input_embeddings():
            fsdp_plan[name] = "free_full_weight"
        # Each direct child of a decoder layer gets its own shard unit
        parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
        parent_module = dict(probe_model.named_modules()).get(parent_name)
        if parent_module is not None and type(parent_module) in block_classes and name != parent_name:
            # Only direct children (one level deeper)
            depth_diff = len(name.split(".")) - len(parent_name.split("."))
            if depth_diff == 1:
                fsdp_plan[name] = "free_full_weight"

    # Final norm + optional lm_head
    decoder_layer_names = {n for n, m in probe_model.named_modules() if type(m) in block_classes}
    final_norm = _find_final_norm(probe_model, decoder_layer_names)
    if final_norm is not None:
        norm_name = {id(m): n for n, m in probe_model.named_modules()}.get(id(final_norm))
        if norm_name:
            fsdp_plan[norm_name] = "keep_full_weight"
    if not tie_word_embeddings:
        output_embed = probe_model.get_output_embeddings()
        if output_embed is not None:
            out_name = {id(m): n for n, m in probe_model.named_modules()}.get(id(output_embed))
            if out_name:
                fsdp_plan[out_name] = "keep_full_weight"
    del probe_model

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = _train_fsdp2(
        rank, config, batches, LR, device_map, device_mesh, dtype, fsdp_plan=fsdp_plan,
    )

    dist.barrier()

    if rank == 0:
        logger.info("")
        _log_comparison_table("Loss per step (manual plan)", ddp_losses, fsdp_losses)
        logger.info("")
        _log_comparison_table("Gradient norm per step (manual plan)", ddp_grad_norms, fsdp_grad_norms)
        logger.info("")

    _assert_ddp_fsdp_match(
        ddp_losses, ddp_grad_norms, ddp_state_dict,
        fsdp_losses, fsdp_grad_norms, fsdp_state_dict,
        label="FSDP2(manual)",
    )


def _test_fsdp2_save_load_impl(rank, config_class, config_dict):
    """Train FSDP2 model, save via DCP, load into fresh model, compare state dicts."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")

    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)
    model = apply_fsdp2(model, device_mesh, fsdp_plan="auto")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()

    state_dict_before = _gather_fsdp2_state_dict(model)

    if rank == 0:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        tmpdir_list = [tmpdir]
    else:
        tmpdir_list = [None]
    dist.broadcast_object_list(tmpdir_list, src=0)
    tmpdir = tmpdir_list[0]

    try:
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        dcp.save({"model": model_state_dict, "optim": optimizer_state_dict}, checkpoint_id=tmpdir)
        dist.barrier()

        set_seed(SEED)
        new_model = AutoModelForCausalLM.from_config(config).to(device_map)
        new_model = apply_fsdp2(new_model, device_mesh, fsdp_plan="auto")
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=LR)

        new_model_state_dict, new_optimizer_state_dict = get_state_dict(new_model, new_optimizer)
        dcp.load({"model": new_model_state_dict, "optim": new_optimizer_state_dict}, checkpoint_id=tmpdir)
        set_state_dict(new_model, new_optimizer, model_state_dict=new_model_state_dict, optim_state_dict=new_optimizer_state_dict)
        dist.barrier()
    finally:
        if rank == 0:
            tmpdir_obj.cleanup()

    state_dict_after = _gather_fsdp2_state_dict(new_model)

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
        logger.info(f"FSDP2 save/load test passed: all {len(state_dict_before)} parameters match exactly.")


# =============================================================================
# Mixin class
# =============================================================================


class FSDPTesterMixin(ABC):
    """
    Mixin for FSDP-related tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - causal_lm_class, base_model_class, etc.

    This mixin adds FSDP-specific tests using that infrastructure.

    Tests included:
      - test_get_transformer_block_classes: CPU-only, meta device (no GPU needed)
      - test_fsdp2_sharding_structure_untied: 2 GPUs, verifies FSDP wrapping structure
      - test_fsdp2_sharding_structure_tied: 2 GPUs, same but with tied embeddings
      - test_fsdp2_auto_plan_vs_ddp: 2 GPUs, compares DDP vs FSDP2 training
      - test_fsdp2_manual_plan_vs_ddp: 2 GPUs, compares DDP vs FSDP2 with manual plan
      - test_fsdp2_save_load: 2 GPUs, save/load checkpoint round-trip
    """

    fsdp_nproc_per_node: int = 2

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

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

    def _skip_if_insufficient_devices(self):
        if backend_device_count(torch_device) < self.fsdp_nproc_per_node:
            self.skipTest(
                f"Need at least {self.fsdp_nproc_per_node} devices, "
                f"have {backend_device_count(torch_device)}"
            )

    def _get_config_for_fsdp(self):
        """Get config class and serialized dict for passing to spawned processes."""
        config = self.model_tester.get_config()
        return type(config), config.to_dict()

    # =========================================================================
    # Test: get_transformer_block_classes (CPU, meta device)
    # =========================================================================

    def test_get_transformer_block_classes(self):
        """get_transformer_block_classes() finds >= 1 block class for the model."""
        config = self.model_tester.get_config()
        model = self._create_model_on_meta(config)

        block_classes = get_transformer_block_classes(model)
        self.assertTrue(len(block_classes) > 0, f"No block classes found for {type(config).__name__}")

        for cls in block_classes:
            count = sum(1 for m in model.modules() if type(m) is cls)
            self.assertGreater(count, 0, f"Block class {cls.__name__} has no instances in model")

    # =========================================================================
    # Test: FSDP2 sharding structure (2 GPUs)
    # =========================================================================

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_sharding_structure_untied(self):
        """Verify FSDP sharding structure with untied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_sharding_structure_impl
        )(config_class, config_dict, False)

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_sharding_structure_tied(self):
        """Verify FSDP sharding structure with tied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_sharding_structure_impl
        )(config_class, config_dict, True)

    # =========================================================================
    # Test: FSDP2 auto plan vs DDP (2 GPUs)
    # =========================================================================

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_auto_plan_vs_ddp_float32_untied(self):
        """DDP vs FSDP2 auto plan: float32, untied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_auto_plan_vs_ddp_impl
        )(config_class, config_dict, torch.float32, False)

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_auto_plan_vs_ddp_bfloat16_untied(self):
        """DDP vs FSDP2 auto plan: bfloat16, untied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_auto_plan_vs_ddp_impl
        )(config_class, config_dict, torch.bfloat16, False)

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_auto_plan_vs_ddp_float32_tied(self):
        """DDP vs FSDP2 auto plan: float32, tied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_auto_plan_vs_ddp_impl
        )(config_class, config_dict, torch.float32, True)

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_auto_plan_vs_ddp_bfloat16_tied(self):
        """DDP vs FSDP2 auto plan: bfloat16, tied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_auto_plan_vs_ddp_impl
        )(config_class, config_dict, torch.bfloat16, True)

    # =========================================================================
    # Test: FSDP2 manual plan vs DDP (2 GPUs)
    # =========================================================================

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_manual_plan_vs_ddp_float32_untied(self):
        """DDP vs FSDP2 manual plan: float32, untied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_manual_plan_vs_ddp_impl
        )(config_class, config_dict, torch.float32, False)

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_manual_plan_vs_ddp_bfloat16_untied(self):
        """DDP vs FSDP2 manual plan: bfloat16, untied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_manual_plan_vs_ddp_impl
        )(config_class, config_dict, torch.bfloat16, False)

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_manual_plan_vs_ddp_float32_tied(self):
        """DDP vs FSDP2 manual plan: float32, tied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_manual_plan_vs_ddp_impl
        )(config_class, config_dict, torch.float32, True)

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_manual_plan_vs_ddp_bfloat16_tied(self):
        """DDP vs FSDP2 manual plan: bfloat16, tied embeddings."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_manual_plan_vs_ddp_impl
        )(config_class, config_dict, torch.bfloat16, True)

    # =========================================================================
    # Test: FSDP2 save/load checkpoint (2 GPUs)
    # =========================================================================

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_save_load(self):
        """Save FSDP2 checkpoint via DCP, load into fresh model, verify exact match."""
        self._skip_if_insufficient_devices()
        config_class, config_dict = self._get_config_for_fsdp()
        _fsdp_init_distributed(world_size=self.fsdp_nproc_per_node)(
            _test_fsdp2_save_load_impl
        )(config_class, config_dict)

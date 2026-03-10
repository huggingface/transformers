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
import traceback
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, is_torch_available
from transformers.testing_utils import (
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
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.tensor import DTensor
    from torch.nn.parallel import DistributedDataParallel as DDP

    from transformers.integrations.fsdp import (
        _find_final_norm,
        apply_fsdp2,
        get_transformer_block_classes,
        initialize_fsdp,
    )


# =============================================================================
# Constants
# =============================================================================

BATCH_SIZE = 2
SEQ_LEN = 64
NUM_STEPS = 20
LR = 3e-4
MIXED_PRECISION_LR = 1e-3
SEED = 42
MIXED_PRECISION_NUM_STEPS = 40
MIXED_PRECISION_REDUCTION_THRESHOLD = 0.4
MIXED_PRECISION_GENERATION_MAX_NEW_TOKENS = 31
MIXED_PRECISION_GENERATION_MATCH_THRESHOLD = 0.8


# =============================================================================
# Distributed helpers (top-level for pickling by mp.spawn)
# =============================================================================


def _fsdp_global_wrapper_batched(rank, test_specs, world_size, port, results_file):
    """Set up distributed environment once and run multiple test functions sequentially.

    This avoids the mp.spawn + NCCL/GLOO init overhead per test by initializing the
    process group once and running all tests within the same spawned processes.
    """
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    _set_determinism(SEED)

    # NOTE(3outeille): will have to do everything through gloo
    # Initialize a dual-backend default group so CUDA tensors use NCCL and CPU
    # tensors (needed by cpu_offload paths) use GLOO.
    try:
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    except Exception as e:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0:
            logger.warning(
                "Falling back to NCCL-only process group init; cpu_offload tests may be skipped. "
                f"Original init error: {e}"
            )
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    results = {}
    for test_name, func, func_args, func_kwargs in test_specs:
        error = None
        try:
            func(rank, *func_args, **func_kwargs)
        except Exception as e:
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        error_flag = torch.tensor([1 if error else 0], device=f"cuda:{rank}")
        dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
        any_failed = error_flag.item() > 0

        if any_failed:
            results[test_name] = error if error else "Failed on another rank"
            if rank == 0:
                print(f"  [FAIL] {test_name}", file=sys.stderr, flush=True)
        else:
            results[test_name] = None

        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        with open(results_file, "w") as f:
            json.dump(results, f)

    dist.barrier()
    dist.destroy_process_group()


def _set_determinism(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
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


def _gather_fsdp2_state_dict(model):
    """Gather FSDP2 sharded parameters into full tensors via DTensor.full_tensor()."""
    state_dict = {}
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, DTensor):
            state_dict[name] = tensor.full_tensor().clone().detach().cpu()
        else:
            state_dict[name] = tensor.clone().detach().cpu()
    return state_dict


def _gather_ddp_state_dict(model):
    return {k: v.clone().detach().cpu() for k, v in model.module.state_dict().items()}


def _build_manual_fsdp_plan(config, device, policy_options=None):
    """Build a default manual FSDP2 plan from model structure."""
    policy_options = policy_options or []
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device)
    named_modules = dict(model.named_modules())
    id_to_name = {id(module): name for name, module in named_modules.items()}
    block_classes = get_transformer_block_classes(model)

    decoder_layer_names = {name for name, module in named_modules.items() if type(module) in block_classes}
    layer_prefixes = {".".join(name.split(".")[:-1]) for name in decoder_layer_names}
    assert layer_prefixes, "Expected at least one decoder layer prefix for manual FSDP plan."

    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )
    embed_name = id_to_name.get(id(input_embed)) if input_embed is not None else None
    output_name = id_to_name.get(id(output_embed)) if output_embed is not None else None
    final_norm = _find_final_norm(model, decoder_layer_names)
    norm_name = id_to_name.get(id(final_norm)) if final_norm is not None else None

    module_plan = {name: ["free_full_weight", *policy_options] for name in layer_prefixes}

    if norm_name:
        module_plan[norm_name] = ["keep_full_weight", *policy_options]

    if weights_tied:
        if embed_name:
            module_plan[embed_name] = ["keep_full_weight", *policy_options]
    else:
        if embed_name:
            module_plan[embed_name] = ["free_full_weight", *policy_options]
        if output_name:
            module_plan[output_name] = ["keep_full_weight", *policy_options]

    del model
    return {"mode": "manual", "modules": module_plan}


def _create_init_state_dict(config, dtype):
    """Create a deterministic initial state dict for reproducible model initialization."""
    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(dtype)
    state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
    del model
    return state_dict


def _save_checkpoint(model, optimizer, tmpdir):
    """Save model, optimizer, and RNG states to a distributed checkpoint."""
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save(
        {
            "model": model_sd,
            "optim": optim_sd,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
        },
        checkpoint_id=tmpdir,
    )


def _load_checkpoint(model, optimizer, tmpdir):
    """Load model, optimizer, and RNG states from a distributed checkpoint."""
    model_sd, optim_sd = get_state_dict(model, optimizer)
    loaded_state = {
        "model": model_sd,
        "optim": optim_sd,
        "cpu_rng_state": torch.empty_like(torch.get_rng_state()),
        "cuda_rng_state": torch.empty_like(torch.cuda.get_rng_state()),
    }
    # MoE models can have sparse optimizer state (experts not selected yet), so
    # allow partial optimizer key restoration instead of failing hard on missing keys.
    dcp.load(loaded_state, checkpoint_id=tmpdir, planner=DefaultLoadPlanner(allow_partial_load=True))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=loaded_state["model"],
        optim_state_dict=loaded_state["optim"],
    )
    torch.set_rng_state(loaded_state["cpu_rng_state"])
    torch.cuda.set_rng_state(loaded_state["cuda_rng_state"])


def train_ddp(rank, config, batches, lr, device, dtype, init_state_dict):
    _set_determinism(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    model.load_state_dict(init_state_dict)
    # MoE/conditional-routing variants) may not use all params on
    # every step, and DDP would otherwise fail. Specifying find_unused_parameters=True allows running backward on a subgraph of the model.
    ddp_model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
    ).to(dtype)
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
    torch.cuda.empty_cache()
    dist.barrier()

    return losses, grad_norms, state_dict


def train_fsdp2(
    rank, config, batches, lr, device_map, device_mesh, dtype, init_state_dict, checkpoint_step, fsdp_plan
):
    # -- Phase 1: Pre-checkpoint run -- train only the first `checkpoint_step` steps, then save
    _set_determinism(SEED)
    pre_ckpt_model = AutoModelForCausalLM.from_config(config).to(device_map).to(dtype)
    pre_ckpt_model.load_state_dict(init_state_dict)
    pre_ckpt_model = apply_fsdp2(pre_ckpt_model, device_mesh, fsdp_plan=fsdp_plan)
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
    tmpdir, tmpdir_obj = _create_shared_tmpdir(rank)
    try:
        _save_checkpoint(pre_ckpt_model, pre_ckpt_optimizer, tmpdir)
        dist.barrier()

        # Intentionally scramble RNG to prove checkpoint restore works
        _set_determinism(SEED + 1234)
        resumed_model = AutoModelForCausalLM.from_config(config).to(device_map).to(dtype)
        resumed_model = apply_fsdp2(resumed_model, device_mesh, fsdp_plan=fsdp_plan)
        resumed_model.train()
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=lr)

        _load_checkpoint(resumed_model, resumed_optimizer, tmpdir)
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
    combined_state_dict = _gather_fsdp2_state_dict(resumed_model)

    return combined_losses, combined_grad_norms, combined_state_dict


# =============================================================================
# Distributed test implementations (top-level for pickling by mp.spawn)
# =============================================================================
def _test_fsdp2_save_load_impl(rank, config_class, config_dict):
    """Train FSDP2 model, save via DCP, load into fresh model, compare state dicts."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)

    batches = _build_repeated_training_batches(config, device, NUM_STEPS)

    auto_plan = {"mode": "auto"}
    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=auto_plan)

    _set_determinism(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)
    model = apply_fsdp2(model, device_mesh, fsdp_plan=auto_plan)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for input_ids, labels in batches:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        output.loss.backward()
        optimizer.step()

    state_dict_before = _gather_fsdp2_state_dict(model)

    tmpdir, tmpdir_obj = _create_shared_tmpdir(rank)
    try:
        _save_checkpoint(model, optimizer, tmpdir)
        dist.barrier()

        _set_determinism(SEED + 1234)
        new_model = AutoModelForCausalLM.from_config(config).to(device_map)
        new_model = apply_fsdp2(new_model, device_mesh, fsdp_plan=auto_plan)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=LR)

        _load_checkpoint(new_model, new_optimizer, tmpdir)
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
        logger.debug(f"FSDP2 save/load test passed: all {len(state_dict_before)} parameters match exactly.")


def _test_fsdp2_sharding_structure_impl(rank, config_class, config_dict, tie_word_embeddings):
    """
    Verify that apply_fsdp2(fsdp_plan={"mode": "auto"}) wraps exactly the right modules.

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

    auto_plan = {"mode": "auto"}
    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=auto_plan)

    set_seed(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)

    block_classes = get_transformer_block_classes(model)
    assert block_classes, "get_transformer_block_classes found no block classes"

    decoder_layer_names = {name for name, module in model.named_modules() if type(module) in block_classes}
    assert len(decoder_layer_names) > 0, "Expected at least one transformer block instance"

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

    expected_targets = {""} | decoder_layer_names | {embed_name} | {norm_name}
    if not weights_tied:
        expected_targets |= {output_name}

    model = apply_fsdp2(model, device_mesh, fsdp_plan=auto_plan)

    actual_targets = {name for name, module in model.named_modules() if type(module).__name__.startswith("FSDP")}

    if rank == 0:
        logger.debug(f"  Weights tied: {weights_tied}")
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


def _test_fsdp2_plan_vs_ddp_impl(
    rank, config_class, config_dict, tie_word_embeddings, plan_mode, policy_options=None, dtype=None
):
    """Validate DDP-vs-FSDP2 trace matching for either auto or manual plan mode."""
    init_test_logger()

    if dtype is None:
        dtype = torch.float32

    policy_options = policy_options or []
    assert "mixed_precision" not in policy_options, (
        "Use the mixed-precision specific tests when enabling mixed_precision policy."
    )

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings

    if plan_mode == "auto":
        fsdp_plan = {
            "mode": "auto",
            "cpu_offload": "cpu_offload" in policy_options,
            "mixed_precision": "mixed_precision" in policy_options,
        }
        test_label = f"FSDP2(auto{'+policies' if policy_options else ''})"
    elif plan_mode == "manual":
        fsdp_plan = _build_manual_fsdp_plan(config, device, policy_options=policy_options)
        test_label = f"FSDP2(manual{'+policies' if policy_options else ''})"
    else:
        raise ValueError(f"Unsupported plan_mode '{plan_mode}'. Expected 'auto' or 'manual'.")

    checkpoint_step = NUM_STEPS // 2
    init_state_dict = _create_init_state_dict(config, dtype)
    batches = _build_repeated_training_batches(config, device, NUM_STEPS)
    ddp_losses, ddp_grad_norms, ddp_state_dict = train_ddp(rank, config, batches, LR, device, dtype, init_state_dict)

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = train_fsdp2(
        rank,
        config,
        batches,
        LR,
        device_map,
        device_mesh,
        dtype,
        init_state_dict=init_state_dict,
        checkpoint_step=checkpoint_step,
        fsdp_plan=fsdp_plan,
    )

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


def _test_fsdp2_plan_cpu_offload_mixed_precision(
    rank, config_class, config_dict, tie_word_embeddings, plan_mode, policy_options=None
):
    """Validate mixed-precision FSDP2 behavior for either auto or manual plan mode."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings
    num_steps = MIXED_PRECISION_NUM_STEPS
    checkpoint_step = num_steps // 2
    lr = MIXED_PRECISION_LR
    policy_options = policy_options or []
    assert "mixed_precision" in policy_options, "Mixed-precision test requires mixed_precision policy option."

    dtype = torch.float32
    init_state_dict = _create_init_state_dict(config, dtype)
    batches = _build_repeated_training_batches(config, device, num_steps)

    if plan_mode == "auto":
        fsdp_plan = {
            "mode": "auto",
            "cpu_offload": "cpu_offload" in policy_options,
            "mixed_precision": "mixed_precision" in policy_options,
        }
        label = "FSDP2(auto + cpu-offload + mixed-precision)"
    elif plan_mode == "manual":
        fsdp_plan = _build_manual_fsdp_plan(config, device, policy_options=policy_options)
        label = "FSDP2(manual + cpu-offload + mixed-precision"
    else:
        raise ValueError(f"Unsupported plan_mode '{plan_mode}'. Expected 'auto' or 'manual'.")

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = train_fsdp2(
        rank,
        config,
        batches,
        lr,
        device_map,
        device_mesh,
        dtype,
        init_state_dict=init_state_dict,
        checkpoint_step=checkpoint_step,
        fsdp_plan=fsdp_plan,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.debug("%s per-step trace:", label)
        for step, (loss, grad_norm) in enumerate(zip(fsdp_losses, fsdp_grad_norms)):
            logger.debug("  step=%d loss=%.6f grad_norm=%.6f", step, loss, grad_norm)

    # Assert loss reduction
    initial_loss, final_loss = fsdp_losses[0], fsdp_losses[-1]
    loss_reduction_ratio = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.debug(
            "%s reduction summary: loss %.6f -> %.6f (%.2f%%)",
            label,
            initial_loss,
            final_loss,
            loss_reduction_ratio * 100.0,
        )
    assert loss_reduction_ratio >= MIXED_PRECISION_REDUCTION_THRESHOLD, (
        f"{label}: expected loss reduction >= {MIXED_PRECISION_REDUCTION_THRESHOLD:.0%}, "
        f"got {loss_reduction_ratio:.2%}"
    )

    # Assert generation matches training pattern
    model = AutoModelForCausalLM.from_config(config).to(device)
    model.load_state_dict(fsdp_state_dict)
    model.eval()

    input_ids, _ = batches[0]
    expected_tokens = input_ids[0].tolist()
    prompt = torch.tensor([[expected_tokens[0]]], device=device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            max_new_tokens=MIXED_PRECISION_GENERATION_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=config.pad_token_id if hasattr(config, "pad_token_id") else 0,
            eos_token_id=0,
            use_cache=getattr(config, "model_type", "") == "recurrent_gemma",
        )[0].tolist()

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.debug("Generation prompt tokens: %s", prompt[0].tolist())
        logger.debug("Generation expected tokens: %s", expected_tokens[: len(generated)])
        logger.debug("Generation generated tokens: %s", generated)

    expected_slice = expected_tokens[: len(generated)]
    num_matches = sum(int(a == b) for a, b in zip(generated, expected_slice))
    match_ratio = num_matches / max(len(expected_slice), 1)
    assert match_ratio >= MIXED_PRECISION_GENERATION_MATCH_THRESHOLD, (
        "Expected generated sequence to match at least "
        f"{MIXED_PRECISION_GENERATION_MATCH_THRESHOLD:.0%} of target tokens after overfitting; "
        f"got {match_ratio:.2%}"
    )
    if rank == 0:
        logger.debug(f"{label} reduction + generation checks passed.")


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
        if backend_device_count(torch_device) < self.fsdp_nproc_per_node:
            self.skipTest(
                f"Need at least {self.fsdp_nproc_per_node} devices, have {backend_device_count(torch_device)}"
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
        return type(config), config.to_diff_dict()

    def _build_fsdp2_subtest_specs(self, config_class, config_dict):
        return [
            ("sharding_structure_untied", _test_fsdp2_sharding_structure_impl, (config_class, config_dict, False), {}),
            ("sharding_structure_tied", _test_fsdp2_sharding_structure_impl, (config_class, config_dict, True), {}),
            ("save_load", _test_fsdp2_save_load_impl, (config_class, config_dict), {}),
            ("auto_plan_untied", _test_fsdp2_plan_vs_ddp_impl, (config_class, config_dict, False, "auto"), {}),
            ("auto_plan_tied", _test_fsdp2_plan_vs_ddp_impl, (config_class, config_dict, True, "auto"), {}),
            ("manual_plan_untied", _test_fsdp2_plan_vs_ddp_impl, (config_class, config_dict, False, "manual"), {}),
            ("manual_plan_tied", _test_fsdp2_plan_vs_ddp_impl, (config_class, config_dict, True, "manual"), {}),
            (
                "auto_plan_untied_cpu_offload_mixed_precision",
                _test_fsdp2_plan_cpu_offload_mixed_precision,
                (config_class, config_dict, False, "auto", ["cpu_offload", "mixed_precision"]),
                {},
            ),
            (
                "auto_plan_tied_cpu_offload_mixed_precision",
                _test_fsdp2_plan_cpu_offload_mixed_precision,
                (config_class, config_dict, True, "auto", ["cpu_offload", "mixed_precision"]),
                {},
            ),
            (
                "manual_plan_untied_cpu_offload_mixed_precision",
                _test_fsdp2_plan_cpu_offload_mixed_precision,
                (config_class, config_dict, False, "manual", ["cpu_offload", "mixed_precision"]),
                {},
            ),
            (
                "manual_plan_tied_cpu_offload_mixed_precision",
                _test_fsdp2_plan_cpu_offload_mixed_precision,
                (config_class, config_dict, True, "manual", ["cpu_offload", "mixed_precision"]),
                {},
            ),
        ]

    def _format_fsdp2_subtest_summary(self, results):
        passed = [name for name, err in results.items() if err is None]
        failed = [name for name, err in results.items() if err is not None]

        summary_lines = [
            "",
            "=" * 60,
            f"  test_fsdp2_all: {len(passed)} passed, {len(failed)} failed (out of {len(results)} subtests)",
            "=" * 60,
        ]
        for name in results:
            is_passed = results[name] is None
            summary_lines.append(f"  {'[PASS]' if is_passed else '[FAIL]'} {name}")
        summary_lines.append("=" * 60)
        return "\n".join(summary_lines)

    # =========================================================================
    # Test: get_transformer_block_classes (CPU, meta device)
    # =========================================================================

    def test_get_transformer_block_classes(self):
        """get_transformer_block_classes() finds >= 1 block class for the model."""
        self._skip_if_fsdp_disabled()
        config = self.model_tester.get_config()
        model = self._create_model_on_meta(config)

        block_classes = get_transformer_block_classes(model)
        self.assertTrue(len(block_classes) > 0, f"No block classes found for {type(config).__name__}")

        for cls in block_classes:
            count = sum(1 for m in model.modules() if type(m) is cls)
            self.assertGreater(count, 0, f"Block class {cls.__name__} has no instances in model")

    # =========================================================================
    # Batched test: all distributed FSDP2 tests in a single mp.spawn
    # =========================================================================

    @require_fsdp
    @require_torch_multi_accelerator
    def test_fsdp2_all(self):
        """Run all distributed FSDP2 tests in a single process group spawn.
        This amortizes the mp.spawn + NCCL init overhead across all
        distributed tests instead of paying it once per test method.
        """
        self._skip_if_insufficient_devices()

        config_class, config_dict = self._get_tiny_config()
        test_specs = self._build_fsdp2_subtest_specs(config_class, config_dict)

        results_file = tempfile.mktemp(suffix=".json")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        try:
            mp.spawn(
                _fsdp_global_wrapper_batched,
                args=(test_specs, self.fsdp_nproc_per_node, port, results_file),
                nprocs=self.fsdp_nproc_per_node,
            )

            with open(results_file) as f:
                results = json.load(f)
        finally:
            if os.path.exists(results_file):
                os.unlink(results_file)

        print(self._format_fsdp2_subtest_summary(results), flush=True)

        for test_name, error in results.items():
            with self.subTest(test_name=test_name):
                if error is not None:
                    self.fail(f"FSDP subtest '{test_name}' failed:\n{error}")

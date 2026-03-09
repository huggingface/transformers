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

    #NOTE(3outeille): will have to do everything through gloo
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
    total = len(test_specs)
    for idx, (test_name, func, func_args, func_kwargs) in enumerate(test_specs, 1):
        if rank == 0:
            print(f"  [{idx}/{total}] Running: {test_name} ...", file=sys.stderr, flush=True)

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
                print(f"  [{idx}/{total}] FAILED:  {test_name}", file=sys.stderr, flush=True)
        else:
            results[test_name] = None
            if rank == 0:
                print(f"  [{idx}/{total}] PASSED:  {test_name}", file=sys.stderr, flush=True)

        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        with open(results_file, "w") as f:
            json.dump(results, f)

    dist.barrier()
    dist.destroy_process_group()


def _get_free_port():
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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


def _create_deterministic_data(batch_size, seq_len, vocab_size, device, seed):
    """Create deterministic random training data using torch.randint."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = input_ids.clone()
    return [(input_ids, labels)]


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


def _run_training_steps(model, optimizer, batches, start_step, end_step):
    losses = []
    grad_norms = []

    for step in range(start_step, end_step):
        input_ids, labels = batches[step]
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = output.loss
        loss.backward()
        grad_norm = _compute_grad_norm(model)
        optimizer.step()

        losses.append(loss.detach().item())
        grad_norms.append(grad_norm)

    return losses, grad_norms


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


def _compute_grad_norm(model):
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            total_norm_sq += grad.data.float().norm(2).item() ** 2
    return total_norm_sq**0.5


def _assert_training_trace_match(
    ref_losses,
    ref_grad_norms,
    ref_state_dict,
    test_losses,
    test_grad_norms,
    test_state_dict,
    ref_label,
    test_label,
    rtol=1e-5,
    atol=1e-5,
):
    for step in range(len(ref_losses)):
        torch.testing.assert_close(
            torch.tensor(ref_losses[step]),
            torch.tensor(test_losses[step]),
            rtol=rtol,
            atol=atol,
            msg=f"Loss mismatch at step {step}: {ref_label}={ref_losses[step]}, {test_label}={test_losses[step]}",
        )
        torch.testing.assert_close(
            torch.tensor(ref_grad_norms[step]),
            torch.tensor(test_grad_norms[step]),
            rtol=rtol,
            atol=atol,
            msg=f"Grad norm mismatch at step {step}: {ref_label}={ref_grad_norms[step]}, {test_label}={test_grad_norms[step]}",
        )

    for key in ref_state_dict:
        assert key in test_state_dict, f"Key {key} missing from {test_label} state dict"
        torch.testing.assert_close(
            ref_state_dict[key],
            test_state_dict[key],
            rtol=rtol,
            atol=atol,
            msg=f"Weight mismatch for {key}: {ref_label} vs {test_label}",
        )


def _assert_training_reduction(losses, label, reduction_threshold=0.7):
    assert len(losses) > 1, f"{label}: expected at least 2 loss values"

    initial_loss, final_loss = losses[0], losses[-1]

    loss_reduction_ratio = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            "%s reduction summary: loss %.6f -> %.6f (%.2f%%)",
            label,
            initial_loss,
            final_loss,
            loss_reduction_ratio * 100.0,
        )

    assert (
        loss_reduction_ratio >= reduction_threshold
    ), f"{label}: expected loss reduction >= {reduction_threshold:.0%}, got {loss_reduction_ratio:.2%}"


def _log_training_trace(losses, grad_norms, label):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    logger.info("%s per-step trace:", label)
    for step, (loss, grad_norm) in enumerate(zip(losses, grad_norms)):
        logger.info("  step=%d loss=%.6f grad_norm=%.6f", step, loss, grad_norm)


def _assert_generation_matches_training_pattern(config, state_dict, batch, device, max_new_tokens_limit):
    """
    Validate generation by checking the model can reproduce the repeated training sequence
    from a short prompt, similar to test_training_overfit.
    """
    model = AutoModelForCausalLM.from_config(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    input_ids, _ = batch
    expected_tokens = input_ids[0].tolist()
    prompt = torch.tensor([[expected_tokens[0]]], device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            max_new_tokens=max_new_tokens_limit,
            do_sample=False,
            pad_token_id=config.pad_token_id if hasattr(config, "pad_token_id") else 0,
            eos_token_id=0,
            use_cache=getattr(config, "model_type", "") == "recurrent_gemma",
        )[0].tolist()

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Generation prompt tokens: %s", prompt[0].tolist())
        logger.info("Generation expected tokens: %s", expected_tokens[: len(generated)])
        logger.info("Generation generated tokens: %s", generated)

    expected_slice = expected_tokens[: len(generated)]
    num_matches = sum(int(a == b) for a, b in zip(generated, expected_slice))
    match_ratio = num_matches / max(len(expected_slice), 1)
    assert match_ratio >= MIXED_PRECISION_GENERATION_MATCH_THRESHOLD, (
        "Expected generated sequence to match at least "
        f"{MIXED_PRECISION_GENERATION_MATCH_THRESHOLD:.0%} of target tokens after overfitting; "
        f"got {match_ratio:.2%}"
    )


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


def _run_ddp_checkpoint_repro(rank, config, batches, lr, device, dtype, init_state_dict, checkpoint_step):
    # -- Phase 1: Reference run -- train all steps without interruption to get ground truth
    _set_determinism(SEED)
    ref_model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    ref_model.load_state_dict(init_state_dict)
    # MoE/conditional-routing variants) may not use all params on
    # every step, and DDP would otherwise fail. Specifying find_unused_parameters=True allows running backward on a subgraph of the model.
    ref_ddp = DDP(
        ref_model,
        device_ids=[rank],
        find_unused_parameters=True,
    ).to(dtype)
    ref_ddp.train()
    ref_optimizer = torch.optim.Adam(ref_ddp.parameters(), lr=lr)
    ref_losses, ref_grad_norms = _run_training_steps(ref_ddp, ref_optimizer, batches, 0, len(batches))
    ref_state_dict = _gather_ddp_state_dict(ref_ddp)

    del ref_optimizer
    del ref_ddp
    del ref_model
    torch.cuda.empty_cache()
    dist.barrier()

    # -- Phase 2: Pre-checkpoint run -- train only the first `checkpoint_step` steps, then save
    _set_determinism(SEED)
    pre_ckpt_model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    pre_ckpt_model.load_state_dict(init_state_dict)
    pre_ckpt_ddp = DDP(
        pre_ckpt_model,
        device_ids=[rank],
        find_unused_parameters=True,
    ).to(dtype)
    pre_ckpt_ddp.train()
    pre_ckpt_optimizer = torch.optim.Adam(pre_ckpt_ddp.parameters(), lr=lr)
    pre_ckpt_losses, pre_ckpt_grad_norms = _run_training_steps(
        pre_ckpt_ddp, pre_ckpt_optimizer, batches, 0, checkpoint_step
    )

    # -- Phase 3: Save checkpoint, then load into a fresh model
    tmpdir, tmpdir_obj = _create_shared_tmpdir(rank)
    try:
        _save_checkpoint(pre_ckpt_ddp, pre_ckpt_optimizer, tmpdir)
        dist.barrier()

        # Intentionally scramble RNG to prove checkpoint restore works
        _set_determinism(SEED + 1234)
        resumed_model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
        resumed_ddp = DDP(
            resumed_model,
            device_ids=[rank],
            find_unused_parameters=True,
        ).to(dtype)
        resumed_ddp.train()
        resumed_optimizer = torch.optim.Adam(resumed_ddp.parameters(), lr=lr)

        _load_checkpoint(resumed_ddp, resumed_optimizer, tmpdir)
        dist.barrier()
    finally:
        if rank == 0:
            tmpdir_obj.cleanup()

    # -- Phase 4: Post-checkpoint run -- continue training the remaining steps from the resumed model
    post_ckpt_losses, post_ckpt_grad_norms = _run_training_steps(
        resumed_ddp,
        resumed_optimizer,
        batches,
        checkpoint_step,
        len(batches),
    )
    combined_losses = pre_ckpt_losses + post_ckpt_losses
    combined_grad_norms = pre_ckpt_grad_norms + post_ckpt_grad_norms
    combined_state_dict = _gather_ddp_state_dict(resumed_ddp)

    # -- Phase 5: Verify that the full uninterrupted run and the save/resume run are identical
    _assert_training_trace_match(
        ref_losses,
        ref_grad_norms,
        ref_state_dict,
        combined_losses,
        combined_grad_norms,
        combined_state_dict,
        ref_label="DDP(full)",
        test_label="DDP(resumed)",
    )
    return ref_losses, ref_grad_norms, ref_state_dict


def _run_fsdp2_checkpoint_repro(
    rank, config, batches, lr, device_map, device_mesh, dtype, init_state_dict, checkpoint_step, fsdp_plan
):
    # -- Phase 1: Reference run -- train all steps without interruption to get ground truth
    _set_determinism(SEED)
    ref_model = AutoModelForCausalLM.from_config(config).to(device_map).to(dtype)
    ref_model.load_state_dict(init_state_dict)
    ref_model = apply_fsdp2(ref_model, device_mesh, fsdp_plan=fsdp_plan)
    ref_model.train()
    ref_optimizer = torch.optim.Adam(ref_model.parameters(), lr=lr)
    ref_losses, ref_grad_norms = _run_training_steps(ref_model, ref_optimizer, batches, 0, len(batches))
    ref_state_dict = _gather_fsdp2_state_dict(ref_model)

    del ref_optimizer
    del ref_model
    torch.cuda.empty_cache()
    dist.barrier()

    # -- Phase 2: Pre-checkpoint run -- train only the first `checkpoint_step` steps, then save
    _set_determinism(SEED)
    pre_ckpt_model = AutoModelForCausalLM.from_config(config).to(device_map).to(dtype)
    pre_ckpt_model.load_state_dict(init_state_dict)
    pre_ckpt_model = apply_fsdp2(pre_ckpt_model, device_mesh, fsdp_plan=fsdp_plan)
    pre_ckpt_model.train()
    pre_ckpt_optimizer = torch.optim.Adam(pre_ckpt_model.parameters(), lr=lr)
    pre_ckpt_losses, pre_ckpt_grad_norms = _run_training_steps(
        pre_ckpt_model, pre_ckpt_optimizer, batches, 0, checkpoint_step
    )

    # -- Phase 3: Save checkpoint, then load into a fresh model
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

    # -- Phase 4: Post-checkpoint run -- continue training the remaining steps from the resumed model
    post_ckpt_losses, post_ckpt_grad_norms = _run_training_steps(
        resumed_model,
        resumed_optimizer,
        batches,
        checkpoint_step,
        len(batches),
    )
    combined_losses = pre_ckpt_losses + post_ckpt_losses
    combined_grad_norms = pre_ckpt_grad_norms + post_ckpt_grad_norms
    combined_state_dict = _gather_fsdp2_state_dict(resumed_model)

    # -- Phase 5: Verify that the full uninterrupted run and the save/resume run are identical
    _assert_training_trace_match(
        ref_losses,
        ref_grad_norms,
        ref_state_dict,
        combined_losses,
        combined_grad_norms,
        combined_state_dict,
        ref_label="FSDP2(full)",
        test_label="FSDP2(resumed)",
    )
    return ref_losses, ref_grad_norms, ref_state_dict


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

    model = apply_fsdp2(model, device_mesh, fsdp_plan="auto")

    actual_targets = {name for name, module in model.named_modules() if type(module).__name__.startswith("FSDP")}

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


def _test_fsdp2_auto_plan_vs_ddp_impl(
    rank, config_class, config_dict, tie_word_embeddings, policy_options=None, dtype=None
):
    """Validate checkpoint reproducibility for both DDP and FSDP2(auto) under same config.

    Optionally accepts policy_options (e.g. ["cpu_offload"]) that don't change
    numerics, so exact DDP trace matching still applies.
    """
    init_test_logger()

    if dtype is None:
        dtype = torch.float32

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings
    checkpoint_step = NUM_STEPS // 2
    policy_options = policy_options or []
    assert "mixed_precision" not in policy_options, "Use _test_fsdp2_auto_plan_mixed_precision_impl for mixed precision."

    fsdp_plan = ["auto", *policy_options] if policy_options else "auto"

    init_state_dict = _create_init_state_dict(config, dtype)

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    ddp_losses, ddp_grad_norms, ddp_state_dict = _run_ddp_checkpoint_repro(
        rank, config, batches, LR, device, dtype, init_state_dict, checkpoint_step
    )

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = _run_fsdp2_checkpoint_repro(
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

    _assert_training_trace_match(
        ddp_losses,
        ddp_grad_norms,
        ddp_state_dict,
        fsdp_losses,
        fsdp_grad_norms,
        fsdp_state_dict,
        ref_label="DDP",
        test_label=f"FSDP2(auto{'+policies' if policy_options else ''})",
    )

    if rank == 0:
        logger.info(f"DDP and FSDP2(auto{'+policies' if policy_options else ''}) checkpoint reproducibility checks passed.")


def _test_fsdp2_auto_plan_mixed_precision_impl(rank, config_class, config_dict, tie_word_embeddings, policy_options=None):
    """Validate FSDP2(auto) with mixed-precision/cpu_offload policies.

    Auto mode accepts a list starting with "auto" followed by policy strings:
      - ["auto", "cpu_offload"]          — auto with CPU offloading
      - ["auto", "mixed_precision"]      — auto with mixed precision (bf16 params, fp32 reduce/output)
      - ["auto", "cpu_offload", "mixed_precision"] — auto with both

    Since mixed precision introduces numerical differences vs DDP (bf16 vs fp32),
    we cannot do exact trace matching. Instead we verify:
      1. FSDP2(auto+policies) checkpoint reproducibility (full run == save/resume run)
      2. Training reduces loss meaningfully (reduction_threshold check)
      3. Generation from the trained model matches training patterns
    """
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings
    num_steps = MIXED_PRECISION_NUM_STEPS
    checkpoint_step = num_steps // 2
    lr = 1e-3
    policy_options = policy_options or []
    assert "mixed_precision" in policy_options, "Auto mixed-precision test requires mixed_precision policy option."

    dtype = torch.float32
    fsdp_plan = ["auto", *policy_options]

    init_state_dict = _create_init_state_dict(config, dtype)

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * num_steps

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = _run_fsdp2_checkpoint_repro(
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

    _log_training_trace(fsdp_losses, fsdp_grad_norms, label="FSDP2(auto+policies)")
    _assert_training_reduction(
        fsdp_losses,
        label="FSDP2(auto+policies)",
        reduction_threshold=MIXED_PRECISION_REDUCTION_THRESHOLD,
    )

    _assert_generation_matches_training_pattern(
        config,
        fsdp_state_dict,
        batches[0],
        device=device,
        max_new_tokens_limit=20,
    )

    if rank == 0:
        logger.info("FSDP2(auto mixed-precision) reduction + generation checks passed.")


def _test_fsdp2_manual_plan_vs_ddp_impl(rank, config_class, config_dict, tie_word_embeddings, policy_options=None):
    """Validate checkpoint reproducibility for DDP and FSDP2(manual plan).
    Uses prefix matching (supported by apply_fsdp2) so that e.g. "model.layers" matches
    "model.layers.0", "model.layers.1", etc.

    Each plan value can be:
      - ``"free_full_weight"`` (reshard after forward)
      - ``"keep_full_weight"`` (do not reshard after forward)
      - a list/tuple combining one of the above with optional policies:
        ``"cpu_offload"``, ``"mixed_precision"``

    Example for a Llama-like model (untied):
       {
           "model.embed_tokens": "free_full_weight",
           "model.layers":       "free_full_weight",   # prefix -> shards each layer
           "model.norm":         "keep_full_weight",
           "lm_head":            "keep_full_weight",
       }

    Example with per-module policies:
       {
           "model.layers": ["free_full_weight", "cpu_offload", "mixed_precision"],
           "model.norm":   "keep_full_weight",
           "lm_head":      "keep_full_weight",
       }

    Example for a multi-group model like BLT (untied):
       {
           "model.local_encoder.embed_tokens":  "free_full_weight",
           "model.local_encoder.layers":        "free_full_weight",
           "model.global_transformer.layers":   "free_full_weight",
           "model.local_decoder.layers":        "free_full_weight",
           "model.global_transformer.norm":     "keep_full_weight",
           "lm_head":                           "keep_full_weight",
       }
    """
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings
    checkpoint_step = NUM_STEPS // 2
    policy_options = policy_options or []
    assert "mixed_precision" not in policy_options, "Use _test_fsdp2_manual_plan_mixed_precision_impl for mixed precision."

    dtype = torch.float32
    init_state_dict = _create_init_state_dict(config, dtype)

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    ddp_losses, ddp_grad_norms, ddp_state_dict = _run_ddp_checkpoint_repro(
        rank, config, batches, LR, device, dtype, init_state_dict, checkpoint_step
    )

    set_seed(SEED)
    probe_model = AutoModelForCausalLM.from_config(config).to(device)
    block_classes = get_transformer_block_classes(probe_model)
    id_to_name = {id(m): n for n, m in probe_model.named_modules()}

    input_embed = probe_model.get_input_embeddings()
    output_embed = probe_model.get_output_embeddings()
    weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )

    # Get unique layer-group prefixes by stripping the index from each decoder layer name.
    # e.g. {"model.layers.0", "model.layers.1"} -> {"model.layers"}
    # For multi-group models like BLT: -> {"model.local_encoder.layers", "model.global_transformer.layers", ...}
    decoder_layer_names = {n for n, m in probe_model.named_modules() if type(m) in block_classes}
    layer_prefixes = {".".join(n.split(".")[:-1]) for n in decoder_layer_names}

    # Find final norm and output embed names
    final_norm = _find_final_norm(probe_model, decoder_layer_names)
    norm_name = id_to_name.get(id(final_norm)) if final_norm is not None else None
    embed_name = id_to_name.get(id(input_embed)) if input_embed is not None else None
    out_name = id_to_name.get(id(output_embed)) if output_embed is not None else None

    fsdp_plan = {}
    if not weights_tied and embed_name:
        fsdp_plan[embed_name] = "free_full_weight"
    fsdp_plan.update(dict.fromkeys(layer_prefixes, "free_full_weight"))
    if norm_name:
        fsdp_plan[norm_name] = "keep_full_weight"
    if weights_tied:
        if embed_name:
            fsdp_plan[embed_name] = "keep_full_weight"
    else:
        if out_name:
            fsdp_plan[out_name] = "keep_full_weight"

    if policy_options:
        first_prefix = sorted(layer_prefixes)[0]
        fsdp_plan[first_prefix] = ["free_full_weight", *policy_options]

    del probe_model

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = _run_fsdp2_checkpoint_repro(
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

    _assert_training_trace_match(
        ddp_losses,
        ddp_grad_norms,
        ddp_state_dict,
        fsdp_losses,
        fsdp_grad_norms,
        fsdp_state_dict,
        ref_label="DDP",
        test_label=f"FSDP2(manual{'+policies' if policy_options else ''})",
    )

    if rank == 0:
        logger.info("DDP and FSDP2(manual) reproducibility + comparison checks passed.")


def _test_fsdp2_manual_plan_mixed_precision_impl(rank, config_class, config_dict, tie_word_embeddings, policy_options=None):
    """Validate FSDP2(manual plan) mixed-precision behavior without DDP numerical comparison."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)
    config.tie_word_embeddings = tie_word_embeddings
    num_steps = MIXED_PRECISION_NUM_STEPS
    checkpoint_step = num_steps // 2
    lr = 1e-3
    policy_options = policy_options or []
    assert "mixed_precision" in policy_options, "Mixed-precision manual test requires mixed_precision policy option."

    dtype = torch.float32
    init_state_dict = _create_init_state_dict(config, dtype)

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * num_steps

    set_seed(SEED)
    probe_model = AutoModelForCausalLM.from_config(config).to(device)
    block_classes = get_transformer_block_classes(probe_model)
    id_to_name = {id(m): n for n, m in probe_model.named_modules()}

    input_embed = probe_model.get_input_embeddings()
    output_embed = probe_model.get_output_embeddings()
    weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )

    decoder_layer_names = {n for n, m in probe_model.named_modules() if type(m) in block_classes}
    layer_prefixes = {".".join(n.split(".")[:-1]) for n in decoder_layer_names}

    final_norm = _find_final_norm(probe_model, decoder_layer_names)
    norm_name = id_to_name.get(id(final_norm)) if final_norm is not None else None
    embed_name = id_to_name.get(id(input_embed)) if input_embed is not None else None
    out_name = id_to_name.get(id(output_embed)) if output_embed is not None else None

    fsdp_plan = {}
    if not weights_tied and embed_name:
        fsdp_plan[embed_name] = "free_full_weight"
    fsdp_plan.update(dict.fromkeys(layer_prefixes, "free_full_weight"))
    if norm_name:
        fsdp_plan[norm_name] = "keep_full_weight"
    if weights_tied:
        if embed_name:
            fsdp_plan[embed_name] = "keep_full_weight"
    else:
        if out_name:
            fsdp_plan[out_name] = "keep_full_weight"

    # Apply mixed/cpu_offload policies across all planned FSDP targets so dtype/offload
    # behavior is consistent across layer, norm, and embedding/output boundaries.
    fsdp_plan = {name: [strategy, *policy_options] for name, strategy in fsdp_plan.items()}

    del probe_model

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan=fsdp_plan)
    fsdp_losses, fsdp_grad_norms, fsdp_state_dict = _run_fsdp2_checkpoint_repro(
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

    _log_training_trace(fsdp_losses, fsdp_grad_norms, label="FSDP2(manual+policies)")
    _assert_training_reduction(
        fsdp_losses,
        label="FSDP2(manual+policies)",
        reduction_threshold=MIXED_PRECISION_REDUCTION_THRESHOLD,
    )

    _assert_generation_matches_training_pattern(
        config,
        fsdp_state_dict,
        batches[0],
        device=device,
        max_new_tokens_limit=20,
    )

    if rank == 0:
        logger.info("FSDP2(manual mixed-precision) reduction + generation checks passed.")


def _test_fsdp2_save_load_impl(rank, config_class, config_dict):
    """Train FSDP2 model, save via DCP, load into fresh model, compare state dicts."""
    init_test_logger()

    device = torch.device(f"cuda:{rank}")
    config = config_class.from_dict(config_dict)

    batches = _create_deterministic_data(BATCH_SIZE, SEQ_LEN, config.vocab_size, device, seed=SEED)
    batches = batches * NUM_STEPS

    device_map, device_mesh, _ = initialize_fsdp(fsdp_plan="auto")

    _set_determinism(SEED)
    model = AutoModelForCausalLM.from_config(config).to(device_map)
    model = apply_fsdp2(model, device_mesh, fsdp_plan="auto")
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
        new_model = apply_fsdp2(new_model, device_mesh, fsdp_plan="auto")
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
      - test_fsdp2_all: 2 GPUs, batches all 11 distributed subtests in a single
        mp.spawn to amortize NCCL init overhead (sharding structure, auto/manual plan
        vs DDP with float32/bfloat16 and tied/untied embeddings, save/load)
    """

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
        self._skip_if_fsdp_disabled()
        if backend_device_count(torch_device) < self.fsdp_nproc_per_node:
            self.skipTest(
                f"Need at least {self.fsdp_nproc_per_node} devices, have {backend_device_count(torch_device)}"
            )

    def _get_config_for_fsdp(self):
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

        config_class, config_dict = self._get_config_for_fsdp()

        test_specs = [
            ("sharding_structure_untied", _test_fsdp2_sharding_structure_impl, (config_class, config_dict, False), {}),
            ("sharding_structure_tied", _test_fsdp2_sharding_structure_impl, (config_class, config_dict, True), {}),
            (
                "auto_plan_untied",
                _test_fsdp2_auto_plan_vs_ddp_impl,
                (config_class, config_dict, False),
                {},
            ),
            (
                "auto_plan_tied",
                _test_fsdp2_auto_plan_vs_ddp_impl,
                (config_class, config_dict, True),
                {},
            ),
            (
                "auto_plan_untied_cpu_offload_mixed_precision",
                _test_fsdp2_auto_plan_mixed_precision_impl,
                (config_class, config_dict, False, ["cpu_offload", "mixed_precision"]),
                {},
            ),
            (
                "auto_plan_tied_cpu_offload_mixed_precision",
                _test_fsdp2_auto_plan_mixed_precision_impl,
                (config_class, config_dict, True, ["cpu_offload", "mixed_precision"]),
                {},
            ),
            (
                "manual_plan_untied",
                _test_fsdp2_manual_plan_vs_ddp_impl,
                (config_class, config_dict, False),
                {},
            ),
            (
                "manual_plan_tied",
                _test_fsdp2_manual_plan_vs_ddp_impl,
                (config_class, config_dict, True),
                {},
            ),
            (
                "manual_plan_untied_cpu_offload_mixed_precision",
                _test_fsdp2_manual_plan_mixed_precision_impl,
                (config_class, config_dict, False, ["cpu_offload", "mixed_precision"]),
                {},
            ),
            (
                "manual_plan_tied_cpu_offload_mixed_precision",
                _test_fsdp2_manual_plan_mixed_precision_impl,
                (config_class, config_dict, True, ["cpu_offload", "mixed_precision"]),
                {},
            ),
            ("save_load", _test_fsdp2_save_load_impl, (config_class, config_dict), {}),
        ]

        results_file = tempfile.mktemp(suffix=".json")
        port = _get_free_port()
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

        passed = [name for name, err in results.items() if err is None]
        failed = {name: err for name, err in results.items() if err is not None}

        summary_lines = [
            "",
            "=" * 60,
            f"  test_fsdp2_all: {len(passed)} passed, {len(failed)} failed (out of {len(results)} subtests)",
            "=" * 60,
        ]
        for name in results:
            status = "PASSED" if results[name] is None else "FAILED"
            summary_lines.append(f"  {'[PASS]' if status == 'PASSED' else '[FAIL]'} {name}")
        summary_lines.append("=" * 60)
        print("\n".join(summary_lines), flush=True)

        for test_name, error in results.items():
            with self.subTest(test_name=test_name):
                if error is not None:
                    self.fail(f"FSDP subtest '{test_name}' failed:\n{error}")

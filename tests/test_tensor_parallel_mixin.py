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
import atexit
import copy
import gc
import multiprocessing
import os
import queue
import socket
import sys
import tempfile
import traceback
from abc import ABC, abstractmethod
from itertools import product

from parameterized import parameterized

from transformers import TorchAoConfig, set_seed
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.tensor_parallel import _get_parameter_plan
from transformers.testing_utils import (
    SINGLE_PROCESS_DEVICES,
    backend_empty_cache,
    is_tensor_parallel_test,
    is_torch_available,
    torch_device,
)
from transformers.utils import is_torch_greater_or_equal, is_torchao_available

from .test_fsdp_mixin import _get_distributed_backend, _get_distributed_device_type, _set_rank_device


if is_torchao_available():
    from torchao.quantization import Float8WeightOnlyConfig


# TODO(3outeille): better guarding
if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.distributed_c10d import GroupMember, _world
    from torch.distributed.tensor import DTensor


# =============================================================================
# Distributed helpers (top-level for pickling to the worker processes)
# =============================================================================


def _find_free_port():
    """Find a free port by binding a socket and releasing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def get_packed_grad_shard(grad, world_size, rank, dim):
    """Get the correct shard of a packed gradient (matching get_packed_weights interleaved logic).

    Packed weights like gate_up_proj are sharded with interleaving:
    Original: [G0 G1 G2 G3 | U0 U1 U2 U3]  (gate | up)
    Rank 0:   [G0 G1 | U0 U1]
    Rank 1:   [G2 G3 | U2 U3]
    """
    total_size = grad.shape[dim]
    # Packed weights have 2 blocks (gate and up)
    block_size = total_size // 2
    # every shard but the last is ceil(block_size / world_size) long, like DTensor's split
    shard_block_size = -(-block_size // world_size)

    # Build interleaved indices
    indices = []
    for block_idx in range(2):  # gate block, then up block
        block_offset = block_idx * block_size
        start = block_offset + min(rank * shard_block_size, block_size)
        stop = block_offset + min((rank + 1) * shard_block_size, block_size)
        indices.extend(range(start, stop))

    # Select along the sharded dimension
    return grad.index_select(dim, torch.tensor(indices, device=grad.device))


def _pool_worker(rank, world_size, port, backend, task_queue, result_queue):
    """Long-lived rank: initializes the process group once, then runs the test functions it receives until stopped."""
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # every TP/EP load builds its device mesh, i.e. new process groups whose sockets and pipes stay open: a rank that
    # runs many tests outgrows the usual 1024 open files
    if sys.platform != "win32":
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        except (ValueError, OSError):  # e.g. an unlimited hard limit on macOS
            pass

    # some backends, e.g. tpu, require the rank to be set before initializing the process group
    _set_rank_device(rank, backend)

    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except Exception:
        result_queue.put((rank, traceback.format_exc()))
        return
    result_queue.put((rank, None))

    device_type = _get_distributed_device_type()
    while True:
        try:
            task = task_queue.get(timeout=5)
        except queue.Empty:
            # the parent can die without stopping us (e.g. pytest-xdist worker killed or hard-exited)
            if not multiprocessing.parent_process().is_alive():
                return
            continue
        if task is None:
            break

        func, func_args, func_kwargs = task
        try:
            func(rank, *func_args, **func_kwargs)
            dist.barrier()
        except BaseException:
            # the other ranks may be stuck in a collective, the parent tears the whole pool down
            result_queue.put((rank, traceback.format_exc()))
            return
        # free the test's models before the next one, as the pool lives for the whole pytest session
        del func, func_args, func_kwargs, task
        gc.collect()
        if device_type != "cpu":
            backend_empty_cache(device_type)
        # and the process groups of the device meshes it built, which would otherwise pile up (with their sockets)
        for group in list(_world.pg_map):
            if group is not GroupMember.WORLD:
                dist.destroy_process_group(group)
        result_queue.put((rank, None))

    dist.destroy_process_group()


class _DistributedWorkerPool:
    """`world_size` ranks with an initialized process group, reused across tests.

    Spawning fresh ranks for every test spends most of its time re-importing torch and transformers in each of them;
    a test only takes a fraction of that once the ranks are warm.
    """

    def __init__(self, world_size: int, backend: str, max_retries: int):
        self.world_size = world_size
        ctx = mp.get_context("spawn")
        for attempt in range(max_retries):
            port = _find_free_port()
            self.result_queue = ctx.Queue()
            self.task_queues = [ctx.Queue() for _ in range(world_size)]
            self.processes = [
                ctx.Process(
                    target=_pool_worker,
                    args=(rank, world_size, port, backend, self.task_queues[rank], self.result_queue),
                )
                for rank in range(world_size)
            ]
            for process in self.processes:
                process.start()
            try:
                self._wait_for_ranks()
                return
            except RuntimeError as e:
                self.terminate()
                if "EADDRINUSE" in str(e) and attempt < max_retries - 1:
                    continue
                raise

    def run(self, func, args, kwargs):
        for task_queue in self.task_queues:
            task_queue.put((func, args, kwargs))
        self._wait_for_ranks()

    def _wait_for_ranks(self):
        """Wait for every rank to report success, raising on the first rank error or crash."""
        pending = set(range(self.world_size))
        while pending:
            try:
                rank, error = self.result_queue.get(timeout=1)
            except queue.Empty:
                dead = {rank: self.processes[rank].exitcode for rank in pending if not self.processes[rank].is_alive()}
                if dead and self.result_queue.empty():
                    raise RuntimeError(f"Distributed ranks exited unexpectedly (rank: exit code): {dead}")
                continue
            if error is not None:
                raise RuntimeError(f"Rank {rank} failed:\n{error}")
            pending.discard(rank)

    def shutdown(self):
        for task_queue in self.task_queues:
            task_queue.put(None)
        for process in self.processes:
            process.join(timeout=10)
        self.terminate()

    def terminate(self):
        for process in self.processes:
            if process.is_alive():
                process.kill()
            process.join()


# one pool per (world_size, backend) and per pytest(-xdist worker) process
_WORKER_POOLS: dict[tuple[int, str], _DistributedWorkerPool] = {}


@atexit.register
def _shutdown_worker_pools():
    for pool in _WORKER_POOLS.values():
        pool.shutdown()
    _WORKER_POOLS.clear()


def _init_distributed(tp: int, max_retries: int = 5, backend: str | None = None):
    """Decorator to run a function on `tp` ranks, reusing the ranks (and their process group) across calls."""
    # default to the current accelerator's backend, as "gloo" cannot carry every accelerator's tensors
    if backend is None:
        backend = _get_distributed_backend()

    def _init_distributed_inner(func):
        def wrapper(*args, **kwargs):
            key = (tp, backend)
            if key not in _WORKER_POOLS:
                _WORKER_POOLS[key] = _DistributedWorkerPool(tp, backend, max_retries)
            try:
                _WORKER_POOLS[key].run(func, args, kwargs)
            except BaseException:
                # the surviving ranks can be stuck in a collective, the next call starts from fresh ranks
                _WORKER_POOLS.pop(key).terminate()
                raise

        return wrapper

    return _init_distributed_inner


def _get_expert_weight_names(model):
    """The routed experts' weight names in `model`: the projections of every experts module
    (`@use_experts_implementation`, which declares its layout via `has_gate`). Empty for a dense model."""
    params = dict(model.named_parameters())  # a tied tensor once, under its first name
    names = []
    for name, module in model.named_modules():
        if hasattr(module, "has_gate"):
            projections = ("gate_up_proj" if module.has_gate else "up_proj", "down_proj")
            names += [f"{name}.{projection}" for projection in projections if f"{name}.{projection}" in params]
    return names


def _load_tp_and_reference_models(model_path, model_class):
    """Load TP model and non-TP reference model for comparison.

    Returns:
        tuple: (model_tp, model_ref, device)
    """
    model_tp = model_class.from_pretrained(
        model_path,
        distributed_config=DistributedConfig(tp_size=dist.get_world_size()),
    )
    dist.barrier()

    device = model_tp.device
    model_ref = model_class.from_pretrained(model_path)
    model_ref = model_ref.to(device)

    return model_tp, model_ref, device


def _verify_tp_sharding(rank, model_tp, model_ref):
    """Verify TP sharding by comparing parameter shapes between TP and reference models.

    Returns:
        list: Names of sharded parameters
    """
    world_size = dist.get_world_size()
    sharded_params = []

    for (name, param), (_, param_full) in zip(model_tp.named_parameters(), model_ref.named_parameters()):
        if param.shape != param_full.shape:
            sharded_params.append(name)
            if rank == 0:
                print(f"[TP Test Debug] TP sharded: {name} - full: {param_full.shape} -> sharded: {param.shape}")

            # Verify sharding is correct
            for dim in range(param.ndim):
                if param.size(dim) != param_full.size(dim):
                    param_plan = _get_parameter_plan(name, model_tp._tp_plan, is_weight=True)
                    if param_plan in ("packed_colwise", "packed_rowwise"):
                        expected_size = param_full.size(dim) // world_size
                        assert param.size(dim) == expected_size, (
                            f"Packed weight {name} sharding incorrect: expected {expected_size}, got {param.size(dim)}"
                        )
                    else:
                        expected_size = (param_full.size(dim) + world_size - 1) // world_size
                        assert param.size(dim) <= expected_size, (
                            f"Weight {name} sharding incorrect: expected <= {expected_size}, got {param.size(dim)}"
                        )
                    break

    return sharded_params


def _reference_grad_shard(name, grad, local_grad, rank, world_size, tp_plan, expert_weights=()):
    """The slice of the unsharded reference gradient `grad` that this rank holds as `local_grad`."""
    # Expert weights hold this rank's experts along dim 0 (`grouped_gemm`), whatever `_tp_plan` still says for them
    if name in expert_weights:
        return grad.narrow(0, rank * local_grad.size(0), local_grad.size(0))
    for dim in range(grad.ndim):
        if grad.size(dim) != local_grad.size(dim):
            if _get_parameter_plan(name, tp_plan, is_weight=True) in ("packed_colwise", "packed_rowwise"):
                return get_packed_grad_shard(grad, world_size, rank, dim)
            # regular slicing: every shard but the last is ceil(size / world_size) long
            return grad.narrow(dim, rank * -(-grad.size(dim) // world_size), local_grad.size(dim))
    return grad


def _to_device(inputs, device):
    return {name: value.to(device) for name, value in inputs.items()}


def _test_tp_forward_impl(_rank, model_path, model_class, atol, rtol, extra_inputs):
    """Implementation for comparing TP and non-TP model outputs."""
    set_seed(0)

    model_tp, model, device = _load_tp_and_reference_models(model_path, model_class)

    _verify_tp_sharding(_rank, model_tp, model)

    model_tp.eval()
    model.eval()

    vocab_size = model.config.get_text_config().vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    extra_inputs = _to_device(extra_inputs, device)

    with torch.no_grad():
        logits = model(input_ids, **extra_inputs).logits
        logits_tp = model_tp(input_ids, **extra_inputs).logits

    diff = (logits - logits_tp).abs()
    assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model outputs differ. Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_tp_backward_impl(rank, model_path, model_class, atol, rtol, extra_inputs):
    """Implementation for comparing TP and non-TP model backward passes."""
    set_seed(0)

    model_tp, model, device = _load_tp_and_reference_models(model_path, model_class)
    model_tp.train()
    model.train()

    vocab_size = model.config.get_text_config().vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    set_seed(0)
    labels = torch.randint(0, vocab_size, (2, 64)).to(device)
    extra_inputs = _to_device(extra_inputs, device)

    loss = model(input_ids, labels=labels, **extra_inputs).loss
    loss.backward()

    loss_tp = model_tp(input_ids, labels=labels, **extra_inputs).loss
    loss_tp.backward()

    assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model losses differ. "
        f"Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, "
        f"Diff: {(loss - loss_tp).abs().item()}"
    )

    # Compare gradients for matching parameters
    world_size = dist.get_world_size()
    failed_grads = {}
    for (name, param), (_, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # A sharded param's grad is a DTensor: take this rank's local shard, since a DTensor
            # reports the *global* shape and can't be compared against a plain tensor.
            if isinstance(grad_tp, DTensor):
                grad_tp = grad_tp.to_local()

            grad = _reference_grad_shard(name, grad, grad_tp, rank, world_size, model_tp._tp_plan)

            if not torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol):
                failed_grads[name] = (grad.cpu() - grad_tp.cpu()).abs().max().item()

    assert not failed_grads, f"Gradients differ for {len(failed_grads)} parameter(s):\n" + "\n".join(
        f"  {name}: max diff = {diff}" for name, diff in failed_grads.items()
    )

    dist.barrier()


def _test_tp_generation_impl(_rank, model_path, model_class, atol, rtol, max_new_tokens, extra_inputs):
    """Implementation for comparing TP and non-TP model generation outputs (direct load path)."""
    set_seed(0)

    model_tp, model, device = _load_tp_and_reference_models(model_path, model_class)
    model_tp.eval()
    model.eval()

    set_seed(0)
    vocab_size = model.config.get_text_config().vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)
    extra_inputs = _to_device(extra_inputs, device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "output_scores": True,
        "return_dict_in_generate": True,
        "use_cache": True,
        **extra_inputs,
    }

    with torch.no_grad():
        output = model.generate(input_ids, **generation_kwargs)
        output_tp = model_tp.generate(input_ids, **generation_kwargs)

    # Compare logits/scores at each generation step
    scores = torch.stack(output.scores)
    scores_tp = torch.stack(output_tp.scores)

    diff = (scores - scores_tp).abs()
    assert torch.allclose(scores, scores_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model generation logits differ (direct load path). "
        f"Max diff: {diff.max().item()} | Mean diff: {diff.mean().item()}"
    )

    # Compare generated token sequences
    assert torch.equal(output.sequences, output_tp.sequences), (
        f"TP and non-TP model generated different token sequences (direct load path). "
        f"Non-TP: {output.sequences.tolist()} | TP: {output_tp.sequences.tolist()}"
    )

    dist.barrier()


def _quantized_weight_names(model):
    """Names of the modules whose weight is quantized, i.e. a tensor subclass (sharded or not)."""
    names = set()
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if isinstance(weight, torch.Tensor) and type(weight) not in (torch.Tensor, torch.nn.Parameter):
            names.add(name)
    return names


def _test_tp_generation_quantized_impl(_rank, model_path, model_class, max_new_tokens, extra_inputs):
    """Implementation for comparing TP+quantized and non-TP quantized generation.

    The TP model quantizes its weight shards, so wherever a layer is sharded along its input dim its float8 scales (and
    rounding) differ from the non-TP model's: the two are different float8 roundings of the same weights. On tiny random
    models that flips greedy near-ties (and MoE routing), and comparing free-running generations position by position
    turns one flip into a mismatch of every later token. Instead, every model scores the non-TP model's generation
    (teacher forcing), and a token the TP model picks must be one that float8 noise can explain: one the non-TP model
    ranks below its own choice by no more than both models' quantization errors allow.
    """
    set_seed(0)

    quantization_config = TorchAoConfig(Float8WeightOnlyConfig())

    model_tp = model_class.from_pretrained(
        model_path,
        distributed_config=DistributedConfig(tp_size=dist.get_world_size()),
        quantization_config=quantization_config,
    )
    dist.barrier()

    device = model_tp.device
    model = model_class.from_pretrained(model_path, quantization_config=quantization_config)
    model = model.to(device)
    model_unquantized = model_class.from_pretrained(model_path).to(device)

    model_tp.eval()
    model.eval()
    model_unquantized.eval()

    quantized, quantized_tp = _quantized_weight_names(model), _quantized_weight_names(model_tp)
    assert quantized and quantized == quantized_tp, (
        f"TP and non-TP models quantize different layers. Only quantized without TP: {sorted(quantized - quantized_tp)}"
        f" | Only quantized with TP: {sorted(quantized_tp - quantized)}"
    )

    vocab_size = model.config.get_text_config().vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)
    extra_inputs = _to_device(extra_inputs, device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "return_dict_in_generate": True,
        "use_cache": True,
        **extra_inputs,
    }

    with torch.no_grad():
        output = model.generate(input_ids, **generation_kwargs)
        output_tp = model_tp.generate(input_ids, **generation_kwargs)

        # the logits predicting each generated token of the non-TP model, given its own previous tokens
        sequence = output.sequences
        generated = slice(input_ids.shape[1] - 1, sequence.shape[1] - 1)
        logits = model(sequence, **extra_inputs).logits[0, generated].float()
        logits_tp = model_tp(sequence, **extra_inputs).logits[0, generated].float()
        logits_unquantized = model_unquantized(sequence, **extra_inputs).logits[0, generated].float()

    # Both quantized models are the unquantized logits plus a quantization error: at most `quantization_noise` for the
    # non-TP model, and assumed no larger than its largest one for the TP model. A token the TP model prefers is then
    # ranked by the non-TP model at most twice the sum of both errors below its own choice.
    quantization_noise = (logits - logits_unquantized).abs().amax(-1)
    tolerance = 2 * (quantization_noise + quantization_noise.max())

    def _ranked_clearly_below(tokens):
        """The positions where the non-TP model scores `tokens` further below its own choice than float8 can explain."""
        positions = torch.arange(len(tokens), device=logits.device)
        gap = logits[positions].amax(-1) - logits[positions, tokens]
        return [
            f"position {i}: gap {gap[i]:.3g} > tolerance {tolerance[i]:.3g}"
            for i in positions[gap > tolerance[positions]].tolist()
        ]

    # the TP model's choices given the same previous tokens: its forward ...
    failures = [f"forward {failure}" for failure in _ranked_clearly_below(logits_tp.argmax(-1))]
    # ... and its cached decoding, up to the first token where its generation departs from the non-TP one (included)
    tokens = sequence[0, input_ids.shape[1] :]
    tokens_tp = output_tp.sequences[0, input_ids.shape[1] :]
    length = min(len(tokens), len(tokens_tp))
    differs = (tokens[:length] != tokens_tp[:length]).nonzero()
    if len(differs):
        failures += [f"decoding {failure}" for failure in _ranked_clearly_below(tokens_tp[: differs[0].item() + 1])]

    assert not failures, (
        "The TP+quantized model picks tokens the non-TP quantized model ranks clearly below its own choice:\n"
        + "\n".join(failures)
        + f"\nNon-TP+quantized: {output.sequences.tolist()}\nTP+quantized: {output_tp.sequences.tolist()}"
    )

    dist.barrier()


def _load_ep_and_reference_models(model_path, model_class):
    """Load EP model and non-EP reference model for comparison."""
    model_ep = model_class.from_pretrained(
        model_path,
        distributed_config=DistributedConfig(tp_size=dist.get_world_size(), ep_size=dist.get_world_size()),
    )
    dist.barrier()

    device = model_ep.device
    model_ref = model_class.from_pretrained(model_path)
    model_ref = model_ref.to(device)

    return model_ep, model_ref, device


def _test_ep_forward_impl(_rank, model_path, model_class, atol, rtol, experts_implementation, extra_inputs):
    """Implementation for comparing EP and non-EP model outputs."""
    set_seed(0)

    model_ep, model_ref, device = _load_ep_and_reference_models(model_path, model_class)

    model_ep.eval()
    model_ref.eval()

    model_ep.set_experts_implementation(experts_implementation)
    model_ref.set_experts_implementation(experts_implementation)

    vocab_size = model_ref.config.get_text_config().vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    extra_inputs = _to_device(extra_inputs, device)

    with torch.no_grad():
        logits_ref = model_ref(input_ids, **extra_inputs).logits
        logits_ep = model_ep(input_ids, **extra_inputs).logits

    diff = (logits_ref - logits_ep).abs()
    assert torch.allclose(logits_ref, logits_ep, atol=atol, rtol=rtol), (
        f"EP and non-EP model outputs differ. Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_ep_backward_impl(rank, model_path, model_class, atol, rtol, experts_implementation, extra_inputs):
    """Implementation for comparing EP and non-EP model backward passes."""
    set_seed(0)

    model_ep, model_ref, device = _load_ep_and_reference_models(model_path, model_class)
    model_ep.train()
    model_ref.train()

    model_ep.set_experts_implementation(experts_implementation)
    model_ref.set_experts_implementation(experts_implementation)

    vocab_size = model_ref.config.get_text_config().vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    labels = torch.randint(0, vocab_size, (2, 64)).to(device)
    extra_inputs = _to_device(extra_inputs, device)

    loss_ref = model_ref(input_ids, labels=labels, **extra_inputs).loss
    loss_ref.backward()

    loss_ep = model_ep(input_ids, labels=labels, **extra_inputs).loss
    loss_ep.backward()

    assert torch.allclose(loss_ref, loss_ep, atol=atol, rtol=rtol), (
        f"EP and non-EP model losses differ. "
        f"Non-EP loss: {loss_ref.item()}, EP loss: {loss_ep.item()}, "
        f"Diff: {(loss_ref - loss_ep).abs().item()}"
    )

    world_size = dist.get_world_size()
    expert_weights = set(_get_expert_weight_names(model_ep))
    failed_grads = {}
    for (name, param), (_, param_ep) in zip(model_ref.named_parameters(), model_ep.named_parameters()):
        if param.grad is None and param_ep.grad is None:
            continue
        if param.grad is None or param_ep.grad is None:
            failed_grads[name] = "no gradient on the " + ("EP" if param_ep.grad is None else "non-EP") + " model"
            continue
        local_grad = param_ep.grad.to_local() if isinstance(param_ep.grad, DTensor) else param_ep.grad
        expected = _reference_grad_shard(
            name, param.grad, local_grad, rank, world_size, model_ep._tp_plan, expert_weights
        ).cpu()
        if not torch.allclose(expected, local_grad.cpu(), atol=atol, rtol=rtol):
            failed_grads[name] = f"max diff = {(expected - local_grad.cpu()).abs().max().item()}"

    assert not failed_grads, f"Gradients differ for {len(failed_grads)} parameter(s):\n" + "\n".join(
        f"  {name}: {failure}" for name, failure in failed_grads.items()
    )

    dist.barrier()


class TensorParallelTesterMixin(ABC):
    """
    Mixin for tensor parallel tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - causal_lm_class (or conditional_generation_class for VLM/ALM testers), base_model_class, etc.

    This mixin adds tensor parallel-specific tests using that infrastructure.
    """

    # ============================================================
    # Configuration (can be overridden per model)
    # ============================================================
    tensor_parallel_size: int = 2
    tensor_parallel_atol: float = 1e-5
    tensor_parallel_rtol: float = 1e-5

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    # ============================================================
    # Helper methods
    # ============================================================
    def _has_ep_plan(self) -> bool:
        """Check if model has an expert parallel plan defined."""
        config = self.model_tester.get_config().get_text_config()
        return hasattr(config, "base_model_ep_plan") and config.base_model_ep_plan is not None

    def _has_tp_plan(self) -> bool:
        """Check if model has a tensor parallel plan defined."""
        config = self.model_tester.get_config().get_text_config()
        return hasattr(config, "base_model_tp_plan") and config.base_model_tp_plan is not None

    def _skip_if_single_process_device(self):
        if torch_device in SINGLE_PROCESS_DEVICES:
            self.skipTest(f"Distributed tests spawn one process per device, which {torch_device!r} doesn't allow")

    def _get_tp_model_class(self):
        """The class the distributed tests run on, through a text-only forward: the tester's `*ForCausalLM`, or a VLM/ALM
        tester's `*ForConditionalGeneration`."""
        return getattr(self.model_tester, "causal_lm_class", None) or getattr(
            self.model_tester, "conditional_generation_class", None
        )

    def _get_tp_config(self, tie_word_embeddings: bool | None = None):
        """The tester's tiny config. Its vocab stays as it is: 99 does not divide by the world size, which is
        the uneven sharding of `lm_head` that real vocabs need."""
        config = self.model_tester.get_config()
        text_config = config.get_text_config()
        if tie_word_embeddings is not None:
            if hasattr(text_config, "tie_word_embeddings"):
                text_config.tie_word_embeddings = tie_word_embeddings
            if hasattr(config, "tie_word_embeddings"):
                config.tie_word_embeddings = tie_word_embeddings
        return config

    def _get_tp_extra_inputs(self, batch_size: int, seq_length: int) -> dict:
        """Inputs besides `input_ids` that the model needs to run on `batch_size` sequences of up to `seq_length`
        tokens (prompt and generated ones), e.g. the audio of a model that cannot run on text alone. None by default."""
        return {}

    def _skip_if_not_supported(self, expert_parallel: bool = False):
        """Check and skip the test if tensor/expert parallel is not supported for this model/environment."""
        parallelism = "Expert" if expert_parallel else "Tensor"
        if not is_torch_greater_or_equal("2.9"):
            self.skipTest(f"{parallelism} parallel tests require torch >= 2.9")

        if torch.cuda.is_available() or torch.xpu.is_available():
            self.skipTest(f"{parallelism} parallel mixin tests are CPU-only and should not run on GPU or XPU machines")

        self._skip_if_single_process_device()

        if os.cpu_count() < self.tensor_parallel_size:
            self.skipTest(
                f"{parallelism} parallel tests require at least {self.tensor_parallel_size} CPUs, "
                f"but only {os.cpu_count()} available"
            )

        model_class = self._get_tp_model_class()
        if model_class is None:
            self.skipTest("Model tester has neither causal_lm_class nor conditional_generation_class")
        if model_class not in self.all_model_classes:
            # e.g. a classifier wrapping another model's `*ForConditionalGeneration`, which that model's tests cover
            self.skipTest(f"{model_class.__name__} is not one of the model classes this test covers")

        if expert_parallel:
            if not self._has_ep_plan():
                self.skipTest("Model does not have an expert parallel plan (base_model_ep_plan)")
        elif not self._has_tp_plan():
            self.skipTest("Model does not have a tensor parallel plan (base_model_tp_plan)")

        # # Skip encoder-decoder models (TP not supported)
        # if getattr(self, "is_encoder_decoder", False):
        #     self.skipTest("TP tests not supported for encoder-decoder models")

        # # Skip VLM models for now
        # config = self.model_tester.get_config()
        # if hasattr(config, "vision_config") and config.vision_config is not None:
        #     self.skipTest("VLM models are not yet supported in TP tests")

    def test_tp_plan_shards_input_embeddings(self):
        """A model whose TP plan shards its decoder must shard its input embeddings too: replicated, they cost
        `vocab x hidden` on every rank, and a tied `lm_head` sharded on the vocabulary would not match the embedding
        weight it shares. A plan that only covers the `lm_head` is not a decoder plan."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            if not any(key.rsplit(".", 1)[-1] != "lm_head" for key in model.tp_plan):
                continue
            embeddings = model.get_input_embeddings()
            if not isinstance(embeddings, torch.nn.Embedding):
                continue
            name = next(name for name, module in model.named_modules() if module is embeddings)
            self.assertIsNotNone(
                _get_parameter_plan(f"{name}.weight", model.tp_plan, is_weight=True),
                f"{model_class.__name__}: the TP plan leaves `{name}` replicated; shard it with `embedding_rowwise`",
            )

    def test_moe_parallel_plans_shard_experts(self):
        """An MoE model's expert weights, the bulk of its parameters, must be sharded by every parallel plan it defines.
        A plan missing them keeps every expert on every rank without changing any output, so no numerical test sees it.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        models = [model_class(copy.deepcopy(config)) for model_class in self.all_model_classes]
        expert_weights = {model: _get_expert_weight_names(model) for model in models}
        if not any(expert_weights.values()):
            self.skipTest("Not an MoE model")

        for model, names in expert_weights.items():
            for kind, plan in (("TP", model._tp_plan), ("EP", model._ep_plan)):
                unsharded = sorted(n for n in names if _get_parameter_plan(n, plan, is_weight=True) is None)
                self.assertFalse(
                    unsharded, f"{type(model).__name__}: the {kind} plan leaves these experts replicated: {unsharded}"
                )

    @parameterized.expand([(False,), (True,)])
    @is_tensor_parallel_test
    def test_tp_forward(self, tie_word_embeddings):
        self._skip_if_not_supported()

        config = self._get_tp_config(tie_word_embeddings=tie_word_embeddings)
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_forward_impl)(
                tmp_dir, model_class, atol, rtol, self._get_tp_extra_inputs(2, 64)
            )

    @is_tensor_parallel_test
    def test_tp_backward(self):
        self._skip_if_not_supported()

        config = self._get_tp_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_backward_impl)(
                tmp_dir, model_class, atol, rtol, self._get_tp_extra_inputs(2, 64)
            )

    @is_tensor_parallel_test
    def test_tp_generation(self):
        # Test TP generation: unfused checkpoint → conversion mapping (if needed) → TP sharding → model → generate
        self._skip_if_not_supported()

        config = self._get_tp_config()

        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol
        max_new_tokens = 25

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_generation_impl)(
                tmp_dir, model_class, atol, rtol, max_new_tokens, self._get_tp_extra_inputs(1, 10 + max_new_tokens)
            )

    @is_tensor_parallel_test
    def test_tp_generation_quantized(self):
        self._skip_if_not_supported()

        if not is_torchao_available():
            self.skipTest("Test requires torchao")

        config = self._get_tp_config()
        model_class = self._get_tp_model_class()
        max_new_tokens = 25

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_generation_quantized_impl)(
                tmp_dir, model_class, max_new_tokens, self._get_tp_extra_inputs(1, 10 + max_new_tokens)
            )

    @parameterized.expand(
        list(
            product(
                [False, True],  # tie_word_embeddings
                ["eager", "grouped_mm", "batched_mm"],  # experts_implementation
            )
        )
    )
    @is_tensor_parallel_test
    def test_ep_forward(self, tie_word_embeddings, experts_implementation):
        self._skip_if_not_supported(expert_parallel=True)

        config = self._get_tp_config(tie_word_embeddings=tie_word_embeddings)
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_ep_forward_impl)(
                tmp_dir, model_class, atol, rtol, experts_implementation, self._get_tp_extra_inputs(2, 64)
            )

    @parameterized.expand([("eager",), ("grouped_mm",), ("batched_mm",)])
    @is_tensor_parallel_test
    def test_ep_backward(self, experts_implementation):
        self._skip_if_not_supported(expert_parallel=True)

        config = self._get_tp_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_ep_backward_impl)(
                tmp_dir, model_class, atol, rtol, experts_implementation, self._get_tp_extra_inputs(2, 64)
            )

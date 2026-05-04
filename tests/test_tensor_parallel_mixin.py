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
import os
import socket
import tempfile
from abc import ABC, abstractmethod

from transformers import TorchAoConfig, set_seed
from transformers.distributed import DistributedConfig
from transformers.integrations.tensor_parallel import _get_parameter_tp_plan, _replicate_dtensor
from transformers.testing_utils import (
    is_tensor_parallel_test,
    is_torch_available,
)
from transformers.utils import is_torch_greater_or_equal, is_torchao_available


if is_torchao_available():
    from torchao.quantization import Float8WeightOnlyConfig


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.tensor import DTensor
    from torch.multiprocessing.spawn import ProcessRaisedException


def _to_local(tensor):
    """Extract local tensor from DTensor, or return as-is for plain tensors."""
    if hasattr(tensor, "to_local"):
        # NOTE(3outeille): With Sequence Parallelism, replicated params (e.g. norm weights) get Partial
        # gradients — each rank holds only its contribution from its sequence shard.
        # We must all-reduce (redistribute to Replicate) before extracting, otherwise
        # we'd compare an incomplete gradient against the full reference.
        # In the case of real training, we will always use SP + FSDP where the last will all-reduce the
        #  Partial gradients for us.

        if isinstance(tensor, DTensor) and any(not p.is_replicate() for p in tensor.placements):
            tensor = _replicate_dtensor(tensor)
        return tensor.to_local()
    return tensor


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
    shard_block_size = block_size // world_size

    # Build interleaved indices
    indices = []
    for block_idx in range(2):  # gate block, then up block
        block_offset = block_idx * block_size
        start = block_offset + rank * shard_block_size
        stop = block_offset + (rank + 1) * shard_block_size
        indices.extend(range(start, stop))

    # Select along the sharded dimension
    return grad.index_select(dim, torch.tensor(indices, device=grad.device))


def _is_packed_colwise_plan(plan) -> bool:
    return plan == "packed_colwise" or getattr(plan, "kind", None) == "packed_colwise"


def _global_wrapper(rank, func, tp, port, func_args, func_kwargs):
    """Wrapper to set up distributed environment and run the test function."""

    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp
    setup_dist_env(rank, world_size, port)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def _init_distributed(tp: int, max_retries: int = 5):
    """Decorator to initialize distributed environment and spawn processes."""

    def _init_distributed_inner(func):
        def wrapper(*args, **kwargs):
            world_size = tp
            for attempt in range(max_retries):
                port = _find_free_port()
                spawn_args = (func, tp, port, args, kwargs)
                try:
                    mp.spawn(_global_wrapper, args=spawn_args, nprocs=world_size)
                    return
                except ProcessRaisedException as e:
                    if "EADDRINUSE" in str(e) and attempt < max_retries - 1:
                        continue
                    raise

        return wrapper

    return _init_distributed_inner


def _load_tp_and_reference_models(model_path, model_class, enable_sequence_parallel=False):
    """Load TP model and non-TP reference model for comparison.

    Returns:
        tuple: (model_tp, model_ref, device)
    """
    tp_size = dist.get_world_size()
    distributed_config = DistributedConfig(
        tp_size=tp_size, tp_plan="auto", enable_sequence_parallel=enable_sequence_parallel
    )
    model_tp = model_class.from_pretrained(
        model_path, distributed_config=distributed_config, attn_implementation="sdpa"
    )
    dist.barrier()

    device = model_tp.device
    model_ref = model_class.from_pretrained(model_path, attn_implementation="sdpa")
    model_ref = model_ref.to(device)

    return model_tp, model_ref, device


def _get_active_tp_plan(model_tp):
    distributed_config = getattr(model_tp.config, "distributed_config", None)
    tp_plan = getattr(distributed_config, "tp_plan", None)

    if tp_plan == "auto":
        return getattr(model_tp, "_tp_plan", None) or {}

    return tp_plan or getattr(model_tp, "_tp_plan", None) or {}


def _verify_tp_sharding(rank, model_tp, model_ref):
    """Verify TP sharding by comparing parameter shapes between TP and reference models.

    For DTensor params, uses the local tensor shape (not the global DTensor shape).

    Returns:
        list: Names of sharded parameters
    """
    world_size = dist.get_world_size()
    sharded_params = []
    tp_plan = _get_active_tp_plan(model_tp)

    for (name, param), (_, param_full) in zip(model_tp.named_parameters(), model_ref.named_parameters()):
        # For DTensor params, get the local shape for comparison
        param_local = _to_local(param)
        if param_local.shape != param_full.shape:
            sharded_params.append(name)
            if rank == 0:
                print(f"[TP Test Debug] TP sharded: {name} - full: {param_full.shape} -> sharded: {param_local.shape}")

            # Verify sharding is correct
            for dim in range(param_local.ndim):
                if param_local.size(dim) != param_full.size(dim):
                    param_plan = _get_parameter_tp_plan(name, tp_plan, is_weight=True)
                    if _is_packed_colwise_plan(param_plan):
                        expected_size = param_full.size(dim) // world_size
                        assert param_local.size(dim) == expected_size, (
                            f"Packed weight {name} sharding incorrect: expected {expected_size}, got {param_local.size(dim)}"
                        )
                    else:
                        expected_size = (param_full.size(dim) + world_size - 1) // world_size
                        assert param_local.size(dim) <= expected_size, (
                            f"Weight {name} sharding incorrect: expected <= {expected_size}, got {param_local.size(dim)}"
                        )
                    break

    return sharded_params


def _test_tp_forward_impl(_rank, model_path, model_class, atol, rtol):
    """Implementation for comparing TP and non-TP model outputs."""
    set_seed(0)

    model_tp, model, device = _load_tp_and_reference_models(model_path, model_class, enable_sequence_parallel=True)

    _verify_tp_sharding(_rank, model_tp, model)

    model_tp.eval()
    model.eval()

    vocab_size = model.config.vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)

    with torch.no_grad():
        logits = model(input_ids).logits
        logits_tp = _to_local(model_tp(input_ids).logits)

    diff = (logits - logits_tp).abs()
    assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model outputs differ. Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_tp_backward_impl(rank, model_path, model_class, atol, rtol):
    """Implementation for comparing TP and non-TP model backward passes."""
    set_seed(0)

    model_tp, model, device = _load_tp_and_reference_models(model_path, model_class, enable_sequence_parallel=True)
    tp_plan = _get_active_tp_plan(model_tp)
    model_tp.train()
    model.train()

    vocab_size = model.config.vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    set_seed(0)
    labels = torch.randint(0, vocab_size, (2, 64)).to(device)

    loss = model(input_ids, labels=labels, use_cache=False).loss
    loss.backward()

    loss_tp = model_tp(input_ids, labels=labels, use_cache=False).loss
    loss_tp.backward()

    loss_tp_local = _to_local(loss_tp)
    assert torch.allclose(loss, loss_tp_local, atol=atol, rtol=rtol), (
        f"TP and non-TP model losses differ. "
        f"Non-TP loss: {loss.item()}, TP loss: {loss_tp_local.item()}, "
        f"Diff: {(loss - loss_tp_local).abs().item()}"
    )

    # Compare gradients for matching parameters
    world_size = dist.get_world_size()

    # Debug: check tied weights and parameter alignment
    failed_grads = {}
    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = _to_local(param_tp.grad)

            # Slice reference gradient to match local shard if parameter is sharded
            if grad.shape != grad_tp.shape:
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
                        param_plan = _get_parameter_tp_plan(name, tp_plan, is_weight=True)
                        if _is_packed_colwise_plan(param_plan):
                            # interleaved slicing
                            grad = get_packed_grad_shard(grad, world_size, rank, dim)
                        else:
                            # regular slicing
                            shard_size = grad_tp.size(dim)
                            start = rank * shard_size
                            grad = grad.narrow(dim, start, shard_size)
                        break

            if not torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol):
                max_diff = (grad.cpu() - grad_tp.cpu()).abs().max().item()
                ref_abs_max = grad.cpu().abs().max().item()
                tp_abs_max = grad_tp.cpu().abs().max().item()
                ratio = tp_abs_max / ref_abs_max if ref_abs_max > 0 else float("inf")
                failed_grads[name] = (max_diff, ref_abs_max, tp_abs_max, ratio)

    assert not failed_grads, f"Gradients differ for {len(failed_grads)} parameter(s):\n" + "\n".join(
        f"  {name}: max_diff={v[0]:.6f}, ref_max={v[1]:.6f}, tp_max={v[2]:.6f}, tp/ref ratio={v[3]:.4f}"
        for name, v in failed_grads.items()
    )

    dist.barrier()


def _test_tp_generation_impl(_rank, model_path, model_class, atol, rtol, max_new_tokens):
    """Implementation for comparing TP and non-TP model generation outputs (direct load path)."""
    set_seed(0)

    model_tp, model, device = _load_tp_and_reference_models(model_path, model_class)
    model_tp.eval()
    model.eval()

    set_seed(0)
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "output_scores": True,
        "return_dict_in_generate": True,
        "use_cache": True,
    }

    with torch.no_grad():
        output = model.generate(input_ids, **generation_kwargs)
        output_tp = model_tp.generate(input_ids, **generation_kwargs)

    # Compare logits/scores at each generation step
    scores = torch.stack(output.scores)
    scores_tp = torch.stack([_to_local(s) for s in output_tp.scores])

    diff = (scores - scores_tp).abs()
    assert torch.allclose(scores, scores_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model generation logits differ (direct load path). "
        f"Max diff: {diff.max().item()} | Mean diff: {diff.mean().item()}"
    )

    # Compare generated token sequences
    sequences_tp = _to_local(output_tp.sequences)
    assert torch.equal(output.sequences, sequences_tp), (
        f"TP and non-TP model generated different token sequences (direct load path). "
        f"Non-TP: {output.sequences.tolist()} | TP: {sequences_tp.tolist()}"
    )

    dist.barrier()


def _test_tp_generation_quantized_impl(_rank, model_path, model_class, max_new_tokens):
    """Implementation for comparing TP+quantized and non-TP quantized generation (sequence equality)."""
    set_seed(0)

    quantization_config = TorchAoConfig(Float8WeightOnlyConfig())

    model_tp = model_class.from_pretrained(
        model_path, distributed_config=DistributedConfig(tp_plan="auto"), quantization_config=quantization_config
    )
    dist.barrier()

    device = model_tp.device
    model = model_class.from_pretrained(model_path, quantization_config=quantization_config)
    model = model.to(device)

    model_tp.eval()
    model.eval()

    vocab_size = model.config.vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "output_scores": True,
        "return_dict_in_generate": True,
        "use_cache": True,
    }

    with torch.no_grad():
        output = model.generate(input_ids, **generation_kwargs)
        output_tp = model_tp.generate(input_ids, **generation_kwargs)

    print(f"[Rank {_rank}] Non-TP-quantized model tokens: {output.sequences[0].tolist()}")
    print(f"[Rank {_rank}] TP-quantized tokens:     {output_tp.sequences[0].tolist()}")
    print(f"[Rank {_rank}] Sequences match: {torch.equal(output.sequences, output_tp.sequences)}")

    # Compare generated token sequences (allow up to 25% mismatch due to Float8 quantization
    # scale differences between full-weight and sharded-weight quantization)
    # NOTE(3outeille): Some models have no perfect match. Investigate better the discrepancy but for now low priority.
    seq = output.sequences[0]
    seq_tp = output_tp.sequences[0]
    min_len = min(len(seq), len(seq_tp))
    match_count = (seq[:min_len] == seq_tp[:min_len]).sum().item()
    match_ratio = match_count / max(len(seq), len(seq_tp))
    assert match_ratio >= 0.75, (
        f"non-TP-quantized + TP-quantized model generated too many different tokens "
        f"(match ratio: {match_ratio:.2%}, threshold: 75%).\n"
        f"Non-TP+quantized: {output.sequences.tolist()} \n TP+quantized: {output_tp.sequences.tolist()}"
    )

    dist.barrier()


class TensorParallelTesterMixin(ABC):
    """
    Mixin for tensor parallel tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - causal_lm_class, base_model_class, etc.

    This mixin adds tensor parallel-specific tests using that infrastructure.
    """

    # ============================================================
    # Configuration (can be overridden per model)
    # ============================================================
    tensor_parallel_size: int = 2
    tensor_parallel_atol: float = 5e-3
    tensor_parallel_rtol: float = 5e-3

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    # ============================================================
    # Helper methods
    # ============================================================
    def _has_tp_plan(self) -> bool:
        """Check if model has a tensor parallel plan defined."""
        config = self.model_tester.get_config()
        return hasattr(config, "base_model_tp_plan") and config.base_model_tp_plan is not None

    def _get_tp_model_class(self):
        """Get the model class to use for TP tests (prefers *ForCausalLM)."""
        if hasattr(self.model_tester, "causal_lm_class") and self.model_tester.causal_lm_class is not None:
            return self.model_tester.causal_lm_class
        return self.all_model_classes[0]

    def _skip_if_not_supported(self):
        """Check and skip test if TP is not supported for this model/environment."""
        if not is_torch_greater_or_equal("2.9"):
            self.skipTest("Tensor parallel tests require torch >= 2.9")

        if torch.cuda.is_available() or torch.xpu.is_available():
            self.skipTest("Tensor parallel mixin tests are CPU-only and should not run on GPU or XPU machines")

        if os.cpu_count() < self.tensor_parallel_size:
            self.skipTest(
                f"Tensor parallel tests require at least {self.tensor_parallel_size} CPUs, "
                f"but only {os.cpu_count()} available"
            )

        if not hasattr(self.model_tester, "causal_lm_class") or self.model_tester.causal_lm_class is None:
            self.skipTest("Model tester does not have causal_lm_class (not using CausalLMModelTester)")

        if not self._has_tp_plan():
            self.skipTest("Model does not have a tensor parallel plan (base_model_tp_plan)")

        # # Skip encoder-decoder models (TP not supported)
        # if getattr(self, "is_encoder_decoder", False):
        #     self.skipTest("TP tests not supported for encoder-decoder models")

        # # Skip VLM models for now
        # config = self.model_tester.get_config()
        # if hasattr(config, "vision_config") and config.vision_config is not None:
        #     self.skipTest("VLM models are not yet supported in TP tests")

    @is_tensor_parallel_test
    def test_tp_forward(self):
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_forward_impl)(tmp_dir, model_class, atol, rtol)

    @is_tensor_parallel_test
    def test_tp_backward(self):
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)

            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_backward_impl)(tmp_dir, model_class, atol, rtol)

    @is_tensor_parallel_test
    def test_tp_generation(self):
        # Test TP generation: unfused checkpoint → conversion mapping (if needed) → TP sharding → model → generate
        self._skip_if_not_supported()

        config = self.model_tester.get_config()

        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol
        max_new_tokens = 25

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_tp_generation_impl)(
                tmp_dir, model_class, atol, rtol, max_new_tokens
            )

    @is_tensor_parallel_test
    def test_tp_generation_quantized(self):
        self._skip_if_not_supported()

        if not is_torchao_available():
            self.skipTest("Test requires torchao")

        self.skipTest("Quantization is not currently supported with distributed training")

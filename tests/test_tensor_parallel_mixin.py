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
from transformers.distributed.sharding_utils import _replicate_dtensor
from transformers.distributed.tensor_parallel import _get_parameter_tp_plan
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


def _global_wrapper(rank, func, tp, port, backend, func_args, func_kwargs):
    """Wrapper to set up distributed environment and run the test function."""

    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp
    setup_dist_env(rank, world_size, port)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def _init_distributed(tp: int, max_retries: int = 5, backend: str = "gloo"):
    """Decorator to initialize distributed environment and spawn processes."""

    def _init_distributed_inner(func):
        def wrapper(*args, **kwargs):
            world_size = tp
            for attempt in range(max_retries):
                port = _find_free_port()
                spawn_args = (func, tp, port, backend, args, kwargs)
                try:
                    mp.spawn(_global_wrapper, args=spawn_args, nprocs=world_size)
                    return
                except ProcessRaisedException as e:
                    if "EADDRINUSE" in str(e) and attempt < max_retries - 1:
                        continue
                    raise

        return wrapper

    return _init_distributed_inner


def _load_distributed_and_reference_models(model_path, model_class, mode: str):
    """Load a distributed model and an unsharded reference model for comparison."""
    tp_size = dist.get_world_size()
    if mode == "ep":
        distributed_config = DistributedConfig(tp_size=tp_size, enable_expert_parallel=True)
    elif mode == "sp":
        distributed_config = DistributedConfig(tp_size=tp_size, enable_sequence_parallel=True)
    else:
        distributed_config = DistributedConfig(tp_size=tp_size, enable_sequence_parallel=False)

    from_pretrained_kwargs = {"distributed_config": distributed_config}
    if mode != "ep":
        from_pretrained_kwargs["attn_implementation"] = "sdpa"

    model_dist = model_class.from_pretrained(model_path, **from_pretrained_kwargs)
    dist.barrier()

    device = model_dist.device
    if mode == "ep":
        model_ref = model_class.from_pretrained(model_path)
    else:
        model_ref = model_class.from_pretrained(model_path, attn_implementation="sdpa")
    model_ref = model_ref.to(device)

    return model_dist, model_ref, device


def _test_forward_impl(_rank, model_path, model_class, atol, rtol, mode: str):
    """Compare distributed and reference model forward outputs for the given mode."""
    assert mode in ("tp", "sp", "ep")
    set_seed(0)

    model_dist, model_ref, device = _load_distributed_and_reference_models(model_path, model_class, mode)

    model_dist.eval()
    model_ref.eval()

    vocab_size = model_ref.config.vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)

    with torch.no_grad():
        logits_ref = model_ref(input_ids).logits
        logits_dist = model_dist(input_ids).logits
        if mode != "ep":
            logits_dist = _to_local(logits_dist)

    diff = (logits_ref - logits_dist).abs()
    assert torch.allclose(logits_ref, logits_dist, atol=atol, rtol=rtol), (
        f"{mode.upper()} and reference model outputs differ. Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _get_packed_grad_shard(grad, world_size, rank, dim):
    """Get the correct shard of a packed gradient (matching get_packed_weights interleaved logic).

    Packed weights (packed_colwise, moe_tp_gate_up_colwise) are sharded with interleaving:
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


def _test_backward_impl(rank, model_path, model_class, atol, rtol, mode: str):
    """Compare distributed and reference model backward passes for the given mode."""
    assert mode in ("tp", "sp", "ep")
    set_seed(0)

    model_dist, model_ref, device = _load_distributed_and_reference_models(model_path, model_class, mode)
    applied_plan = getattr(model_dist, f"_{mode}_plan", None) or {}
    model_dist.train()
    model_ref.train()

    vocab_size = model_ref.config.vocab_size
    set_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
    set_seed(0)
    labels = torch.randint(0, vocab_size, (2, 64)).to(device)

    loss_ref = model_ref(input_ids, labels=labels, use_cache=False).loss
    loss_ref.backward()

    loss_dist = model_dist(input_ids, labels=labels, use_cache=False).loss
    loss_dist.backward()

    if mode == "ep":
        loss_dist_value = loss_dist
    else:
        loss_dist_value = _to_local(loss_dist)

    assert torch.allclose(loss_ref, loss_dist_value, atol=atol, rtol=rtol), (
        f"{mode.upper()} and reference model losses differ. "
        f"Reference loss: {loss_ref.item()}, {mode.upper()} loss: {loss_dist_value.item()}, "
        f"Diff: {(loss_ref - loss_dist_value).abs().item()}"
    )

    # Compare parameter gradients
    world_size = dist.get_world_size()
    failed_grads = []
    for (name, param), (_, param_dist) in zip(model_ref.named_parameters(), model_dist.named_parameters()):
        if param.grad is None or param_dist.grad is None:
            continue

        grad = param.grad
        grad_dist = _to_local(param_dist.grad)

        if grad.shape != grad_dist.shape:
            for dim in range(grad.ndim):
                if grad.size(dim) != grad_dist.size(dim):
                    param_plan = _get_parameter_tp_plan(name, applied_plan, is_weight=True)
                    if param_plan in ("packed_colwise", "moe_tp_gate_up_colwise"):
                        grad = _get_packed_grad_shard(grad, world_size, rank, dim)
                    else:
                        shard_size = grad_dist.size(dim)
                        start = rank * shard_size
                        grad = grad.narrow(dim, start, shard_size)
                    break

        try:
            torch.testing.assert_close(grad, grad_dist, atol=atol, rtol=rtol)
        except AssertionError as e:
            failed_grads.append(f"{name}: {e}")

    assert not failed_grads, "Gradients differ:\n" + "\n".join(failed_grads)

    dist.barrier()


def _test_generation_impl(_rank, model_path, model_class, atol, rtol, mode: str, max_new_tokens):
    """Compare distributed and reference model generation outputs for the given mode."""
    assert mode in ("tp", "ep")
    set_seed(0)

    model_dist, model_ref, device = _load_distributed_and_reference_models(model_path, model_class, mode)
    model_dist.eval()
    model_ref.eval()

    set_seed(0)
    vocab_size = model_ref.config.vocab_size
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
        output = model_ref.generate(input_ids, **generation_kwargs)
        output_dist = model_dist.generate(input_ids, **generation_kwargs)

    # Compare logits/scores at each generation step
    scores = torch.stack(output.scores)
    if mode == "ep":
        scores_dist = torch.stack(output_dist.scores)
    else:
        scores_dist = torch.stack([_to_local(s) for s in output_dist.scores])

    diff = (scores - scores_dist).abs()
    assert torch.allclose(scores, scores_dist, atol=atol, rtol=rtol), (
        f"{mode.upper()} and reference model generation logits differ. "
        f"Max diff: {diff.max().item()} | Mean diff: {diff.mean().item()}"
    )

    # Compare generated token sequences
    sequences_dist = output_dist.sequences if mode == "ep" else _to_local(output_dist.sequences)
    assert torch.equal(output.sequences, sequences_dist), (
        f"{mode.upper()} and reference model generated different token sequences. "
        f"Reference: {output.sequences.tolist()} | {mode.upper()}: {sequences_dist.tolist()}"
    )

    dist.barrier()


def _test_generation_quantized_impl(_rank, model_path, model_class, max_new_tokens):
    """Implementation for comparing TP+quantized and non-TP quantized generation (sequence equality)."""
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

    Distributed test modes:
      - "tp": _tp_plan, enable_sequence_parallel=False — inference-style TP.
      - "sp": _sp_plan with per-layer MLP entries — training-style sequence parallel.
      - "ep": _ep_plan, enable_expert_parallel=True — expert parallel on MoE weights.
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
    def _has_ep_plan(self) -> bool:
        """Check if model has an expert parallel plan defined."""
        config = self.model_tester.get_config()
        return hasattr(config, "base_model_ep_plan") and config.base_model_ep_plan is not None

    def _has_tp_plan(self) -> bool:
        """Check if model has a tensor parallel plan defined."""
        config = self.model_tester.get_config()
        return hasattr(config, "base_model_tp_plan") and config.base_model_tp_plan is not None

    def _has_sp_plan(self) -> bool:
        config = self.model_tester.get_config()
        return getattr(config, "base_model_sp_plan", None) and config.base_model_sp_plan is not None

    def _get_model_class(self):
        """Get the model class to use for TP tests (prefers *ForCausalLM)."""
        if hasattr(self.model_tester, "causal_lm_class") and self.model_tester.causal_lm_class is not None:
            return self.model_tester.causal_lm_class
        return self.all_model_classes[0]

    def _assert_mixed_mlp_layers(self, model, config):
        """If the config interleaves dense and sparse/MoE layers, fail fast unless the tiny test
        model actually built both kinds.
        """
        layer_types = getattr(config, "mlp_layer_types", None)
        if layer_types is not None:
            # MoE marker is 'sparse' or 'moe' depending on the model; dense is always 'dense'.
            intends_mixed = "dense" in layer_types and any(t != "dense" for t in layer_types)
        else:
            sparse_step = getattr(config, "decoder_sparse_step", 1) or 1
            intends_mixed = sparse_step > 1 or bool(getattr(config, "mlp_only_layers", None))
        if not intends_mixed:
            return

        kinds = {
            "moe" if hasattr(module, "experts") else "dense"
            for name, module in model.named_modules()
            if name.endswith(".mlp")
        }
        # Empty means this model's feed-forward isn't named `.mlp` (e.g. `block_sparse_moe`); the
        # structural check doesn't apply, so skip rather than spuriously fail its TP test.
        if not kinds:
            return
        self.assertEqual(
            kinds,
            {"dense", "moe"},
            f"{type(model).__name__}: config interleaves dense/MoE mlp layers but the tiny test config "
            f"built only {kinds}. Adjust the model tester (num_hidden_layers / decoder_sparse_step / "
            f"mlp_layer_types) so both kinds exist.",
        )

    def _skip_if_mode_not_supported(self, mode: str) -> None:
        """Check and skip test if the mode is not supported for this model/environment."""
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

        if mode == "tp" and not self._has_tp_plan():
            self.skipTest("Model does not have a tensor parallel plan (base_model_tp_plan)")
        elif mode == "sp" and not self._has_sp_plan():
            self.skipTest("Model does not have a sequence parallel plan (base_model_sp_plan)")
        elif mode == "ep" and not self._has_ep_plan():
            self.skipTest("Model does not have an expert parallel plan (base_model_ep_plan)")

        # # Skip encoder-decoder models (TP not supported)
        # if getattr(self, "is_encoder_decoder", False):
        #     self.skipTest("TP tests not supported for encoder-decoder models")

    @is_tensor_parallel_test
    def test_tp_forward(self):
        """Tensor parallel forward (_tp_plan, no sequence parallel)."""
        self._skip_if_mode_not_supported("tp")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            self._assert_mixed_mlp_layers(model, config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_forward_impl)(tmp_dir, model_class, atol, rtol, "tp")

    @is_tensor_parallel_test
    def test_tp_backward(self):
        """Tensor parallel backward (_tp_plan, no sequence parallel)."""
        self._skip_if_mode_not_supported("tp")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            self._assert_mixed_mlp_layers(model, config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_backward_impl)(
                tmp_dir, model_class, atol, rtol, "tp"
            )

    @is_tensor_parallel_test
    def test_sp_forward(self):
        """Sequence-parallel forward (_sp_plan including per-layer MLP)."""
        self._skip_if_mode_not_supported("sp")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            self._assert_mixed_mlp_layers(model, config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_forward_impl)(tmp_dir, model_class, atol, rtol, "sp")

    @is_tensor_parallel_test
    def test_sp_backward(self):
        """Sequence-parallel backward (_sp_plan including per-layer MLP)."""
        self._skip_if_mode_not_supported("sp")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            self._assert_mixed_mlp_layers(model, config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_backward_impl)(
                tmp_dir, model_class, atol, rtol, "sp"
            )

    @is_tensor_parallel_test
    def test_ep_forward(self):
        """Expert-parallel forward (_ep_plan)."""
        self._skip_if_mode_not_supported("ep")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_forward_impl)(tmp_dir, model_class, atol, rtol, "ep")

    @is_tensor_parallel_test
    def test_ep_backward(self):
        """Expert-parallel backward (_ep_plan)."""
        self._skip_if_mode_not_supported("ep")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_backward_impl)(
                tmp_dir, model_class, atol, rtol, "ep"
            )

    @is_tensor_parallel_test
    def test_tp_generation(self):
        """Tensor parallel generation (_tp_plan)."""
        self._skip_if_mode_not_supported("tp")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_generation_impl)(
                tmp_dir, model_class, atol, rtol, "tp", 25
            )

    @is_tensor_parallel_test
    def test_ep_generation(self):
        """Expert-parallel generation (_ep_plan)."""
        self._skip_if_mode_not_supported("ep")
        config = self.model_tester.get_config()
        model_class = self._get_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        with tempfile.TemporaryDirectory() as tmp_dir:
            set_seed(42)
            model = model_class(config)
            model.save_pretrained(tmp_dir, save_original_format=True)
            _init_distributed(tp=self.tensor_parallel_size)(_test_generation_impl)(
                tmp_dir, model_class, atol, rtol, "ep", 25
            )

    @is_tensor_parallel_test
    def test_tp_generation_quantized(self):
        self._skip_if_mode_not_supported("tp")

        if not is_torchao_available():
            self.skipTest("Test requires torchao")

        self.skipTest("Quantization is not currently supported with distributed training (dtensor)")

    # TODO(3outeille): add test_tp_ep_forward, test_tp_ep_backward, test_tp_ep_generation, test_tp_ep_generation_quantized
    # TODO(3outeille): add test_sp_ep_forward, test_sp_ep_backward

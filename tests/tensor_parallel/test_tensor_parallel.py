# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# Run all tests: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py
# Run dense tests: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py -k "dense"
# Run MoE tests: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py -k "moe"
# Collect tests: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py --collect-only
import os
import tempfile
import warnings

import pytest
from safetensors import safe_open

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, is_torch_available, set_seed
from transformers.integrations.tensor_parallel import get_packed_weights, repack_weights
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    get_torch_dist_unique_port,
    require_huggingface_hub_greater_or_equal,
    require_torch_multi_accelerator,
    torch_device,
)
from transformers.utils import is_torch_greater_or_equal


# Tensor parallel tests require torch >= 2.9 for proper torch.compile support with distributed collectives
# Newer versions of PyTorch has torch.library.register_autograd in https://github.com/pytorch/pytorch/blob/8bcedd6e6029cce5f3a3731dd59be4941414c731/torch/distributed/_functional_collectives.py#L630
# that fix the warning "autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it"
# NOTE(3outeille): need to double check if it works with older version of torch
pytestmark = pytest.mark.skipif(
    not is_torch_greater_or_equal("2.9"),
    reason="Tensor parallel tests require torch >= 2.9 for torch.compile support with distributed collectives",
)


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


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


def global_wrapper(rank, func, tp, port, func_args, func_kwargs):
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


def init_distributed(tp: int):
    def _init_distributed(func):
        def wrapper(*args, **kwargs):
            world_size = tp
            port = get_torch_dist_unique_port()
            spawn_args = (func, tp, port, args, kwargs)
            mp.spawn(global_wrapper, args=spawn_args, nprocs=world_size)

        return wrapper

    return _init_distributed


def skip_if_insufficient_devices(nproc_per_node):
    """Skip test if there aren't enough devices available."""
    if backend_device_count(torch_device) < nproc_per_node:
        pytest.skip(f"Need at least {nproc_per_node} devices, have {backend_device_count(torch_device)}")


class TestTensorParallelUtils(TestCasePlus):
    def test_packed_unpacked_conversion(self):
        WORLD_SIZE = 2
        PACKED_BLOCK_SIZE = 800
        SHARDING_DIM = 2
        NUM_BLOCKS = 2

        original_packed_weights = torch.randn(4, 512, 2 * PACKED_BLOCK_SIZE)
        original_packed_weights.get_dtype = lambda: "F32"  # get_packed_weights expects PySlice object
        empty_param = torch.empty(4, 512, 2 * PACKED_BLOCK_SIZE)

        class MockDeviceMesh:
            def size(self):
                return WORLD_SIZE

        mock_mesh = (
            MockDeviceMesh()
        )  # get_packed_weights only calls `.size()`, do this to avoid doing actual distributed run

        packed_weights_0 = get_packed_weights(original_packed_weights, empty_param, mock_mesh, 0, SHARDING_DIM)
        packed_weights_1 = get_packed_weights(original_packed_weights, empty_param, mock_mesh, 1, SHARDING_DIM)

        # simulate all gather of sharded weights
        packed_weights = torch.cat([packed_weights_0, packed_weights_1], dim=SHARDING_DIM)
        unpacked_weights = repack_weights(packed_weights, SHARDING_DIM, WORLD_SIZE, NUM_BLOCKS)

        assert torch.allclose(unpacked_weights, original_packed_weights)


class TestTensorParallelProperties(TestCasePlus):
    def test_tp_plan_property_setter_getter(self):
        """Test that tp_plan property can be set and retrieved correctly."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting empty plan
        model.tp_plan = {}
        self.assertEqual(model.tp_plan, {})

        # Test setting a valid plan
        valid_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        model.tp_plan = valid_plan
        self.assertEqual(model.tp_plan, valid_plan)

        # Test updating the plan
        model.tp_plan.update({"model.layers.*.self_attn.k_proj": "colwise"})
        expected_plan = {"model.layers.*.self_attn.q_proj": "colwise", "model.layers.*.self_attn.k_proj": "colwise"}
        self.assertEqual(model.tp_plan, expected_plan)

        # Test overriding existing entry
        model.tp_plan.update({"model.layers.*.self_attn.q_proj": "rowwise"})
        expected_plan = {
            "model.layers.*.self_attn.q_proj": "rowwise",
            "model.layers.*.self_attn.k_proj": "colwise",
        }
        self.assertEqual(model.tp_plan, expected_plan)

    def test_tp_plan_validation_invalid_style(self):
        """Test that invalid parallel styles are rejected."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test invalid parallel style
        with self.assertRaises(ValueError) as context:
            model.tp_plan = {"layers.*.self_attn.q_proj": "invalid_style"}

        self.assertIn("Unsupported tensor parallel style 'invalid_style'", str(context.exception))
        self.assertIn("Supported styles are", str(context.exception))

    def test_tp_plan_validation_nonexistent_layer_warning(self):
        """Test that warnings are issued for non-existent layer patterns."""

        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test warning for non-existent layer pattern
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.tp_plan = {"nonexistent.*.layer": "colwise"}

            # Check that a warning was issued
            self.assertTrue(len(w) > 0)
            warning_message = str(w[0].message)
            self.assertIn("Layer pattern 'nonexistent.*.layer' does not match any parameters", warning_message)

    def test_tp_plan_valid_layer_patterns(self):
        """Test that valid layer patterns are accepted without warnings."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test valid layer patterns that should match the model structure
        valid_plans = [
            {"model.layers.*.self_attn.q_proj": "colwise"},
            {"model.layers.*.self_attn.k_proj": "rowwise"},
            {"model.layers.*.mlp.gate_proj": "colwise"},
        ]

        for plan in valid_plans:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model.tp_plan = plan

                # Filter out any warnings that are not about layer patterns
                layer_warnings = [
                    warning
                    for warning in w
                    if "Layer pattern" in str(warning.message)
                    and "does not match any parameters" in str(warning.message)
                ]

                # Should not have layer pattern warnings for valid patterns
                self.assertEqual(
                    len(layer_warnings),
                    0,
                    f"Unexpected warning for valid pattern {plan}: {[str(w.message) for w in layer_warnings]}",
                )

        # Verify the final plan was set correctly
        self.assertEqual(model.tp_plan, valid_plans[-1])

    def test_tp_plan_none_handling(self):
        """Test that None values are handled correctly."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting None
        model.tp_plan = None
        self.assertEqual(model.tp_plan, {})

        # Test setting a plan after None
        model.tp_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        self.assertEqual(model.tp_plan, {"model.layers.*.self_attn.q_proj": "colwise"})


# ====== TEST FUNCTIONS ======
def _test_model_dense_forward_impl(rank, mode, dtype=torch.float32):
    """Implementation for comparing TP and non-TP model outputs."""
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    set_seed(42)

    atol, rtol = (1e-5, 1e-5)

    # Load tokenizer and prepare inputs - same for both models
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    prompt = "Can I help"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Load TP model first to determine device
    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, tp_plan="auto")
    dist.barrier()
    if mode == "eval":
        model_tp.eval()
    else:
        model_tp.train()

    # Load non-TP model and move to same device as TP model
    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)

    if mode == "eval":
        model.eval()
    else:
        model.train()

    # Prepare inputs on the same device
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        outputs_tp = model_tp(input_ids)
        logits_tp = outputs_tp.logits

    diff = (logits - logits_tp).abs()
    assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model outputs differ (dtype={dtype}). "
        f"Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_model_dense_backward_pass_impl(rank, dtype=torch.float32):
    """Implementation for comparing TP and non-TP model backward passes."""
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    set_seed(42)

    # Set tolerance based on dtype
    atol, rtol = (1e-5, 1e-5)

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, tp_plan="auto")
    dist.barrier()
    model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)
    model.train()

    batch_size, seq_length = 2, 1024
    set_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length)).to(device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_length)).to(device)

    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    outputs_tp = model_tp(input_ids, labels=labels)
    loss_tp = outputs_tp.loss
    loss_tp.backward()

    assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model losses differ (dtype={dtype}). Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, Diff: {(loss - loss_tp).abs().item()}"
    )

    # Compare gradients for matching parameters
    # Note: TP model may have sharded parameters, so we slice the reference gradient to match
    world_size = dist.get_world_size()
    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # Slice reference gradient to match local shard if parameter is sharded
            if grad.shape != grad_tp.shape:
                # Find the dimension that differs and slice accordingly
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
                        # Packed weights (gate_up_proj) use interleaved sharding
                        if "gate_up_proj" in name:
                            grad = get_packed_grad_shard(grad, world_size, rank, dim)
                        else:
                            # Regular weights use simple chunking
                            shard_size = grad_tp.size(dim)
                            start = rank * shard_size
                            grad = grad.narrow(dim, start, shard_size)
                        break

            assert torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol), (
                f"Gradients differ for parameter {name} (dtype={dtype}). Max diff: {(grad.cpu() - grad_tp.cpu()).abs().max().item()} | Min diff: {(grad.cpu() - grad_tp.cpu()).abs().min().item()}"
            )

    dist.barrier()


def _test_model_dense_forward_compile_impl(rank, mode, dtype=torch.float32):
    """Implementation for comparing TP and non-TP model outputs with torch.compile."""
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    set_seed(42)

    # Set tolerance based on dtype
    atol, rtol = (1e-5, 1e-5)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    prompt = "Can I help"
    inputs = tokenizer(prompt, return_tensors="pt")

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, tp_plan="auto")
    dist.barrier()
    if mode == "eval":
        model_tp.eval()
    else:
        model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)

    if mode == "eval":
        model.eval()
    else:
        model.train()

    # Compile both models
    model.forward = torch.compile(model.forward)
    model_tp.forward = torch.compile(model_tp.forward)

    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        outputs_tp = model_tp(input_ids)
        logits_tp = outputs_tp.logits

    assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model outputs differ (dtype={dtype}). Max diff: {(logits - logits_tp).abs().max().item()} | Min diff: {(logits - logits_tp).abs().min().item()}"
    )

    dist.barrier()


def _test_model_dense_backward_compile_impl(rank, dtype=torch.float32):
    """Implementation for comparing TP and non-TP model backward passes with torch.compile."""
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    set_seed(42)

    # Set tolerance based on dtype
    atol, rtol = (1e-5, 1e-5)

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, tp_plan="auto")
    dist.barrier()
    model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)
    model.train()

    # Compile both models
    model.forward = torch.compile(model.forward)
    model_tp.forward = torch.compile(model_tp.forward)

    batch_size, seq_length = 2, 1024
    set_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length)).to(device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_length)).to(device)

    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    outputs_tp = model_tp(input_ids, labels=labels)
    loss_tp = outputs_tp.loss
    loss_tp.backward()

    assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP model losses differ (dtype={dtype}). Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, Diff: {(loss - loss_tp).abs().item()}"
    )

    # Compare gradients for matching parameters
    world_size = dist.get_world_size()
    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # Slice reference gradient to match local shard if parameter is sharded
            if grad.shape != grad_tp.shape:
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
                        # Packed weights (gate_up_proj) use interleaved sharding
                        if "gate_up_proj" in name:
                            grad = get_packed_grad_shard(grad, world_size, rank, dim)
                        else:
                            # Regular weights use simple chunking
                            shard_size = grad_tp.size(dim)
                            start = rank * shard_size
                            grad = grad.narrow(dim, start, shard_size)
                        break

            assert torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol), (
                f"Gradients differ for parameter {name} (dtype={dtype}). Max diff: {(grad.cpu() - grad_tp.cpu()).abs().max().item()}"
            )

    dist.barrier()


def _test_model_dense_save_impl(rank, tmp_dir):
    """Implementation of test_model_save for distributed execution."""
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    if dist.is_initialized():
        kwargs = {"tp_plan": "auto"}
        result_dir = f"{tmp_dir}/tp"
    else:
        kwargs = {}
        result_dir = f"{tmp_dir}/nontp"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.save_pretrained(result_dir)


# ====== DENSE MODEL TESTS ======
@pytest.mark.parametrize("nproc_per_node", [2])
@pytest.mark.parametrize("mode", ["train", "eval"])
@require_torch_multi_accelerator
def test_model_dense_forward(nproc_per_node, mode):
    """Test that TP and non-TP models produce the same outputs."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(tp=nproc_per_node)(_test_model_dense_forward_impl)(mode, torch.float32)


@pytest.mark.parametrize("nproc_per_node", [2])
@require_torch_multi_accelerator
def test_model_dense_backward_pass(nproc_per_node):
    """Test that TP and non-TP models produce the same gradients."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(tp=nproc_per_node)(_test_model_dense_backward_pass_impl)(torch.float32)


@pytest.mark.parametrize("nproc_per_node", [2])
@pytest.mark.parametrize("mode", ["train", "eval"])
@require_torch_multi_accelerator
def test_model_dense_forward_compile(nproc_per_node, mode):
    """Test that TP and non-TP models produce the same outputs with torch.compile."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(tp=nproc_per_node)(_test_model_dense_forward_compile_impl)(mode, torch.float32)


@pytest.mark.parametrize("nproc_per_node", [2])
@require_torch_multi_accelerator
def test_model_dense_backward_compile(nproc_per_node):
    """Test that TP and non-TP models produce the same gradients with torch.compile."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(tp=nproc_per_node)(_test_model_dense_backward_compile_impl)(torch.float32)


@pytest.mark.parametrize("nproc_per_node", [2])
@require_huggingface_hub_greater_or_equal("0.31.4")
@require_torch_multi_accelerator
def test_model_dense_save(nproc_per_node):
    """Test that TP model can be saved and matches non-TP version."""
    skip_if_insufficient_devices(nproc_per_node)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # First run with TP (distributed)
        init_distributed(tp=nproc_per_node)(_test_model_dense_save_impl)(tmp_dir)

        # Then run without TP (non-distributed)
        _test_model_dense_save_impl(0, tmp_dir)

        non_tp_model_path = os.path.join(tmp_dir, "nontp")
        tp_model_path = os.path.join(tmp_dir, "tp")

        for filename in os.listdir(non_tp_model_path):
            if not filename.endswith(".safetensors"):
                continue

            non_tp_model = safe_open(os.path.join(non_tp_model_path, filename), device="cpu", framework="pt")
            tp_model = safe_open(os.path.join(tp_model_path, filename), device="cpu", framework="pt")
            for non_tp_key in non_tp_model.keys():
                non_tp_tensor = non_tp_model.get_tensor(non_tp_key)
                tp_tensor = tp_model.get_tensor(non_tp_key)
                assert torch.allclose(non_tp_tensor, tp_tensor), f"Tensor with key: {non_tp_key} does not match"
                del non_tp_tensor, tp_tensor


def _test_model_moe_forward_impl(rank, mode, dtype=torch.float32):
    """Implementation for comparing TP and non-TP MoE model outputs."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    set_seed(42)

    # Set tolerance based on dtype
    atol, rtol = (1e-5, 1e-5)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    prompt = "Can I help"
    inputs = tokenizer(prompt, return_tensors="pt")

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, tp_plan="auto")
    dist.barrier()
    if mode == "eval":
        model_tp.eval()
    else:
        model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)

    if mode == "eval":
        model.eval()
    else:
        model.train()

    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        outputs_tp = model_tp(input_ids)
        logits_tp = outputs_tp.logits

    diff = (logits - logits_tp).abs()
    assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP MoE model outputs differ (dtype={dtype}). "
        f"Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_model_moe_backward_pass_impl(rank, dtype=torch.float32):
    """Implementation for comparing TP and non-TP MoE model backward passes."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    set_seed(42)

    atol, rtol = (1e-5, 1e-5)

    config = AutoConfig.from_pretrained(model_id)

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=dtype, tp_plan="auto")
    dist.barrier()
    model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=dtype)
    model = model.to(device)
    model.train()

    batch_size, seq_length = 2, 1024
    set_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)

    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    outputs_tp = model_tp(input_ids, labels=labels)
    loss_tp = outputs_tp.loss
    loss_tp.backward()

    assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP MoE model losses differ (dtype={dtype}). Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, Diff: {(loss - loss_tp).abs().item()}"
    )

    # Compare gradients for matching parameters
    world_size = dist.get_world_size()

    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # Slice reference gradient to match local shard if parameter is sharded
            if grad.shape != grad_tp.shape:
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
                        if "gate_up_proj" in name:
                            grad = get_packed_grad_shard(grad, world_size, rank, dim)
                        else:
                            shard_size = grad_tp.size(dim)
                            start = rank * shard_size
                            grad = grad.narrow(dim, start, shard_size)
                        break

            assert torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol), (
                f"Gradients differ for parameter {name} (dtype={dtype}). Max diff: {(grad.cpu() - grad_tp.cpu()).abs().max().item()}"
            )

    dist.barrier()


def _test_model_moe_forward_compile_impl(rank, mode, dtype=torch.float32, experts_implementation=None):
    """Implementation for comparing TP and non-TP MoE model outputs with torch.compile."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    set_seed(42)

    if dtype == torch.bfloat16:
        atol, rtol = (5e-3, 5e-3)
    else:
        atol, rtol = (1e-5, 1e-5)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    prompt = "Can I help"
    inputs = tokenizer(prompt, return_tensors="pt")

    model_tp = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, tp_plan="auto", experts_implementation=experts_implementation
    )
    dist.barrier()
    if mode == "eval":
        model_tp.eval()
    else:
        model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, experts_implementation=experts_implementation)
    model = model.to(device)

    if mode == "eval":
        model.eval()
    else:
        model.train()

    # Compile both models
    model.forward = torch.compile(model.forward)
    model_tp.forward = torch.compile(model_tp.forward)

    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        outputs_tp = model_tp(input_ids)
        logits_tp = outputs_tp.logits

    assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP MoE model outputs differ (dtype={dtype}). Max diff: {(logits - logits_tp).abs().max().item()} | Min diff: {(logits - logits_tp).abs().min().item()}"
    )

    dist.barrier()


def _test_model_moe_backward_compile_impl(rank, dtype=torch.float32, experts_implementation=None):
    """Implementation for comparing TP and non-TP MoE model backward passes with torch.compile."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    set_seed(42)

    # bfloat16 has lower precision
    if dtype == torch.bfloat16:
        atol, rtol = (1e-3, 1e-3)
    else:
        atol, rtol = (1e-5, 1e-5)

    config = AutoConfig.from_pretrained(model_id)

    model_tp = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, dtype=dtype, tp_plan="auto", experts_implementation=experts_implementation
    )
    dist.barrier()
    model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, dtype=dtype, experts_implementation=experts_implementation
    )
    model = model.to(device)
    model.train()

    model.forward = torch.compile(model.forward)
    model_tp.forward = torch.compile(model_tp.forward)

    batch_size, seq_length = 2, 1024
    set_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length)).to(device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_length)).to(device)

    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    outputs_tp = model_tp(input_ids, labels=labels)
    loss_tp = outputs_tp.loss
    loss_tp.backward()

    assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP MoE model losses differ (dtype={dtype}). Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, Diff: {(loss - loss_tp).abs().item()}"
    )

    # Compare gradients for matching parameters
    world_size = dist.get_world_size()

    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # Slice reference gradient to match local shard if parameter is sharded
            if grad.shape != grad_tp.shape:
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
                        if "gate_up_proj" in name:
                            grad = get_packed_grad_shard(grad, world_size, rank, dim)
                        else:
                            shard_size = grad_tp.size(dim)
                            start = rank * shard_size
                            grad = grad.narrow(dim, start, shard_size)
                        break

            assert torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol), (
                f"Gradients differ for parameter {name} (dtype={dtype}). Max diff: {(grad.cpu() - grad_tp.cpu()).abs().max().item()}"
            )

    dist.barrier()


def _test_model_moe_save_impl(rank, tmp_dir):
    """Implementation of test_model_save for MoE model distributed execution."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    if dist.is_initialized():
        kwargs = {"tp_plan": "auto"}
        result_dir = f"{tmp_dir}/tp"
    else:
        kwargs = {}
        result_dir = f"{tmp_dir}/nontp"

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", **kwargs)
    model.save_pretrained(result_dir)


# ====== MOE MODEL TESTS ======
@pytest.mark.parametrize("nproc_per_node", [2])
@pytest.mark.parametrize("mode", ["train", "eval"])
@require_torch_multi_accelerator
def test_model_moe_forward(nproc_per_node, mode):
    """Test that TP and non-TP MoE models produce the same outputs."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(tp=nproc_per_node)(_test_model_moe_forward_impl)(mode, torch.float32)


@pytest.mark.parametrize("nproc_per_node", [2])
@require_torch_multi_accelerator
def test_model_moe_backward_pass(nproc_per_node):
    """Test that TP and non-TP MoE models produce the same gradients."""
    skip_if_insufficient_devices(nproc_per_node)
    init_distributed(tp=nproc_per_node)(_test_model_moe_backward_pass_impl)(torch.float32)


@pytest.mark.parametrize("nproc_per_node", [2])
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("experts_implementation", ["batched_mm", "grouped_mm"])
@require_torch_multi_accelerator
def test_model_moe_forward_compile(nproc_per_node, mode, dtype, experts_implementation):
    """Test that TP and non-TP MoE models produce the same outputs with torch.compile."""
    skip_if_insufficient_devices(nproc_per_node)
    dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    init_distributed(tp=nproc_per_node)(_test_model_moe_forward_compile_impl)(
        mode, dtype, experts_implementation=experts_implementation
    )


@pytest.mark.parametrize("nproc_per_node", [2])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("experts_implementation", ["batched_mm", "grouped_mm"])
@require_torch_multi_accelerator
def test_model_moe_backward_compile(nproc_per_node, dtype, experts_implementation):
    """Test that TP and non-TP MoE models produce the same gradients with torch.compile."""
    skip_if_insufficient_devices(nproc_per_node)
    dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    init_distributed(tp=nproc_per_node)(_test_model_moe_backward_compile_impl)(
        dtype, experts_implementation=experts_implementation
    )


@pytest.mark.parametrize("nproc_per_node", [2])
@require_huggingface_hub_greater_or_equal("0.31.4")
@require_torch_multi_accelerator
def test_model_moe_save(nproc_per_node):
    """Test that TP MoE model can be saved and matches non-TP version."""
    skip_if_insufficient_devices(nproc_per_node)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # First run with TP (distributed)
        init_distributed(tp=nproc_per_node)(_test_model_moe_save_impl)(tmp_dir)

        # Then run without TP (non-distributed)
        _test_model_moe_save_impl(0, tmp_dir)

        non_tp_model_path = os.path.join(tmp_dir, "nontp")
        tp_model_path = os.path.join(tmp_dir, "tp")

        for filename in os.listdir(non_tp_model_path):
            if not filename.endswith(".safetensors"):
                continue

            non_tp_model = safe_open(os.path.join(non_tp_model_path, filename), device="cpu", framework="pt")
            tp_model = safe_open(os.path.join(tp_model_path, filename), device="cpu", framework="pt")
            for non_tp_key in non_tp_model.keys():
                non_tp_tensor = non_tp_model.get_tensor(non_tp_key)
                tp_tensor = tp_model.get_tensor(non_tp_key)
                assert torch.allclose(non_tp_tensor, tp_tensor), f"Tensor with key: {non_tp_key} does not match"
                del non_tp_tensor, tp_tensor

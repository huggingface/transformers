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
# Run specific config: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py -k "2Proc"
# Run multiple configs: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py -k "2Proc or 4Proc"
# Run spefic test: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py::TestTensorParallel2Proc::test_model_dense_forward_train
# Run tests with a specific prefix: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py::TestTensorParallel2Proc -k "forward"
import os
import tempfile
import warnings

from safetensors import safe_open

from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_available
from transformers.integrations.tensor_parallel import get_packed_weights, get_tensor_shard, repack_weights
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    get_torch_dist_unique_port,
    require_huggingface_hub_greater_or_equal,
    require_torch_multi_accelerator,
    torch_device,
)


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


def global_wrapper(rank, func, tp, port, func_args, func_kwargs):
    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp
    setup_dist_env(rank, world_size, port)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(rank)
        dist.init_process_group(backend="xccl", rank=rank, world_size=world_size)
    else:
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
        model_id = "JackFram/llama-68m"
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
        model.tp_plan.update({"model.layers.*.self_attn.q_proj": "colwise_rep"})
        expected_plan = {
            "model.layers.*.self_attn.q_proj": "colwise_rep",
            "model.layers.*.self_attn.k_proj": "colwise",
        }
        self.assertEqual(model.tp_plan, expected_plan)

    def test_tp_plan_validation_invalid_style(self):
        """Test that invalid parallel styles are rejected."""
        model_id = "JackFram/llama-68m"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test invalid parallel style
        with self.assertRaises(ValueError) as context:
            model.tp_plan = {"layers.*.self_attn.q_proj": "invalid_style"}

        self.assertIn("Unsupported tensor parallel style 'invalid_style'", str(context.exception))
        self.assertIn("Supported styles are", str(context.exception))

    def test_tp_plan_validation_nonexistent_layer_warning(self):
        """Test that warnings are issued for non-existent layer patterns."""

        model_id = "JackFram/llama-68m"
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
        model_id = "JackFram/llama-68m"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test valid layer patterns that should match the model structure
        valid_plans = [
            {"model.layers.*.self_attn.q_proj": "colwise"},
            {"model.layers.*.self_attn.k_proj": "rowwise"},
            {"model.layers.*.mlp.gate_proj": "colwise_rep"},
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
        model_id = "JackFram/llama-68m"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting None
        model.tp_plan = None
        self.assertEqual(model.tp_plan, {})

        # Test setting a plan after None
        model.tp_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        self.assertEqual(model.tp_plan, {"model.layers.*.self_attn.q_proj": "colwise"})


# ====== TEST FUNCTIONS ======
def _test_model_dense_forward_impl(rank, mode):
    """Implementation for comparing TP and non-TP model outputs."""
    model_id = "JackFram/llama-68m"

    # Ensure same random seed for reproducibility
    torch.manual_seed(0)

    # Load tokenizer and prepare inputs - same for both models
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    prompt = "Can I help"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Load TP model first to determine device
    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", tp_plan="auto")
    dist.barrier()
    if mode == "eval":
        model_tp.eval()
    else:
        model_tp.train()

    # Load non-TP model and move to same device as TP model
    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    model = model.to(device)

    if mode == "eval":
        model.eval()
    else:
        model.train()

    # Prepare inputs on the same device
    input_ids = inputs.input_ids.to(device)

    # Run forward pass on both models
    with torch.no_grad():
        # Non-TP model output
        outputs = model(input_ids)
        logits = outputs.logits

        # TP model output
        outputs_tp = model_tp(input_ids)
        logits_tp = outputs_tp.logits

    # Compare outputs - they should match
    assert torch.allclose(logits, logits_tp, atol=1e-5, rtol=1e-5), (
        f"TP and non-TP model outputs differ. Max diff: {(logits - logits_tp).abs().max().item()} | Min diff: {(logits - logits_tp).abs().min().item()}"
    )

    dist.barrier()


def _test_model_dense_backward_pass_impl(rank):
    """Implementation for comparing TP and non-TP model backward passes."""
    model_id = "JackFram/llama-68m"

    torch.manual_seed(0)

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, tp_plan="auto")
    dist.barrier()
    model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = model.to(device)
    model.train()

    batch_size, seq_length = 2, 10
    torch.manual_seed(42)  # Different seed for inputs to ensure they're deterministic
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)

    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    outputs_tp = model_tp(input_ids, labels=labels)
    loss_tp = outputs_tp.loss
    loss_tp.backward()

    assert torch.allclose(loss, loss_tp, atol=1e-5, rtol=1e-5), (
        f"TP and non-TP model losses differ. Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, Diff: {(loss - loss_tp).abs().item()}"
    )

    # Compare gradients for matching parameters
    # Note: TP model may have sharded parameters (DTensors), so we slice the reference gradient to match
    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            if isinstance(param_tp.data, dist.tensor.DTensor):
                placement = param_tp.data.placements[0]
                if hasattr(placement, "dim") and placement.dim is not None:
                    grad_shard = get_tensor_shard(grad, grad, param_tp.data.device_mesh, rank, placement.dim)
                else:
                    grad_shard = grad
            else:
                grad_shard = grad

            grad_tp_local = grad_tp.to_local() if isinstance(grad_tp, dist.tensor.DTensor) else grad_tp

            assert torch.allclose(grad_shard.cpu(), grad_tp_local.cpu(), atol=1e-5, rtol=1e-5), (
                f"Gradients differ for parameter {name}. Max diff: {(grad_shard.cpu() - grad_tp_local.cpu()).abs().max().item()} | Min diff: {(grad_shard.cpu() - grad_tp_local.cpu()).abs().min().item()}"
            )

    dist.barrier()


def _test_model_dense_forward_compile_impl(rank, mode):
    """Implementation for comparing TP and non-TP model outputs with torch.compile."""
    model_id = "JackFram/llama-68m"

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    prompt = "Can I help"
    inputs = tokenizer(prompt, return_tensors="pt")

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", tp_plan="auto")
    dist.barrier()
    if mode == "eval":
        model_tp.eval()
    else:
        model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
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

    assert torch.allclose(logits, logits_tp, atol=1e-5, rtol=1e-5), (
        f"TP and non-TP model outputs differ. Max diff: {(logits - logits_tp).abs().max().item()} | Min diff: {(logits - logits_tp).abs().min().item()}"
    )

    dist.barrier()


def _test_model_dense_save_impl(rank, tmp_dir):
    """Implementation of test_model_save for distributed execution."""
    model_id = "JackFram/llama-68m"

    if dist.is_initialized():
        kwargs = {"tp_plan": "auto"}
        result_dir = f"{tmp_dir}/tp"
    else:
        kwargs = {}
        result_dir = f"{tmp_dir}/nontp"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.save_pretrained(result_dir)


class TestTensorParallelBase(TestCasePlus):
    """Base class for tensor parallel tests. Subclasses must set nproc_per_node."""

    nproc_per_node = None

    @require_torch_multi_accelerator
    def test_model_dense_forward_eval(self):
        """Test that TP and non-TP models produce the same outputs in eval mode."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_impl)("eval")

    @require_torch_multi_accelerator
    def test_model_dense_forward_train(self):
        """Test that TP and non-TP models produce the same outputs in train mode."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_impl)("train")

    @require_torch_multi_accelerator
    def test_model_dense_backward_pass(self):
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_backward_pass_impl)()

    @require_torch_multi_accelerator
    def test_model_dense_forward_compile_eval(self):
        """Test that TP and non-TP models produce the same outputs with torch.compile in eval mode."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_compile_impl)("eval")

    @require_torch_multi_accelerator
    def test_model_dense_forward_compile_train(self):
        """Test that TP and non-TP models produce the same outputs with torch.compile in train mode."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_compile_impl)("train")

    @require_huggingface_hub_greater_or_equal("0.31.4")
    @require_torch_multi_accelerator
    def test_model_dense_save(self):
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # First run with TP (distributed)
            init_distributed(tp=self.nproc_per_node)(_test_model_dense_save_impl)(tmp_dir)

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


class TestTensorParallel2Proc(TestTensorParallelBase):
    """Test tensor parallel with 2 processes."""

    nproc_per_node = 2


class TestTensorParallel4Proc(TestTensorParallelBase):
    """Test tensor parallel with 4 processes."""

    nproc_per_node = 4

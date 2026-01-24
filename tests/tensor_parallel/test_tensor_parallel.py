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
# Run spefic test: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py::TestTensorParallelDense2Proc::test_model_dense_forward_train
# Run tests with a specific prefix: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py::TestTensorParallelDense2Proc -k "forward"
# Run MoE tests only: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py -k "Moe"
# Run dense tests only: RUN_SLOW=1 pytest -v tests/tensor_parallel/test_tensor_parallel.py -k "TestTensorParallelDense2Proc or TestTensorParallelDense4Proc"
import os
import tempfile
import warnings

from safetensors import safe_open

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, is_torch_available
from transformers.integrations.tensor_parallel import get_packed_weights, repack_weights
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


class DebugLogger:
    """
    A debug utility that attaches hooks to model layers to capture activation statistics.
    Useful for comparing outputs between TP and non-TP models layer by layer.
    """

    def __init__(self, model, name="model", rank=0, world_size=2, log_dir=None):
        self.model = model
        self.name = name
        self.rank = rank
        self.world_size = world_size
        self.log_dir = log_dir
        self.hooks = []
        self.activations = {}
        self.raw_tensors = {}  # Store raw tensors for slice comparison
        self.layer_order = []

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"rank_{rank}.log")
            # Clear the file
            with open(self.log_file, "w") as f:
                f.write(f"=== Debug Log for {name} (Rank {rank}) ===\n\n")
        else:
            self.log_file = None

    def _log(self, msg):
        """Log message to file and/or stdout."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")
        print(f"[Rank {self.rank}] {msg}")

    def _get_tensor_stats(self, tensor, num_first_values=5):
        """Get statistics and first values of a tensor."""
        if tensor is None:
            return "None"
        if not isinstance(tensor, torch.Tensor):
            return f"Non-tensor: {type(tensor)}"

        flat = tensor.detach().float().flatten()
        stats = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": flat.mean().item(),
            "std": flat.std().item() if flat.numel() > 1 else 0.0,
            "min": flat.min().item(),
            "max": flat.max().item(),
            "abs_mean": flat.abs().mean().item(),
            "first_values": flat[:num_first_values].tolist(),
            "last_values": flat[-num_first_values:].tolist(),
        }
        return stats

    def _format_stats(self, stats, indent=2):
        """Format stats dictionary for logging."""
        if isinstance(stats, str):
            return stats
        indent_str = " " * indent
        lines = [
            f"{indent_str}shape: {stats['shape']}, dtype: {stats['dtype']}",
            f"{indent_str}mean: {stats['mean']:.6f}, std: {stats['std']:.6f}",
            f"{indent_str}min: {stats['min']:.6f}, max: {stats['max']:.6f}, abs_mean: {stats['abs_mean']:.6f}",
            f"{indent_str}first_5: {[f'{v:.6f}' for v in stats['first_values']]}",
            f"{indent_str}last_5: {[f'{v:.6f}' for v in stats['last_values']]}",
        ]
        return "\n".join(lines)

    def _process_tensor_or_tuple(self, data, prefix=""):
        """Process a tensor or tuple of tensors and return stats."""
        if data is None:
            return {"None": "None"}
        if isinstance(data, torch.Tensor):
            return {prefix or "tensor": self._get_tensor_stats(data)}
        if isinstance(data, (tuple, list)):
            result = {}
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    result[f"{prefix}[{i}]"] = self._get_tensor_stats(item)
                elif item is not None:
                    result[f"{prefix}[{i}]"] = f"type={type(item).__name__}"
            return result
        return {prefix or "data": f"type={type(data).__name__}"}

    def _make_hook(self, layer_name):
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            # Store activations
            self.activations[layer_name] = {
                "input": self._process_tensor_or_tuple(input, "input"),
                "output": self._process_tensor_or_tuple(output, "output"),
            }
            # Store raw output tensor for slice comparison
            if isinstance(output, torch.Tensor):
                self.raw_tensors[layer_name] = output.detach().clone()
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                self.raw_tensors[layer_name] = output[0].detach().clone()

            if layer_name not in self.layer_order:
                self.layer_order.append(layer_name)

        return hook

    def attach_hooks(self, include_patterns=None, exclude_patterns=None):
        """
        Attach forward hooks to model layers.

        Args:
            include_patterns: List of patterns to include (e.g., ["layers", "embed"])
            exclude_patterns: List of patterns to exclude (e.g., ["dropout"])
        """
        include_patterns = include_patterns or []
        exclude_patterns = exclude_patterns or ["dropout"]

        for name, module in self.model.named_modules():
            # Skip empty name (root module)
            if not name:
                continue

            # Check exclude patterns
            if any(pat in name.lower() for pat in exclude_patterns):
                continue

            # Check include patterns (if specified, only include matching)
            if include_patterns and not any(pat in name.lower() for pat in include_patterns):
                continue

            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

        self._log(f"Attached {len(self.hooks)} hooks to {self.name}")

    def remove_hooks(self):
        """Remove all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}
        self.raw_tensors = {}
        self.layer_order = []

    def log_activations(self, title=""):
        """Log all stored activations."""
        self._log(f"\n{'=' * 60}")
        self._log(f"Activations for {self.name} {title}")
        self._log(f"{'=' * 60}")

        for layer_name in self.layer_order:
            act = self.activations.get(layer_name, {})
            self._log(f"\n--- {layer_name} ---")

            # Log output stats (usually more important)
            if "output" in act:
                self._log("  OUTPUT:")
                for key, stats in act["output"].items():
                    self._log(f"    {key}:")
                    self._log(self._format_stats(stats, indent=6))

    def compare_with(self, other_logger, title="", atol=1e-5, rtol=1e-5):
        """
        Compare activations with another DebugLogger and log differences.
        Also compares sliced non-TP tensors with TP tensors for sharded layers.
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"Comparing {self.name} vs {other_logger.name} {title}")
        self._log(f"Tolerances: atol={atol}, rtol={rtol}")
        self._log(f"{'=' * 60}")

        # Get common layers
        my_layers = set(self.activations.keys())
        other_layers = set(other_logger.activations.keys())
        common_layers = my_layers & other_layers

        self._log(f"Layers in {self.name} only: {my_layers - other_layers}")
        self._log(f"Layers in {other_logger.name} only: {other_layers - my_layers}")
        self._log(f"Common layers: {len(common_layers)}")

        # Compare common layers
        for layer_name in self.layer_order:
            if layer_name not in common_layers:
                continue

            my_act = self.activations[layer_name]
            other_act = other_logger.activations[layer_name]

            # Compare outputs
            if "output" in my_act and "output" in other_act:
                for key in my_act["output"]:
                    if key in other_act["output"]:
                        my_stats = my_act["output"][key]
                        other_stats = other_act["output"][key]

                        if isinstance(my_stats, dict) and isinstance(other_stats, dict):
                            my_shape = my_stats["shape"]
                            other_shape = other_stats["shape"]

                            mean_diff = abs(my_stats["mean"] - other_stats["mean"])
                            max_diff = abs(my_stats["max"] - other_stats["max"])

                            # Check if shapes differ (sharded layer)
                            is_sharded = my_shape != other_shape

                            self._log(f"\n--- {layer_name} / {key} ---")
                            self._log(f"  {self.name} shape: {my_shape}")
                            self._log(f"  {other_logger.name} shape: {other_shape}")
                            self._log(f"  {self.name}: mean={my_stats['mean']:.6f}, max={my_stats['max']:.6f}")
                            self._log(
                                f"  {other_logger.name}: mean={other_stats['mean']:.6f}, max={other_stats['max']:.6f}"
                            )
                            self._log(f"  STAT DIFF: mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}")

                            # For sharded layers, compare sliced non-TP tensor with TP tensor
                            if (
                                is_sharded
                                and layer_name in self.raw_tensors
                                and layer_name in other_logger.raw_tensors
                            ):
                                my_tensor = self.raw_tensors[layer_name]  # non-TP (full)
                                other_tensor = other_logger.raw_tensors[layer_name]  # TP (sharded)

                                # Find sharding dimension and slice
                                for dim in range(my_tensor.ndim):
                                    if my_tensor.size(dim) != other_tensor.size(dim):
                                        shard_size = other_tensor.size(dim)
                                        start = other_logger.rank * shard_size
                                        my_slice = my_tensor.narrow(dim, start, shard_size)

                                        # Compare the slice
                                        slice_match = torch.allclose(my_slice, other_tensor, atol=atol, rtol=rtol)
                                        slice_diff = (my_slice - other_tensor).abs()

                                        status = "✓ MATCH" if slice_match else "✗ MISMATCH"
                                        self._log(f"  SLICE COMPARISON (dim={dim}, rank={other_logger.rank}):")
                                        self._log(
                                            f"    {self.name}[{start}:{start + shard_size}] vs {other_logger.name}: {status}"
                                        )
                                        self._log(
                                            f"    slice max_diff={slice_diff.max().item():.6f}, "
                                            f"mean_diff={slice_diff.float().mean().item():.6f}"
                                        )
                                        self._log(f"    {self.name} slice first_5: {my_slice.flatten()[:5].tolist()}")
                                        self._log(
                                            f"    {other_logger.name} first_5: {other_tensor.flatten()[:5].tolist()}"
                                        )
                                        break
                            else:
                                # Same shape - direct comparison
                                if layer_name in self.raw_tensors and layer_name in other_logger.raw_tensors:
                                    my_tensor = self.raw_tensors[layer_name]
                                    other_tensor = other_logger.raw_tensors[layer_name]
                                    direct_match = torch.allclose(my_tensor, other_tensor, atol=atol, rtol=rtol)
                                    direct_diff = (my_tensor - other_tensor).abs()

                                    status = "✓ MATCH" if direct_match else "✗ MISMATCH"
                                    self._log(f"  DIRECT COMPARISON: {status}")
                                    self._log(
                                        f"    max_diff={direct_diff.max().item():.6f}, "
                                        f"mean_diff={direct_diff.float().mean().item():.6f}"
                                    )


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

    # Create dtype-specific log directory
    dtype_name = str(dtype).replace("torch.", "")
    log_dir = f"tp_debug_logs_forward_{dtype_name}"

    # Get world size
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Ensure same random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set tolerance based on dtype
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

    # Setup DebugLoggers for both models
    debug_model = DebugLogger(model, name="non-TP", rank=rank, world_size=world_size, log_dir=log_dir)
    debug_model_tp = DebugLogger(model_tp, name="TP", rank=rank, world_size=world_size, log_dir=log_dir)

    # Attach hooks to capture activations - focus on key layers
    debug_model.attach_hooks(include_patterns=["embed", "layers.0", "layers.1", "norm", "lm_head"])
    debug_model_tp.attach_hooks(include_patterns=["embed", "layers.0", "layers.1", "norm", "lm_head"])

    debug_model._log(f"\n{'=' * 80}")
    debug_model._log(f"Test: mode={mode}, dtype={dtype}")
    debug_model._log(f"Tolerances: atol={atol}, rtol={rtol}")
    debug_model._log(f"World size: {world_size}, Rank: {rank}")
    debug_model._log(f"Input tokens: {inputs.input_ids.tolist()}")
    debug_model._log(f"TP model tp_plan: {model_tp.tp_plan}")
    debug_model._log(f"{'=' * 80}")

    # Prepare inputs on the same device
    input_ids = inputs.input_ids.to(device)

    # Run forward pass on both models with hooks
    with torch.no_grad():
        # Non-TP model output
        debug_model.clear_activations()
        outputs = model(input_ids)
        logits = outputs.logits

        # TP model output
        debug_model_tp.clear_activations()
        outputs_tp = model_tp(input_ids)
        logits_tp = outputs_tp.logits

    # Log activations
    debug_model.log_activations(title=f"(mode={mode}, dtype={dtype})")
    debug_model_tp.log_activations(title=f"(mode={mode}, dtype={dtype})")

    # Compare activations between models (pass tolerances for match/mismatch check)
    debug_model.compare_with(debug_model_tp, title=f"(mode={mode}, dtype={dtype})", atol=atol, rtol=rtol)

    # Clean up hooks
    debug_model.remove_hooks()
    debug_model_tp.remove_hooks()

    # Final logits comparison
    diff = (logits - logits_tp).abs()
    is_close = torch.allclose(logits, logits_tp, atol=atol, rtol=rtol)
    final_status = "✓ MATCH" if is_close else "✗ MISMATCH"

    debug_model._log(f"\n{'=' * 60}")
    debug_model._log(f"FINAL LOGITS COMPARISON: {final_status}")
    debug_model._log(f"{'=' * 60}")
    debug_model._log(
        f"Non-TP logits: shape={logits.shape}, mean={logits.float().mean().item():.6f}, "
        f"std={logits.float().std().item():.6f}"
    )
    debug_model._log(
        f"TP logits: shape={logits_tp.shape}, mean={logits_tp.float().mean().item():.6f}, "
        f"std={logits_tp.float().std().item():.6f}"
    )
    debug_model._log(
        f"Diff: max={diff.max().item():.6f}, min={diff.min().item():.6f}, mean={diff.float().mean().item():.6f}"
    )

    # First 10 values comparison
    debug_model._log("\nFirst 10 logits (flattened):")
    debug_model._log(f"  Non-TP: {logits.flatten()[:10].tolist()}")
    debug_model._log(f"  TP:     {logits_tp.flatten()[:10].tolist()}")

    # Find where the max differences are
    max_diff_idx = diff.argmax()
    debug_model._log(f"\nMax diff location (flat idx): {max_diff_idx.item()}")
    debug_model._log(f"  Non-TP value: {logits.flatten()[max_diff_idx].item():.6f}")
    debug_model._log(f"  TP value: {logits_tp.flatten()[max_diff_idx].item():.6f}")

    debug_model._log(f"\ntorch.allclose result: {is_close}")

    # Compare outputs - they should match
    assert is_close, (
        f"TP and non-TP model outputs differ (dtype={dtype}). "
        f"Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_model_dense_backward_pass_impl(rank, dtype=torch.float32):
    """Implementation for comparing TP and non-TP model backward passes."""
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set tolerance based on dtype
    atol, rtol = (1e-5, 1e-5)

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, tp_plan="auto")
    dist.barrier()
    model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device)
    model.train()

    batch_size, seq_length = 2, 10
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
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
    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # Slice reference gradient to match local shard if parameter is sharded
            if grad.shape != grad_tp.shape:
                # Find the dimension that differs and slice accordingly
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
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

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

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

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

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

    batch_size, seq_length = 2, 10
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
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
    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # Slice reference gradient to match local shard if parameter is sharded
            if grad.shape != grad_tp.shape:
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
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


class TestTensorParallelDenseBase(TestCasePlus):
    """Base class for tensor parallel tests. Subclasses must set nproc_per_node."""

    nproc_per_node = None

    @require_torch_multi_accelerator
    def test_model_dense_forward_eval_float32(self):
        """Test that TP and non-TP models produce the same outputs in eval mode (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_impl)("eval", torch.float32)

    @require_torch_multi_accelerator
    def test_model_dense_forward_eval_bfloat16(self):
        """Test that TP and non-TP models produce the same outputs in eval mode (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_impl)("eval", torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_dense_forward_train_float32(self):
        """Test that TP and non-TP models produce the same outputs in train mode (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_impl)("train", torch.float32)

    @require_torch_multi_accelerator
    def test_model_dense_forward_train_bfloat16(self):
        """Test that TP and non-TP models produce the same outputs in train mode (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_impl)("train", torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_dense_backward_pass_float32(self):
        """Test that TP and non-TP models produce the same gradients (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_backward_pass_impl)(torch.float32)

    @require_torch_multi_accelerator
    def test_model_dense_backward_pass_bfloat16(self):
        """Test that TP and non-TP models produce the same gradients (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_backward_pass_impl)(torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_dense_forward_compile_eval_float32(self):
        """Test that TP and non-TP models produce the same outputs with torch.compile in eval mode (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_compile_impl)("eval", torch.float32)

    @require_torch_multi_accelerator
    def test_model_dense_forward_compile_eval_bfloat16(self):
        """Test that TP and non-TP models produce the same outputs with torch.compile in eval mode (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_compile_impl)("eval", torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_dense_forward_compile_train_float32(self):
        """Test that TP and non-TP models produce the same outputs with torch.compile in train mode (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_compile_impl)("train", torch.float32)

    @require_torch_multi_accelerator
    def test_model_dense_forward_compile_train_bfloat16(self):
        """Test that TP and non-TP models produce the same outputs with torch.compile in train mode (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_forward_compile_impl)("train", torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_dense_backward_compile_float32(self):
        """Test that TP and non-TP models produce the same gradients with torch.compile (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_backward_compile_impl)(torch.float32)

    @require_torch_multi_accelerator
    def test_model_dense_backward_compile_bfloat16(self):
        """Test that TP and non-TP models produce the same gradients with torch.compile (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_dense_backward_compile_impl)(torch.bfloat16)

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


class TestTensorParallelDense2Proc(TestTensorParallelDenseBase):
    """Test tensor parallel dense model with 2 processes."""

    nproc_per_node = 2


class TestTensorParallelDense4Proc(TestTensorParallelDenseBase):
    """Test tensor parallel dense model with 4 processes."""

    nproc_per_node = 4


# ====== MOE MODEL TEST FUNCTIONS ======
def _test_model_moe_forward_impl(rank, mode, dtype=torch.float32):
    """Implementation for comparing TP and non-TP MoE model outputs."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    # Create dtype-specific log directory
    dtype_name = str(dtype).replace("torch.", "")
    log_dir = f"tp_debug_logs_moe_forward_{dtype_name}"

    # Get world size
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Ensure same random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set tolerance based on dtype
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

    # Setup DebugLoggers for both models
    debug_model = DebugLogger(model, name="non-TP", rank=rank, world_size=world_size, log_dir=log_dir)
    debug_model_tp = DebugLogger(model_tp, name="TP", rank=rank, world_size=world_size, log_dir=log_dir)

    # Attach hooks to capture activations - focus on key layers including MoE-specific ones
    debug_model.attach_hooks(
        include_patterns=["embed", "layers.0", "layers.1", "norm", "lm_head", "router", "experts"]
    )
    debug_model_tp.attach_hooks(
        include_patterns=["embed", "layers.0", "layers.1", "norm", "lm_head", "router", "experts"]
    )

    debug_model._log(f"\n{'=' * 80}")
    debug_model._log(f"MoE Test: mode={mode}, dtype={dtype}")
    debug_model._log(f"Tolerances: atol={atol}, rtol={rtol}")
    debug_model._log(f"World size: {world_size}, Rank: {rank}")
    debug_model._log(f"Input tokens: {inputs.input_ids.tolist()}")
    debug_model._log(f"TP model tp_plan: {model_tp.tp_plan}")
    debug_model._log(f"{'=' * 80}")

    # Prepare inputs on the same device
    input_ids = inputs.input_ids.to(device)

    # Run forward pass on both models with hooks
    with torch.no_grad():
        # Non-TP model output
        debug_model.clear_activations()
        outputs = model(input_ids)
        logits = outputs.logits

        # TP model output
        debug_model_tp.clear_activations()
        outputs_tp = model_tp(input_ids)
        logits_tp = outputs_tp.logits

    # Log activations
    debug_model.log_activations(title=f"(mode={mode}, dtype={dtype})")
    debug_model_tp.log_activations(title=f"(mode={mode}, dtype={dtype})")

    # Compare activations between models (pass tolerances for match/mismatch check)
    debug_model.compare_with(debug_model_tp, title=f"(mode={mode}, dtype={dtype})", atol=atol, rtol=rtol)

    # Clean up hooks
    debug_model.remove_hooks()
    debug_model_tp.remove_hooks()

    # Final logits comparison
    diff = (logits - logits_tp).abs()
    is_close = torch.allclose(logits, logits_tp, atol=atol, rtol=rtol)
    final_status = "✓ MATCH" if is_close else "✗ MISMATCH"

    debug_model._log(f"\n{'=' * 60}")
    debug_model._log(f"FINAL LOGITS COMPARISON: {final_status}")
    debug_model._log(f"{'=' * 60}")
    debug_model._log(
        f"Non-TP logits: shape={logits.shape}, mean={logits.float().mean().item():.6f}, "
        f"std={logits.float().std().item():.6f}"
    )
    debug_model._log(
        f"TP logits: shape={logits_tp.shape}, mean={logits_tp.float().mean().item():.6f}, "
        f"std={logits_tp.float().std().item():.6f}"
    )
    debug_model._log(
        f"Diff: max={diff.max().item():.6f}, min={diff.min().item():.6f}, mean={diff.float().mean().item():.6f}"
    )

    # First 10 values comparison
    debug_model._log("\nFirst 10 logits (flattened):")
    debug_model._log(f"  Non-TP: {logits.flatten()[:10].tolist()}")
    debug_model._log(f"  TP:     {logits_tp.flatten()[:10].tolist()}")

    # Find where the max differences are
    max_diff_idx = diff.argmax()
    debug_model._log(f"\nMax diff location (flat idx): {max_diff_idx.item()}")
    debug_model._log(f"  Non-TP value: {logits.flatten()[max_diff_idx].item():.6f}")
    debug_model._log(f"  TP value: {logits_tp.flatten()[max_diff_idx].item():.6f}")

    debug_model._log(f"\ntorch.allclose result: {is_close}")

    # Compare outputs - they should match
    assert is_close, (
        f"TP and non-TP MoE model outputs differ (dtype={dtype}). "
        f"Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
    )

    dist.barrier()


def _test_model_moe_backward_pass_impl(rank, dtype=torch.float32):
    """Implementation for comparing TP and non-TP MoE model backward passes."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set tolerance based on dtype
    atol, rtol = (1e-5, 1e-5)

    # Disable weight tying to avoid gradient mismatch between replicated embed_tokens
    # and sharded lm_head when weights are tied
    config = AutoConfig.from_pretrained(model_id)
    config.tie_word_embeddings = False

    model_tp = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=dtype, tp_plan="auto")
    dist.barrier()
    model_tp.train()

    device = model_tp.device
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype=dtype)
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

    assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
        f"TP and non-TP MoE model losses differ (dtype={dtype}). Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, Diff: {(loss - loss_tp).abs().item()}"
    )

    # ANSI color codes for terminal output
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Compare gradients for matching parameters
    world_size = dist.get_world_size()

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

    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad  # Full gradient from non-TP model
            grad_tp = param_tp.grad  # Gradient from TP model (may be sharded)

            # Determine if this param is sharded by comparing shapes
            is_sharded = grad.shape != grad_tp.shape
            shard_dim = None
            grad_shard = grad

            if is_sharded:
                # Find which dimension is sharded
                for dim in range(len(grad.shape)):
                    if grad.shape[dim] != grad_tp.shape[dim]:
                        shard_dim = dim
                        break

                if shard_dim is not None:
                    # Packed weights (gate_up_proj) use interleaved sharding
                    if "gate_up_proj" in name:
                        grad_shard = get_packed_grad_shard(grad, world_size, rank, shard_dim)
                    else:
                        # Regular weights use simple chunking
                        chunks = torch.chunk(grad, world_size, dim=shard_dim)
                        grad_shard = chunks[rank]

            # Detailed logging for all parameters to trace divergence
            if rank == 0:
                # Check if this is a key MoE component
                is_moe = any(x in name for x in ["mlp.experts", "mlp.gate"])
                is_attn = "self_attn" in name
                is_embed = "embed_tokens" in name
                is_norm = "layernorm" in name or "norm" in name
                is_lm_head = "lm_head" in name

                # Color-code by layer type
                if is_moe:
                    layer_color, layer_type = MAGENTA, "MoE"
                elif is_attn:
                    layer_color, layer_type = BLUE, "Attn"
                elif is_embed:
                    layer_color, layer_type = WHITE, "Embed"
                elif is_lm_head:
                    layer_color, layer_type = WHITE, "LMHead"
                elif is_norm:
                    layer_color, layer_type = YELLOW, "Norm"
                else:
                    layer_color, layer_type = RESET, "Other"

                # Check shapes match after sharding
                if grad_shard.shape != grad_tp.shape:
                    print(f"\n{layer_color}{BOLD}[{layer_type}]{RESET} {name}")
                    print(
                        f"  {RED}SHAPE MISMATCH:{RESET} non-TP={grad.shape}, TP={grad_tp.shape}, shard={grad_shard.shape}"
                    )
                    continue

                diff = (grad_shard.cpu() - grad_tp.cpu()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                # Color-code diff severity
                if max_diff < 1e-5:
                    diff_color = GREEN
                    status = "✓"
                elif max_diff < 1e-3:
                    diff_color = YELLOW
                    status = "~"
                else:
                    diff_color = RED
                    status = "✗"

                shard_info = f" [sharded dim={shard_dim}]" if is_sharded else ""
                print(f"\n{layer_color}{BOLD}[{layer_type}]{RESET} {name}{shard_info}")
                print(
                    f"  {GREEN}Non-TP:{RESET} min={grad_shard.min().item():.6e}, max={grad_shard.max().item():.6e}, mean={grad_shard.mean().item():.6e}"
                )
                print(
                    f"  {MAGENTA}TP:{RESET}     min={grad_tp.min().item():.6e}, max={grad_tp.max().item():.6e}, mean={grad_tp.mean().item():.6e}"
                )
                print(
                    f"  {diff_color}{BOLD}Diff [{status}]:{RESET} max={diff_color}{max_diff:.6e}{RESET}, mean={mean_diff:.6e}"
                )

            # Skip assertion to see all gradient comparisons

    dist.barrier()


def _test_model_moe_forward_compile_impl(rank, mode, dtype=torch.float32, experts_implementation=None):
    """Implementation for comparing TP and non-TP MoE model outputs with torch.compile."""
    model_id = "hf-internal-testing/tiny-random-MixtralForCausalLM"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set tolerance based on dtype
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

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set tolerance based on dtype
    atol, rtol = (1e-5, 1e-5)

    config = AutoConfig.from_pretrained(model_id)
    config.tie_word_embeddings = False

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

    # Compile both models
    model.forward = torch.compile(model.forward)
    model_tp.forward = torch.compile(model_tp.forward)

    batch_size, seq_length = 2, 10
    torch.manual_seed(42)
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
    for (name, param), (name_tp, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
        if param.grad is not None and param_tp.grad is not None:
            grad = param.grad
            grad_tp = param_tp.grad

            # Handle sharded parameters - slice reference gradient to match local shard
            if grad.shape != grad_tp.shape:
                for dim in range(grad.ndim):
                    if grad.size(dim) != grad_tp.size(dim):
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


class TestTensorParallelMoeBase(TestCasePlus):
    """Base class for MoE tensor parallel tests. Subclasses must set nproc_per_node."""

    nproc_per_node = None

    @require_torch_multi_accelerator
    def test_model_moe_forward_eval_float32(self):
        """Test that TP and non-TP MoE models produce the same outputs in eval mode (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_impl)("eval", torch.float32)

    @require_torch_multi_accelerator
    def test_model_moe_forward_eval_bfloat16(self):
        """Test that TP and non-TP MoE models produce the same outputs in eval mode (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_impl)("eval", torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_moe_forward_train_float32(self):
        """Test that TP and non-TP MoE models produce the same outputs in train mode (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_impl)("train", torch.float32)

    @require_torch_multi_accelerator
    def test_model_moe_forward_train_bfloat16(self):
        """Test that TP and non-TP MoE models produce the same outputs in train mode (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_impl)("train", torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_moe_backward_pass_float32(self):
        """Test that TP and non-TP MoE models produce the same gradients (float32)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_backward_pass_impl)(torch.float32)

    @require_torch_multi_accelerator
    def test_model_moe_backward_pass_bfloat16(self):
        """Test that TP and non-TP MoE models produce the same gradients (bfloat16)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_backward_pass_impl)(torch.bfloat16)

    @require_torch_multi_accelerator
    def test_model_moe_forward_compile_eval_float32_batched_mm(self):
        """Test that TP and non-TP MoE models produce the same outputs with torch.compile in eval mode (float32, batched_mm)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_compile_impl)(
            "eval", torch.float32, experts_implementation="batched_mm"
        )

    @require_torch_multi_accelerator
    def test_model_moe_forward_compile_eval_bfloat16_grouped_mm(self):
        """Test that TP and non-TP MoE models produce the same outputs with torch.compile in eval mode (bfloat16, grouped_mm)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_compile_impl)(
            "eval", torch.bfloat16, experts_implementation="grouped_mm"
        )

    @require_torch_multi_accelerator
    def test_model_moe_forward_compile_train_float32_batched_mm(self):
        """Test that TP and non-TP MoE models produce the same outputs with torch.compile in train mode (float32, batched_mm)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_compile_impl)(
            "train", torch.float32, experts_implementation="batched_mm"
        )

    @require_torch_multi_accelerator
    def test_model_moe_forward_compile_train_bfloat16_grouped_mm(self):
        """Test that TP and non-TP MoE models produce the same outputs with torch.compile in train mode (bfloat16, grouped_mm)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_forward_compile_impl)(
            "train", torch.bfloat16, experts_implementation="grouped_mm"
        )

    @require_torch_multi_accelerator
    def test_model_moe_backward_compile_float32_batched_mm(self):
        """Test that TP and non-TP MoE models produce the same gradients with torch.compile (float32, batched_mm)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_backward_compile_impl)(
            torch.float32, experts_implementation="batched_mm"
        )

    @require_torch_multi_accelerator
    def test_model_moe_backward_compile_bfloat16_grouped_mm(self):
        """Test that TP and non-TP MoE models produce the same gradients with torch.compile (bfloat16, grouped_mm)."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        init_distributed(tp=self.nproc_per_node)(_test_model_moe_backward_compile_impl)(
            torch.bfloat16, experts_implementation="grouped_mm"
        )

    @require_huggingface_hub_greater_or_equal("0.31.4")
    @require_torch_multi_accelerator
    def test_model_moe_save(self):
        """Test that TP MoE model can be saved and matches non-TP version."""
        if self.nproc_per_node is None:
            self.skipTest("nproc_per_node not set")
        if backend_device_count(torch_device) < self.nproc_per_node:
            self.skipTest(f"Need at least {self.nproc_per_node} devices, have {backend_device_count(torch_device)}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # First run with TP (distributed)
            init_distributed(tp=self.nproc_per_node)(_test_model_moe_save_impl)(tmp_dir)

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


class TestTensorParallelMoe2Proc(TestTensorParallelMoeBase):
    """Test MoE tensor parallel with 2 processes."""

    nproc_per_node = 2


class TestTensorParallelMoe4Proc(TestTensorParallelMoeBase):
    """Test MoE tensor parallel with 4 processes."""

    nproc_per_node = 4

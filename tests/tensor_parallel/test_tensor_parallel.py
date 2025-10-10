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

# Run the test: CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/tensor_parallel/test_tensor_parallel.py

import os
import tempfile
import textwrap

from transformers import is_torch_available
from transformers.integrations.tensor_parallel import get_packed_weights, repack_weights
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    require_huggingface_hub_greater_or_equal,
    require_torch_multi_accelerator,
    torch_device,
    torchrun,
)


if is_torch_available():
    import torch


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


class TestTensorParallel(TestCasePlus):
    nproc_per_node = 2

    def test_model_forward(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            import os
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "JackFram/llama-68m"

            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", tp_plan="auto")
            torch.distributed.barrier()

            has_dtensor = 0
            for name, parameter in model.named_parameters():
                if isinstance(parameter.data, torch.distributed.tensor.DTensor):
                    has_dtensor = 1
                    break

            assert has_dtensor == 1, "TP model must has DTensor"

            tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
            prompt = "Can I help"

            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            outputs = model(inputs)

            next_token_logits = outputs[0][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            response = tokenizer.decode(next_token)
            assert response == "with"

            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            """
        )
        torchrun(script_to_run, self.nproc_per_node, env=self.get_env())

    def test_model_backward_pass(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            import os
            from transformers import AutoModelForCausalLM
            from torch import nn

            model_id = "JackFram/llama-68m"

            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, tp_plan="auto")
            torch.distributed.barrier()

            # Dummy forward and backward pass
            # Note that loss.backward() will fail if there is a bug in the TP implementation
            inputs = torch.randint(0, model.config.vocab_size, (2, 10), device=model.device)
            labels = torch.randint(0, model.config.vocab_size, (2, 10), device=model.device)
            loss = model(inputs, labels=labels).loss
            loss.backward()

            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            """
        )
        torchrun(script_to_run, self.nproc_per_node, env=self.get_env())

    def test_model_generate(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            import os
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "JackFram/llama-68m"

            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", tp_plan="auto")
            torch.distributed.barrier()

            model.forward = torch.compile(model.forward)

            has_dtensor = 0
            for name, parameter in model.named_parameters():
                if isinstance(parameter.data, torch.distributed.tensor.DTensor):
                    has_dtensor = 1
                    break

            assert has_dtensor == 1, "TP model must has DTensor"

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            prompt = "Can I help"

            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            outputs = model.generate(inputs, max_new_tokens=10, cache_implementation="static")

            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            assert output_text[0].startswith(prompt), f"Expected output to start with '{prompt}', got '{output_text[0]}'"

            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            """
        )
        torchrun(script_to_run, self.nproc_per_node, env=self.get_env())

    @require_huggingface_hub_greater_or_equal("0.31.4")
    def test_model_save(self):
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as tmp_dir:
            for is_torchrun in [True, False]:
                script_to_run = textwrap.dedent(
                    f"""
                    import torch
                    import os
                    from transformers import AutoModelForCausalLM

                    model_id = "JackFram/llama-68m"
                    kwargs = dict()

                    if os.environ.get("RANK", None) is not None:
                        kwargs["tp_plan"] = "auto"
                        result_dir = "{tmp_dir}/tp"
                    else:
                        result_dir = "{tmp_dir}/nontp"

                    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                    model.save_pretrained(result_dir)
                    """
                )
                torchrun(script_to_run, self.nproc_per_node, is_torchrun=is_torchrun, env=self.get_env())

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


class TestTensorParallelProperties(TestCasePlus):
    def test_tp_plan_property_setter_getter(self):
        """Test that tp_plan property can be set and retrieved correctly."""
        from transformers import AutoModelForCausalLM

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
        from transformers import AutoModelForCausalLM

        model_id = "JackFram/llama-68m"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test invalid parallel style
        with self.assertRaises(ValueError) as context:
            model.tp_plan = {"layers.*.self_attn.q_proj": "invalid_style"}

        self.assertIn("Unsupported tensor parallel style 'invalid_style'", str(context.exception))
        self.assertIn("Supported styles are", str(context.exception))

    def test_tp_plan_validation_nonexistent_layer_warning(self):
        """Test that warnings are issued for non-existent layer patterns."""
        import warnings

        from transformers import AutoModelForCausalLM

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
        import warnings

        from transformers import AutoModelForCausalLM

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
        from transformers import AutoModelForCausalLM

        model_id = "JackFram/llama-68m"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting None
        model.tp_plan = None
        self.assertEqual(model.tp_plan, {})

        # Test setting a plan after None
        model.tp_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        self.assertEqual(model.tp_plan, {"model.layers.*.self_attn.q_proj": "colwise"})


@require_torch_multi_accelerator
class TestTensorParallelAccelerator(TestTensorParallel):
    nproc_per_node = backend_device_count(torch_device)

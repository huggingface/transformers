# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GptOss model."""

import difflib
import inspect
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_deterministic_for_xpu,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        GptOssModel,
    )

    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        NUM_GPUS = torch.xpu.device_count()
    else:
        NUM_GPUS = 0


class GptOssModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GptOssModel


@require_torch
class GptOssModelTest(CausalLMModelTest, unittest.TestCase):
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = GptOssModelTester

    @require_kernels
    @pytest.mark.flash_attn_test
    @require_torch_gpu
    def test_default_flash_implementation_auto_correction(self):
        """
        Tests that setting attn_implementation="flash_attention_2" during model initialization
        automatically corrects to the model's `_compatible_flash_implementations`.
        """
        from kernels import get_kernel

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        expected_kernel = "kernels-community/vllm-flash-attn3"
        flash = get_kernel(expected_kernel)
        if flash is None:
            self.skipTest(f"{expected_kernel} is not available, skipping auto-correction test.")

        # Option 1: Auto correction on setting config on init
        config._attn_implementation = "flash_attention_2"
        tmp_model = GptOssModel(config).to(device=torch_device, dtype=torch.bfloat16)
        self.assertEqual(tmp_model.config._attn_implementation, expected_kernel)

        # Option 2: Auto correction on load time
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_model.save_pretrained(tmp_dir_name)
            model = GptOssModel.from_pretrained(tmp_dir_name, attn_implementation="flash_attention_2").to(
                device=torch_device
            )
            self.assertEqual(model.config._attn_implementation, expected_kernel)

        # Option 3: Auto correction on `set_attn_implementation`
        model.set_attn_implementation("eager")
        self.assertEqual(model.config._attn_implementation, "eager")
        model.set_attn_implementation("flash_attention_2")
        self.assertEqual(model.config._attn_implementation, expected_kernel)

        # Verify model still works
        with torch.no_grad():
            output = model(**inputs_dict)
        self.assertIsNotNone(output)

    @unittest.skip("GptOss's forcefully disables sdpa due to Sink")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("GptOss's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip("GptOss eager/FA2 attention outputs are expected to be different")
    def test_flash_attn_2_equivalence(self):
        pass

    @unittest.skip("Most probably because of the MOE, the moe and router does not ignore padding tokens")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("GptOss does not support flex officially")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skipIf(torch_device == "cpu", "GptOss does not support flex officially")
    def test_generate_compile_model_forward_fullgraph(self):
        return super().test_generate_compile_model_forward_fullgraph()

    def test_reverse_loading_mapping(self, check_keys_were_modified=False):
        super().test_reverse_loading_mapping(check_keys_were_modified)


RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/gpt_oss/integration_tests.json"


# ------------------------
# Worker function for distributed torchrun
# ------------------------
def distributed_worker(quantized, model_size, kernels, attn_impl, mode):
    """This is the function that will be executed by torchrun workers."""
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.testing_utils import torch_device

    def generate_config_key(quantized, model, kernels, attn_impl, mode):
        """Generate a key for the restructured integration test results."""
        return f"device={torch_device}|quantized={str(quantized).lower()}|model={model}|kernels={str(kernels).lower()}|attn_impl={attn_impl}|mode={mode}"

    input_text = [
        "Roses are red, violets",
        "How are you? Tell me the name of the president of",
    ]

    # Convert args
    quantized = quantized.lower() == "true"
    kernels = kernels.lower() == "true"

    # Distributed model loading
    model_id = f"openai/gpt-oss-{model_size}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        tp_plan="auto",  # distributed inference
        use_kernels=kernels,
    ).to(torch_device)
    model.set_attn_implementation(attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    # Inference
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(torch_device)
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    output_texts = tokenizer.batch_decode(output, skip_special_tokens=False)

    # Only rank 0 writes results and validates against expected outputs
    if int(os.environ.get("RANK", "0")) == 0:
        # Generate key to look up expected outputs
        key = generate_config_key(quantized, model_size, kernels, attn_impl, mode)

        # Load expected outputs from restructured JSON
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                expected_results = json.load(f)

            # Check if we have expected results for this configuration
            if key in expected_results:
                expected_outputs = expected_results[key]

                # Compare actual outputs with expected outputs
                assert len(output_texts) == len(expected_outputs), f"Output length mismatch for {key}"

                for i, (actual, expected) in enumerate(zip(output_texts, expected_outputs)):
                    actual_stripped = actual.strip()
                    expected_stripped = expected.strip()

                    # Make lengths match by taking minimum length to be resilient to generation differences
                    min_length = min(len(actual_stripped), len(expected_stripped))
                    actual_truncated = actual_stripped[:min_length]
                    expected_truncated = expected_stripped[:min_length]

                    if actual_truncated != expected_truncated:
                        diff = "\n".join(
                            difflib.unified_diff(
                                expected_truncated.splitlines(keepends=True),
                                actual_truncated.splitlines(keepends=True),
                                fromfile=f"expected[{i}]",
                                tofile=f"actual[{i}]",
                                lineterm="",
                            )
                        )
                        raise AssertionError(
                            f"Output mismatch at index {i} for {key}:\n"
                            f"Expected: '{expected_stripped}'\n"
                            f"Actual:   '{actual_stripped}'\n"
                            f"Diff (truncated to min length {min_length}):\n{diff}"
                        )

                print(f"✓ Outputs match expected results for {key}")
            else:
                print(f"Warning: No expected results found for configuration: {key}")
        else:
            print(f"Warning: Results file {RESULTS_PATH} not found")


@slow
@require_torch_accelerator
class GptOssIntegrationTest(unittest.TestCase):
    input_text = [
        "Roses are red, violets",
        "How are you? Tell me the name of the president of",
    ]

    @staticmethod
    def generate_config_key(quantized, model, kernels, attn_impl, mode):
        """Generate a key for the restructured integration test results."""
        return f"device={torch_device}|quantized={str(quantized).lower()}|model={model}|kernels={str(kernels).lower()}|attn_impl={attn_impl}|mode={mode}"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    # ------------------------
    # Non-distributed inference
    # ------------------------
    @staticmethod
    def load_and_forward(model_id, attn_implementation, input_text, mode="eval", **pretrained_kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation,
            **pretrained_kwargs,
        )

        # Set the correct mode
        if mode == "train":
            model.train()
        else:
            model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_text

    # ------------------------
    # Distributed inference using inspect
    # ------------------------
    @staticmethod
    def run_distributed_test(quantized, model, kernels, attn_impl, mode):
        """Launch torchrun using a temporary worker file generated from inspect.getsource()."""
        import textwrap

        # Extract worker function source dynamically
        worker_src = inspect.getsource(distributed_worker)

        # Create a temp file that calls the worker
        script_code = f"""
import sys
import json

RESULTS_PATH = "{RESULTS_PATH}"

{worker_src}

if __name__ == "__main__":
    distributed_worker("{quantized}", "{model}", "{kernels}", "{attn_impl}", "{mode}")
"""
        # Dedent for proper formatting
        script_code = textwrap.dedent(script_code)

        # Write to temp file
        with tempfile.NamedTemporaryFile("w", suffix="_worker.py", delete=False) as tmp:
            tmp.write(script_code)
            tmp_path = tmp.name

        # Launch torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={NUM_GPUS}",
            tmp_path,
        ]
        subprocess.run(cmd, check=True)

        # Cleanup
        os.remove(tmp_path)

    # ------------------------
    # Shared parameterization
    # ------------------------
    PARAMETERS = [
        (False, "20b", False, "eager", "eval"),
        (False, "20b", False, "eager", "train"),
        (False, "20b", False, "kernels-community/vllm-flash-attn3", "eval"),
        (False, "20b", False, "kernels-community/vllm-flash-attn3", "train"),
        (False, "20b", True, "eager", "eval"),
        (False, "20b", True, "eager", "train"),
        (False, "20b", True, "kernels-community/vllm-flash-attn3", "eval"),
        (False, "20b", True, "kernels-community/vllm-flash-attn3", "train"),
        (True, "20b", False, "eager", "eval"),
        (True, "20b", False, "eager", "train"),
        (True, "20b", False, "kernels-community/vllm-flash-attn3", "eval"),
        (True, "20b", False, "kernels-community/vllm-flash-attn3", "train"),
        (True, "20b", True, "eager", "eval"),
        (True, "20b", True, "eager", "train"),
        (True, "20b", True, "kernels-community/vllm-flash-attn3", "eval"),
        (True, "20b", True, "kernels-community/vllm-flash-attn3", "train"),
        (False, "120b", False, "eager", "eval"),
        (False, "120b", False, "eager", "train"),
        (False, "120b", False, "kernels-community/vllm-flash-attn3", "eval"),
        (False, "120b", False, "kernels-community/vllm-flash-attn3", "train"),
        (False, "120b", True, "eager", "eval"),
        (False, "120b", True, "eager", "train"),
        (False, "120b", True, "kernels-community/vllm-flash-attn3", "eval"),
        (False, "120b", True, "kernels-community/vllm-flash-attn3", "train"),
        (True, "120b", False, "eager", "eval"),
        (True, "120b", False, "eager", "train"),
        (True, "120b", False, "kernels-community/vllm-flash-attn3", "eval"),
        (True, "120b", False, "kernels-community/vllm-flash-attn3", "train"),
        (True, "120b", True, "eager", "eval"),
        (True, "120b", True, "eager", "train"),
        (True, "120b", True, "kernels-community/vllm-flash-attn3", "eval"),
        (True, "120b", True, "kernels-community/vllm-flash-attn3", "train"),
    ]

    # ------------------------
    # Non-distributed test
    # ------------------------
    @parameterized.expand(PARAMETERS)
    @require_deterministic_for_xpu
    def test_model_outputs(self, quantized, model, kernels, attn_impl, mode):
        if torch_device == "xpu" and attn_impl == "kernels-community/vllm-flash-attn3":
            self.skipTest("flash attention 3 is not supported on XPU yet.")

        model_id = f"openai/gpt-oss-{model}"
        output_texts = self.load_and_forward(
            model_id,
            attn_impl,
            self.input_text,
            mode=mode,
            use_kernels=kernels,
        )

        # Generate key to look up expected outputs
        key = self.generate_config_key(quantized, model, kernels, attn_impl, mode)

        # Load expected outputs from restructured JSON
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                expected_results = json.load(f)

            # Check if we have expected results for this configuration
            if key in expected_results:
                expected_outputs = expected_results[key]

                # Compare actual outputs with expected outputs
                self.assertEqual(len(output_texts), len(expected_outputs), f"Output length mismatch for {key}")

                for i, (actual, expected) in enumerate(zip(output_texts, expected_outputs)):
                    actual_stripped = actual.strip()
                    expected_stripped = expected.strip()

                    # Make lengths match by taking minimum length to be resilient to generation differences
                    min_length = min(len(actual_stripped), len(expected_stripped))
                    actual_truncated = actual_stripped[:min_length]
                    expected_truncated = expected_stripped[:min_length]

                    if actual_truncated != expected_truncated:
                        diff = "\n".join(
                            difflib.unified_diff(
                                expected_truncated.splitlines(keepends=True),
                                actual_truncated.splitlines(keepends=True),
                                fromfile=f"expected[{i}]",
                                tofile=f"actual[{i}]",
                                lineterm="",
                            )
                        )
                        self.fail(
                            f"Output mismatch at index {i} for {key}:\n"
                            f"Expected: '{expected_stripped}'\n"
                            f"Actual:   '{actual_stripped}'\n"
                            f"Diff (truncated to min length {min_length}):\n{diff}"
                        )
            else:
                # If no expected results exist, this is a new configuration
                # We could optionally add it to the results file here
                print(f"Warning: No expected results found for configuration: {key}")

        self.assertIsInstance(output_texts, list)
        self.assertTrue(all(isinstance(x, str) for x in output_texts))

    # ------------------------
    # Distributed test
    # ------------------------
    @parameterized.expand(PARAMETERS)
    def test_model_outputs_distributed(self, quantized, model, kernels, attn_impl, mode):
        if torch_device == "xpu" and attn_impl == "kernels-community/vllm-flash-attn3":
            self.skipTest("flash attention 3 is not supported on XPU yet.")

        self.run_distributed_test(quantized, model, kernels, attn_impl, mode)

    # ------------------------
    # Training test
    # ------------------------
    @parameterized.expand(PARAMETERS)
    def test_training_step(self, quantized, model, kernels, attn_impl, mode):
        if mode != "train":
            self.skipTest("This test is only for training mode.")

        if quantized:
            self.skipTest("Training test for quantized models is not supported.")

        model_id = f"openai/gpt-oss-{model}"

        model_obj = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            use_kernels=kernels,
        )
        model_obj.train()

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(model_obj.device)
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model_obj(**inputs)
        loss = outputs.loss
        self.assertIsNotNone(loss)

        loss.backward()

        # Check that gradients were computed for all parameters that have a grad field
        for name, param in model_obj.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter '{name}' did not receive a gradient.")
                # Check that gradients are not all zero
                self.assertTrue(
                    torch.sum(torch.abs(param.grad)).item() > 0, f"Gradient for parameter '{name}' is all zeros."
                )

    def test_model_matches_original_20b(self):
        input_text = "Roses are red, violets"

        original_output = "Roses are red, violets are blue, I love you, and I love you too."
        original_logprobs = torch.tensor(
            [
                -0.037353515625,
                -0.08154296875,
                -1.21875,
                -1.953125,
                -2.234375,
                -0.96875,
                -1.546875,
                -1.640625,
                -0.93359375,
                -1.609375,
                -1.625,
                -0.85546875,
                -1.7265625,
                -0.7421875,
                -2.078125,
                -0.006561279296875,
                -0.10498046875,
                -0.1767578125,
                -0.1240234375,
                -0.099609375,
            ]
        )

        model_id = "openai/gpt-oss-20b"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(input_text)["input_ids"]

        num_generated_tokens = 0
        with torch.no_grad():
            for i in range(12):
                tensors = torch.as_tensor(tokens, dtype=torch.int32, device=model.device).unsqueeze(0)
                logits = model(tensors).logits[0]

                predicted_token = torch.argmax(logits[-1, :], dim=-1).item()
                logprobs = torch.log_softmax(logits[-1, :], dim=-1)
                selected_logprobs = logprobs[predicted_token]

                tokens.append(predicted_token)
                num_generated_tokens += 1
                decoded_token = tokenizer.decode([predicted_token])
                logprob_differences = selected_logprobs - original_logprobs[i]

                print(
                    f"Generated token: {repr(decoded_token)}, logprob: {selected_logprobs}, logprob differences: {logprob_differences}"
                )
                torch.testing.assert_close(
                    selected_logprobs.cpu().to(original_logprobs.dtype), original_logprobs[i], atol=1e-1, rtol=1e-1
                )

        decoded_string = tokenizer.decode(tokens)
        self.assertTrue(original_output.startswith(decoded_string))

    def test_model_matches_original_120b(self):
        input_text = "Roses are red, violets"

        original_output = """Roses are red, violets are blue,
I am a language model, not a human being"""
        original_logprobs = torch.tensor(
            [
                -0.90234375,
                -0.66015625,
                -1.546875,
                -2.703125,
                -2.078125,
                -1.21875,
                -2.484375,
                -0.031982421875,
                -0.84765625,
                -1.890625,
                -0.1923828125,
                -2.046875,
                -1.65625,
                -1.3515625,
                -1.1640625,
                -0.3671875,
                -1.9921875,
                -1.5390625,
                -1.46875,
                -0.85546875,
            ]
        )

        model_id = "openai/gpt-oss-120b"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(input_text)["input_ids"]

        num_generated_tokens = 0
        with torch.no_grad():
            for i in range(12):
                tensors = torch.as_tensor(tokens, dtype=torch.int32, device=model.device).unsqueeze(0)
                logits = model(tensors).logits[0]

                predicted_token = torch.argmax(logits[-1, :], dim=-1).item()
                logprobs = torch.log_softmax(logits[-1, :], dim=-1)
                selected_logprobs = logprobs[predicted_token]

                tokens.append(predicted_token)
                num_generated_tokens += 1
                decoded_token = tokenizer.decode([predicted_token])
                logprob_differences = selected_logprobs - original_logprobs[i]

                print(
                    f"Generated token: {repr(decoded_token)}, logprob: {selected_logprobs}, logprob differences: {logprob_differences}"
                )
                torch.testing.assert_close(
                    selected_logprobs.cpu().to(original_logprobs.dtype), original_logprobs[i], atol=1e-1, rtol=1e-1
                )

        decoded_string = tokenizer.decode(tokens)
        self.assertTrue(original_output.startswith(decoded_string))

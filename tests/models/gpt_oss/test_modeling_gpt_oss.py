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
    GptOssConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        GptOssForCausalLM,
        GptOssModel,
    )

    NUM_GPUS = torch.cuda.device_count()


class GptOssModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = GptOssConfig
        base_model_class = GptOssModel
        causal_lm_class = GptOssForCausalLM

    pipeline_model_mapping = (
        {
            "feature-extraction": GptOssModel,
            "text-generation": GptOssForCausalLM,
        }
        if is_torch_available()
        else {}
    )


@require_torch
class GptOssModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (GptOssModel, GptOssForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": GptOssModel,
            "text-generation": GptOssForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = GptOssModelTester

    def setUp(self):
        self.model_tester = GptOssModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GptOssConfig, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("GptOss's forcefully disables sdpa due to Sink")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("GptOss's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("GptOss has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("GptOss has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("GptOss has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("GptOss has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesn't work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("GptOss has HybridCache which auto-compiles. Compile and FA2 don't work together.")
    def test_eager_matches_fa2_generate(self):
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


RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/gpt_oss/integration_tests.json"


# ------------------------
# Worker function for distributed torchrun
# ------------------------
def distributed_worker(quantized, model_size, kernels, attn_impl, mode):
    """This is the function that will be executed by torchrun workers."""
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.testing_utils import torch_device

    input_text = [
        "Roses are red, violets",
        "How are you? Tell me the name of the president of",
    ]

    # Convert args
    quantized = quantized.lower() == "true"
    kernels = kernels.lower() == "true"

    # Distributed model loading
    model_id = f"/fsx/vb/new-oai/gpt-oss-{model_size}-trfs"
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

    # Only rank 0 writes results
    if int(os.environ.get("RANK", "0")) == 0:
        result_entry = {
            "quantized": quantized,
            "model": model_size,
            "kernels": kernels,
            "attn_impl": attn_impl,
            "mode": mode,
            "outputs": output_texts,
        }

        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                results = json.load(f)
        else:
            results = []
        results.append(result_entry)

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)


@slow
@require_torch_accelerator
class GptOssIntegrationTest(unittest.TestCase):
    input_text = [
        "Roses are red, violets",
        "How are you? Tell me the name of the president of",
    ]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    # ------------------------
    # Non-distributed inference
    # ------------------------
    @staticmethod
    def load_and_forward(model_id, attn_implementation, input_text, **pretrained_kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation,
            **pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)
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
        (False, "120b", False, "eager", "eval"),
        (False, "120b", False, "eager", "train"),
        (False, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (False, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
        (False, "120b", True, "eager", "eval"),
        (False, "120b", True, "eager", "train"),
        (False, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (False, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
        (True, "120b", False, "eager", "eval"),
        (True, "120b", False, "eager", "train"),
        (True, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (True, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
        (True, "120b", True, "eager", "eval"),
        (True, "120b", True, "eager", "train"),
        (True, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (True, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
        (False, "20b", False, "eager", "eval"),
        (False, "20b", False, "eager", "train"),
        (False, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (False, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
        (False, "20b", True, "eager", "eval"),
        (False, "20b", True, "eager", "train"),
        (False, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (False, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
        (True, "20b", False, "eager", "eval"),
        (True, "20b", False, "eager", "train"),
        (True, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (True, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
        (True, "20b", True, "eager", "eval"),
        (True, "20b", True, "eager", "train"),
        (True, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
        (True, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
    ]

    # ------------------------
    # Non-distributed test
    # ------------------------
    @parameterized.expand(PARAMETERS)
    @require_read_token
    def test_model_outputs(self, quantized, model, kernels, attn_impl, mode):
        model_id = f"/fsx/vb/new-oai/gpt-oss-{model}-trfs"
        output_texts = self.load_and_forward(
            model_id,
            attn_impl,
            self.input_text,
            use_kernels=kernels,
        )

        result_entry = {
            "quantized": quantized,
            "model": model,
            "kernels": kernels,
            "attn_impl": attn_impl,
            "mode": mode,
            "outputs": output_texts,
        }

        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                results = json.load(f)
        else:
            results = []
        results.append(result_entry)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        self.assertIsInstance(output_texts, list)
        self.assertTrue(all(isinstance(x, str) for x in output_texts))

    # ------------------------
    # Distributed test
    # ------------------------
    @parameterized.expand(PARAMETERS)
    @require_read_token
    def test_model_outputs_distributed(self, quantized, model, kernels, attn_impl, mode):
        self.run_distributed_test(quantized, model, kernels, attn_impl, mode)

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

        model_id = "/fsx/vb/new-oai/gpt-oss-20b-trfs"

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

        model_id = "/fsx/vb/new-oai/gpt-oss-120b-trfs"

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

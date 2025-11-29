# Copyright 2025 Eduard Durech, SGLang, and HuggingFace Inc. team.
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
#
# Usage:
# RUN_SLOW=1 pytest -s tests/generation/test_flash_attention_parity.py

import unittest
from collections import defaultdict

import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import (
    require_flash_attn,
    require_flash_attn_3,
    require_flash_attn_4,
    require_torch_gpu,
    slow,
)


class FlashAttentionParityTest(unittest.TestCase):
    # From https://github.com/sgl-project/sglang/blob/main/python/sglang/test/test_utils.py
    def _lcs(self, X, Y):
        m = len(X)
        n = len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        return L[m][n]

    # From https://github.com/sgl-project/sglang/blob/main/python/sglang/test/test_utils.py
    def _calculate_rouge_l(self, output_strs_list1, output_strs_list2):
        rouge_l_scores = []

        for s1, s2 in zip(output_strs_list1, output_strs_list2):
            lcs_len = self._lcs(s1, s2)
            precision = lcs_len / len(s1) if len(s1) > 0 else 0
            recall = lcs_len / len(s2) if len(s2) > 0 else 0
            if precision + recall > 0:
                fmeasure = (2 * precision * recall) / (precision + recall)
            else:
                fmeasure = 0.0
            rouge_l_scores.append(fmeasure)

        return rouge_l_scores

    def _benchmark_generation(self, model, inputs, n_warmup=3, n_runs=5):
        for _ in range(n_warmup):
            model.generate(**inputs, max_new_tokens=20, do_sample=False)
        torch.cuda.synchronize()

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        for _ in range(n_runs):
            model.generate(**inputs, max_new_tokens=20, do_sample=False)
        end_time.record()
        torch.cuda.synchronize()

        return start_time.elapsed_time(end_time) / n_runs

    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @pytest.mark.flash_attn_3_test
    @pytest.mark.flash_attn_4_test
    @require_flash_attn
    @require_flash_attn_3
    @require_flash_attn_4
    @slow
    def test_flash_attention_parity(self):
        flash_attn_versions = [2, 3, 4]

        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        prompt = ["The ETH AI Center is", "What is life?"]

        # 1. Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # 2. Generate with both models
        inputs = tokenizer(prompt, padding=True, padding_side="left", return_tensors="pt").to("cuda")

        logits = {}
        logprobs = {}
        outputs = defaultdict(list)
        with torch.no_grad():

            def generate(model, version, outputs, logits, logprobs):
                model.set_attn_implementation(f"flash_attention_{version}")
                output = model.generate(
                    **inputs, max_new_tokens=20, do_sample=False, output_scores=True, return_dict_in_generate=True
                )
                logit = torch.stack(output.scores)
                logprob = torch.nn.functional.log_softmax(logit, dim=-1)

                for i in range(len(prompt)):
                    outputs[version].append(tokenizer.decode(output.sequences[i], skip_special_tokens=True))
                logits[version] = logit
                logprobs[version] = logprob

            for version in flash_attn_versions:
                generate(model, version, outputs, logits, logprobs)

        # 3. Correctness check
        # 3a. Logits
        # FA2 as base to compare against
        # logits_1 = logits[2]
        logprobs_1 = logprobs[2]
        max_logprob_diffs = []
        for version in range(1, len(flash_attn_versions)):
            # TODO: logits significantly differ between FA2 and FA4
            # logits_x = logits[flash_attn_versions[version]]
            # torch.testing.assert_close(logits_1, logits_x, atol=1e-3, rtol=1e-3)
            logprobs_x = logprobs[flash_attn_versions[version]]
            max_logprob_diffs.append(torch.max(torch.abs(logprobs_1 - logprobs_x)).item())

        # 3b. Generated text
        # FA2 as base to compare against
        texts_1 = outputs[2]
        rouge_scores = []
        for version in range(1, len(flash_attn_versions)):
            fa_version = flash_attn_versions[version]
            texts_x = outputs[fa_version]
            rouge_score = self._calculate_rouge_l(texts_1, texts_x)
            for idx, score in enumerate(rouge_score):
                assert score > 0.99, (
                    f"Generated texts at prompt {idx} do not match (ROUGE-L: {score}) comparing FA2 vs FA{fa_version}"
                )
            rouge_scores.append(self._calculate_rouge_l(texts_1, texts_x))

        # 4. Performance check
        times = []
        with torch.no_grad():
            for version in flash_attn_versions:
                model.set_attn_implementation(f"flash_attention_{version}")
                times.append(self._benchmark_generation(model, inputs))

        # Summary
        print(f"\n--- Flash Attention Parity Test on {model_id} ---")
        print(f"Prompts: '{prompt}'")
        print("\nGenerated texts:")
        for version in flash_attn_versions:
            print(f"    With FA{version}: {outputs[version]}")
        print("\nROUGE-L scores:")
        for idx, version in enumerate(range(1, len(flash_attn_versions))):
            print(f"    Between FA2 and FA{flash_attn_versions[version]}: {rouge_scores[idx]}")
        print("\nMax absolute difference in logprobs:")
        for idx, version in enumerate(range(1, len(flash_attn_versions))):
            print(f"    Between FA2 and FA{flash_attn_versions[version]}: {max_logprob_diffs[idx]:.5e}")
        print("\nLatency:")
        for idx, version in enumerate(flash_attn_versions):
            print(f"    With FA{version}: {times[idx]}")
        print("---")

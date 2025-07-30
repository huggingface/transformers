# Copyright 2025 Eduard Durech and SGLang team.
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

import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import require_flash_attn, require_flash_attn_3, require_torch_gpu, slow


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

    @pytest.mark.flash_attn_3_test
    @require_torch_gpu
    @require_flash_attn
    @require_flash_attn_3
    @slow
    def test_flash_attention_2_3_parity(self):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        prompt = "The ETH AI Center is"

        # 1. Load FA2 model and tokenizer
        model_2 = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 2. Load FA3 model
        try:
            model_3 = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_3",
            ).to("cuda")
        except (ValueError, ImportError) as e:
            pytest.skip(f"Could not load Flash Attention 3 model, skipping test. Error: {e}")

        # 3. Generate with both models
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_2 = model_2.generate(
                **inputs, max_new_tokens=20, do_sample=False, output_scores=True, return_dict_in_generate=True
            )
            output_3 = model_3.generate(
                **inputs, max_new_tokens=20, do_sample=False, output_scores=True, return_dict_in_generate=True
            )

        # 4. Correctness check
        # 4a. Logits
        logits_2 = torch.stack(output_2.scores)
        logits_3 = torch.stack(output_3.scores)
        torch.testing.assert_close(logits_2, logits_3, atol=1e-3, rtol=1e-3)
        logprobs_2 = torch.nn.functional.log_softmax(logits_2, dim=-1)
        logprobs_3 = torch.nn.functional.log_softmax(logits_3, dim=-1)
        max_logprob_diff = torch.max(torch.abs(logprobs_2 - logprobs_3)).item()

        # 4b. Generated text
        text_2 = tokenizer.decode(output_2.sequences[0], skip_special_tokens=True)
        text_3 = tokenizer.decode(output_3.sequences[0], skip_special_tokens=True)
        rouge_score = self._calculate_rouge_l([text_2], [text_3])[0]
        assert rouge_score > 0.99, f"Generated texts do not match (ROUGE-L: {rouge_score})"

        # 5. Performance check
        with torch.no_grad():
            time_2 = self._benchmark_generation(model_2, inputs)
            time_3 = self._benchmark_generation(model_3, inputs)

        print(f"\n--- Flash Attention {2, 3} Parity Test on {model_id} ---")
        print(f"Prompt: '{prompt}'")
        print(f"Generated text with Flash Attention 2: {text_2}")
        print(f"Generated text with Flash Attention 3: {text_3}")
        print(f"ROUGE-L: {rouge_score}")
        print(f"Max absolute difference in logprobs: {max_logprob_diff:.5e}")
        print(f"Flash Attention 2 latency: {time_2:.2f} ms")
        print(f"Flash Attention 3 latency: {time_3:.2f} ms")
        print(f"Speed-up: {time_2 / time_3:.2f}x")
        print("---")

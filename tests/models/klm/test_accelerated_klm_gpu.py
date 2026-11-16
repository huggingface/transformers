# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Actual CUDA kernel qualification at KLM's smallest accelerated recurrent head dimension."""

import contextlib
import copy
import json
import math
import sys
import unittest
from unittest.mock import patch

from transformers import is_torch_available
from transformers.utils import is_kernels_available


if is_torch_available():
    import torch

    from transformers import KlmConfig, KlmForCausalLM, KlmModel
    from transformers.models.klm import modeling_klm as klm


CUDA_FLA = is_torch_available() and torch.cuda.is_available() and klm.is_klm_fla_available
SMOKE_METRICS = []
FLASH_REVISION = "291664546de978ab2bb730554252e5e2c1099e4f"


@unittest.skipUnless(CUDA_FLA, "Requires CUDA and the optional FLA KLM kernels")
class KlmAcceleratedCudaTest(unittest.TestCase):
    @staticmethod
    def _error(actual, reference):
        squared_error = squared_norm = maximum = 0.0
        for a, b in zip(actual, reference, strict=True):
            a, b = a.detach().double().cpu(), b.detach().double().cpu()
            difference = a - b
            squared_error += difference.square().sum().item()
            squared_norm += b.square().sum().item()
            maximum = max(maximum, difference.abs().max().item())
        return {"relative_l2": math.sqrt(squared_error / max(squared_norm, 1e-30)), "max_abs": maximum}

    def _setup_case(self, amp, causal_lm=False):
        torch.manual_seed(3407)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            linear_num_heads=2,
            linear_head_dim=8,
            num_bigrams=16,
            num_trigrams=8,
            ngram_embedding_size=8,
            layer_types=["linear_attention", "full_attention"] * 2,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        reference = (KlmForCausalLM(config) if causal_lm else KlmModel(config)).cuda().train()
        for name, parameter in reference.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter, std=0.1)
        core = reference.model if causal_lm else reference
        core.embed_ngrams.bigram_keys.copy_(torch.arange(16, device="cuda") + 32)
        core.embed_ngrams.trigram_keys.copy_(torch.arange(8, device="cuda") + 32**2 + 32 * 5)
        core._use_fla_kernels = False
        accelerated = copy.deepcopy(reference)
        (accelerated.model if causal_lm else accelerated)._use_fla_kernels = True
        if amp:
            accelerated.set_attn_implementation(f"kernels-community/flash-attn2@{FLASH_REVISION}")
        for model in (reference, accelerated):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        generator = torch.Generator().manual_seed(3407)
        batches = []
        for length in (19, 29):
            ids = torch.randint(3, 32, (2, length), generator=generator)
            mask = torch.ones_like(ids, dtype=torch.bool)
            mask[0, :3] = False
            mask[1, -4:] = False
            ids[:, 6:8] = 2
            ids[:, 12] = 2
            ids[~mask] = 0
            direction = torch.randn(2, length, 32, generator=generator)
            batches.append((ids.cuda(), mask.cuda(), direction.cuda()))
        return reference, accelerated, batches

    def _check(self, amp):
        reference, accelerated, batches = self._setup_case(amp)

        def run(model):
            losses, outputs = [], []
            for ids, mask, direction in batches:
                precision = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
                with precision:
                    hidden = model(ids, attention_mask=mask, use_cache=False).last_hidden_state
                    outputs.append(hidden[mask].detach().cpu())
                    losses.append((hidden[mask].float() * direction[mask]).mean())
            (sum(losses) / len(losses)).backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                gradients[name] = parameter.grad.detach().cpu()
            return outputs, gradients

        expected_outputs, expected_gradients = run(reference)
        with patch.object(klm, "klm_chunk_gdn2", wraps=klm.klm_chunk_gdn2) as recurrence:
            actual_outputs, actual_gradients = run(accelerated)
        self.assertGreater(recurrence.call_count, 0)
        self.assertTrue(all(call.args[0].shape[-1] == 8 for call in recurrence.call_args_list))
        if amp:
            real_token_counts = {int(mask.sum()) for _, mask, _ in batches}
            self.assertTrue(all(call.args[0].shape[1] in real_token_counts for call in recurrence.call_args_list))
        self.assertEqual(actual_gradients.keys(), expected_gradients.keys())
        output_error = self._error(actual_outputs, expected_outputs)
        gradient_error = self._error(
            actual_gradients.values(), [expected_gradients[name] for name in actual_gradients]
        )
        output_limit, gradient_limit = (0.02, 0.04) if amp else (0.002, 0.01)
        result = {
            "precision": "amp" if amp else "fp32",
            "linear_head_dim": 8,
            "forward_shapes": [list(ids.shape) for ids, _, _ in batches],
            "gradient_tensors": len(actual_gradients),
            "fla_recurrence_calls": recurrence.call_count,
            "output_error": output_error,
            "gradient_error": gradient_error,
            "gates": {"output_relative_l2": output_limit, "gradient_relative_l2": gradient_limit},
        }
        SMOKE_METRICS.append(result)
        print("GPU_SMOKE_CASE " + json.dumps(result), flush=True)
        self.assertLessEqual(output_error["relative_l2"], output_limit)
        self.assertLessEqual(gradient_error["relative_l2"], gradient_limit)

    def test_small_recurrent_head_fp32(self):
        self._check(amp=False)

    @unittest.skipUnless(is_kernels_available(), "Requires the FlashAttention Hub kernel interface")
    def test_small_recurrent_head_amp_flash(self):
        self._check(amp=True)

    def _check_causal_lm(self, amp):
        for external in (False, True):
            with self.subTest(loss="external" if external else "builtin"):
                reference, accelerated, batches = self._setup_case(amp, causal_lm=True)

                def run(model):
                    losses, outputs = [], []
                    for ids, mask, _ in batches:
                        labels = ids.masked_fill(~mask, -100)
                        precision = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
                        with precision:
                            result = model(
                                ids, attention_mask=mask, labels=None if external else labels, use_cache=False
                            )
                            outputs.append(result.logits.detach().cpu())
                            loss = (
                                torch.nn.functional.cross_entropy(
                                    result.logits[:, :-1].float().reshape(-1, model.config.vocab_size),
                                    labels[:, 1:].reshape(-1),
                                )
                                if external
                                else result.loss
                            )
                            losses.append(loss)
                    loss = sum(losses) / len(losses)
                    loss.backward()
                    gradients = {}
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        gradients[name] = parameter.grad.detach().cpu()
                    return outputs, gradients, loss.detach().item()

                expected_outputs, expected_gradients, expected_loss = run(reference)
                with patch.object(klm, "klm_chunk_gdn2", wraps=klm.klm_chunk_gdn2) as recurrence:
                    actual_outputs, actual_gradients, actual_loss = run(accelerated)
                self.assertGreater(recurrence.call_count, 0)
                full_token_counts = {ids.numel() for ids, _, _ in batches}
                self.assertTrue(all(call.args[0].shape[1] in full_token_counts for call in recurrence.call_args_list))
                self.assertEqual(actual_gradients.keys(), expected_gradients.keys())
                output_error = self._error(actual_outputs, expected_outputs)
                gradient_error = self._error(
                    actual_gradients.values(), [expected_gradients[name] for name in actual_gradients]
                )
                output_limit, gradient_limit = (0.02, 0.04) if amp else (0.002, 0.01)
                loss_error = abs(actual_loss - expected_loss) / max(abs(expected_loss), 1e-30)
                result = {
                    "precision": "amp" if amp else "fp32",
                    "model": "causal_lm",
                    "loss_kind": "external_shifted" if external else "builtin",
                    "gradient_tensors": len(actual_gradients),
                    "fla_recurrence_calls": recurrence.call_count,
                    "padding_predictors_retained": True,
                    "output_error": output_error,
                    "gradient_error": gradient_error,
                    "loss_relative_error": loss_error,
                    "gates": {"output_relative_l2": output_limit, "gradient_relative_l2": gradient_limit},
                }
                SMOKE_METRICS.append(result)
                print("GPU_SMOKE_CASE " + json.dumps(result), flush=True)
                self.assertLessEqual(loss_error, output_limit)
                self.assertLessEqual(output_error["relative_l2"], output_limit)
                self.assertLessEqual(gradient_error["relative_l2"], gradient_limit)

    def test_causal_lm_padding_losses_fp32(self):
        self._check_causal_lm(amp=False)

    @unittest.skipUnless(is_kernels_available(), "Requires the FlashAttention Hub kernel interface")
    def test_causal_lm_padding_losses_amp_flash(self):
        self._check_causal_lm(amp=True)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and not result.skipped
    print(
        "GPU_SMOKE "
        + json.dumps(
            {
                "passed": passed,
                "tests_run": result.testsRun,
                "skipped": len(result.skipped),
                "checks": SMOKE_METRICS,
            }
        ),
        flush=True,
    )
    sys.exit(0 if passed else 1)

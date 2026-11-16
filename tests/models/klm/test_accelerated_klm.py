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
"""CPU semantic checks for the packed training integration, independent of CUDA kernels."""

import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import DynamicCache, KlmConfig, KlmForCausalLM, KlmModel
    from transformers.models.klm import modeling_klm as klm


@require_torch
class KlmKernelEligibilityTest(unittest.TestCase):
    def test_offload_and_unsupported_widths_use_reference_operators(self):
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_heads=2,
            linear_head_dim=16,
            num_bigrams=8,
            num_trigrams=8,
            ngram_embedding_size=4,
            layer_types=["linear_attention", "full_attention"],
        )
        model = KlmModel(config)
        model._use_fla_kernels = True
        cuda_input = SimpleNamespace(device=torch.device("cuda"))
        self.assertTrue(model._can_use_fla(cuda_input, None))
        model.norm._hf_hook = object()
        self.assertFalse(model._can_use_fla(cuda_input, None))
        del model.norm._hf_hook
        for width in (4, 257):
            model.config.linear_head_dim = width
            self.assertFalse(model._can_use_fla(cuda_input, None))


@require_torch
class KlmAcceleratedIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        config = KlmConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            linear_num_heads=2,
            linear_head_dim=16,
            num_bigrams=16,
            num_trigrams=8,
            ngram_embedding_size=8,
            layer_types=["linear_attention", "full_attention"] * 2,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        self.reference = KlmModel(config).train()
        for name, parameter in self.reference.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter, std=0.1)
        self.reference.embed_ngrams.bigram_keys.copy_(torch.arange(16) + 32)
        self.reference.embed_ngrams.trigram_keys.copy_(torch.arange(8) + 32**2 + 32 * 5)
        self.reference._use_fla_kernels = False
        self.accelerated = copy.deepcopy(self.reference)
        self.accelerated._use_fla_kernels = True
        # Construct with the installed CPU backend, then replace FlashAttention with an independent segmented SDPA.
        self.accelerated.config._attn_implementation = "flash_attention_2"
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(
            patch.object(
                KlmModel,
                "_can_use_fla",
                lambda model, ids, cache: model._use_fla_kernels and cache is None,
            )
        )
        self.context.enter_context(
            patch.object(
                KlmModel,
                "_can_use_flash_attention",
                lambda model, ids: model.config._attn_implementation == "flash_attention_2",
            )
        )
        self.attnres = self.context.enter_context(
            patch.object(klm, "klm_fused_attnres", side_effect=self.attnres_reference)
        )
        self.context.enter_context(patch.object(klm, "klm_causal_conv1d", side_effect=self.conv_reference))
        self.gdn = self.context.enter_context(patch.object(klm, "klm_chunk_gdn2", side_effect=self.gdn_reference))
        self.flash = self.context.enter_context(
            patch.dict(
                klm.ALL_ATTENTION_FUNCTIONS._local_mapping,
                {"flash_attention_2": self.flash_reference},
            )
        )

    @staticmethod
    def attnres_reference(query, residuals, rms_weight, output_rms_weight=None, rms_eps=1e-6, **kwargs):
        states = torch.stack(residuals, dim=-2).float()
        normalized = F.rms_norm(states, (states.shape[-1],), eps=rms_eps)
        scores = (normalized * (query * rms_weight).float()).sum(-1)
        output = (scores.softmax(-1).unsqueeze(-1) * states).sum(-2)
        if output_rms_weight is not None:
            output = F.rms_norm(output, (output.shape[-1],), output_rms_weight.float(), eps=rms_eps)
        return output.to(residuals[0].dtype)

    @staticmethod
    def conv_reference(x, weight, activation=None, cu_seqlens=None, cu_seqlens_cpu=None, **kwargs):
        flat = x.reshape(-1, x.shape[-1])
        offsets = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
        outputs = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            part = flat[start:end].T.unsqueeze(0)
            result = F.conv1d(part, weight[:, None], padding=weight.shape[-1] - 1, groups=flat.shape[-1])
            outputs.append(result[..., : end - start].squeeze(0).T)
        output = torch.cat(outputs).reshape_as(x)
        return (F.silu(output) if activation == "silu" else output), None

    @staticmethod
    def gdn_reference(q, k, v, g, b, w, cu_seqlens=None, cu_seqlens_cpu=None, scale=None, **kwargs):
        offsets = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        starts = torch.zeros(q.shape[:2], dtype=torch.bool, device=q.device)
        starts.reshape(-1)[offsets[:-1].long()] = True
        state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        outputs = []
        for index in range(q.shape[1]):
            state = state * g[:, index].exp().unsqueeze(-1)
            state = state.masked_fill(starts[:, index, None, None, None], 0)
            erased = ((b[:, index] * k[:, index]).unsqueeze(-1) * state).sum(-2)
            state = state + k[:, index].unsqueeze(-1) * (w[:, index] * v[:, index] - erased).unsqueeze(-2)
            outputs.append((q[:, index].unsqueeze(-1) * state).sum(-2) * (scale or q.shape[-1] ** -0.5))
        return torch.stack(outputs, dim=1), None

    @staticmethod
    def flash_reference(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        offsets = kwargs["cu_seq_lens_q"].tolist()
        length = query.shape[-2]
        mask = torch.zeros((length, length), dtype=torch.bool, device=query.device)
        for start, end in zip(offsets[:-1], offsets[1:]):
            mask[start:end, start:end] = torch.ones((end - start, end - start), dtype=torch.bool).tril()
        groups = query.shape[1] // key.shape[1]
        result = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, 1),
            value.repeat_interleave(groups, 1),
            attn_mask=mask,
            dropout_p=dropout,
            scale=scaling,
        )
        return result.transpose(1, 2), None

    def inputs(self, kind="padding"):
        ids = torch.tensor([[0, 0, 1, 5, 6, 2, 2, 7, 8], [1, 5, 2, 7, 8, 9, 0, 0, 0]])
        mask = ids != 0
        if kind == "empty_row":
            ids[0] = 0
            mask[0] = False
        elif kind == "holes":
            ids[1, 3] = 0
            mask[1, 3] = False
        elif kind == "empty":
            ids.zero_()
            mask.zero_()
        return ids, mask

    def assert_gradients_match(self):
        actual = dict(self.accelerated.named_parameters())
        for name, expected in self.reference.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(expected.grad)
                self.assertIsNotNone(actual[name].grad)
                torch.testing.assert_close(actual[name].grad, expected.grad, rtol=3e-4, atol=3e-6)

    def test_packed_outputs_and_all_gradients(self):
        for kind in ("padding", "empty_row", "holes"):
            with self.subTest(kind=kind):
                self.reference.zero_grad(set_to_none=True)
                self.accelerated.zero_grad(set_to_none=True)
                ids, mask = self.inputs(kind)
                direction = torch.randn(*ids.shape, 32)
                expected = self.reference(ids, attention_mask=mask).last_hidden_state
                actual = self.accelerated(ids, attention_mask=mask).last_hidden_state
                self.assertEqual(actual.shape, expected.shape)
                torch.testing.assert_close(actual[mask], expected[mask], rtol=3e-4, atol=3e-6)
                (expected[mask] * direction[mask]).mean().backward()
                (actual[mask] * direction[mask]).mean().backward()
                self.assert_gradients_match()
        self.assertGreater(self.attnres.call_count, 0)

    def test_checkpoint_multiple_forwards_before_backward(self):
        self.accelerated.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        expected_loss = actual_loss = 0
        for kind in ("padding", "holes"):
            ids, mask = self.inputs(kind)
            direction = torch.randn(*ids.shape, 32)
            expected = self.reference(ids, attention_mask=mask).last_hidden_state
            actual = self.accelerated(ids, attention_mask=mask).last_hidden_state
            expected_loss += (expected[mask] * direction[mask]).mean()
            actual_loss += (actual[mask] * direction[mask]).mean()
        expected_loss.backward()
        actual_loss.backward()
        self.assert_gradients_match()

    def test_causal_lm_retains_padding_queries_that_predict_supervised_tokens(self):
        reference = KlmForCausalLM(copy.deepcopy(self.reference.config)).train()
        reference.model.load_state_dict(self.reference.state_dict())
        reference.model._use_fla_kernels = False
        accelerated = copy.deepcopy(reference)
        accelerated.model._use_fla_kernels = True
        accelerated.config._attn_implementation = "flash_attention_2"
        ids, mask = self.inputs()
        labels = ids.masked_fill(~mask, -100)
        expected = reference(ids, attention_mask=mask, labels=labels)
        actual = accelerated(ids, attention_mask=mask, labels=labels)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
        actual.loss.backward()
        expected.loss.backward()
        self.reference, self.accelerated = reference, accelerated
        self.assert_gradients_match()
        self.assertEqual(self.gdn.call_args.args[0].shape[1], ids.numel())

    def test_hidden_state_capture_preserves_padded_shapes(self):
        ids, mask = self.inputs()
        expected = self.reference(ids, attention_mask=mask, output_hidden_states=True)
        actual = self.accelerated(ids, attention_mask=mask, output_hidden_states=True)
        self.assertEqual(self.attnres.call_count, 0)
        self.assertEqual(len(actual.hidden_states), len(expected.hidden_states))
        for x, y in zip(actual.hidden_states, expected.hidden_states):
            self.assertEqual(x.shape, (*ids.shape, 32))
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)

    def test_external_causal_loss_retains_padding_position_logits(self):
        reference = KlmForCausalLM(copy.deepcopy(self.reference.config)).train()
        reference.model.load_state_dict(self.reference.state_dict())
        reference.model._use_fla_kernels = False
        accelerated = copy.deepcopy(reference)
        accelerated.model._use_fla_kernels = True
        accelerated.config._attn_implementation = "flash_attention_2"
        ids, mask = self.inputs()
        labels = ids.masked_fill(~mask, -100)
        expected = reference(ids, attention_mask=mask).logits
        actual = accelerated(ids, attention_mask=mask).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for logits in (expected, actual):
            F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)).backward()
        self.reference, self.accelerated = reference, accelerated
        self.assert_gradients_match()
        self.assertEqual(self.gdn.call_args.args[0].shape[1], ids.numel())

    def test_cache_never_uses_training_kernels(self):
        ids, mask = self.inputs()
        expected_cache, actual_cache = (
            DynamicCache(config=self.reference.config),
            DynamicCache(config=self.accelerated.config),
        )
        for start, end in ((0, 4), (4, 9)):
            expected = self.reference(
                ids[:, start:end], attention_mask=mask[:, :end], cache=expected_cache, use_cache=True
            )
            actual = self.accelerated(
                ids[:, start:end], attention_mask=mask[:, :end], cache=actual_cache, use_cache=True
            )
            torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
        self.assertEqual(self.attnres.call_count, 0)

    def test_attention_capture_uses_portable_eager_path(self):
        ids, mask = self.inputs()
        self.reference.config._attn_implementation = "eager"
        expected = self.reference(ids, attention_mask=mask, output_attentions=True)
        actual = self.accelerated(ids, attention_mask=mask, output_attentions=True)
        self.assertEqual(self.attnres.call_count, 0)
        self.assertEqual(len(actual.attentions), len(expected.attentions))
        for x, y in zip(actual.attentions, expected.attentions):
            self.assertEqual(x.shape, (ids.shape[0], 4, ids.shape[1], ids.shape[1]))
            torch.testing.assert_close(x, y)

    def test_all_masked_input_falls_back(self):
        ids, mask = self.inputs("empty")
        expected = self.reference(ids, attention_mask=mask).last_hidden_state
        actual = self.accelerated(ids, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(actual, expected)
        self.assertEqual(self.attnres.call_count, 0)

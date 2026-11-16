# Copyright 2026 The KLM authors and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch KLM model."""

import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers import KlmConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, DynamicCache, KlmForCausalLM, KlmModel
    from transformers.models.klm.modeling_klm import (
        KlmRotaryEmbedding,
        torch_chunk_gated_delta_rule_2,
        torch_recurrent_gated_delta_rule_2,
    )


class KlmModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = KlmConfig
        base_model_class = KlmModel
        causal_lm_class = KlmForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["linear_attention", "full_attention"]
        self.linear_num_heads = 2
        self.linear_head_dim = 8
        self.linear_conv_kernel_dim = 4
        self.num_bigrams = 24
        self.num_trigrams = 16
        self.ngram_embedding_size = 8
        self.hidden_act = "silu"


@require_torch
class KlmModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = KlmModelTester
    rotary_embedding_layer = KlmRotaryEmbedding if is_torch_available() else None

    def _get_model(self, config=None):
        if config is None:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.eos_token_id = 2
        model = KlmModel(config).to(torch_device).eval()
        # Every block residual starts as a plain mean, random queries make the test sensitive to the order of states
        for name, parameter in model.named_parameters():
            if name.endswith("_residual.query"):
                torch.nn.init.normal_(parameter)
        return model

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        """Overwritten: only the first linear attention layer uses its second conv state, for the document state."""
        self.assertIsInstance(cache, DynamicCache)
        self.assertEqual(len(cache), config.num_hidden_layers)
        qkv_dim = config.linear_num_heads * config.linear_head_dim
        k_shape, v_shape = self._get_attention_shape(batch_size, seq_length, config)
        document_layer_idx = config.layer_types.index("linear_attention")
        for layer_idx, layer in enumerate(cache.layers):
            if config.layer_types[layer_idx] == "full_attention":
                self._check_attention_shapes(layer, seq_length, k_shape, v_shape)
                continue
            self.assertEqual(layer.conv_states[0].shape, (batch_size, 3 * qkv_dim, config.linear_conv_kernel_dim))
            self.assertEqual(
                layer.recurrent_states[0].shape,
                (batch_size, config.linear_num_heads, config.linear_head_dim, config.linear_head_dim),
            )
            if layer_idx == document_layer_idx:
                self.assertEqual(layer.conv_states[1].shape, (batch_size, 3, 1))
            else:
                self.assertIsNone(layer.conv_states[1])

    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip("KLM needs `input_ids` to look up n-grams and to find where documents start")
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass

    @unittest.skip("KLM needs `input_ids` to look up n-grams and to find where documents start")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("KLM needs `input_ids` to look up n-grams and to find where documents start")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip("Layers read all the previous hidden states, which only non-reentrant checkpointing can pass")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    def test_attention_outputs(self):
        """Needs to be overwritten as KLM alternates between attention layers and gated deltanet layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = self.model_tester.seq_length
        num_attention_layers = sum(layer_type == "full_attention" for layer_type in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(outputs.attentions), num_attention_layers)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len]
            )

    def test_chunk_gated_delta_rule_matches_recurrent(self):
        """The chunked kernel has to give the outputs and final state of the token by token one."""
        batch_size, seq_len, num_heads, head_dim = 3, 75, 2, 8
        shape = (batch_size, seq_len, num_heads, head_dim)
        query, key, value = torch.randn(3, *shape, dtype=torch.float64, device=torch_device)
        decay = -3 * torch.rand(shape, dtype=torch.float64, device=torch_device)
        erase, write = torch.rand(2, *shape, dtype=torch.float64, device=torch_device)
        # Documents that start inside a chunk, one right after another, and on a chunk boundary
        document_starts = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=torch_device)
        document_starts[0, [9, 10, 40]] = True
        document_starts[1, [32, 64]] = True
        initial_state = torch.randn(batch_size, num_heads, head_dim, head_dim, device=torch_device)

        for state in (None, initial_state):
            inputs = (query, key, value, decay, erase, write, document_starts, state)
            expected_output, expected_state = torch_recurrent_gated_delta_rule_2(*inputs)
            for chunk_size in (1, 7, 32, 128):
                output, final_state = torch_chunk_gated_delta_rule_2(*inputs, chunk_size=chunk_size)
                torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(final_state, expected_state, rtol=1e-4, atol=1e-5)

    def test_documents_are_independent(self):
        """Whatever follows an end of sequence token is read as if nothing came before it."""
        model = self._get_model()
        vocab_size, eos_token_id = model.config.vocab_size, model.config.eos_token_id
        first = ids_tensor((2, 9), vocab_size - 3).to(torch_device) + 3
        second = ids_tensor((2, 40), vocab_size - 3).to(torch_device) + 3
        eos = torch.full((2, 1), eos_token_id, device=torch_device)

        for implementation in ("eager", "sdpa"):
            model.set_attn_implementation(implementation)
            with torch.no_grad():
                expected = model(second).last_hidden_state
                output = model(torch.cat([first, eos, second], dim=1)).last_hidden_state[:, -second.shape[1] :]
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_left_padding_is_not_read(self):
        """The first token after left padding starts a document, so the padding cannot change the outputs."""
        model = self._get_model()
        input_ids = ids_tensor((2, 20), model.config.vocab_size - 3).to(torch_device) + 3
        padding = torch.full((2, 6), model.config.pad_token_id, device=torch_device)
        attention_mask = torch.cat([torch.zeros_like(padding), torch.ones_like(input_ids)], dim=1)

        with torch.no_grad():
            expected = model(input_ids).last_hidden_state
            output = model(torch.cat([padding, input_ids], dim=1), attention_mask=attention_mask).last_hidden_state
        torch.testing.assert_close(output[:, padding.shape[1] :], expected, rtol=1e-4, atol=1e-4)

    def test_cached_forward_matches_full_forward(self):
        """
        Feeding a sequence in pieces with a cache has to give the outputs of a single forward, whether a piece is one
        token or several, and whether documents end inside a piece or between two of them.
        """
        model = self._get_model()
        input_ids = ids_tensor((2, 60), model.config.vocab_size - 3).to(torch_device) + 3
        input_ids[0, [4, 5, 33]] = model.config.eos_token_id
        input_ids[1, [17, 41]] = model.config.eos_token_id

        with torch.no_grad():
            expected = model(input_ids).last_hidden_state
            for pieces in ([1] * 60, [18, 1, 1, 22, 1, 17], [5, 55]):
                cache = DynamicCache(config=model.config)
                outputs = [
                    model(piece, cache=cache, use_cache=True).last_hidden_state
                    for piece in input_ids.split(pieces, dim=1)
                ]
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-4, atol=1e-4)

    def test_autocast(self):
        """The recurrence of the linear attention layers stays in float32 under autocast."""
        torch.manual_seed(0)
        model = self._get_model(self.model_tester.get_config())
        input_ids = torch.arange(80, device=torch_device).reshape(2, 40) % model.config.vocab_size

        def check_recurrence(*args, **kwargs):
            self.assertFalse(torch.is_autocast_enabled(torch_device))
            output, state = torch_chunk_gated_delta_rule_2(*args, **kwargs)
            self.assertEqual(output.dtype, torch.float32)
            self.assertEqual(state.dtype, torch.float32)
            # Compare the same projected operands, so this checks recurrence precision independently of BF16
            # projection rounding and its amplification by the randomly initialized residual routing.
            expected_output, expected_state = torch_recurrent_gated_delta_rule_2(*args, **kwargs)
            torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-5)
            return output, state

        with patch(
            "transformers.models.klm.modeling_klm.torch_chunk_gated_delta_rule_2", side_effect=check_recurrence
        ) as recurrence:
            with torch.no_grad(), torch.autocast(torch_device, dtype=torch.bfloat16):
                output = model(input_ids).last_hidden_state
        self.assertEqual(recurrence.call_count, model.config.layer_types.count("linear_attention"))
        self.assertTrue(torch.isfinite(output).all())

    @parameterized.expand([("single_token", 1), ("shorter_padding", 2), ("longer_padding", 3), ("inside_document", 4)])
    def test_cached_left_padding(self, _, chunk_size):
        model = self._get_model()
        input_ids = torch.tensor([[0, 0, 0, 5, 6, 7], [0, 0, 3, 8, 9, 4]], device=torch_device)
        attention_mask = input_ids.ne(0)
        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            cache = DynamicCache(config=model.config)
            outputs = []
            for start in range(0, input_ids.shape[1], chunk_size):
                end = min(start + chunk_size, input_ids.shape[1])
                outputs.append(
                    model(
                        input_ids[:, start:end], attention_mask=attention_mask[:, :end], cache=cache, use_cache=True
                    ).last_hidden_state
                )
        actual = torch.cat(outputs, dim=1)
        torch.testing.assert_close(actual[attention_mask], expected[attention_mask], rtol=1e-4, atol=1e-4)

    def test_generate_chunked_prefill_left_padding(self):
        torch.manual_seed(41)
        config = self.model_tester.get_config()
        model = KlmForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[0, 0, 0, 5, 6, 7], [0, 0, 3, 8, 9, 4]], device=torch_device)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0),
            "max_new_tokens": 3,
            "do_sample": False,
            "eos_token_id": None,
            "pad_token_id": 0,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        with torch.no_grad():
            expected = model.generate(**kwargs)
            actual = model.generate(**kwargs, prefill_chunk_size=3)
        torch.testing.assert_close(actual.sequences, expected.sequences)
        for actual_scores, expected_scores in zip(actual.scores, expected.scores):
            torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-4, atol=1e-4)

    def test_ngram_embeddings(self):
        """N-grams are looked up by key and do not reach back past the start of a document."""
        model = self._get_model()
        embed_ngrams, vocab_size = model.embed_ngrams, model.config.vocab_size
        input_ids = torch.tensor([[5, 6, 7, 2, 5, 6, 7, 8]], device=torch_device)
        document_starts = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.bool, device=torch_device)
        previous_ids = torch.zeros((1, 2), dtype=torch.long, device=torch_device)
        # Known n-grams: (5, 6), (7, 8) and (5, 6, 7). The bank is sorted, the unused keys are larger than any n-gram.
        bigram_keys = torch.full_like(embed_ngrams.bigram_keys, vocab_size**3)
        bigram_keys[:2] = torch.tensor([5 * vocab_size + 6, 7 * vocab_size + 8])
        trigram_keys = torch.full_like(embed_ngrams.trigram_keys, vocab_size**3)
        trigram_keys[0] = (5 * vocab_size + 6) * vocab_size + 7
        embed_ngrams.bigram_keys.copy_(bigram_keys)
        embed_ngrams.trigram_keys.copy_(trigram_keys)

        with torch.no_grad():
            output = embed_ngrams(input_ids, previous_ids, document_starts)
            unknown = embed_ngrams.bigram_proj(embed_ngrams.bigram_embeddings.weight[0])
            unknown = unknown + embed_ngrams.trigram_proj(embed_ngrams.trigram_embeddings.weight[0])
            bigram = embed_ngrams.bigram_proj(embed_ngrams.bigram_embeddings.weight[1:3])
            trigram = embed_ngrams.trigram_proj(embed_ngrams.trigram_embeddings.weight[:2])
        # Positions 0 and 4 have no previous token, position 3 is not a known n-gram
        for position in (0, 3, 4):
            torch.testing.assert_close(output[0, position], unknown)
        # (5, 6) is known at positions 1 and 5, both times without a known trigram
        torch.testing.assert_close(output[0, 1], bigram[0] + trigram[0])
        torch.testing.assert_close(output[0, 5], output[0, 1])
        # (5, 6, 7) is known at positions 2 and 6, where (6, 7) is not a known bigram
        torch.testing.assert_close(output[0, 2], unknown - trigram[0] + trigram[1])
        torch.testing.assert_close(output[0, 6], output[0, 2])
        # (7, 8) is known at position 7, where (6, 7, 8) is not a known trigram
        torch.testing.assert_close(output[0, 7], bigram[1] + trigram[0])


@require_torch
class KlmIntegrationTest(unittest.TestCase):
    # The original checkpoint, converted with `convert_klm_weights_to_hf.py`
    checkpoint = "MinimaLabs/KLM-1.6"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = KlmForCausalLM.from_pretrained(self.checkpoint).to(torch_device)
        # Two documents in a single sequence
        text = "The capital of France is Paris.<eos>Photosynthesis is the process by which plants"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(torch_device)
        with torch.no_grad():
            logits = model(input_ids).logits.float()

        EXPECTED_MEAN = torch.tensor(
            [[18.7292, -2.1244, 4.1339, -6.1907, 3.4924, 1.3876, 0.4001, -0.5738]], device=torch_device
        )
        torch.testing.assert_close(logits.mean(-1)[:, -8:], EXPECTED_MEAN, rtol=1e-3, atol=1e-3)
        EXPECTED_SLICE = torch.tensor(
            [-7.6502, -7.7844, 6.0391, 2.5945, 4.9552, -1.9173, -5.5932, -1.0267, 3.2742, 7.5289], device=torch_device
        )
        torch.testing.assert_close(logits[0, -1, :10], EXPECTED_SLICE, rtol=1e-3, atol=1e-3)

    @slow
    def test_model_greedy_generation(self):
        EXPECTED_TEXT_COMPLETIONS = [
            "Photosynthesis is the process by which plants turn light into chemical energy. The process is carried out by the chloroplasts of the plant cells. The chloropl",
            "Once upon a time there was a small fox who lived at the edge of a forest. The forest was full of tall trees and bushes, but the fox loved to hide in the",
        ]
        prompts = [
            "Photosynthesis is the process by which plants turn light into chemical energy.",
            "Once upon a time there was a small fox who lived at the edge of a forest.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, padding_side="left")
        model = KlmForCausalLM.from_pretrained(self.checkpoint).to(torch_device)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(torch_device)
        generated_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETIONS, text)

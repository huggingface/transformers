# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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


import unittest

from transformers import is_torch_available
from transformers.file_utils import cached_property

from .utils import require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import MarianForConditionalGeneration, MarianSentencePieceTokenizer
    from transformers.sinusoidal_positional_embeddings import SinusoidalPositionalEmbedding


@require_torch
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "opus/marian-en-de"
        cls.tokenizer = MarianSentencePieceTokenizer.from_pretrained(cls.model_name)
        cls.eos_token_id = cls.tokenizer.eos_token_id
        return cls

    @cached_property
    def model(self):
        return MarianForConditionalGeneration.from_pretrained(self.model_name).to(torch_device)

    @slow
    def test_forward(self):
        src, tgt = ["I am a small frog"], ["▁Ich ▁bin ▁ein ▁kleiner ▁Fro sch"]
        expected = [38, 121, 14, 697, 38848, 0]

        model_inputs: dict = self.tokenizer.prepare_translation_batch(src, tgt_texts=tgt)
        self.assertListEqual(expected, model_inputs["input_ids"][0].tolist())

        desired_keys = {
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        }
        self.assertSetEqual(desired_keys, set(model_inputs.keys()))
        with torch.no_grad():
            logits, *enc_features = self.model(**model_inputs)
        max_indices = logits.argmax(-1)
        predicted_words = self.tokenizer.decode_batch(max_indices)
        print(predicted_words)

    @slow
    def test_repl_generate(self):
        src = ["I am a small frog", "Hello"]
        model_inputs: dict = self.tokenizer.prepare_translation_batch(src).to(torch_device)
        generated_ids = self.model.generate(model_inputs["input_ids"], num_beams=2,)
        generated_words = self.tokenizer.decode_batch(generated_ids)[0]
        expected_words = "Ich bin ein kleiner Frosch"
        self.assertEqual(expected_words, generated_words)

    def test_marian_equivalence(self):
        input_ids = self.tokenizer.prepare_translation_batch(["I am a small frog"])["input_ids"][0].to(torch_device)
        expected = [38, 121, 14, 697, 38848, 0]
        self.assertListEqual(expected, input_ids.tolist())

    def test_pad_not_split(self):
        input_ids_w_pad = self.tokenizer.prepare_translation_batch(["I am a small frog <pad>"])["input_ids"][0]
        expected_w_pad = [38, 121, 14, 697, 38848, self.tokenizer.pad_token_id, 0]  # pad
        self.assertListEqual(expected_w_pad, input_ids_w_pad.tolist())


@require_torch
class TestSinusoidalPositionalEmbeddings(unittest.TestCase):
    def test_positional_emb_cache_logic(self):
        pad = 1
        input_ids = torch.tensor([[4, 10]], dtype=torch.long, device=torch_device)
        emb1 = SinusoidalPositionalEmbedding(init_size=32, embedding_dim=6, padding_idx=pad).to(torch_device)
        no_cache = emb1(input_ids, use_cache=False)
        yes_cache = emb1(input_ids, use_cache=True)
        self.assertEqual((1, 1, 6), yes_cache.shape)  # extra dim to allow broadcasting, feel free to delete!
        self.assertListEqual(no_cache[-1].tolist(), yes_cache[0][0].tolist())

    def test_odd_embed_dim(self):
        with self.assertRaises(NotImplementedError):
            SinusoidalPositionalEmbedding(init_size=4, embedding_dim=5, padding_idx=0).to(torch_device)

        # odd init_size is allowed
        SinusoidalPositionalEmbedding(init_size=5, embedding_dim=4, padding_idx=0).to(torch_device)

    def test_positional_emb_weights_against_marian(self):
        """SinusoidalPositionalEmbeddings."""
        pad = 1
        input_ids = torch.tensor([[4, 10] * 3], dtype=torch.long, device=torch_device)

        emb1 = SinusoidalPositionalEmbedding(init_size=512, embedding_dim=512, padding_idx=pad).to(torch_device)
        self.assertEqual(0.84147096, emb1.weight[1, 0])

        marian_results = [
            [0, 0, 0, 0, 0],
            [0.84147096, 0.82177866, 0.80180490, 0.78165019, 0.76140374],
            [0.90929741, 0.93651021, 0.95829457, 0.97505713, 0.98720258],
        ]
        weights = emb1.weight.data[:3, :5].tolist()
        for i, (expected, actual) in enumerate(zip(marian_results, weights)):
            for j in range(5):
                print(f"position {i}, {j}")
                self.assertAlmostEqual(expected[j], actual[j], places=3)

        # test that forward pass is just a lookup
        input_ids = torch.tensor([[4, 10, pad, pad, pad]], dtype=torch.long, device=torch_device)
        no_cache_pad_zero = emb1(input_ids)

        self.assertTrue(torch.allclose(torch.Tensor(marian_results), no_cache_pad_zero[:3, :5], atol=1e-3))

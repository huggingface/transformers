# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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
from pathlib import Path

from transformers import is_torch_available
from transformers.marian2hf import main

from .utils import require_torch, torch_device


if is_torch_available():
    import torch
    from transformers import MarianModel, MarianSPTokenizer, BartConfig
    from transformers.sinusoidal_positional_embeddings import SinusoidalPositionalEmbedding, assert_valid_pos_emb

LOCAL_PATH = "/Users/shleifer/transformers_fork/converted-en-de/"
LOCAL_MARIAN = "/Users/shleifer/transformers_fork/en-de/"



class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dest_dir = Path("converted-en-de")
        dest_dir.mkdir(exist_ok=True)
        #main(Path(LOCAL_MARIAN), dest_dir)
        cls.tokenizer = MarianSPTokenizer.from_pretrained(dest_dir.name)
        cls.model = MarianModel.from_pretrained(dest_dir.name)
        cls.config: BartConfig = cls.model.config
        cls.dest_dir = dest_dir
        cls.eos_token_id = cls.model.config.eos_token_id
        return cls

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls.dest_dir)

    def test_forward(self):
        #src, tgt = ["dinner", "life"], ["Abendessen", "Leben"]
        src, tgt = ["I am a small frog"], ['▁Ich ▁bin ▁ein ▁kleiner ▁Fro sch']
        expected = [38, 121, 14, 697, 38848, 0]

        model_inputs: dict = self.tokenizer.prepare_translation_batch(src, tgt_texts=tgt)
        self.assertListEqual(expected, model_inputs['input_ids'][0].tolist())
        shapes = {k: v.shape for k, v in model_inputs.items()}

        desired_keys = {
            "input_ids",
            "attention_mask",
            # "token_type_ids",
            "decoder_input_ids",
            "decoder_attention_mask",
            # "decoder_token_type_ids",
        }
        self.assertSetEqual(desired_keys, set(model_inputs.keys()))
        with torch.no_grad():
            logits, *enc_features = self.model(**model_inputs)
        max_indices = logits.argmax(-1)
        #print(max_indices)
    def test_repl_generate(self):
        #src, tgt = ["dinner", "life"], ["Abendessen", "Leben"]
        src, tgt = ["I am a small frog"], ['▁Ich ▁bin ▁ein ▁kleiner ▁Fro sch']
        expected = [38, 121, 14, 697, 38848, 0]

        model_inputs: dict = self.tokenizer.prepare_translation_batch(src)
        generated_ids = self.model.generate(
            model_inputs['input_ids'],
            num_beams=2,
            decoder_start_token_id=self.model.config.pad_token_id,
        )
        generated_words = self.tokenizer.decode_batch(generated_ids)
        print(generated_words)





    def test_tokenizer(self):
        input_ids = self.tokenizer.prepare_translation_batch(["I am a small frog"])['input_ids'][0]
        # expected = [444, 982, 111, 34045, 1, 0]   # marian produces this, see invocation issue.
        expected = [38, 121, 14, 697, 38848, 0]
        self.assertListEqual(expected, input_ids.tolist())
        input_ids_w_pad = self.tokenizer.prepare_translation_batch(["I am a small frog <pad>"])['input_ids'][0]
        expected_w_pad =  [38, 121, 14, 697, 38848, self.tokenizer.pad_token_id, 0]  # pad goes before EOS.
        self.assertListEqual(expected_w_pad, input_ids_w_pad.tolist())

    def test_generate(self):
        """Should produce a good translation."""
        src, tgt = ["What's for dinner?", "life"], ["Was gibt es zum Abendessen", "Leben"]
        model_inputs: dict = self.tokenizer.prepare_translation_batch(src)
        result_ids = self.model.generate(
            **model_inputs, num_beams=6, decoder_start_token_id=self.eos_token_id, no_repeat_ngram_size=3
        )
        print(result_ids)
        predicted_de_text = [self.tokenizer.decode(r) for r in result_ids]
        self.assertListEqual(predicted_de_text, tgt)


@require_torch
class FastTests(unittest.TestCase):
    def test_positional_embeddings(self):

        pad = 1
        input_ids = torch.tensor([[4, 10]], dtype=torch.long, device=torch_device)
        emb1 = SinusoidalPositionalEmbedding(10, pad, init_size=32).to(torch_device)
        no_cache = emb1(input_ids, use_cache=False)
        yes_cache = emb1(input_ids, use_cache=True)
        self.assertListEqual(no_cache[0,-1:].tolist(),  yes_cache[0].tolist())

    def test_pos_v2(self):
        """SinusoidalPositionalEmbeddings."""
        pad = 1
        input_ids = torch.tensor([[4, 10]* 3], dtype=torch.long, device=torch_device)
        emb1 = SinusoidalPositionalEmbedding(512, pad, init_size=512).to(torch_device)

        marian_results = [[0, 0, 0, 0, 0],
                          [0.84147096, 0.82177866, 0.80180490, 0.78165019, 0.76140374],
                          [0.90929741, 0.93651021, 0.95829457, 0.97505713, 0.98720258]
                          ]
        weights = emb1.weight.data[:3, :5].tolist()
        for i, (expected, actual) in enumerate(zip(marian_results, weights)):
            for j in range(5):
                print(f'position {i}, {j}')
                self.assertAlmostEqual(expected[j], actual[j], places=3)


        # test that forward pass is just a lookup
        input_ids = torch.tensor([[4, 10, pad, pad, pad]], dtype=torch.long, device=torch_device)
        no_cache_pad_zero = emb1(input_ids)

        self.assertTrue(torch.allclose(torch.Tensor(marian_results), no_cache_pad_zero[:3, :5], atol=1e-3))

        # emb0 = SinusoidalPositionalEmbedding(10, pad, init_size=32).to(torch_device)
        # assert (emb1.weights == emb0.weights).all()

        # position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=torch_device)
        # hard_coded = emb0.weights.index_select(0, position_ids).unsqueeze(0)

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

from transformers.file_utils import is_torch_available

from .utils import require_torch, torch_device


if is_torch_available():
    import torch
    from transformers.sinusoidal_positional_embeddings import SinusoidalPositionalEmbedding


@require_torch
class TestSinusoidalPositionalEmbeddings(unittest.TestCase):
    desired_weights = [
        [0, 0, 0, 0, 0],
        [0.84147096, 0.82177866, 0.80180490, 0.78165019, 0.76140374],
        [0.90929741, 0.93651021, 0.95829457, 0.97505713, 0.98720258],
    ]

    def test_positional_emb_cache_logic(self):
        pad = 1
        input_ids = torch.tensor([[4, 10]], dtype=torch.long, device=torch_device)
        emb1 = SinusoidalPositionalEmbedding(num_positions=32, embedding_dim=6, padding_idx=pad).to(torch_device)
        no_cache = emb1(input_ids, use_cache=False)
        yes_cache = emb1(input_ids, use_cache=True)
        self.assertEqual((1, 1, 6), yes_cache.shape)  # extra dim to allow broadcasting, feel free to delete!
        self.assertListEqual(no_cache[-1].tolist(), yes_cache[0][0].tolist())

    def test_odd_embed_dim(self):
        with self.assertRaises(NotImplementedError):
            SinusoidalPositionalEmbedding(num_positions=4, embedding_dim=5, padding_idx=0).to(torch_device)

        # odd init_size is allowed
        SinusoidalPositionalEmbedding(num_positions=5, embedding_dim=4, padding_idx=0).to(torch_device)

    def test_positional_emb_weights_against_marian(self):
        pad = 1
        emb1 = SinusoidalPositionalEmbedding(num_positions=512, embedding_dim=512, padding_idx=pad).to(torch_device)
        weights = emb1.weight.data[:3, :5].tolist()
        for i, (expected_weight, actual_weight) in enumerate(zip(self.desired_weights, weights)):
            for j in range(5):
                self.assertAlmostEqual(expected_weight[j], actual_weight[j], places=3)

        # test that forward pass is just a lookup, there is no ignore padding logic
        input_ids = torch.tensor([[4, 10, pad, pad, pad]], dtype=torch.long, device=torch_device)
        no_cache_pad_zero = emb1(input_ids)
        self.assertTrue(torch.allclose(torch.Tensor(self.desired_weights), no_cache_pad_zero[:3, :5], atol=1e-3))

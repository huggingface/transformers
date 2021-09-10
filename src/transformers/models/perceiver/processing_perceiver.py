# coding=utf-8
# Copyright 2021 Deepmind and The HuggingFace Inc. team.
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
"""
IO pre- and post-processor classes for Perceiver.
"""
import torch.nn as nn
import torch

class PerceiverTextPreprocessor(nn.Module):
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.d_model)
        self.position_embeddings = nn.Embedding(seq_len, config.d_model)
        self.seq_len = seq_len

    def __call__(self, inputs):

        embeddings = self.embeddings(inputs)
        position_ids = torch.arange(0, self.seq_len)
        embeddings = embeddings + self.position_embeddings(position_ids)

        return embeddings

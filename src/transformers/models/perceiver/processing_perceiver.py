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
import torch
import torch.nn as nn


class PerceiverTextPreprocessor(nn.Module):
    """Text preprocessing for Perceiver Encoder."""

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


class PerceiverTextPostprocessor(nn.Module):
    """Module to decode embeddings."""

    def __init__(self, config, embedding_matrix):
        """Constructs the module."""
        super().__init__()
        self.classifier = embedding_matrix
        self.vocab_size, self.d_model = embedding_matrix.weight.shape
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))

    def __call__(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        output = torch.matmul(
            hidden_states.reshape([-1, self.d_model]), torch.transpose(self.classifier)  # Flatten batch dim
        )
        output = output + bias

        return output.reshape([batch_size, seq_len, self.vocab_size])

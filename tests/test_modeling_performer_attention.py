import math
import torch
import unittest
from transformers import PerformerAttention, PerformerAttentionConfig

class PerformerAttentionTest(unittest.TestCase):
    def test_relu_noncausal_attention_block_output(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu'
        ))

        query = torch.ones(batch_size, length, dim)
        key = torch.ones(batch_size, length, dim)
        value = torch.ones(batch_size, length, dim)
        attention_block_output = attention(query, key, value)[0]

        self.assertListEqual(list(attention_block_output.shape), [batch_size, length, dim])

    def test_relu_causal_attention_block_output_shape(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu',
            causal=True
        ))

        query = torch.ones(batch_size, length, dim)
        key = torch.ones(batch_size, length, dim)
        value = torch.ones(batch_size, length, dim)
        attention_block_output = attention(query, key, value)[0]

        self.assertListEqual(list(attention_block_output.shape), [batch_size, length, dim])

    def test_softmax_noncausal_attention_block_output_shape(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4
        num_random_features = 350

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='exp',
            num_random_features=num_random_features
        ))

        query = torch.ones(batch_size, length, dim)
        key = torch.ones(batch_size, length, dim)
        value = torch.ones(batch_size, length, dim)

        attention_block_output = attention(query, key, value)[0]
        self.assertListEqual(list(attention_block_output.shape), [batch_size, length, dim])

    def test_softmax_noncausal_attention_block_output(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 10

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='exp',
            num_random_features=30000,
            use_linear_layers=False
        ))

        query = torch.randn(batch_size, length, dim)
        key = torch.randn(batch_size, length, dim)
        value = torch.randn(batch_size, length, dim)

        performer_attention_output = attention(query, key, value)[0]

        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(float(dim))
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=1)
        softmax_output = attention_scores @ value

        errors = softmax_output - performer_attention_output
        mse = torch.mean(errors ** 2)
        bias = torch.mean(errors)

        self.assertLess(mse, 0.1)
        self.assertLess(torch.abs(bias), 0.025)

    def test_fast_attention(self):
        hidden_size = 64
        num_heads = 4
        dropout = 0.5

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=hidden_size,
            num_heads=num_heads,
            attention_dropout=dropout
        ))

        length = 2
        x = torch.ones([1, length, hidden_size])
        y = attention(x, x, x)[0]
        self.assertEqual(y.shape, (1, length, hidden_size))

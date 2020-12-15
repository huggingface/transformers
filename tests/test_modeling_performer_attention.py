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

        query = torch.ones(batch_size, length, num_heads, dim)
        key = torch.ones(batch_size, length, num_heads, dim)
        value = torch.ones(batch_size, length, num_heads, dim)
        attention_block_output = attention(query, key, value)

        self.assertListEqual(attention_block_output.get_shape().as_list(),
                             [batch_size, length, num_heads, dim])

    def test_relu_causal_attention_block_output_shape(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu'
        ))

        query = torch.ones(batch_size, length, num_heads, dim)
        key = torch.ones(batch_size, length, num_heads, dim)
        value = torch.ones(batch_size, length, num_heads, dim)
        attention_block_output = attention(query, key, value)

        self.assertListEqual(attention_block_output.get_shape().as_list(),
                             [batch_size, length, num_heads, dim])

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

        query = torch.ones(batch_size, length, num_heads, dim)
        key = torch.ones(batch_size, length, num_heads, dim)
        value = torch.ones(batch_size, length, num_heads, dim)

        attention_block_output = attention(query, key, value)
        self.assertListEqual(attention_block_output.get_shape().as_list(),
                             [batch_size, length, num_heads, dim])

    def test_softmax_noncausal_attention_block_output(self):
        batch_size = 1
        length = 2
        num_heads = 1
        dim = 8
        num_random_features = 30000

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu',
            num_random_features=num_random_features
        ))

        query = torch.randn(batch_size, length, num_heads, dim)
        key = torch.randn(batch_size, length, num_heads, dim)
        value = torch.randn(batch_size, length, num_heads, dim)

        attention_block_output = attention(query, key, value)

        query = query * 1.0 / math.sqrt(float(dim))
        attention_scores = torch.einsum("BXHD,BYHD->BXYH", query, key)
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=2)
        exact_attention_block_output = torch.einsum("BXYH,BYHD->BXHD", attention_scores, value)
        max_error = 2.0
        error = torch.abs(
            (exact_attention_block_output - attention_block_output) /
            exact_attention_block_output)
        self.assertLess(torch.max(torch.abs(error)), max_error)

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
        y = attention(x, training=True)
        self.assertEqual(y.shape, (
            1,
            length,
            64,
        ))

    def test_custom_causal_gradients(self):
        L = 64
        B = 128
        H = 4
        D = 64
        M = 128
        qs = torch.randn(L, B, H, M, requires_grad=True)
        ks = torch.randn(L, B, H, M, requires_grad=True)
        vs = torch.randn(L, B, H, D, requires_grad=True)
        coefs = torch.randn(vs.shape)

        attention = PerformerAttention(PerformerAttentionConfig(
            d_model=H * D,
            num_heads=H,
            num_random_features=M,
            causal=True
        ))
        output = attention.compute_attention_with_projected_queries_and_keys(qs, ks, vs)

        loss = torch.sum(output * coefs)
        loss.backward()

        self.assertListEqual([L, B, H, M], qs.grad.get_shape().as_list())
        self.assertListEqual([L, B, H, M], ks.grad.get_shape().as_list())
        self.assertListEqual([L, B, H, D], vs.grad.get_shape().as_list())

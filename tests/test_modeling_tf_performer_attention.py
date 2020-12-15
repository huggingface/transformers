import math
import tensorflow as tf
import unittest
from transformers import PerformerAttentionConfig
from ..src.transformers.modeling_tf_performer_attention import TFPerformerAttention

class PerformerAttentionTest(unittest.TestCase):
    def test_relu_noncausal_attention_block_output(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu'
        ))

        query = tf.ones(batch_size, length, num_heads, dim)
        key = tf.ones(batch_size, length, num_heads, dim)
        value = tf.ones(batch_size, length, num_heads, dim)
        attention_block_output = attention(query, key, value)

        self.assertListEqual(attention_block_output.get_shape().as_list(),
                             [batch_size, length, num_heads, dim])

    def test_relu_causal_attention_block_output_shape(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu'
        ))

        query = tf.ones(batch_size, length, num_heads, dim)
        key = tf.ones(batch_size, length, num_heads, dim)
        value = tf.ones(batch_size, length, num_heads, dim)
        attention_block_output = attention(query, key, value)

        self.assertListEqual(attention_block_output.get_shape().as_list(),
                             [batch_size, length, num_heads, dim])

    def test_softmax_noncausal_attention_block_output_shape(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4
        num_random_features = 350

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='exp',
            num_random_features=num_random_features
        ))

        query = tf.ones(batch_size, length, num_heads, dim)
        key = tf.ones(batch_size, length, num_heads, dim)
        value = tf.ones(batch_size, length, num_heads, dim)

        attention_block_output = attention(query, key, value)
        self.assertListEqual(attention_block_output.get_shape().as_list(),
                             [batch_size, length, num_heads, dim])

    def test_softmax_noncausal_attention_block_output(self):
        batch_size = 1
        length = 2
        num_heads = 1
        dim = 8
        num_random_features = 30000

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu',
            num_random_features=num_random_features
        ))

        query = tf.random.normal(batch_size, length, num_heads, dim)
        key = tf.random.normal(batch_size, length, num_heads, dim)
        value = tf.random.normal(batch_size, length, num_heads, dim)

        attention_block_output = attention(query, key, value)

        query = query * 1.0 / math.sqrt(float(dim))
        attention_scores = tf.einsum("BXHD,BYHD->BXYH", query, key)
        attention_scores = tf.nn.softmax(attention_scores, dim=2)
        exact_attention_block_output = tf.einsum("BXYH,BYHD->BXHD", attention_scores, value)
        max_error = 2.0
        error = tf.abs(
            (exact_attention_block_output - attention_block_output) /
            exact_attention_block_output)
        self.assertLess(tf.reduce_max(tf.abs(error)), max_error)

    def test_fast_attention(self):
        hidden_size = 64
        num_heads = 4
        dropout = 0.5

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=hidden_size,
            num_heads=num_heads,
            attention_dropout=dropout
        ))

        length = 2
        x = tf.ones([1, length, hidden_size])
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
        qs = tf.random.normal([L, B, H, M])
        ks = tf.random.normal([L, B, H, M])
        vs = tf.random.normal([L, B, H, D])
        coefs = tf.random.normal(vs.shape)

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=H * D,
            num_heads=H,
            num_random_features=M,
            causal=True
        ))

        with tf.GradientTape() as tape:
            tape.watch([qs, ks, vs])
            output = attention.compute_attention_with_projected_queries_and_keys(qs, ks, vs)

            loss = tf.reduce_sum(output * coefs)

        grads1 = tape.gradient(loss, [qs, ks, vs])
        self.assertListEqual([L, B, H, M], grads1[0].get_shape().as_list())
        self.assertListEqual([L, B, H, M], grads1[1].get_shape().as_list())
        self.assertListEqual([L, B, H, D], grads1[2].get_shape().as_list())

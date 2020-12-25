import math
import tensorflow as tf
import unittest
from transformers import PerformerAttentionConfig, TFPerformerAttention

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

        query = tf.ones((batch_size, length, dim))
        key = tf.ones((batch_size, length, dim))
        value = tf.ones((batch_size, length, dim))
        attention_block_output = attention(query, key, value)[0]

        self.assertListEqual(list(attention_block_output.shape),
                             [batch_size, length, dim])

    def test_relu_causal_attention_block_output_shape(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 4

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='relu',
            causal=True
        ))

        query = tf.ones((batch_size, length, dim))
        key = tf.ones((batch_size, length, dim))
        value = tf.ones((batch_size, length, dim))
        attention_block_output = attention(query, key, value)[0]

        self.assertListEqual(list(attention_block_output.shape),
                             [batch_size, length, dim])

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

        query = tf.ones((batch_size, length, dim))
        key = tf.ones((batch_size, length, dim))
        value = tf.ones((batch_size, length, dim))

        attention_block_output = attention(query, key, value)[0]
        self.assertListEqual(list(attention_block_output.shape),
                             [batch_size, length, dim])

    def test_softmax_noncausal_attention_block_output(self):
        batch_size = 1
        length = 2
        num_heads = 1
        dim = 8
        num_random_features = 30000

        attention = TFPerformerAttention(PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='exp',
            num_random_features=num_random_features,
            use_linear_layers=False
        ))

        query = tf.random.normal((batch_size, length, dim))
        key = tf.random.normal((batch_size, length, dim))
        value = tf.random.normal((batch_size, length, dim))

        attention_block_output = attention(query, key, value)[0]

        query /= math.sqrt(float(dim))
        attention_scores = tf.einsum("BXD,BYD->BXY", query, key)
        attention_scores = tf.nn.softmax(attention_scores, axis=2)
        exact_attention_block_output = tf.einsum("BXY,BYD->BXD", attention_scores, value)
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
        y = attention(x, x, x, training=True)[0]
        self.assertEqual(y.shape, (1, length, 64))

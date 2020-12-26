import math
import random
import tensorflow as tf
import unittest
from dataclasses import fields
from itertools import product
from transformers import PerformerAttentionConfig, TFPerformerAttention, PerformerKernel, OrthogonalFeatureAlgorithm


class PerformerAttentionTest(unittest.TestCase):
    def test_output_shape(self):
        param_names = ['kernel_type', 'orthogonal_feature_algorithm']
        legal_values = [PerformerKernel, OrthogonalFeatureAlgorithm]  # Enum classes are iterable

        # Get all boolean config options
        for x in fields(PerformerAttentionConfig):
            if x.type == bool:
                legal_values.append((False, True))
                param_names.append(x.name)

        # Exhaustive grid search of possible config options
        for values in product(*legal_values):
            kwargs = dict(zip(param_names, values))

            d_model = random.randint(2, 10)
            batch_size = random.randint(1, 4)
            length = 1 if kwargs.get('use_recurrent_decoding') else random.randint(1, 10)

            try:
                attention = TFPerformerAttention(PerformerAttentionConfig(d_model=d_model, num_heads=1, **kwargs))
            except AssertionError:
                # Skip illegal kwargs combinations
                continue

            with self.subTest(msg=repr(kwargs)):
                q, k, v = [tf.random.normal((batch_size, length, d_model)) for _ in range(3)]
                output = attention(q, k, v)[0]

                self.assertListEqual(list(output.shape), [batch_size, length, d_model])

    def test_softmax_noncausal_attention_block_output(self):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 10
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

        performer_attention_output = attention(query, key, value)[0]

        attention_scores = query @ tf.linalg.matrix_transpose(key) / math.sqrt(float(dim))
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        softmax_output = attention_scores @ value

        errors = softmax_output - performer_attention_output
        mse = tf.math.reduce_mean(errors ** 2)
        bias = tf.math.reduce_mean(errors)

        self.assertLess(mse, 0.1)
        self.assertLess(tf.abs(bias), 0.025)

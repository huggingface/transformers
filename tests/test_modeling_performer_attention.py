import math
import numpy as np
import random
import unittest
from dataclasses import fields
from itertools import product
from transformers import (
    PerformerAttentionConfig, PerformerKernel, OrthogonalFeatureAlgorithm,
    is_torch_available, is_tf_available
)

if is_torch_available():
    import torch
    from transformers import PerformerAttention
if is_tf_available():
    import tensorflow as tf
    from transformers import TFPerformerAttention


class PerformerAttentionTest(unittest.TestCase):
    @unittest.skipIf(not is_torch_available(), reason="PyTorch not available")
    def test_output_shape_pytorch(self):
        self._test_output_shape_for_library('pt')

    @unittest.skipIf(not is_tf_available(), reason="TensorFlow not available")
    def test_output_shape_tensorflow(self):
        self._test_output_shape_for_library('tf')

    @unittest.skipIf(not is_torch_available(), reason="PyTorch not available")
    def test_softmax_noncausal_attention_output_pytorch(self):
        self._test_softmax_noncausal_attention_output_for_library('pt')

    @unittest.skipIf(not is_tf_available(), reason="TensorFlow not available")
    def test_softmax_noncausal_attention_output_tensorflow(self):
        self._test_softmax_noncausal_attention_output_for_library('tf')

    def _test_output_shape_for_library(self, library: str = 'pt'):
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
            num_heads = random.choice([i for i in range(1, d_model) if not d_model % i])    # Factors of d_model
            length = 1 if kwargs.get('use_recurrent_decoding') else random.randint(1, 10)
            config = PerformerAttentionConfig(d_model=d_model, num_heads=num_heads, **kwargs)

            # PyTorch specific stuff
            if library == 'pt':
                attn_class = PerformerAttention
                rand_tensor_func = lambda: torch.randn(batch_size, length, d_model)

            # TensorFlow specific stuff
            else:
                attn_class = TFPerformerAttention
                rand_tensor_func = lambda: tf.random.normal((batch_size, length, d_model))

            try:
                attention = attn_class(config)
            except AssertionError:
                # Skip illegal kwargs combinations
                pass
            else:
                with self.subTest(**kwargs):
                    q, k, v = [rand_tensor_func() for _ in range(3)]
                    output = attention(q, k, v)[0]

                    self.assertListEqual(list(output.shape), [batch_size, length, d_model])

    def _test_softmax_noncausal_attention_output_for_library(self, library: str = 'pt'):
        batch_size = 1
        length = 10
        num_heads = 1
        dim = 10

        config = PerformerAttentionConfig(
            d_model=dim,
            num_heads=num_heads,
            kernel_type='exp',
            num_random_features=30000,
            use_linear_layers=False
        )
        # PyTorch-specific stuff
        if library == 'pt':
            pt_attention = PerformerAttention(config)

            q, k, v = [torch.randn(batch_size, length, dim) for _ in range(3)]
            performer_attention_output = pt_attention(q, k, v)[0]

            attention_scores = q @ k.transpose(-2, -1) / math.sqrt(float(dim))
            attention_scores = torch.nn.functional.softmax(attention_scores, dim=1)

        # TensorFlow-specific stuff
        else:
            tf_attention = TFPerformerAttention(config)

            q, k, v = [tf.random.normal((batch_size, length, dim)) for _ in range(3)]
            performer_attention_output = tf_attention(q, k, v)[0]

            attention_scores = q @ tf.linalg.matrix_transpose(k) / math.sqrt(float(dim))
            attention_scores = tf.nn.softmax(attention_scores, axis=1)

        softmax_output = (attention_scores @ v).numpy()

        errors = softmax_output - performer_attention_output
        mse = np.mean(errors ** 2)
        bias = np.mean(errors)

        self.assertLess(mse, 0.1)
        self.assertLess(np.abs(bias), 0.025)

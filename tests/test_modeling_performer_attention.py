import math
import numpy as np
import random
import unittest
from dataclasses import fields
from itertools import product
from transformers.testing_utils import parse_flag_from_env
from transformers import (
    PerformerAttentionConfig, PerformerKernel, OrthogonalFeatureAlgorithm, performer_supporting_models_and_configs,
    is_torch_available, is_tf_available
)
from typing import Iterator, Tuple

if is_torch_available():
    import torch
    from transformers import PerformerAttention
if is_tf_available():
    import tensorflow as tf
    from transformers import TFPerformerAttention


_run_nondeterministic_tests = parse_flag_from_env("RUN_NONDETERMINISTIC_TESTS", default=False)


class PerformerAttentionTest(unittest.TestCase):
    # Check that setting attention_type='performer' actually makes the model use (TF)PerformerAttention
    def test_performer_models(self):
        def _model_is_tf(model_cls):
            return is_tf_available() and issubclass(model_cls, tf.keras.layers.Layer)

        for model_class, model_config in performer_supporting_models_and_configs():
            try:
                model = model_class(model_config(attention_type='performer'))

            # The TapasModel requires the torch-scatter library, and we shouldn't fail this test just because
            # the user doesn't have that library installed
            except ImportError:
                pass
            else:
                with self.subTest(model=model_class):
                    self.assertIsNotNone(model)

                    # It turns out that it's very non-trivial to do this type of recursive iteration of sublayers in
                    # TensorFlow, so we just don't bother to do the check for those models
                    if not _model_is_tf(model_class):
                        self.assertTrue(any((isinstance(module, PerformerAttention) for module in model.modules())))

    @unittest.skipIf(not is_torch_available(), "PyTorch not available")
    def test_output_shape_pytorch(self):
        self._test_output_shape_for_library('pt')

    @unittest.skipIf(not is_tf_available(), "TensorFlow not available")
    def test_output_shape_tensorflow(self):
        self._test_output_shape_for_library('tf')

    @unittest.skipUnless(_run_nondeterministic_tests, "This can fail randomly if we draw an 'unlucky' set of features.")
    def test_softmax_noncausal_attention_output_pytorch(self):
        self._test_softmax_noncausal_attention_output_for_library('pt')

    @unittest.skipUnless(_run_nondeterministic_tests, "This can fail randomly if we draw an 'unlucky' set of features.")
    def test_softmax_noncausal_attention_output_tensorflow(self):
        self._test_softmax_noncausal_attention_output_for_library('tf')

    @unittest.skipUnless(is_torch_available() and is_tf_available(), "Both PyTorch and TensorFlow must be available")
    @torch.no_grad()
    def test_pytorch_tensorflow_parity(self):
        for config, batch, seq_len in self._iterate_config_options():
            # This option leads to random test failures due to the TFPerformerAttention object randomly redrawing
            # features right after we set its features to be equal to those of the PyTorch object, so we just skip it
            if config.redraw_stochastically:
                continue

            try:
                pt_attention = PerformerAttention(config)
            except AssertionError:
                continue

            try:
                tf_attention = TFPerformerAttention(config)
            except AssertionError:
                continue

            # Copy the weights from the PyTorch object to the TensorFlow one
            for name, param in pt_attention.named_parameters():
                pt_value = param.data.numpy()

                # Get the corresponding param (tf.Variable) in the TensorFlow object
                obj = tf_attention
                for key in name.split('.'):
                    if key.isnumeric():
                        obj = obj[int(key)]
                    elif key == "weight":
                        # Note that we have to transpose the weights when converting to TF to get the same output
                        obj.kernel_initializer = tf.constant_initializer(pt_value.T)
                    elif key == "bias":
                        obj.bias_initializer = tf.constant_initializer(pt_value)
                    else:
                        obj = getattr(obj, key)

            # Test that the two modules produce the same output, within numerical error
            with self.subTest(**config.to_dict()):
                q, k, v = (torch.randn(batch, seq_len, config.d_model) for _ in range(3))
                tf_q, tf_k, tf_v = (tf.constant(x.numpy()) for x in (q, k, v))
                pt_output = pt_attention(q, k, v)[0]

                tf_attention.random_features = tf.constant(pt_attention.random_features.numpy())
                tf_output = tf_attention(tf_q, tf_k, tf_v)[0]

                self.assertTrue(np.allclose(pt_output.numpy(), tf_output.numpy(), atol=2e-4))
                self.assertListEqual(list(pt_output.shape), [batch, seq_len, config.d_model])

    # Exhaustive grid search of possible config options (and a random search of batch sizes and seq lengths)
    @staticmethod
    def _iterate_config_options() -> Iterator[Tuple[PerformerAttentionConfig, int, int]]:
        param_names = ['kernel_type', 'orthogonal_feature_algorithm']
        legal_values = [PerformerKernel, OrthogonalFeatureAlgorithm]  # Enum classes are iterable

        # Get all boolean config options
        for x in fields(PerformerAttentionConfig):
            if x.type == bool:
                legal_values.append((False, True))
                param_names.append(x.name)

        for values in product(*legal_values):
            kwargs = dict(zip(param_names, values))

            d_model = random.randint(2, 10)
            batch_size = random.randint(1, 4)
            num_heads = random.choice([i for i in range(1, d_model) if not d_model % i])  # Factors of d_model
            length = 1 if kwargs.get('use_recurrent_decoding') else random.randint(1, 10)
            yield PerformerAttentionConfig(d_model=d_model, num_heads=num_heads, **kwargs), batch_size, length

    @torch.no_grad()
    def _test_output_shape_for_library(self, library: str = 'pt'):
        for config, batch_size, length in self._iterate_config_options():
            d_model = config.d_model

            # PyTorch specific stuff
            if library == 'pt':
                attn_class = PerformerAttention

                def rand_tensor_func():
                    return torch.randn(batch_size, length, d_model)

            # TensorFlow specific stuff
            else:
                attn_class = TFPerformerAttention

                def rand_tensor_func():
                    return tf.random.normal((batch_size, length, d_model))

            try:
                attention = attn_class(config)
            except AssertionError:
                # Skip illegal kwargs combinations
                pass
            else:
                with self.subTest(**config.to_dict()):
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

        errors = softmax_output - performer_attention_output.numpy()
        mse = np.mean(errors ** 2)
        bias = np.mean(errors)

        self.assertLess(mse, 0.1)
        self.assertLess(np.abs(bias), 0.025)

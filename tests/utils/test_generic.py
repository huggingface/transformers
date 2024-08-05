# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
import warnings

import numpy as np

from transformers.testing_utils import require_flax, require_tf, require_torch
from transformers.utils import (
    expand_dims,
    filter_out_non_signature_kwargs,
    flatten_dict,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    reshape,
    squeeze,
    transpose,
)


if is_flax_available():
    import jax.numpy as jnp

if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch


class GenericTester(unittest.TestCase):
    def test_flatten_dict(self):
        input_dict = {
            "task_specific_params": {
                "summarization": {"length_penalty": 1.0, "max_length": 128, "min_length": 12, "num_beams": 4},
                "summarization_cnn": {"length_penalty": 2.0, "max_length": 142, "min_length": 56, "num_beams": 4},
                "summarization_xsum": {"length_penalty": 1.0, "max_length": 62, "min_length": 11, "num_beams": 6},
            }
        }
        expected_dict = {
            "task_specific_params.summarization.length_penalty": 1.0,
            "task_specific_params.summarization.max_length": 128,
            "task_specific_params.summarization.min_length": 12,
            "task_specific_params.summarization.num_beams": 4,
            "task_specific_params.summarization_cnn.length_penalty": 2.0,
            "task_specific_params.summarization_cnn.max_length": 142,
            "task_specific_params.summarization_cnn.min_length": 56,
            "task_specific_params.summarization_cnn.num_beams": 4,
            "task_specific_params.summarization_xsum.length_penalty": 1.0,
            "task_specific_params.summarization_xsum.max_length": 62,
            "task_specific_params.summarization_xsum.min_length": 11,
            "task_specific_params.summarization_xsum.num_beams": 6,
        }

        self.assertEqual(flatten_dict(input_dict), expected_dict)

    def test_transpose_numpy(self):
        x = np.random.randn(3, 4)
        self.assertTrue(np.allclose(transpose(x), x.transpose()))

        x = np.random.randn(3, 4, 5)
        self.assertTrue(np.allclose(transpose(x, axes=(1, 2, 0)), x.transpose((1, 2, 0))))

    @require_torch
    def test_transpose_torch(self):
        x = np.random.randn(3, 4)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(transpose(x), transpose(t).numpy()))

        x = np.random.randn(3, 4, 5)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(transpose(x, axes=(1, 2, 0)), transpose(t, axes=(1, 2, 0)).numpy()))

    @require_tf
    def test_transpose_tf(self):
        x = np.random.randn(3, 4)
        t = tf.constant(x)
        self.assertTrue(np.allclose(transpose(x), transpose(t).numpy()))

        x = np.random.randn(3, 4, 5)
        t = tf.constant(x)
        self.assertTrue(np.allclose(transpose(x, axes=(1, 2, 0)), transpose(t, axes=(1, 2, 0)).numpy()))

    @require_flax
    def test_transpose_flax(self):
        x = np.random.randn(3, 4)
        t = jnp.array(x)
        self.assertTrue(np.allclose(transpose(x), np.asarray(transpose(t))))

        x = np.random.randn(3, 4, 5)
        t = jnp.array(x)
        self.assertTrue(np.allclose(transpose(x, axes=(1, 2, 0)), np.asarray(transpose(t, axes=(1, 2, 0)))))

    def test_reshape_numpy(self):
        x = np.random.randn(3, 4)
        self.assertTrue(np.allclose(reshape(x, (4, 3)), np.reshape(x, (4, 3))))

        x = np.random.randn(3, 4, 5)
        self.assertTrue(np.allclose(reshape(x, (12, 5)), np.reshape(x, (12, 5))))

    @require_torch
    def test_reshape_torch(self):
        x = np.random.randn(3, 4)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(reshape(x, (4, 3)), reshape(t, (4, 3)).numpy()))

        x = np.random.randn(3, 4, 5)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(reshape(x, (12, 5)), reshape(t, (12, 5)).numpy()))

    @require_tf
    def test_reshape_tf(self):
        x = np.random.randn(3, 4)
        t = tf.constant(x)
        self.assertTrue(np.allclose(reshape(x, (4, 3)), reshape(t, (4, 3)).numpy()))

        x = np.random.randn(3, 4, 5)
        t = tf.constant(x)
        self.assertTrue(np.allclose(reshape(x, (12, 5)), reshape(t, (12, 5)).numpy()))

    @require_flax
    def test_reshape_flax(self):
        x = np.random.randn(3, 4)
        t = jnp.array(x)
        self.assertTrue(np.allclose(reshape(x, (4, 3)), np.asarray(reshape(t, (4, 3)))))

        x = np.random.randn(3, 4, 5)
        t = jnp.array(x)
        self.assertTrue(np.allclose(reshape(x, (12, 5)), np.asarray(reshape(t, (12, 5)))))

    def test_squeeze_numpy(self):
        x = np.random.randn(1, 3, 4)
        self.assertTrue(np.allclose(squeeze(x), np.squeeze(x)))

        x = np.random.randn(1, 4, 1, 5)
        self.assertTrue(np.allclose(squeeze(x, axis=2), np.squeeze(x, axis=2)))

    @require_torch
    def test_squeeze_torch(self):
        x = np.random.randn(1, 3, 4)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(squeeze(x), squeeze(t).numpy()))

        x = np.random.randn(1, 4, 1, 5)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(squeeze(x, axis=2), squeeze(t, axis=2).numpy()))

    @require_tf
    def test_squeeze_tf(self):
        x = np.random.randn(1, 3, 4)
        t = tf.constant(x)
        self.assertTrue(np.allclose(squeeze(x), squeeze(t).numpy()))

        x = np.random.randn(1, 4, 1, 5)
        t = tf.constant(x)
        self.assertTrue(np.allclose(squeeze(x, axis=2), squeeze(t, axis=2).numpy()))

    @require_flax
    def test_squeeze_flax(self):
        x = np.random.randn(1, 3, 4)
        t = jnp.array(x)
        self.assertTrue(np.allclose(squeeze(x), np.asarray(squeeze(t))))

        x = np.random.randn(1, 4, 1, 5)
        t = jnp.array(x)
        self.assertTrue(np.allclose(squeeze(x, axis=2), np.asarray(squeeze(t, axis=2))))

    def test_expand_dims_numpy(self):
        x = np.random.randn(3, 4)
        self.assertTrue(np.allclose(expand_dims(x, axis=1), np.expand_dims(x, axis=1)))

    @require_torch
    def test_expand_dims_torch(self):
        x = np.random.randn(3, 4)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(expand_dims(x, axis=1), expand_dims(t, axis=1).numpy()))

    @require_tf
    def test_expand_dims_tf(self):
        x = np.random.randn(3, 4)
        t = tf.constant(x)
        self.assertTrue(np.allclose(expand_dims(x, axis=1), expand_dims(t, axis=1).numpy()))

    @require_flax
    def test_expand_dims_flax(self):
        x = np.random.randn(3, 4)
        t = jnp.array(x)
        self.assertTrue(np.allclose(expand_dims(x, axis=1), np.asarray(expand_dims(t, axis=1))))


class ValidationDecoratorTester(unittest.TestCase):
    def test_cases_no_warning(self):
        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter("always")

            # basic test
            @filter_out_non_signature_kwargs()
            def func1(a):
                return a

            result = func1(1)
            self.assertEqual(result, 1)

            # include extra kwarg
            @filter_out_non_signature_kwargs(extra=["extra_arg"])
            def func2(a, **kwargs):
                return a, kwargs

            a, kwargs = func2(1)
            self.assertEqual(a, 1)
            self.assertEqual(kwargs, {})

            a, kwargs = func2(1, extra_arg=2)
            self.assertEqual(a, 1)
            self.assertEqual(kwargs, {"extra_arg": 2})

            # multiple extra kwargs
            @filter_out_non_signature_kwargs(extra=["extra_arg", "extra_arg2"])
            def func3(a, **kwargs):
                return a, kwargs

            a, kwargs = func3(2)
            self.assertEqual(a, 2)
            self.assertEqual(kwargs, {})

            a, kwargs = func3(3, extra_arg2=3)
            self.assertEqual(a, 3)
            self.assertEqual(kwargs, {"extra_arg2": 3})

            a, kwargs = func3(1, extra_arg=2, extra_arg2=3)
            self.assertEqual(a, 1)
            self.assertEqual(kwargs, {"extra_arg": 2, "extra_arg2": 3})

            # Check that no warnings were raised
            self.assertEqual(len(raised_warnings), 0, f"Warning raised: {[w.message for w in raised_warnings]}")

    def test_cases_with_warnings(self):
        @filter_out_non_signature_kwargs()
        def func1(a):
            return a

        with self.assertWarns(UserWarning):
            func1(1, extra_arg=2)

        @filter_out_non_signature_kwargs(extra=["extra_arg"])
        def func2(a, **kwargs):
            return kwargs

        with self.assertWarns(UserWarning):
            kwargs = func2(1, extra_arg=2, extra_arg2=3)
        self.assertEqual(kwargs, {"extra_arg": 2})

        @filter_out_non_signature_kwargs(extra=["extra_arg", "extra_arg2"])
        def func3(a, **kwargs):
            return kwargs

        with self.assertWarns(UserWarning):
            kwargs = func3(1, extra_arg=2, extra_arg2=3, extra_arg3=4)
        self.assertEqual(kwargs, {"extra_arg": 2, "extra_arg2": 3})

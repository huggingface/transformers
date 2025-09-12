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
import pytest

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.testing_utils import require_torch
from transformers.utils import (
    can_return_tuple,
    expand_dims,
    filter_out_non_signature_kwargs,
    flatten_dict,
    is_torch_available,
    reshape,
    squeeze,
    to_py_obj,
    transpose,
)


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

    @require_torch
    def test_reshape_torch(self):
        x = np.random.randn(3, 4)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(reshape(x, (4, 3)), reshape(t, (4, 3)).numpy()))

        x = np.random.randn(3, 4, 5)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(reshape(x, (12, 5)), reshape(t, (12, 5)).numpy()))

    @require_torch
    def test_squeeze_torch(self):
        x = np.random.randn(1, 3, 4)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(squeeze(x), squeeze(t).numpy()))

        x = np.random.randn(1, 4, 1, 5)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(squeeze(x, axis=2), squeeze(t, axis=2).numpy()))

    def test_expand_dims_numpy(self):
        x = np.random.randn(3, 4)
        self.assertTrue(np.allclose(expand_dims(x, axis=1), np.expand_dims(x, axis=1)))

    @require_torch
    def test_expand_dims_torch(self):
        x = np.random.randn(3, 4)
        t = torch.tensor(x)
        self.assertTrue(np.allclose(expand_dims(x, axis=1), expand_dims(t, axis=1).numpy()))

    def test_to_py_obj_native(self):
        self.assertTrue(to_py_obj(1) == 1)
        self.assertTrue(to_py_obj([1, 2, 3]) == [1, 2, 3])
        self.assertTrue(to_py_obj([((1.0, 1.1), 1.2), (2, 3)]) == [[[1.0, 1.1], 1.2], [2, 3]])

    def test_to_py_obj_numpy(self):
        x1 = [[1, 2, 3], [4, 5, 6]]
        t1 = np.array(x1)
        self.assertTrue(to_py_obj(t1) == x1)

        x2 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t2 = np.array(x2)
        self.assertTrue(to_py_obj(t2) == x2)

        self.assertTrue(to_py_obj([t1, t2]) == [x1, x2])

    @require_torch
    def test_to_py_obj_torch(self):
        x1 = [[1, 2, 3], [4, 5, 6]]
        t1 = torch.tensor(x1)
        self.assertTrue(to_py_obj(t1) == x1)

        x2 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t2 = torch.tensor(x2)
        self.assertTrue(to_py_obj(t2) == x2)

        self.assertTrue(to_py_obj([t1, t2]) == [x1, x2])


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


@require_torch
class CanReturnTupleDecoratorTester(unittest.TestCase):
    def _get_model(self, config, store_config=True, raise_in_forward=False):
        # Simple model class for testing can_return_tuple decorator.
        class SimpleTestModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                if store_config:
                    self.config = config

            @can_return_tuple
            def forward(self, x):
                if raise_in_forward:
                    raise ValueError("Test error")
                return BaseModelOutput(
                    last_hidden_state=x,
                    hidden_states=None,
                    attentions=None,
                )

        return SimpleTestModel(config)

    def test_decorator_eager(self):
        """Test that the can_return_tuple decorator works with eager mode."""

        # test nothing is set
        config = PretrainedConfig()
        model = self._get_model(config)
        inputs = torch.tensor(10)
        output = model(inputs)
        self.assertIsInstance(
            output, BaseModelOutput, "output should be a BaseModelOutput when return_dict is not set"
        )

        # test all explicit cases
        for config_return_dict in [True, False, None]:
            for return_dict in [True, False, None]:
                config = PretrainedConfig(return_dict=config_return_dict)
                model = self._get_model(config)
                output = model(torch.tensor(10), return_dict=return_dict)

                expected_type = (
                    tuple
                    if return_dict is False
                    else (tuple if config_return_dict is False and return_dict is None else BaseModelOutput)
                )
                if config_return_dict is None and return_dict is None:
                    expected_type = tuple
                message = f"output should be a {expected_type.__name__} when config.use_return_dict={config_return_dict} and return_dict={return_dict}"
                self.assertIsInstance(output, expected_type, message)

    @pytest.mark.torch_compile_test
    def test_decorator_compiled(self):
        """Test that the can_return_tuple decorator works with compiled mode."""
        config = PretrainedConfig()

        # Output object
        model = self._get_model(config)
        compiled_model = torch.compile(model)
        output = compiled_model(torch.tensor(10))
        self.assertIsInstance(output, BaseModelOutput)

        # Tuple output
        model = self._get_model(config)
        compiled_model = torch.compile(model)
        output = compiled_model(torch.tensor(10), return_dict=False)
        self.assertIsInstance(output, tuple)

    @pytest.mark.torch_export_test
    def test_decorator_torch_export(self):
        """Test that the can_return_tuple decorator works with torch.export."""
        config = PretrainedConfig()
        model = self._get_model(config)
        torch.export.export(model, args=(torch.tensor(10),))

    def test_decorator_torchscript(self):
        """Test that the can_return_tuple decorator works with torch.jit.trace."""
        config = PretrainedConfig(return_dict=False)
        model = self._get_model(config)
        inputs = torch.tensor(10)
        traced_module = torch.jit.trace(model, inputs)
        output = traced_module(inputs)
        self.assertIsInstance(output, tuple)

    def test_attribute_cleanup(self):
        """Test that the `_is_top_level_module` attribute is removed after the forward call."""

        config = PretrainedConfig(return_dict=False)
        inputs = torch.tensor(10)

        # working case
        model = self._get_model(config)
        output = model(inputs)

        self.assertIsInstance(output, tuple)
        for name, module in model.named_modules():
            self.assertFalse(
                hasattr(module, "_is_top_level_module"),
                f"Module `{name}` should not have `_is_top_level_module` attribute",
            )

        # model without config
        no_config_model = self._get_model(config, store_config=False)
        output = no_config_model(inputs)

        self.assertIsInstance(output, BaseModelOutput)
        for name, module in no_config_model.named_modules():
            self.assertFalse(
                hasattr(module, "_is_top_level_module"),
                f"Module `{name}` should not have `_is_top_level_module` attribute",
            )

        # model with raise in forward
        model_with_raise = self._get_model(config, raise_in_forward=True)
        with self.assertRaises(ValueError):
            model_with_raise(inputs)

        for name, module in model_with_raise.named_modules():
            self.assertFalse(
                hasattr(module, "_is_top_level_module"),
                f"Module `{name}` should not have `_is_top_level_module` attribute",
            )

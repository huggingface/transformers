# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional
from unittest import mock

from transformers import is_tf_available, is_torch_available, pipeline
from transformers.pipelines import Pipeline
from transformers.testing_utils import _run_slow_tests, is_pipeline_test, require_tf, require_torch, slow
from transformers.tokenization_utils_base import to_py_obj


VALID_INPUTS = ["A simple string", ["list of strings"]]


@is_pipeline_test
class CustomInputPipelineCommonMixin:
    pipeline_task = None
    pipeline_loading_kwargs = {}  # Additional kwargs to load the pipeline with
    pipeline_running_kwargs = {}  # Additional kwargs to run the pipeline with
    small_models = []  # Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator
    valid_inputs = VALID_INPUTS  # Some inputs which are valid to compare fast and slow tokenizers

    def setUp(self) -> None:
        if not is_tf_available() and not is_torch_available():
            return  # Currently no JAX pipelines

        # Download needed checkpoints
        models = self.small_models
        if _run_slow_tests:
            models = models + self.large_models

        for model_name in models:
            if is_torch_available():
                pipeline(
                    self.pipeline_task,
                    model=model_name,
                    tokenizer=model_name,
                    framework="pt",
                    **self.pipeline_loading_kwargs,
                )
            if is_tf_available():
                pipeline(
                    self.pipeline_task,
                    model=model_name,
                    tokenizer=model_name,
                    framework="tf",
                    **self.pipeline_loading_kwargs,
                )

    @require_torch
    @slow
    def test_pt_defaults(self):
        pipeline(self.pipeline_task, framework="pt", **self.pipeline_loading_kwargs)

    @require_tf
    @slow
    def test_tf_defaults(self):
        pipeline(self.pipeline_task, framework="tf", **self.pipeline_loading_kwargs)

    @require_torch
    def test_torch_small(self):
        for model_name in self.small_models:
            nlp = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                **self.pipeline_loading_kwargs,
            )
            self._test_pipeline(nlp)

    @require_tf
    def test_tf_small(self):
        for model_name in self.small_models:
            nlp = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="tf",
                **self.pipeline_loading_kwargs,
            )
            self._test_pipeline(nlp)

    @require_torch
    @slow
    def test_torch_large(self):
        for model_name in self.large_models:
            nlp = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                **self.pipeline_loading_kwargs,
            )
            self._test_pipeline(nlp)

    @require_tf
    @slow
    def test_tf_large(self):
        for model_name in self.large_models:
            nlp = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="tf",
                **self.pipeline_loading_kwargs,
            )
            self._test_pipeline(nlp)

    def _test_pipeline(self, nlp: Pipeline):
        raise NotImplementedError

    @require_torch
    def test_compare_slow_fast_torch(self):
        for model_name in self.small_models:
            nlp_slow = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                use_fast=False,
                **self.pipeline_loading_kwargs,
            )
            nlp_fast = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="pt",
                use_fast=True,
                **self.pipeline_loading_kwargs,
            )
            self._compare_slow_fast_pipelines(nlp_slow, nlp_fast, method="forward")

    @require_tf
    def test_compare_slow_fast_tf(self):
        for model_name in self.small_models:
            nlp_slow = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="tf",
                use_fast=False,
                **self.pipeline_loading_kwargs,
            )
            nlp_fast = pipeline(
                task=self.pipeline_task,
                model=model_name,
                tokenizer=model_name,
                framework="tf",
                use_fast=True,
                **self.pipeline_loading_kwargs,
            )
            self._compare_slow_fast_pipelines(nlp_slow, nlp_fast, method="call")

    def _compare_slow_fast_pipelines(self, nlp_slow: Pipeline, nlp_fast: Pipeline, method: str):
        """We check that the inputs to the models forward passes are identical for
        slow and fast tokenizers.
        """
        with mock.patch.object(
            nlp_slow.model, method, wraps=getattr(nlp_slow.model, method)
        ) as mock_slow, mock.patch.object(nlp_fast.model, method, wraps=getattr(nlp_fast.model, method)) as mock_fast:
            for inputs in self.valid_inputs:
                if isinstance(inputs, dict):
                    inputs.update(self.pipeline_running_kwargs)
                    _ = nlp_slow(**inputs)
                    _ = nlp_fast(**inputs)
                else:
                    _ = nlp_slow(inputs, **self.pipeline_running_kwargs)
                    _ = nlp_fast(inputs, **self.pipeline_running_kwargs)

                mock_slow.assert_called()
                mock_fast.assert_called()

                self.assertEqual(len(mock_slow.call_args_list), len(mock_fast.call_args_list))
                for mock_slow_call_args, mock_fast_call_args in zip(
                    mock_slow.call_args_list, mock_slow.call_args_list
                ):
                    slow_call_args, slow_call_kwargs = mock_slow_call_args
                    fast_call_args, fast_call_kwargs = mock_fast_call_args

                    slow_call_args, slow_call_kwargs = to_py_obj(slow_call_args), to_py_obj(slow_call_kwargs)
                    fast_call_args, fast_call_kwargs = to_py_obj(fast_call_args), to_py_obj(fast_call_kwargs)

                    self.assertEqual(slow_call_args, fast_call_args)
                    self.assertDictEqual(slow_call_kwargs, fast_call_kwargs)


@is_pipeline_test
class MonoInputPipelineCommonMixin(CustomInputPipelineCommonMixin):
    """A version of the CustomInputPipelineCommonMixin
    with a predefined `_test_pipeline` method.
    """

    mandatory_keys = {}  # Keys which should be in the output
    invalid_inputs = [None]  # inputs which are not allowed
    expected_multi_result: Optional[List] = None
    expected_check_keys: Optional[List[str]] = None

    def _test_pipeline(self, nlp: Pipeline):
        self.assertIsNotNone(nlp)

        mono_result = nlp(self.valid_inputs[0], **self.pipeline_running_kwargs)
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in self.mandatory_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [nlp(input, **self.pipeline_running_kwargs) for input in self.valid_inputs]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if self.expected_multi_result is not None:
            for result, expect in zip(multi_result, self.expected_multi_result):
                for key in self.expected_check_keys or []:
                    self.assertEqual(
                        set([o[key] for o in result]),
                        set([o[key] for o in expect]),
                    )

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in self.mandatory_keys:
                self.assertIn(key, result)

        self.assertRaises(Exception, nlp, self.invalid_inputs)

from typing import List, Optional

from transformers import is_tf_available, is_torch_available, pipeline

# from transformers.pipelines import DefaultArgumentHandler, Pipeline
from transformers.pipelines import Pipeline
from transformers.testing_utils import _run_slow_tests, is_pipeline_test, require_tf, require_torch, slow


VALID_INPUTS = ["A simple string", ["list of strings"]]


@is_pipeline_test
class CustomInputPipelineCommonMixin:
    pipeline_task = None
    pipeline_loading_kwargs = {}
    small_models = None  # Models tested without the @slow decorator
    large_models = None  # Models tested with the @slow decorator

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
        pipeline(self.pipeline_task, framework="pt")

    @require_tf
    @slow
    def test_tf_defaults(self):
        pipeline(self.pipeline_task, framework="tf")

    @require_torch
    def test_torch_small(self):
        for model_name in self.small_models:
            nlp = pipeline(task=self.pipeline_task, model=model_name, tokenizer=model_name, framework="pt")
            self._test_pipeline(nlp)

    @require_tf
    def test_tf_small(self):
        for model_name in self.small_models:
            nlp = pipeline(task=self.pipeline_task, model=model_name, tokenizer=model_name, framework="tf")
            self._test_pipeline(nlp)

    @require_torch
    @slow
    def test_torch_large(self):
        for model_name in self.large_models:
            nlp = pipeline(task=self.pipeline_task, model=model_name, tokenizer=model_name, framework="pt")
            self._test_pipeline(nlp)

    @require_tf
    @slow
    def test_tf_large(self):
        for model_name in self.large_models:
            nlp = pipeline(task=self.pipeline_task, model=model_name, tokenizer=model_name, framework="tf")
            self._test_pipeline(nlp)

    def _test_pipeline(self, nlp: Pipeline):
        raise NotImplementedError


@is_pipeline_test
class MonoInputPipelineCommonMixin:
    pipeline_task = None
    pipeline_loading_kwargs = {}  # Additional kwargs to load the pipeline with
    pipeline_running_kwargs = {}  # Additional kwargs to run the pipeline with
    small_models = []  # Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator
    mandatory_keys = {}  # Keys which should be in the output
    valid_inputs = VALID_INPUTS  # inputs which are valid
    invalid_inputs = [None]  # inputs which are not allowed
    expected_multi_result: Optional[List] = None
    expected_check_keys: Optional[List[str]] = None

    def setUp(self) -> None:
        if not is_tf_available() and not is_torch_available():
            return  # Currently no JAX pipelines

        for model_name in self.small_models:
            pipeline(self.pipeline_task, model=model_name, tokenizer=model_name, **self.pipeline_loading_kwargs)
        for model_name in self.large_models:
            pipeline(self.pipeline_task, model=model_name, tokenizer=model_name, **self.pipeline_loading_kwargs)

    @require_torch
    @slow
    def test_pt_defaults_loads(self):
        pipeline(self.pipeline_task, framework="pt", **self.pipeline_loading_kwargs)

    @require_tf
    @slow
    def test_tf_defaults_loads(self):
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


# @is_pipeline_test
# class DefaultArgumentHandlerTestCase(unittest.TestCase):
#     def setUp(self) -> None:
#         self.handler = DefaultArgumentHandler()
#
#     def test_kwargs_x(self):
#         mono_data = {"X": "This is a sample input"}
#         mono_args = self.handler(**mono_data)
#
#         self.assertTrue(isinstance(mono_args, list))
#         self.assertEqual(len(mono_args), 1)
#
#         multi_data = {"x": ["This is a sample input", "This is a second sample input"]}
#         multi_args = self.handler(**multi_data)
#
#         self.assertTrue(isinstance(multi_args, list))
#         self.assertEqual(len(multi_args), 2)
#
#     def test_kwargs_data(self):
#         mono_data = {"data": "This is a sample input"}
#         mono_args = self.handler(**mono_data)
#
#         self.assertTrue(isinstance(mono_args, list))
#         self.assertEqual(len(mono_args), 1)
#
#         multi_data = {"data": ["This is a sample input", "This is a second sample input"]}
#         multi_args = self.handler(**multi_data)
#
#         self.assertTrue(isinstance(multi_args, list))
#         self.assertEqual(len(multi_args), 2)
#
#     def test_multi_kwargs(self):
#         mono_data = {"data": "This is a sample input", "X": "This is a sample input 2"}
#         mono_args = self.handler(**mono_data)
#
#         self.assertTrue(isinstance(mono_args, list))
#         self.assertEqual(len(mono_args), 2)
#
#         multi_data = {
#             "data": ["This is a sample input", "This is a second sample input"],
#             "test": ["This is a sample input 2", "This is a second sample input 2"],
#         }
#         multi_args = self.handler(**multi_data)
#
#         self.assertTrue(isinstance(multi_args, list))
#         self.assertEqual(len(multi_args), 4)
#
#     def test_args(self):
#         mono_data = "This is a sample input"
#         mono_args = self.handler(mono_data)
#
#         self.assertTrue(isinstance(mono_args, list))
#         self.assertEqual(len(mono_args), 1)
#
#         mono_data = ["This is a sample input"]
#         mono_args = self.handler(mono_data)
#
#         self.assertTrue(isinstance(mono_args, list))
#         self.assertEqual(len(mono_args), 1)
#
#         multi_data = ["This is a sample input", "This is a second sample input"]
#         multi_args = self.handler(multi_data)
#
#         self.assertTrue(isinstance(multi_args, list))
#         self.assertEqual(len(multi_args), 2)
#
#         multi_data = ["This is a sample input", "This is a second sample input"]
#         multi_args = self.handler(*multi_data)
#
#         self.assertTrue(isinstance(multi_args, list))
#         self.assertEqual(len(multi_args), 2)

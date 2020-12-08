import unittest

import numpy as np

from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification, pipeline
from transformers.testing_utils import slow

from .test_pipelines_common import MonoInputPipelineCommonMixin


VALID_INPUTS = ["I really disagree with what you've said.", ["I love you."]]


class SentimentAnalysisPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "sentiment-analysis"
    small_models = ["distilbert-base-cased"]  # Default model - Models tested without the @slow decorator
    large_models = [None]  # Models tested with the @slow decorator
    mandatory_keys = {"label", "score"}  # Keys which should be in the output

    @slow
    def test_function_to_apply(self):
        for model_name in self.small_models:
            string_input, string_list_input = VALID_INPUTS
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            model = DistilBertForSequenceClassification(DistilBertConfig.from_pretrained(model_name))
            model.eval()
            classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

            def check_output(output):
                # This model does not have a sequence-classification head, so results are random
                self.assertTrue(isinstance(output, list))
                self.assertEqual(len(output), 1)
                d = string_output[0]
                self.assertEqual(set(d.keys()), {"label", "score"})
                self.assertEqual(type(d["label"]), str)
                self.assertEqual(type(d["score"]), float)

            def pipeline_call_argument(argument=None):
                _string_output = classifier(string_input, function_to_apply=argument)
                _string_list_output = classifier(string_list_input, function_to_apply=argument)
                check_output(_string_output)
                return _string_output, _string_list_output

            def pipeline_init_argument(argument=None):
                _classifier = pipeline(
                    task="sentiment-analysis", model=model, tokenizer=tokenizer, function_to_apply=argument
                )
                _string_output = _classifier(string_input)
                _string_list_output = _classifier(string_list_input)
                check_output(_string_output)
                return _string_output, _string_list_output

            def pipeline_model_argument(argument=None):
                model.config.task_specific_params = {"sentiment-analysis": {"function_to_apply": argument}}
                _classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
                _string_output = _classifier(string_input)
                _string_list_output = _classifier(string_list_input)
                check_output(_string_output)
                return _string_output, _string_list_output

            string_output = classifier(string_input)
            string_list_output = classifier(string_list_input)

            string_output_default = pipeline_call_argument("default")
            string_output_sigmoid = pipeline_call_argument("sigmoid")
            string_output_softmax = pipeline_call_argument("softmax")
            string_output_none = pipeline_call_argument("none")

            string_output_init_default = pipeline_init_argument("default")
            string_output_init_sigmoid = pipeline_init_argument("sigmoid")
            string_output_init_softmax = pipeline_init_argument("softmax")
            string_output_init_none = pipeline_init_argument("none")

            string_output_model_default = pipeline_model_argument("default")
            string_output_model_sigmoid = pipeline_model_argument("sigmoid")
            string_output_model_softmax = pipeline_model_argument("softmax")
            string_output_model_none = pipeline_model_argument("none")

            should_be_equal = [
                (
                    (string_output, string_list_output),
                    string_output_default,
                    string_output_init_default,
                    string_output_model_default,
                ),
                (string_output_sigmoid, string_output_init_sigmoid, string_output_model_sigmoid),
                (string_output_softmax, string_output_init_softmax, string_output_model_softmax),
                (string_output_none, string_output_init_none, string_output_model_none),
            ]

            for tuples_containing_equal_values in should_be_equal:
                # Retrieve each tuple from the list
                for tuple_value_0 in tuples_containing_equal_values:
                    for tuple_value_1 in tuples_containing_equal_values:
                        if tuple_value_0 is not tuple_value_1:
                            # Compare each tuple value with all the others, as long as they're not the same object
                            for pipeline_output_0, pipeline_output_1 in zip(tuple_value_0, tuple_value_1):
                                # Compare all outputs (call, init, model argument)
                                for example_result_0, example_result_1 in zip(pipeline_output_0, pipeline_output_1):
                                    # Iterate through the results
                                    self.assertTrue(
                                        np.allclose(example_result_0["score"], example_result_1["score"], atol=1e-6)
                                    )

    @slow
    def test_function_to_apply_error(self):
        for model_name in self.small_models:
            string_input, _ = VALID_INPUTS
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            model = DistilBertForSequenceClassification(DistilBertConfig.from_pretrained(model_name))
            model.eval()
            classifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

            with self.assertRaises(ValueError):
                classifier(string_input, function_to_apply="logits")

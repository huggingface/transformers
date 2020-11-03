import unittest

from transformers import AutoTokenizer, pipeline
from transformers.pipelines import Pipeline
from transformers.testing_utils import require_tf, require_torch

from .test_pipelines_common import CustomInputPipelineCommonMixin


VALID_INPUTS = ["A simple string", ["list of strings"]]


class NerPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "ner"
    small_models = [
        "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
    ]  # Default model - Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator

    def _test_pipeline(self, nlp: Pipeline):
        output_keys = {"entity", "word", "score"}
        if nlp.grouped_entities:
            output_keys = {"entity_group", "word", "score"}

        ungrouped_ner_inputs = [
            [
                {"entity": "B-PER", "index": 1, "score": 0.9994944930076599, "is_subword": False, "word": "Cons"},
                {"entity": "B-PER", "index": 2, "score": 0.8025449514389038, "is_subword": True, "word": "##uelo"},
                {"entity": "I-PER", "index": 3, "score": 0.9993102550506592, "is_subword": False, "word": "Ara"},
                {"entity": "I-PER", "index": 4, "score": 0.9993743896484375, "is_subword": True, "word": "##új"},
                {"entity": "I-PER", "index": 5, "score": 0.9992871880531311, "is_subword": True, "word": "##o"},
                {"entity": "I-PER", "index": 6, "score": 0.9993029236793518, "is_subword": False, "word": "No"},
                {"entity": "I-PER", "index": 7, "score": 0.9981776475906372, "is_subword": True, "word": "##guera"},
                {"entity": "B-PER", "index": 15, "score": 0.9998136162757874, "is_subword": False, "word": "Andrés"},
                {"entity": "I-PER", "index": 16, "score": 0.999740719795227, "is_subword": False, "word": "Pas"},
                {"entity": "I-PER", "index": 17, "score": 0.9997414350509644, "is_subword": True, "word": "##tran"},
                {"entity": "I-PER", "index": 18, "score": 0.9996136426925659, "is_subword": True, "word": "##a"},
                {"entity": "B-ORG", "index": 28, "score": 0.9989739060401917, "is_subword": False, "word": "Far"},
                {"entity": "I-ORG", "index": 29, "score": 0.7188422083854675, "is_subword": True, "word": "##c"},
            ],
            [
                {"entity": "I-PER", "index": 1, "score": 0.9968166351318359, "is_subword": False, "word": "En"},
                {"entity": "I-PER", "index": 2, "score": 0.9957635998725891, "is_subword": True, "word": "##zo"},
                {"entity": "I-ORG", "index": 7, "score": 0.9986497163772583, "is_subword": False, "word": "UN"},
            ],
        ]

        expected_grouped_ner_results = [
            [
                {"entity_group": "PER", "score": 0.999369223912557, "word": "Consuelo Araújo Noguera"},
                {"entity_group": "PER", "score": 0.9997771680355072, "word": "Andrés Pastrana"},
                {"entity_group": "ORG", "score": 0.9989739060401917, "word": "Farc"},
            ],
            [
                {"entity_group": "PER", "score": 0.9968166351318359, "word": "Enzo"},
                {"entity_group": "ORG", "score": 0.9986497163772583, "word": "UN"},
            ],
        ]

        expected_grouped_ner_results_w_subword = [
            [
                {"entity_group": "PER", "score": 0.9994944930076599, "word": "Cons"},
                {"entity_group": "PER", "score": 0.9663328925768534, "word": "##uelo Araújo Noguera"},
                {"entity_group": "PER", "score": 0.9997273534536362, "word": "Andrés Pastrana"},
                {"entity_group": "ORG", "score": 0.8589080572128296, "word": "Farc"},
            ],
            [
                {"entity_group": "PER", "score": 0.9962901175022125, "word": "Enzo"},
                {"entity_group": "ORG", "score": 0.9986497163772583, "word": "UN"},
            ],
        ]

        self.assertIsNotNone(nlp)

        mono_result = nlp(VALID_INPUTS[0])
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [nlp(input) for input in VALID_INPUTS]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

        if nlp.grouped_entities:
            if nlp.ignore_subwords:
                for ungrouped_input, grouped_result in zip(ungrouped_ner_inputs, expected_grouped_ner_results):
                    self.assertEqual(nlp.group_entities(ungrouped_input), grouped_result)
            else:
                for ungrouped_input, grouped_result in zip(
                    ungrouped_ner_inputs, expected_grouped_ner_results_w_subword
                ):
                    self.assertEqual(nlp.group_entities(ungrouped_input), grouped_result)

    @require_tf
    def test_tf_only(self):
        model_name = "Narsil/small"  # This model only has a TensorFlow version
        # We test that if we don't specificy framework='tf', it gets detected automatically
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        nlp = pipeline(task="ner", model=model_name, tokenizer=tokenizer)
        self._test_pipeline(nlp)

    #         offset=tokenizer(VALID_INPUTS[0],return_offsets_mapping=True)['offset_mapping']
    #         pipeline_running_kwargs = {"offset_mapping"}  # Additional kwargs to run the pipeline with

    @require_tf
    def test_tf_defaults(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="tf")
        self._test_pipeline(nlp)

    @require_tf
    def test_tf_small(self):
        for model_name in self.small_models:
            print(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(
                task="ner",
                model=model_name,
                tokenizer=tokenizer,
                framework="tf",
                grouped_entities=True,
                ignore_subwords=True,
            )
            self._test_pipeline(nlp)

            for model_name in self.small_models:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                nlp = pipeline(
                    task="ner",
                    model=model_name,
                    tokenizer=tokenizer,
                    framework="tf",
                    grouped_entities=True,
                    ignore_subwords=False,
                )
                self._test_pipeline(nlp)

    @require_torch
    def test_pt_defaults(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(task="ner", model=model_name, tokenizer=tokenizer)
            self._test_pipeline(nlp)

    @require_torch
    def test_torch_small(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, grouped_entities=True, ignore_subwords=True
            )
            self._test_pipeline(nlp)

        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, grouped_entities=True, ignore_subwords=False
            )
            self._test_pipeline(nlp)

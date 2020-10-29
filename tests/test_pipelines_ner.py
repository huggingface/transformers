import unittest

from transformers import pipeline
from transformers.pipelines import Pipeline
from transformers.testing_utils import require_tf

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

        ungrouped_ner_inputs = [
            [
                {"entity": "B-PER", "index": 1, "score": 0.9994944930076599, "word": "Cons"},
                {"entity": "B-PER", "index": 2, "score": 0.8025449514389038, "word": "##uelo"},
                {"entity": "I-PER", "index": 3, "score": 0.9993102550506592, "word": "Ara"},
                {"entity": "I-PER", "index": 4, "score": 0.9993743896484375, "word": "##új"},
                {"entity": "I-PER", "index": 5, "score": 0.9992871880531311, "word": "##o"},
                {"entity": "I-PER", "index": 6, "score": 0.9993029236793518, "word": "No"},
                {"entity": "I-PER", "index": 7, "score": 0.9981776475906372, "word": "##guera"},
                {"entity": "B-PER", "index": 15, "score": 0.9998136162757874, "word": "Andrés"},
                {"entity": "I-PER", "index": 16, "score": 0.999740719795227, "word": "Pas"},
                {"entity": "I-PER", "index": 17, "score": 0.9997414350509644, "word": "##tran"},
                {"entity": "I-PER", "index": 18, "score": 0.9996136426925659, "word": "##a"},
                {"entity": "B-ORG", "index": 28, "score": 0.9989739060401917, "word": "Far"},
                {"entity": "I-ORG", "index": 29, "score": 0.7188422083854675, "word": "##c"},
            ],
            [
                {"entity": "I-PER", "index": 1, "score": 0.9968166351318359, "word": "En"},
                {"entity": "I-PER", "index": 2, "score": 0.9957635998725891, "word": "##zo"},
                {"entity": "I-ORG", "index": 7, "score": 0.9986497163772583, "word": "UN"},
            ],
        ]
        expected_grouped_ner_results = [
            [
                {"entity_group": "B-PER", "score": 0.9710702640669686, "word": "Consuelo Araújo Noguera"},
                {"entity_group": "B-PER", "score": 0.9997273534536362, "word": "Andrés Pastrana"},
                {"entity_group": "B-ORG", "score": 0.8589080572128296, "word": "Farc"},
            ],
            [
                {"entity_group": "I-PER", "score": 0.9962901175022125, "word": "Enzo"},
                {"entity_group": "I-ORG", "score": 0.9986497163772583, "word": "UN"},
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

        for ungrouped_input, grouped_result in zip(ungrouped_ner_inputs, expected_grouped_ner_results):
            self.assertEqual(nlp.group_entities(ungrouped_input), grouped_result)

    @require_tf
    def test_tf_only(self):
        model_name = "Narsil/small"  # This model only has a TensorFlow version
        # We test that if we don't specificy framework='tf', it gets detected automatically
        nlp = pipeline(task="ner", model=model_name, tokenizer=model_name)
        self._test_pipeline(nlp)

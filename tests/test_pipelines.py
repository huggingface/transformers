import unittest
from typing import Iterable, List, Optional

from transformers import pipeline
from transformers.pipelines import SUPPORTED_TASKS, DefaultArgumentHandler, Pipeline

from .utils import require_tf, require_torch, slow


VALID_INPUTS = ["A simple string", ["list of strings"]]

NER_FINETUNED_MODELS = ["sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"]

# xlnet-base-cased disabled for now, since it crashes TF2
FEATURE_EXTRACT_FINETUNED_MODELS = ["sshleifer/tiny-distilbert-base-cased"]
TEXT_CLASSIF_FINETUNED_MODELS = ["sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"]
TEXT_GENERATION_FINETUNED_MODELS = ["sshleifer/tiny-ctrl"]

FILL_MASK_FINETUNED_MODELS = ["sshleifer/tiny-distilroberta-base"]
LARGE_FILL_MASK_FINETUNED_MODELS = ["distilroberta-base"]  # @slow

SUMMARIZATION_FINETUNED_MODELS = ["sshleifer/bart-tiny-random", "patrickvonplaten/t5-tiny-random"]
TF_SUMMARIZATION_FINETUNED_MODELS = ["patrickvonplaten/t5-tiny-random"]

TRANSLATION_FINETUNED_MODELS = [
    ("patrickvonplaten/t5-tiny-random", "translation_en_to_de"),
    ("patrickvonplaten/t5-tiny-random", "translation_en_to_ro"),
]
TF_TRANSLATION_FINETUNED_MODELS = [("patrickvonplaten/t5-tiny-random", "translation_en_to_fr")]

expected_fill_mask_result = [
    [
        {"sequence": "<s> My name is:</s>", "score": 0.009954338893294334, "token": 35},
        {"sequence": "<s> My name is John</s>", "score": 0.0080940006300807, "token": 610},
    ],
    [
        {"sequence": "<s> The largest city in France is Paris</s>", "score": 0.3185044229030609, "token": 2201},
        {"sequence": "<s> The largest city in France is Lyon</s>", "score": 0.21112334728240967, "token": 12790},
    ],
]
SUMMARIZATION_KWARGS = dict(num_beams=2, min_length=2, max_length=5)


class DefaultArgumentHandlerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = DefaultArgumentHandler()

    def test_kwargs_x(self):
        mono_data = {"X": "This is a sample input"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = {"x": ["This is a sample input", "This is a second sample input"]}
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

    def test_kwargs_data(self):
        mono_data = {"data": "This is a sample input"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = {"data": ["This is a sample input", "This is a second sample input"]}
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

    def test_multi_kwargs(self):
        mono_data = {"data": "This is a sample input", "X": "This is a sample input 2"}
        mono_args = self.handler(**mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 2)

        multi_data = {
            "data": ["This is a sample input", "This is a second sample input"],
            "test": ["This is a sample input 2", "This is a second sample input 2"],
        }
        multi_args = self.handler(**multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 4)

    def test_args(self):
        mono_data = "This is a sample input"
        mono_args = self.handler(mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        mono_data = ["This is a sample input"]
        mono_args = self.handler(mono_data)

        self.assertTrue(isinstance(mono_args, list))
        self.assertEqual(len(mono_args), 1)

        multi_data = ["This is a sample input", "This is a second sample input"]
        multi_args = self.handler(multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)

        multi_data = ["This is a sample input", "This is a second sample input"]
        multi_args = self.handler(*multi_data)

        self.assertTrue(isinstance(multi_args, list))
        self.assertEqual(len(multi_args), 2)


class MonoColumnInputTestCase(unittest.TestCase):
    def _test_mono_column_pipeline(
        self,
        nlp: Pipeline,
        valid_inputs: List,
        output_keys: Iterable[str],
        invalid_inputs: List = [None],
        expected_multi_result: Optional[List] = None,
        expected_check_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0], **kwargs)
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [nlp(input) for input in valid_inputs]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if expected_multi_result is not None:
            for result, expect in zip(multi_result, expected_multi_result):
                for key in expected_check_keys or []:
                    self.assertEqual(
                        set([o[key] for o in result]), set([o[key] for o in expect]),
                    )

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

        self.assertRaises(Exception, nlp, invalid_inputs)

    @require_torch
    def test_torch_ner(self):
        mandatory_keys = {"entity", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_torch
    def test_ner_grouped(self):
        mandatory_keys = {"entity_group", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name, grouped_entities=True)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_tf
    def test_tf_ner(self):
        mandatory_keys = {"entity", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_tf
    def test_tf_ner_grouped(self):
        mandatory_keys = {"entity_group", "word", "score"}
        for model_name in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model_name, tokenizer=model_name, framework="tf", grouped_entities=True)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_torch
    def test_torch_sentiment_analysis(self):
        mandatory_keys = {"label", "score"}
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="sentiment-analysis", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_tf
    def test_tf_sentiment_analysis(self):
        mandatory_keys = {"label", "score"}
        for model_name in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="sentiment-analysis", model=model_name, tokenizer=model_name, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys)

    @require_torch
    def test_torch_feature_extraction(self):
        for model_name in FEATURE_EXTRACT_FINETUNED_MODELS:
            nlp = pipeline(task="feature-extraction", model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})

    @require_tf
    def test_tf_feature_extraction(self):
        for model_name in FEATURE_EXTRACT_FINETUNED_MODELS:
            nlp = pipeline(task="feature-extraction", model=model_name, tokenizer=model_name, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})

    @require_torch
    def test_torch_fill_mask(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        for model_name in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="pt", topk=2,)
            self._test_mono_column_pipeline(nlp, valid_inputs, mandatory_keys, expected_check_keys=["sequence"])

    @require_tf
    def test_tf_fill_mask(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        for model_name in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf", topk=2,)
            self._test_mono_column_pipeline(nlp, valid_inputs, mandatory_keys, expected_check_keys=["sequence"])

    @require_torch
    @slow
    def test_torch_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        for model_name in LARGE_FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="pt", topk=2,)
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                mandatory_keys,
                expected_multi_result=expected_fill_mask_result,
                expected_check_keys=["sequence"],
            )

    @require_tf
    @slow
    def test_tf_fill_mask_results(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        for model_name in LARGE_FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model_name, tokenizer=model_name, framework="tf", topk=2)
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                mandatory_keys,
                expected_multi_result=expected_fill_mask_result,
                expected_check_keys=["sequence"],
            )

    @require_torch
    def test_torch_summarization(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["summary_text"]
        for model in SUMMARIZATION_FINETUNED_MODELS:
            nlp = pipeline(task="summarization", model=model, tokenizer=model)
            self._test_mono_column_pipeline(
                nlp, VALID_INPUTS, mandatory_keys, invalid_inputs=invalid_inputs, **SUMMARIZATION_KWARGS
            )

    @slow
    @require_tf
    def test_tf_summarization(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["summary_text"]
        for model_name in TF_SUMMARIZATION_FINETUNED_MODELS:
            nlp = pipeline(task="summarization", model=model_name, tokenizer=model_name, framework="tf",)
            self._test_mono_column_pipeline(
                nlp, VALID_INPUTS, mandatory_keys, invalid_inputs=invalid_inputs, **SUMMARIZATION_KWARGS
            )

    @require_torch
    def test_torch_translation(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["translation_text"]
        for model_name, task in TRANSLATION_FINETUNED_MODELS:
            nlp = pipeline(task=task, model=model_name, tokenizer=model_name)
            self._test_mono_column_pipeline(
                nlp, VALID_INPUTS, mandatory_keys, invalid_inputs,
            )

    @require_tf
    @slow
    def test_tf_translation(self):
        invalid_inputs = [4, "<mask>"]
        mandatory_keys = ["translation_text"]
        for model, task in TF_TRANSLATION_FINETUNED_MODELS:
            nlp = pipeline(task=task, model=model, tokenizer=model, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, mandatory_keys, invalid_inputs=invalid_inputs)

    @require_torch
    def test_torch_text_generation(self):
        for model_name in TEXT_GENERATION_FINETUNED_MODELS:
            nlp = pipeline(task="text-generation", model=model_name, tokenizer=model_name, framework="pt")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})

    @require_tf
    def test_tf_text_generation(self):
        for model_name in TEXT_GENERATION_FINETUNED_MODELS:
            nlp = pipeline(task="text-generation", model=model_name, tokenizer=model_name, framework="tf")
            self._test_mono_column_pipeline(nlp, VALID_INPUTS, {})


QA_FINETUNED_MODELS = ["sshleifer/tiny-distilbert-base-cased-distilled-squad"]


class QAPipelineTests(unittest.TestCase):
    def _test_qa_pipeline(self, nlp):
        output_keys = {"score", "answer", "start", "end"}
        valid_inputs = [
            {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."},
            {
                "question": "In what field is HuggingFace working ?",
                "context": "HuggingFace is a startup based in New-York founded in Paris which is trying to solve NLP.",
            },
        ]
        invalid_inputs = [
            {"question": "", "context": "This is a test to try empty question edge case"},
            {"question": None, "context": "This is a test to try empty question edge case"},
            {"question": "What is does with empty context ?", "context": ""},
            {"question": "What is does with empty context ?", "context": None},
        ]
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, dict)

        for key in output_keys:
            self.assertIn(key, mono_result)

        multi_result = nlp(valid_inputs)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)
        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, bad_input)
        self.assertRaises(Exception, nlp, invalid_inputs)

    @require_torch
    def test_torch_question_answering(self):
        for model_name in QA_FINETUNED_MODELS:
            nlp = pipeline(task="question-answering", model=model_name, tokenizer=model_name)
            self._test_qa_pipeline(nlp)

    @require_tf
    def test_tf_question_answering(self):
        for model_name in QA_FINETUNED_MODELS:
            nlp = pipeline(task="question-answering", model=model_name, tokenizer=model_name, framework="tf")
            self._test_qa_pipeline(nlp)


class PipelineCommonTests(unittest.TestCase):

    pipelines = SUPPORTED_TASKS.keys()

    @slow
    @require_tf
    def test_tf_defaults(self):
        # Test that pipelines can be correctly loaded without any argument
        for task in self.pipelines:
            with self.subTest(msg="Testing TF defaults with TF and {}".format(task)):
                pipeline(task, framework="tf")

    @slow
    @require_torch
    def test_pt_defaults(self):
        # Test that pipelines can be correctly loaded without any argument
        for task in self.pipelines:
            with self.subTest(msg="Testing Torch defaults with PyTorch and {}".format(task)):
                pipeline(task, framework="pt")

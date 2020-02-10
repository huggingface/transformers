import unittest
from typing import Iterable, List, Optional

from transformers import pipeline
from transformers.pipelines import Pipeline

from .utils import require_tf, require_torch


QA_FINETUNED_MODELS = {
    ("bert-base-uncased", "bert-large-uncased-whole-word-masking-finetuned-squad", None),
    ("bert-base-cased", "bert-large-cased-whole-word-masking-finetuned-squad", None),
    ("bert-base-cased", "distilbert-base-cased-distilled-squad", None),
}

TF_QA_FINETUNED_MODELS = {
    ("bert-base-uncased", "bert-large-uncased-whole-word-masking-finetuned-squad", None),
    ("bert-base-cased", "bert-large-cased-whole-word-masking-finetuned-squad", None),
    ("bert-base-cased", "distilbert-base-cased-distilled-squad", None),
}

TF_NER_FINETUNED_MODELS = {
    (
        "bert-base-cased",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
    )
}

NER_FINETUNED_MODELS = {
    (
        "bert-base-cased",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
    )
}

FEATURE_EXTRACT_FINETUNED_MODELS = {
    ("bert-base-cased", "bert-base-cased", None),
    # ('xlnet-base-cased', 'xlnet-base-cased', None), # Disabled for now as it crash for TF2
    ("distilbert-base-cased", "distilbert-base-cased", None),
}

TF_FEATURE_EXTRACT_FINETUNED_MODELS = {
    ("bert-base-cased", "bert-base-cased", None),
    # ('xlnet-base-cased', 'xlnet-base-cased', None), # Disabled for now as it crash for TF2
    ("distilbert-base-cased", "distilbert-base-cased", None),
}

TF_TEXT_CLASSIF_FINETUNED_MODELS = {
    (
        "bert-base-uncased",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "distilbert-base-uncased-finetuned-sst-2-english",
    )
}

TEXT_CLASSIF_FINETUNED_MODELS = {
    (
        "bert-base-uncased",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "distilbert-base-uncased-finetuned-sst-2-english",
    )
}

FILL_MASK_FINETUNED_MODELS = {
    ("distilroberta-base", "distilroberta-base", None),
}

TF_FILL_MASK_FINETUNED_MODELS = {
    ("distilroberta-base", "distilroberta-base", None),
}


class MonoColumnInputTestCase(unittest.TestCase):
    def _test_mono_column_pipeline(
        self,
        nlp: Pipeline,
        valid_inputs: List,
        invalid_inputs: List,
        output_keys: Iterable[str],
        expected_multi_result: Optional[List] = None,
        expected_check_keys: Optional[List[str]] = None,
    ):
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = nlp(valid_inputs)
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
    def test_ner(self):
        mandatory_keys = {"entity", "word", "score"}
        valid_inputs = ["HuggingFace is solving NLP one commit at a time.", "HuggingFace is based in New-York & Paris"]
        invalid_inputs = [None]
        for tokenizer, model, config in NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model, config=config, tokenizer=tokenizer)
            self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

    @require_tf
    def test_tf_ner(self):
        mandatory_keys = {"entity", "word", "score"}
        valid_inputs = ["HuggingFace is solving NLP one commit at a time.", "HuggingFace is based in New-York & Paris"]
        invalid_inputs = [None]
        for tokenizer, model, config in TF_NER_FINETUNED_MODELS:
            nlp = pipeline(task="ner", model=model, config=config, tokenizer=tokenizer)
            self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

    @require_torch
    def test_sentiment_analysis(self):
        mandatory_keys = {"label", "score"}
        valid_inputs = ["HuggingFace is solving NLP one commit at a time.", "HuggingFace is based in New-York & Paris"]
        invalid_inputs = [None]
        for tokenizer, model, config in TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="sentiment-analysis", model=model, config=config, tokenizer=tokenizer)
            self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

    @require_tf
    def test_tf_sentiment_analysis(self):
        mandatory_keys = {"label", "score"}
        valid_inputs = ["HuggingFace is solving NLP one commit at a time.", "HuggingFace is based in New-York & Paris"]
        invalid_inputs = [None]
        for tokenizer, model, config in TF_TEXT_CLASSIF_FINETUNED_MODELS:
            nlp = pipeline(task="sentiment-analysis", model=model, config=config, tokenizer=tokenizer)
            self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

    @require_torch
    def test_feature_extraction(self):
        valid_inputs = ["HuggingFace is solving NLP one commit at a time.", "HuggingFace is based in New-York & Paris"]
        invalid_inputs = [None]
        for tokenizer, model, config in FEATURE_EXTRACT_FINETUNED_MODELS:
            nlp = pipeline(task="feature-extraction", model=model, config=config, tokenizer=tokenizer)
            self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, {})

    @require_tf
    def test_tf_feature_extraction(self):
        valid_inputs = ["HuggingFace is solving NLP one commit at a time.", "HuggingFace is based in New-York & Paris"]
        invalid_inputs = [None]
        for tokenizer, model, config in TF_FEATURE_EXTRACT_FINETUNED_MODELS:
            nlp = pipeline(task="feature-extraction", model=model, config=config, tokenizer=tokenizer)
            self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, {})

    @require_torch
    def test_fill_mask(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        invalid_inputs = [None]
        expected_multi_result = [
            [
                {"score": 0.008698059245944023, "sequence": "<s>My name is John</s>", "token": 610},
                {"score": 0.007750614080578089, "sequence": "<s>My name is Chris</s>", "token": 1573},
            ],
            [
                {"score": 0.2721288502216339, "sequence": "<s>The largest city in France is Paris</s>", "token": 2201},
                {
                    "score": 0.19764970242977142,
                    "sequence": "<s>The largest city in France is Lyon</s>",
                    "token": 12790,
                },
            ],
        ]
        for tokenizer, model, config in FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model, config=config, tokenizer=tokenizer, topk=2)
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                invalid_inputs,
                mandatory_keys,
                expected_multi_result=expected_multi_result,
                expected_check_keys=["sequence"],
            )

    @require_tf
    def test_tf_fill_mask(self):
        mandatory_keys = {"sequence", "score", "token"}
        valid_inputs = [
            "My name is <mask>",
            "The largest city in France is <mask>",
        ]
        invalid_inputs = [None]
        expected_multi_result = [
            [
                {"score": 0.008698059245944023, "sequence": "<s>My name is John</s>", "token": 610},
                {"score": 0.007750614080578089, "sequence": "<s>My name is Chris</s>", "token": 1573},
            ],
            [
                {"score": 0.2721288502216339, "sequence": "<s>The largest city in France is Paris</s>", "token": 2201},
                {
                    "score": 0.19764970242977142,
                    "sequence": "<s>The largest city in France is Lyon</s>",
                    "token": 12790,
                },
            ],
        ]
        for tokenizer, model, config in TF_FILL_MASK_FINETUNED_MODELS:
            nlp = pipeline(task="fill-mask", model=model, config=config, tokenizer=tokenizer, topk=2)
            self._test_mono_column_pipeline(
                nlp,
                valid_inputs,
                invalid_inputs,
                mandatory_keys,
                expected_multi_result=expected_multi_result,
                expected_check_keys=["sequence"],
            )


class MultiColumnInputTestCase(unittest.TestCase):
    def _test_multicolumn_pipeline(self, nlp, valid_inputs: list, invalid_inputs: list, output_keys: Iterable[str]):
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

        self.assertRaises(Exception, nlp, invalid_inputs[0])
        self.assertRaises(Exception, nlp, invalid_inputs)

    @require_torch
    def test_question_answering(self):
        mandatory_output_keys = {"score", "answer", "start", "end"}
        valid_samples = [
            {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."},
            {
                "question": "In what field is HuggingFace working ?",
                "context": "HuggingFace is a startup based in New-York founded in Paris which is trying to solve NLP.",
            },
        ]
        invalid_samples = [
            {"question": "", "context": "This is a test to try empty question edge case"},
            {"question": None, "context": "This is a test to try empty question edge case"},
            {"question": "What is does with empty context ?", "context": ""},
            {"question": "What is does with empty context ?", "context": None},
        ]

        for tokenizer, model, config in QA_FINETUNED_MODELS:
            nlp = pipeline(task="question-answering", model=model, config=config, tokenizer=tokenizer)
            self._test_multicolumn_pipeline(nlp, valid_samples, invalid_samples, mandatory_output_keys)

    @require_tf
    def test_tf_question_answering(self):
        mandatory_output_keys = {"score", "answer", "start", "end"}
        valid_samples = [
            {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."},
            {
                "question": "In what field is HuggingFace working ?",
                "context": "HuggingFace is a startup based in New-York founded in Paris which is trying to solve NLP.",
            },
        ]
        invalid_samples = [
            {"question": "", "context": "This is a test to try empty question edge case"},
            {"question": None, "context": "This is a test to try empty question edge case"},
            {"question": "What is does with empty context ?", "context": ""},
            {"question": "What is does with empty context ?", "context": None},
        ]

        for tokenizer, model, config in TF_QA_FINETUNED_MODELS:
            nlp = pipeline(task="question-answering", model=model, config=config, tokenizer=tokenizer)
            self._test_multicolumn_pipeline(nlp, valid_samples, invalid_samples, mandatory_output_keys)

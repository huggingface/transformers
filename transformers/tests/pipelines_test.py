import unittest
from unittest.mock import patch

from typing import Iterable

from transformers import pipeline
from transformers.tests.utils import require_tf, require_torch

QA_FINETUNED_MODELS = {
    ('bert-base-uncased', 'bert-large-uncased-whole-word-masking-finetuned-squad', None),
    ('bert-base-cased', 'bert-large-cased-whole-word-masking-finetuned-squad', None),
    ('bert-base-uncased', 'distilbert-base-uncased-distilled-squad', None)
}

NER_FINETUNED_MODELS = {
    (
        'bert-base-cased',
        'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-finetuned-conll03-english-pytorch_model.bin',
        'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-finetuned-conll03-english-config.json'
    )
}

FEATURE_EXTRACT_FINETUNED_MODELS = {
   ('bert-base-cased', 'bert-base-cased', None),
   # ('xlnet-base-cased', 'xlnet-base-cased', None), # Disabled for now as it crash for TF2
   ('distilbert-base-uncased', 'distilbert-base-uncased', None)
}

TEXT_CLASSIF_FINETUNED_MODELS = {
    (
        'bert-base-uncased',
        'https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-finetuned-sst-2-english-pytorch_model.bin',
        'https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-finetuned-sst-2-english-config.json'
    )
}


@require_tf
def tf_pipeline(*args, **kwargs):
    return pipeline(**kwargs)


@require_torch
def torch_pipeline(*args, **kwargs):
    return pipeline(**kwargs)


class MonoColumnInputTestCase(unittest.TestCase):
    def _test_mono_column_pipeline(self, nlp, valid_inputs: list, invalid_inputs: list, output_keys: Iterable[str]):
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

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

        self.assertRaises(Exception, nlp, invalid_inputs)

    def test_ner(self):
        mandatory_keys = {'entity', 'word', 'score'}
        valid_inputs = ['HuggingFace is solving NLP one commit at a time.', 'HuggingFace is based in New-York & Paris']
        invalid_inputs = [None]
        for tokenizer, model, config in NER_FINETUNED_MODELS:
            with patch('transformers.pipelines.is_torch_available', return_value=False):
                nlp = tf_pipeline(task='ner', model=model, config=config, tokenizer=tokenizer)
                self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

            with patch('transformers.pipelines.is_tf_available', return_value=False):
                nlp = torch_pipeline(task='ner', model=model, config=config, tokenizer=tokenizer)
                self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

    def test_sentiment_analysis(self):
        mandatory_keys = {'label'}
        valid_inputs = ['HuggingFace is solving NLP one commit at a time.', 'HuggingFace is based in New-York & Paris']
        invalid_inputs = [None]
        for tokenizer, model, config in TEXT_CLASSIF_FINETUNED_MODELS:
            with patch('transformers.pipelines.is_torch_available', return_value=False):
                nlp = tf_pipeline(task='sentiment-analysis', model=model, config=config, tokenizer=tokenizer)
                self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

            with patch('transformers.pipelines.is_tf_available', return_value=False):
                nlp = torch_pipeline(task='sentiment-analysis', model=model, config=config, tokenizer=tokenizer)
                self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, mandatory_keys)

    def test_features_extraction(self):
        valid_inputs = ['HuggingFace is solving NLP one commit at a time.', 'HuggingFace is based in New-York & Paris']
        invalid_inputs = [None]
        for tokenizer, model, config in FEATURE_EXTRACT_FINETUNED_MODELS:
            with patch('transformers.pipelines.is_torch_available', return_value=False):
                nlp = tf_pipeline(task='sentiment-analysis', model=model, config=config, tokenizer=tokenizer)
                self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, {})

            with patch('transformers.pipelines.is_tf_available', return_value=False):
                nlp = torch_pipeline(task='sentiment-analysis', model=model, config=config, tokenizer=tokenizer)
                self._test_mono_column_pipeline(nlp, valid_inputs, invalid_inputs, {})


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

    def test_question_answering(self):
        mandatory_output_keys = {'score', 'answer', 'start', 'end'}
        valid_samples = [
            {'question': 'Where was HuggingFace founded ?', 'context': 'HuggingFace was founded in Paris.'},
            {
                'question': 'In what field is HuggingFace working ?',
                'context': 'HuggingFace is a startup based in New-York founded in Paris which is trying to solve NLP.'
            }
        ]
        invalid_samples = [
            {'question': '', 'context': 'This is a test to try empty question edge case'},
            {'question': None, 'context': 'This is a test to try empty question edge case'},
            {'question': 'What is does with empty context ?', 'context': ''},
            {'question': 'What is does with empty context ?', 'context': None},
        ]

        for tokenizer, model, config in QA_FINETUNED_MODELS:

            # Test for Tensorflow
            with patch('transformers.pipelines.is_torch_available', return_value=False):
                nlp = pipeline(task='question-answering', model=model, config=config, tokenizer=tokenizer)
                self._test_multicolumn_pipeline(nlp, valid_samples, invalid_samples, mandatory_output_keys)

            # Test for PyTorch
            with patch('transformers.pipelines.is_tf_available', return_value=False):
                nlp = pipeline(task='question-answering', model=model, config=config, tokenizer=tokenizer)
                self._test_multicolumn_pipeline(nlp, valid_samples, invalid_samples, mandatory_output_keys)


if __name__ == '__main__':
    unittest.main()

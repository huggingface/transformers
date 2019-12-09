import unittest
from unittest.mock import patch


QA_FINETUNED_MODELS = {
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    'bert-large-cased-whole-word-masking-finetuned-squad',
    'distilbert-base-uncased-distilled-squad',

}


class QuestionAnsweringPipelineTest(unittest.TestCase):
    def check_answer_structure(self, answer, batch, topk):
        self.assertIsInstance(answer, list)
        self.assertEqual(len(answer), batch)
        self.assertIsInstance(answer[0], list)
        self.assertEqual(len(answer[0]), topk)
        self.assertIsInstance(answer[0][0], dict)

        for item in answer[0]:
            self.assertTrue('start' in item)
            self.assertTrue('end' in item)
            self.assertTrue('score' in item)
            self.assertTrue('answer' in item)

    def question_answering_pipeline(self, nlp):
        # Simple case with topk = 1, no batching
        a = nlp(question='What is the name of the company I\'m working for ?', context='I\'m working for Huggingface.')
        self.check_answer_structure(a, 1, 1)

        # Simple case with topk = 2, no batching
        a = nlp(question='What is the name of the company I\'m working for ?', context='I\'m working for Huggingface.', topk=2)
        self.check_answer_structure(a, 1, 2)

        # Batch case with topk = 1
        a = nlp(question=['What is the name of the company I\'m working for ?', 'Where is the company based ?'],
                context=['I\'m working for Huggingface.', 'The company is based in New York and Paris'])
        self.check_answer_structure(a, 2, 1)

        # Batch case with topk = 2
        a = nlp(question=['What is the name of the company I\'m working for ?', 'Where is the company based ?'],
                context=['I\'m working for Huggingface.', 'The company is based in New York and Paris'], topk=2)
        self.check_answer_structure(a, 2, 2)

    @patch('transformers.pipelines.is_torch_available', return_value=False)
    def test_tf_models(self, is_torch_available):
        from transformers import pipeline
        for model in QA_FINETUNED_MODELS:
            self.question_answering_pipeline(pipeline('question-answering', model))

    @patch('transformers.pipelines.is_tf_available', return_value=False)
    @patch('transformers.tokenization_utils.is_tf_available', return_value=False)
    def test_torch_models(self, is_tf_available, _):
        from transformers import pipeline
        for model in QA_FINETUNED_MODELS:
            self.question_answering_pipeline(pipeline('question-answering', model))


class AutoPipelineTest(unittest.TestCase):
    @patch('transformers.pipelines.is_torch_available', return_value=False)
    def test_tf_qa(self, is_torch_available):
        from transformers import pipeline
        from transformers.pipelines import QuestionAnsweringPipeline
        from transformers.modeling_tf_utils import TFPreTrainedModel
        for model in QA_FINETUNED_MODELS:
            nlp = pipeline('question-answering', model)
            self.assertIsInstance(nlp, QuestionAnsweringPipeline)
            self.assertIsInstance(nlp.model, TFPreTrainedModel)

    @patch('transformers.pipelines.is_tf_available', return_value=False)
    def test_torch_qa(self, is_tf_available):
        from transformers import pipeline
        from transformers.pipelines import QuestionAnsweringPipeline
        from transformers.modeling_utils import PreTrainedModel
        for model in QA_FINETUNED_MODELS:
            nlp = pipeline('question-answering', model)
            self.assertIsInstance(nlp, QuestionAnsweringPipeline)
            self.assertIsInstance(nlp.model, PreTrainedModel)


if __name__ == '__main__':
    unittest.main()

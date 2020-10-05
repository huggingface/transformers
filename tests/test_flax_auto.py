import unittest

from transformers import AutoConfig, is_flax_available, BertConfig
from transformers.modeling_flax_bert import FlaxBertModel
from transformers.modeling_flax_roberta import FlaxRobertaModel
from transformers.testing_utils import require_flax, slow


if is_flax_available():
    from transformers.modeling_flax_auto import FlaxAutoModel


@require_flax
class FlaxAutoModelTest(unittest.TestCase):
    @slow
    def test_bert_from_pretrained(self):
        for model_name in ["bert-base-cased", "bert-large-uncased"]:
            with self.subTest(model_name):
                config = AutoConfig.from_pretrained(model_name)
                self.assertIsNotNone(config)
                self.assertIsInstance(config, BertConfig)

                model = FlaxAutoModel.from_pretrained(model_name)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, FlaxBertModel)

    @slow
    def test_roberta_from_pretrained(self):
        for model_name in ["roberta-base-cased", "roberta-large-uncased"]:
            with self.subTest(model_name):
                config = AutoConfig.from_pretrained(model_name)
                self.assertIsNotNone(config)
                self.assertIsInstance(config, BertConfig)

                model = FlaxAutoModel.from_pretrained(model_name)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, FlaxRobertaModel)

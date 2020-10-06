import unittest

from transformers import AutoConfig, AutoTokenizer, BertConfig, TensorType, is_flax_available
from transformers.testing_utils import require_flax, slow


if is_flax_available():
    import jax
    from transformers.modeling_flax_auto import FlaxAutoModel
    from transformers.modeling_flax_bert import FlaxBertModel
    from transformers.modeling_flax_roberta import FlaxRobertaModel


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

    @slow
    def test_bert_jax_jit(self):
        for model_name in ["bert-base-cased", "bert-large-uncased"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = FlaxBertModel.from_pretrained(model_name)
            tokens = tokenizer("Do you support jax jitted function?", return_tensors=TensorType.JAX)

            @jax.jit
            def eval(**kwargs):
                return model(**kwargs)

            eval(**tokens).block_until_ready()

    @slow
    def test_roberta_jax_jit(self):
        for model_name in ["roberta-base-cased", "roberta-large-uncased"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = FlaxRobertaModel.from_pretrained(model_name)
            tokens = tokenizer("Do you support jax jitted function?", return_tensors=TensorType.JAX)

            @jax.jit
            def eval(**kwargs):
                return model(**kwargs)

            eval(**tokens).block_until_ready()

import unittest

from transformers import AutoConfig, is_flax_available

from tests.utils import require_flax, slow

if is_flax_available():
    from transformers.modeling_flax_auto import FlaxAutoModel, MODEL_MAPPING


@require_flax
class FlaxAutoModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        for model_config, model_class in MODEL_MAPPING.items():

            for model_name in model_class.pretrained_model_archive_map.keys():
                with self.subTest(model_name):
                    config = AutoConfig.from_pretrained(model_name)
                    self.assertIsNotNone(config)
                    self.assertIsInstance(config, model_config)

                    model = FlaxAutoModel.from_pretrained(model_name)
                    self.assertIsNotNone(model)
                    self.assertIsInstance(model, model_class)
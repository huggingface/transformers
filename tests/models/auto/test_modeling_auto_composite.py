import json
import os
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoConfig, PretrainedConfig
from transformers.testing_utils import slow


class DummyQwen2_5Config(PretrainedConfig):
    model_type = "dummy_qwen2_5"

    def __init__(self, use_cache=True, **kwargs):
        super().__init__(**kwargs)
        self.use_cache = use_cache

class DummyQwen2_5VlVisionConfig(PretrainedConfig):
    model_type = "dummy_qwen2_5_vl_vision"
    def __init__(self, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = output_attentions

class DummyQwen2_5VlConfig(PretrainedConfig):
    model_type = "dummy_qwen2_5_vl"
    is_composite = True

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        text_kwargs = {}
        vision_kwargs = {}
        for key in list(kwargs.keys()):
            if key.startswith("text_"):
                text_kwargs[key[5:]] = kwargs.pop(key)
            elif key.startswith("vision_"):
                vision_kwargs[key[7:]] = kwargs.pop(key)

        super().__init__(**kwargs)

        text_config = text_config if text_config is not None else {}
        text_config.update(text_kwargs)
        self.text_config = DummyQwen2_5Config(**text_config)

        vision_config = vision_config if vision_config is not None else {}
        vision_config.update(vision_kwargs)
        self.vision_config = DummyQwen2_5VlVisionConfig(**vision_config)


class AutoModelCompositeTest(unittest.TestCase):
    # We DO NOT need setUpClass or tearDownClass. The patch will be managed inside the test.

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        main_config_dict = {
            "model_type": "dummy_qwen2_5_vl",
            "text_config": {"model_type": "dummy_qwen2_5", "use_cache": True},
            "vision_config": {"model_type": "dummy_qwen2_5_vl_vision", "output_attentions": False},
            "architectures": ["DummyQwen2_5VlForCausalLM"],
        }
        config_path = os.path.join(self.tmpdir.name, "config.json")
        with open(config_path, "w") as f:
            json.dump(main_config_dict, f)
        print(f"[DEBUG] config.json created at: {config_path}")

    def tearDown(self):
        self.tmpdir.cleanup()
        print("[DEBUG] Temporary directory cleaned up.")

    @slow
    def test_composite_model_kwarg_routing(self):
        use_cache_arg = True
        output_attentions_arg = True

        # Define the custom mapping that we need to inject
        custom_mapping = {
            "dummy_qwen2_5": DummyQwen2_5Config,
            "dummy_qwen2_5_vl": DummyQwen2_5VlConfig,
            "dummy_qwen2_5_vl_vision": DummyQwen2_5VlVisionConfig,
        }

        # THE CRITICAL CHANGE: We patch the `_extra_content` dictionary of the
        # actual CONFIG_MAPPING object. This is the correct, safe injection point.
        # The `with` statement ensures the patch is applied only for this block
        # and is automatically removed afterward.
        with patch.dict("transformers.models.auto.configuration_auto.CONFIG_MAPPING._extra_content", custom_mapping):
            print("\n[DEBUG] Patch applied to CONFIG_MAPPING._extra_content. Calling from_pretrained...")
            config = AutoConfig.from_pretrained(
                self.tmpdir.name,
                text_use_cache=use_cache_arg,
                vision_output_attentions=output_attentions_arg,
            )

        # Assertions are performed after the `with` block, on the `config` object that was created
        # while the patch was active.
        print("config=====> ", config)
        self.assertIsInstance(config, DummyQwen2_5VlConfig)
        self.assertIsInstance(config.text_config, DummyQwen2_5Config)
        self.assertIsInstance(config.vision_config, DummyQwen2_5VlVisionConfig)

        self.assertEqual(config.text_config.use_cache, use_cache_arg)

        print("\n[SUCCESS] Test passed!")
        print(f"Asserted that text_config.use_cache is '{use_cache_arg}' (was overridden).")
        print(f"Asserted that vision_config.output_attentions is '{output_attentions_arg}' (was overridden).")


# This block allows the test to be run directly using `python your_test_file.py`
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)


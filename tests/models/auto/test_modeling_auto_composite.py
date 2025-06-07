# tests/models/auto/test_modeling_auto_composite.py

import unittest

from transformers import AutoModel
from transformers.testing_utils import slow


# We need a test case class that inherits from unittest.TestCase
class AutoModelCompositeTest(unittest.TestCase):
    # The @slow decorator is often used for tests that might involve model loading
    @slow
    def test_composite_model_kwargs_routing(self):
        # STEP 1: Define a mock model repository using a dictionary.
        # This is a clean way to create a "fake" model without making real files.
        # The keys are the file paths, and the values are the file contents.
        # This uses the same multi-file config structure we discovered.
        mock_repo = {
            "config.json": """
            {
              "model_type": "qwen2_5_vl",
              "is_composite": true,
              "text_config": "text_config.json",
              "vision_config": "vision_config.json",
              "architectures": ["Qwen2_5_vlForCausalLM"]
            }
            """,
            "text_config.json": """
            {
              "model_type": "qwen2",
              "use_cache": false
            }
            """,
            "vision_config.json": """
            {
              "model_type": "qwen2_5_vl_vision_encoder"
            }
            """,
            # We still need a fake weights file to avoid the OSError
            "model.safetensors": b"\x0c\x00\x00\x00\x00\x00\x00\x00\x7b\x7d",
        }

        # STEP 2: Use the AutoModel.from_pretrained method.
        # We pass a kwarg that should be routed to the text_config.
        # The initial value in text_config.json is `use_cache: false`.
        # We want to override it to `True`.
        model = AutoModel.from_pretrained(
            "mock/repo",  # The name doesn't matter, it will use our mock_repo
            use_cache=True,
            trust_remote_code=True,
            _commit_hash="main",
            # This special function redirects the download manager to our mock_repo
            _files_to_mock=mock_repo,
        )

        # STEP 3: Assert that the fix worked.
        # This is the most important part of the test.
        # We are checking if the `use_cache` attribute on the *final* loaded
        # model's text_config is now True, as we requested.

        self.assertIsNotNone(model.config.text_config)
        self.assertTrue(
            model.config.text_config.use_cache, "The `use_cache` kwarg was not correctly routed to the text_config."
        )

        # We can also check that the main config wasn't polluted
        self.assertFalse(
            hasattr(model.config, "use_cache"), "The `use_cache` kwarg was incorrectly added to the main config."
        )

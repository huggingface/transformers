# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved
"""
Tests for the model_builder module.
"""

import unittest
from unittest.mock import patch

import torch


class TestModelBuilder(unittest.TestCase):
    """Test cases for the model_builder module."""

    def test_build_sam3_image_model(self):
        """Test that build_sam3_image_model creates a model with expected structure."""
        # This is a placeholder test that would need to be implemented
        # with proper mocking of dependencies
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_demo import Sam3ImageInteractiveDemo

        # For now, we'll just assert that the function exists
        self.assertTrue(callable(build_sam3_image_model))

        # In a real implementation, we would do something like:
        bpe_path = "assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path)
        self.assertIsInstance(model, Sam3ImageInteractiveDemo)

        # etc.


if __name__ == "__main__":
    unittest.main()

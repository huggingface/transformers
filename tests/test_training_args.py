import unittest
import tempfile
import os
from transformers import TrainingArguments


class TestTrainingArguments(unittest.TestCase):
    def test_default_output_dir(self):
        """Test that output_dir defaults to 'tmp_trainer' when not specified."""
        args = TrainingArguments(output_dir=None)
        self.assertEqual(args.output_dir, "tmp_trainer")

    def test_custom_output_dir(self):
        """Test that output_dir is respected when specified."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(output_dir=tmp_dir)
            self.assertEqual(args.output_dir, tmp_dir)
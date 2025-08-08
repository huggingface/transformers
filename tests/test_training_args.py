import os
import tempfile
from tempfile import TemporaryDirectory

from transformers import TrainingArguments
from transformers.testing_utils import TestCasePlus, is_accelerate_available, require_accelerate


if is_accelerate_available():
    from accelerate.utils import patch_environment


class TestTrainingArguments(TestCasePlus):
    def test_default_output_dir(self):
        """Test that output_dir defaults to 'trainer_output' when not specified."""
        args = TrainingArguments(output_dir=None)
        self.assertEqual(args.output_dir, "trainer_output")

    def test_custom_output_dir(self):
        """Test that output_dir is respected when specified."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(output_dir=tmp_dir)
            self.assertEqual(args.output_dir, tmp_dir)

    def test_output_dir_creation(self):
        """Test that output_dir is created only when needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "test_output")

            # Directory should not exist before creating args
            self.assertFalse(os.path.exists(output_dir))

            # Create args with save_strategy="no" - should not create directory
            args = TrainingArguments(
                output_dir=output_dir,
                do_train=True,
                save_strategy="no",
                report_to=None,
            )
            self.assertFalse(os.path.exists(output_dir))

            # Now set save_strategy="steps" - should create directory when needed
            args.save_strategy = "steps"
            args.save_steps = 1
            self.assertFalse(os.path.exists(output_dir))  # Still shouldn't exist

            # Directory should be created when actually needed (e.g. in Trainer)

    def test_torch_empty_cache_steps_requirements(self):
        """Test that torch_empty_cache_steps is a positive integer or None."""

        # None is acceptable (feature is disabled):
        args = TrainingArguments(torch_empty_cache_steps=None)
        self.assertIsNone(args.torch_empty_cache_steps)

        # non-int is unacceptable:
        with self.assertRaises(ValueError):
            TrainingArguments(torch_empty_cache_steps=1.0)
        with self.assertRaises(ValueError):
            TrainingArguments(torch_empty_cache_steps="none")

        # negative int is unacceptable:
        with self.assertRaises(ValueError):
            TrainingArguments(torch_empty_cache_steps=-1)

        # zero is unacceptable:
        with self.assertRaises(ValueError):
            TrainingArguments(torch_empty_cache_steps=0)

        # positive int is acceptable:
        args = TrainingArguments(torch_empty_cache_steps=1)
        self.assertEqual(args.torch_empty_cache_steps, 1)

    @require_accelerate
    def test_mixed_precision(self):
        with TemporaryDirectory() as temp_dir:
            # First with no env
            TrainingArguments(fp16=True, output_dir=temp_dir)
            args = TrainingArguments(output_dir=temp_dir, fp16=False)
            self.assertEqual(args.fp16, False)
            self.assertEqual(args.mixed_precision_dtype, None)

            # Then with env
            with patch_environment(accelerate_mixed_precision="fp16"):
                args = TrainingArguments(output_dir=temp_dir)
                self.assertEqual(args.fp16, True)
                self.assertEqual(args.mixed_precision_dtype, "fp16")
            # # Then as argument
            args = TrainingArguments(output_dir=temp_dir, mixed_precision_dtype="fp16")
            self.assertEqual(args.fp16, True)
            self.assertEqual(args.mixed_precision_dtype, "fp16")

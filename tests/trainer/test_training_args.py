import os
import tempfile
import unittest

from transformers import TrainingArguments
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import HubStrategy, IntervalStrategy, SaveStrategy, SchedulerType
from transformers.training_args import OptimizerNames


class TestTrainingArguments(unittest.TestCase):
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

    def test_output_dir_expands_user(self):
        """Test that ~ in output_dir is expanded to the user's home directory."""
        args = TrainingArguments(output_dir="~/foo", report_to=None)
        self.assertEqual(args.output_dir, os.path.expanduser("~/foo"))

    def test_enum_coercions(self):
        """Test that string values are correctly converted to their enum types."""
        args = TrainingArguments(
            output_dir="tmp",
            eval_strategy="steps",
            eval_steps=10,
            logging_strategy="steps",
            save_strategy="epoch",
            hub_strategy="end",
            lr_scheduler_type="linear",
            optim="adamw_torch",
            report_to=None,
        )
        self.assertEqual(args.eval_strategy, IntervalStrategy.STEPS)
        self.assertEqual(args.logging_strategy, IntervalStrategy.STEPS)
        self.assertEqual(args.save_strategy, SaveStrategy.EPOCH)
        self.assertEqual(args.hub_strategy, HubStrategy.END)
        self.assertEqual(args.lr_scheduler_type, SchedulerType.LINEAR)
        self.assertEqual(args.optim, OptimizerNames.ADAMW_TORCH)

        # Invalid string should raise ValueError
        with self.assertRaises(ValueError):
            TrainingArguments(output_dir="tmp", eval_strategy="invalid_strategy", report_to=None)

    def test_do_eval_auto_enabled(self):
        """Test that do_eval is automatically set to True when eval_strategy is not 'no'."""
        args = TrainingArguments(
            output_dir="tmp",
            do_eval=False,
            eval_strategy="steps",
            eval_steps=10,
            report_to=None,
        )
        self.assertTrue(args.do_eval)

    def test_eval_steps_fallback_to_logging_steps(self):
        """Test that eval_steps falls back to logging_steps when not specified."""
        args = TrainingArguments(
            output_dir="tmp",
            eval_strategy="steps",
            logging_steps=10,
            report_to=None,
        )
        self.assertEqual(args.eval_steps, 10)

    def test_eval_steps_required_when_strategy_steps(self):
        """Test that eval_strategy='steps' with logging_steps=0 raises ValueError."""
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                eval_strategy="steps",
                logging_steps=0,
                report_to=None,
            )

    def test_logging_steps_required_nonzero(self):
        """Test that logging_strategy='steps' with logging_steps=0 raises ValueError."""
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                logging_strategy="steps",
                logging_steps=0,
                report_to=None,
            )

    def test_steps_must_be_integer_when_greater_than_one(self):
        """Test that fractional steps >1 raise ValueError, but <=1 are allowed."""
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                logging_strategy="steps",
                logging_steps=10.5,
                report_to=None,
            )
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                eval_strategy="steps",
                eval_steps=10.5,
                report_to=None,
            )
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                save_strategy="steps",
                save_steps=10.5,
                report_to=None,
            )
        # Fractional values <=1 (ratios) are allowed
        args = TrainingArguments(
            output_dir="tmp",
            logging_strategy="steps",
            logging_steps=0.5,
            report_to=None,
        )
        self.assertEqual(args.logging_steps, 0.5)

    def test_load_best_model_requires_matching_strategies(self):
        """Test load_best_model_at_end validation for strategy and step compatibility."""
        # Mismatched eval/save strategy should raise
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                load_best_model_at_end=True,
                eval_strategy="steps",
                eval_steps=10,
                save_strategy="epoch",
                report_to=None,
            )

        # save_steps not a multiple of eval_steps should raise
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                load_best_model_at_end=True,
                eval_strategy="steps",
                eval_steps=10,
                save_strategy="steps",
                save_steps=15,
                report_to=None,
            )

        # Valid: matching strategies with compatible steps should not raise
        args = TrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=20,
            report_to=None,
        )
        self.assertTrue(args.load_best_model_at_end)

    def test_metric_for_best_model_defaults(self):
        """Test default metric_for_best_model and greater_is_better behavior."""
        # load_best_model_at_end with no metric → defaults to "loss"
        args = TrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to=None,
        )
        self.assertEqual(args.metric_for_best_model, "loss")
        self.assertFalse(args.greater_is_better)

        # metric ending in "loss" → greater_is_better is False
        args = TrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            report_to=None,
        )
        self.assertFalse(args.greater_is_better)

        # metric not ending in "loss" → greater_is_better is True
        args = TrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="accuracy",
            report_to=None,
        )
        self.assertTrue(args.greater_is_better)

    def test_fp16_bf16_mutual_exclusivity(self):
        """Test that fp16 and bf16 cannot both be True."""
        with self.assertRaises(ValueError):
            TrainingArguments(output_dir="tmp", fp16=True, bf16=True, report_to=None)
        with self.assertRaises(ValueError):
            TrainingArguments(output_dir="tmp", fp16_full_eval=True, bf16_full_eval=True, report_to=None)

    def test_reduce_on_plateau_requires_eval(self):
        """Test that reduce_lr_on_plateau scheduler requires an eval strategy."""
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                lr_scheduler_type="reduce_lr_on_plateau",
                eval_strategy="no",
                report_to=None,
            )

    def test_torch_compile_auto_enable(self):
        """Test that torch_compile is auto-enabled when mode or backend is set."""
        args = TrainingArguments(
            output_dir="tmp",
            torch_compile_mode="reduce-overhead",
            report_to=None,
        )
        self.assertTrue(args.torch_compile)

        args = TrainingArguments(
            output_dir="tmp",
            torch_compile_backend="inductor",
            report_to=None,
        )
        self.assertTrue(args.torch_compile)

        # Default backend when torch_compile=True
        args = TrainingArguments(
            output_dir="tmp",
            torch_compile=True,
            report_to=None,
        )
        self.assertEqual(args.torch_compile_backend, "inductor")

    def test_report_to_none_handling(self):
        """Test report_to normalization for 'none' and string values."""
        args = TrainingArguments(output_dir="tmp", report_to="none")
        self.assertEqual(args.report_to, [])

        args = TrainingArguments(output_dir="tmp", report_to=["none"])
        self.assertEqual(args.report_to, [])

        args = TrainingArguments(output_dir="tmp", report_to="tensorboard")
        self.assertEqual(args.report_to, ["tensorboard"])

    def test_warmup_steps_validation(self):
        """Test warmup_steps validation for negative values."""
        with self.assertRaises(ValueError):
            TrainingArguments(output_dir="tmp", warmup_steps=-1, report_to=None)

        # Zero and fractional values are valid
        args = TrainingArguments(output_dir="tmp", warmup_steps=0, report_to=None)
        self.assertEqual(args.warmup_steps, 0)

        args = TrainingArguments(output_dir="tmp", warmup_steps=0.5, report_to=None)
        self.assertEqual(args.warmup_steps, 0.5)

    def test_debug_option_parsing(self):
        """Test debug string parsing into DebugOption enum list."""
        args = TrainingArguments(output_dir="tmp", debug="underflow_overflow", report_to=None)
        self.assertEqual(args.debug, [DebugOption.UNDERFLOW_OVERFLOW])

        args = TrainingArguments(output_dir="tmp", debug=None, report_to=None)
        self.assertEqual(args.debug, [])

    def test_dataloader_prefetch_requires_workers(self):
        """Test that dataloader_prefetch_factor requires num_workers > 0."""
        with self.assertRaises(ValueError):
            TrainingArguments(
                output_dir="tmp",
                dataloader_prefetch_factor=2,
                dataloader_num_workers=0,
                report_to=None,
            )
        # Valid: prefetch with workers > 0
        args = TrainingArguments(
            output_dir="tmp",
            dataloader_prefetch_factor=2,
            dataloader_num_workers=2,
            report_to=None,
        )
        self.assertEqual(args.dataloader_prefetch_factor, 2)

    def test_use_cpu_disables_pin_memory(self):
        """Test that use_cpu=True disables dataloader_pin_memory."""
        args = TrainingArguments(output_dir="tmp", use_cpu=True, report_to=None)
        self.assertFalse(args.dataloader_pin_memory)

    def test_include_num_input_tokens_seen_coercion(self):
        """Test bool-to-string coercion for include_num_input_tokens_seen."""
        args = TrainingArguments(output_dir="tmp", include_num_input_tokens_seen=True, report_to=None)
        self.assertEqual(args.include_num_input_tokens_seen, "all")

        args = TrainingArguments(output_dir="tmp", include_num_input_tokens_seen=False, report_to=None)
        self.assertEqual(args.include_num_input_tokens_seen, "no")

    def test_dict_field_parsing(self):
        """Test that JSON string dict fields are parsed into dicts."""
        args = TrainingArguments(output_dir="tmp", lr_scheduler_kwargs='{"factor": 0.5}', report_to=None)
        self.assertEqual(args.lr_scheduler_kwargs, {"factor": 0.5})

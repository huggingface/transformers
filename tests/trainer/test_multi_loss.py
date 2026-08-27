import tempfile

from torch import nn

from transformers import Trainer, TrainingArguments, is_torch_available
from transformers.testing_utils import TestCasePlus, require_torch


if is_torch_available():
    from .trainer_test_utils import RegressionDataset


class MultiLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 1)
        self.config = None

    def forward(self, input_x, labels=None, **kwargs):
        logits = self.classifier(input_x.unsqueeze(-1))
        if labels is None:
            return {"logits": logits}

        # Main loss
        loss = nn.functional.mse_loss(logits.squeeze(), labels)

        # Additional components
        loss_part_a = loss * 0.1
        loss_part_b = loss * 0.2

        return {"loss": loss, "loss_part_a": loss_part_a, "loss_part_b": loss_part_b, "logits": logits}


@require_torch
class TrainerMultiLossTest(TestCasePlus):
    def test_multi_loss_logging(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = MultiLossModel()
            train_dataset = RegressionDataset(length=16)

            args = TrainingArguments(
                output_dir=tmp_dir,
                logging_steps=1,
                max_steps=3,
                logging_loss_components=True,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
            )

            trainer.train()

            # Check log history
            log_history = trainer.state.log_history
            # Usually the last log is train metrics, earlier ones are step logs
            step_logs = [log for log in log_history if "loss" in log and "train_runtime" not in log]

            self.assertGreater(len(step_logs), 0)
            for log in step_logs:
                self.assertIn("loss", log)
                self.assertIn("loss_part_a", log)
                self.assertIn("loss_part_b", log)
                self.assertIsInstance(log["loss_part_a"], float)
                self.assertIsInstance(log["loss_part_b"], float)

    def test_multi_loss_logging_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = MultiLossModel()
            train_dataset = RegressionDataset(length=16)

            args = TrainingArguments(
                output_dir=tmp_dir,
                logging_steps=1,
                max_steps=3,
                logging_loss_components=False,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
            )

            trainer.train()

            # Check log history
            log_history = trainer.state.log_history
            step_logs = [log for log in log_history if "loss" in log and "train_runtime" not in log]

            self.assertGreater(len(step_logs), 0)
            for log in step_logs:
                self.assertIn("loss", log)
                self.assertNotIn("loss_part_a", log)
                self.assertNotIn("loss_part_b", log)

import tempfile

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments


class TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([idx, idx + 1, idx + 2], dtype=torch.long),
            "labels": torch.tensor([1], dtype=torch.long),
        }


class ModelWithSkipLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.called_with_skip_logits = False

    def forward(self, input_ids=None, labels=None, skip_logits=None, **kwargs):
        assert skip_logits is True
        self.called_with_skip_logits = True
        return {"loss": torch.tensor(0.0)}


def test_trainer_sets_skip_logits_for_loss_only_eval_when_liger_enabled():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)

        model = ModelWithSkipLogits()
        ds = TinyDataset()

        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                per_device_eval_batch_size=2,
                do_train=False,
                do_eval=True,
                prediction_loss_only=True,
                use_liger_kernel=True,
                report_to=[],
                disable_tqdm=True,
            )
            trainer = Trainer(model=model, args=args, eval_dataset=ds)
            trainer.evaluate()

        assert model.called_with_skip_logits is True


class ReturnLossNoLabelsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_skip_logits = []

    def forward(self, input_ids=None, return_loss=None, skip_logits=None, **kwargs):
        self.seen_skip_logits.append(skip_logits)
        if return_loss:
            return {"loss": torch.tensor(0.0)}
        return {"logits": torch.zeros((input_ids.shape[0], 2))}


def test_trainer_does_not_set_skip_logits_when_no_labels_but_return_loss_true():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)

        model = ReturnLossNoLabelsModel()

        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                per_device_eval_batch_size=2,
                do_train=False,
                do_eval=True,
                prediction_loss_only=True,
                use_liger_kernel=True,
                report_to=[],
                disable_tqdm=True,
            )
            trainer = Trainer(model=model, args=args)

            trainer.label_names = []
            trainer.prediction_step(
                trainer.model,
                {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long), "return_loss": True},
                prediction_loss_only=True,
            )

        assert model.seen_skip_logits[-1] is None

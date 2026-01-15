import torch
import pytest
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
from transformers.utils import is_torch_bf16_gpu_available


@pytest.mark.skipif(
    not is_torch_bf16_gpu_available(),
    reason="bf16 requires supported GPU",
)
def test_trainer_train_and_eval_dtype_match():
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        torch_dtype=torch.bfloat16,
    )

    dataset = Dataset.from_dict(
        {
            "input_ids": [[101, 102], [101, 102]],
            "attention_mask": [[1, 1]],
            "labels": [0, 1],
        }
    )

    args = TrainingArguments(
        output_dir="tmp",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        bf16=True,
        logging_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    train_batch = next(iter(trainer.get_train_dataloader()))
    train_loss = trainer.training_step(trainer.model, train_batch)

    eval_batch = next(iter(trainer.get_eval_dataloader()))
    eval_loss, logits, _ = trainer.prediction_step(
        trainer.model,
        eval_batch,
        prediction_loss_only=False,
    )

    assert train_loss.dtype == logits.dtype

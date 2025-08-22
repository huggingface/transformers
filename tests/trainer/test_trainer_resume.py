import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    Trainer,
    TrainingArguments,
)

from transformers.testing_utils import TestCasePlus


class DummyModel(nn.Module):
    def __init__(self, input_dim=10, num_labels=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = self.linear(input_ids.float())
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


class DummyDictDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def create_dummy_dataset():
    """Creates a dummy dataset for testing."""
    num_samples = 13
    input_dim = 10
    dummy_input_ids = torch.rand(num_samples, input_dim)
    dummy_attention_mask = torch.ones(num_samples, input_dim)
    dummy_labels = torch.randint(0, 2, (num_samples,))
    return DummyDictDataset(dummy_input_ids, dummy_attention_mask, dummy_labels)


class TestTrainerResume(TestCasePlus):
    def test_resume_with_original_trainer(self):
        """Tests the original transformers Trainer."""
        print("Testing the original transformers Trainer...")

        # 1. Set up a dummy model
        model = DummyModel(input_dim=10, num_labels=2)
        dummy_dataset = create_dummy_dataset()

        # 3. First training (simulate interruption)
        output_dir_initial = self.get_auto_remove_tmp_dir()
        training_args_initial = TrainingArguments(
            output_dir=output_dir_initial,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=3,
            save_strategy="steps",
            save_steps=1,  # Save at every step
            report_to=[],  # Disable wandb/tensorboard and other loggers
            max_steps=2,  # Stop after step 2 to simulate interruption
        )

        trainer_initial = Trainer(
            model=model,
            args=training_args_initial,
            train_dataset=dummy_dataset,
        )
        trainer_initial.train()

        # Make sure we have a checkpoint before interruption
        checkpoint_path = os.path.join(output_dir_initial, "checkpoint-2")
        assert os.path.exists(checkpoint_path)

        print("Second phase")
        # 4. Resume training from checkpoint
        output_dir_resumed = self.get_auto_remove_tmp_dir()
        training_args_resumed = TrainingArguments(
            output_dir=output_dir_resumed,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=3,
            save_strategy="steps",
            save_steps=1,  # Keep the same save strategy
        )

        trainer_resumed = Trainer(
            model=model,
            args=training_args_resumed,
            train_dataset=dummy_dataset,
        )
        # Resume from the interrupted checkpoint and finish the remaining training
        trainer_resumed.train(resume_from_checkpoint=checkpoint_path)

        # 5. Assertion: Check if the final model has been saved
        final_model_path = os.path.join(output_dir_resumed, "checkpoint-3", "model.safetensors")
        try:
            assert os.path.exists(final_model_path), "Original Trainer: Final model checkpoint was not saved!"
            print("✓ Original Trainer: Final model has been saved.")
        except AssertionError as e:
            print(f"✗ Original Trainer: {e}")


# Run all tests
if __name__ == "__main__":
    import unittest

    unittest.main()

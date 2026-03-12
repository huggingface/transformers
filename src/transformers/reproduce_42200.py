# reproduce_42200.py
# Reproduces issue #42200 - prediction_step returns tuple instead of torch.Tensor
# Expected: crash or numpy memory error when compute_metrics is provided
# without preprocess_logits_for_metrics

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class DummyDataset(Dataset):
    """Dummy dataset with >850 rows as described in the issue."""
    def __init__(self, size=900, seq_len=16, vocab_size=100):
        self.data = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "labels": self.data[idx],
        }


def compute_metrics(eval_pred):
    """
    Intentionally does NOT use preprocess_logits_for_metrics.
    This is the exact scenario described in the issue.
    If logits is a tuple instead of a tensor, np.argmax will
    either crash or try to allocate a huge array.
    """
    logits, labels = eval_pred

    # Print the type so we can see the bug clearly
    print(f"\n>>> logits type: {type(logits)}")
    if isinstance(logits, tuple):
        print(f">>> BUG CONFIRMED: logits is a tuple of {len(logits)} elements")
        print(f">>> First element shape: {logits[0].shape}")
    else:
        print(f">>> logits shape: {logits.shape}")

    # This is what a normal user would write — it will blow up if logits is a tuple
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


def main():
    # Use a tiny LLaMA model so it runs fast on CPU
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = LlamaForCausalLM(config)

    train_dataset = DummyDataset(size=10)   # small train set
    eval_dataset  = DummyDataset(size=900)  # >850 rows as issue requires

    args = TrainingArguments(
        output_dir="./tmp_reproduce",
        eval_on_start=True,                 # triggers eval immediately — key to reproducing
        max_steps=1,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,    # provided
        # preprocess_logits_for_metrics intentionally NOT provided
    )

    print("Starting training — eval runs first due to eval_on_start=True...")
    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
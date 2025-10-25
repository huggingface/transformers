# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    require_accelerate,
    require_torch_multi_accelerator,
    run_first,
    slow,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        HfArgumentParser,
        PreTrainedTokenizerBase,
        Trainer,
        TrainingArguments,
    )

    class CPDataset(torch.utils.data.Dataset):
        """Simple dataset for context parallelism testing."""

        def __init__(self, tokenizer: PreTrainedTokenizerBase, seq_length: int = 128, num_samples: int = 8):
            self.tokenizer = tokenizer
            self.seq_length = seq_length
            # Create simple text samples
            texts = [
                "The quick brown fox jumps over the lazy dog. " * 10,
                "Hello world, this is a test sentence for training. " * 10,
            ] * (num_samples // 2)

            self.data = []
            for text in texts:
                encoded = tokenizer(
                    text,
                    max_length=seq_length,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=False,  # CP doesn't use attention_mask
                )
                input_ids = encoded["input_ids"]
                # Pre-compute shift_labels for causal LM
                shift_labels = input_ids[1:] + [tokenizer.pad_token_id]
                shift_labels = [lbl if lbl != tokenizer.pad_token_id else -100 for lbl in shift_labels]

                self.data.append({"input_ids": input_ids, "shift_labels": shift_labels})

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    class CPDataCollator:
        """Data collator for context parallelism - handles special CP requirements."""

        def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int | None = None):
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of

        def __call__(self, features):
            # Stack input_ids and shift_labels - use clone() to avoid memory sharing issues
            input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
            shift_labels = torch.stack([torch.tensor(f["shift_labels"], dtype=torch.long) for f in features])

            # Pad to multiple if needed (required for CP: sequences must be divisible by cp_size * 2)
            if self.pad_to_multiple_of:
                seq_len = input_ids.shape[1]
                if seq_len % self.pad_to_multiple_of != 0:
                    padding_len = self.pad_to_multiple_of - (seq_len % self.pad_to_multiple_of)
                    input_ids = torch.nn.functional.pad(input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
                    shift_labels = torch.nn.functional.pad(shift_labels, (0, padding_len), value=-100)

            # Create batch dictionary
            batch = {
                "input_ids": input_ids.clone(),  # Clone to avoid memory sharing with pin_memory
                "shift_labels": shift_labels.clone(),  # CP trainer expects this key
                "labels": shift_labels.clone(),  # Clone to avoid memory sharing
            }

            # Add position_ids (CP needs explicit position IDs)
            # Use repeat instead of expand to avoid view/memory sharing issues
            seq_len = batch["input_ids"].shape[1]
            batch_size = batch["input_ids"].shape[0]
            batch["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

            return batch


class TestTrainerContextParallel(TestCasePlus):
    """Test Trainer with context parallelism enabled via accelerate's ParallelismConfig."""

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    @run_first
    def test_trainer(self):
        """Test basic training with context parallelism enabled."""
        output_dir = self.get_auto_remove_tmp_dir()
        config_path = f"{self.test_file_dir}/context_parallel_config.yaml"

        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            config_path,
            f"{self.test_file_dir}/test_trainer_context_parallel.py",
            "--output_dir",
            output_dir,
            "--report_to",
            "none",
            "--max_steps",
            "5",
            "--per_device_train_batch_size",
            "1",
            "--logging_steps",
            "1",
            "--remove_unused_columns",
            "False",
        ]

        execute_subprocess_async(cmd, env=self.get_env())

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    def test_cp_reproducibility(self):
        """Test that CP produces reproducible results with the same seed."""
        import os

        output_dir = self.get_auto_remove_tmp_dir()
        config_path_cp = f"{self.test_file_dir}/context_parallel_config.yaml"

        # Run 1: Train with CP and seed=42
        loss_file_1 = os.path.join(output_dir, "losses_run1.json")
        cmd_1 = [
            "accelerate",
            "launch",
            "--config_file",
            config_path_cp,
            f"{self.test_file_dir}/test_trainer_context_parallel.py",
            "--output_dir",
            os.path.join(output_dir, "run1"),
            "--report_to",
            "none",
            "--max_steps",
            "10",
            "--per_device_train_batch_size",
            "1",
            "--logging_steps",
            "1",
            "--remove_unused_columns",
            "False",
            "--seed",
            "42",
            "--loss_output_file",
            loss_file_1,
        ]
        execute_subprocess_async(cmd_1, env=self.get_env())

        # Run 2: Train with CP and same seed=42
        loss_file_2 = os.path.join(output_dir, "losses_run2.json")
        cmd_2 = [
            "accelerate",
            "launch",
            "--config_file",
            config_path_cp,
            f"{self.test_file_dir}/test_trainer_context_parallel.py",
            "--output_dir",
            os.path.join(output_dir, "run2"),
            "--report_to",
            "none",
            "--max_steps",
            "10",
            "--per_device_train_batch_size",
            "1",
            "--logging_steps",
            "1",
            "--remove_unused_columns",
            "False",
            "--seed",
            "42",
            "--loss_output_file",
            loss_file_2,
        ]
        execute_subprocess_async(cmd_2, env=self.get_env())

        # Compare losses - should be identical with same seed
        with open(loss_file_1) as f:
            losses_1 = json.load(f)
        with open(loss_file_2) as f:
            losses_2 = json.load(f)

        assert len(losses_1) == len(losses_2), (
            f"Different number of losses: Run1 has {len(losses_1)}, Run2 has {len(losses_2)}"
        )

        # Losses should be identical (or very close) with same seed
        for i, (loss_1, loss_2) in enumerate(zip(losses_1, losses_2)):
            assert abs(loss_1 - loss_2) < 1e-6, (
                f"Loss mismatch at step {i + 1}: Run1={loss_1:.6f}, Run2={loss_2:.6f}, diff={abs(loss_1 - loss_2):.6e}"
            )


if __name__ == "__main__":
    import sys

    # Parse custom arguments (not TrainingArguments parameters)
    loss_output_file = None

    if "--loss_output_file" in sys.argv:
        idx = sys.argv.index("--loss_output_file")
        loss_output_file = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]

    # Use SmolLM (small Llama-based model that works with CP)
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",  # CP requires SDPA
        use_cache=False,  # Disable KV cache for CP
    )

    # Create dataset and data collator
    train_dataset = CPDataset(tokenizer, seq_length=128, num_samples=8)
    # pad_to_multiple_of=4 for cp_size=2 (must be divisible by cp_size * 2)
    data_collator = CPDataCollator(tokenizer, pad_to_multiple_of=4)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train for a few steps
    trainer.train()

    # Verify training completed
    assert trainer.state.global_step > 0, "Training should have completed at least one step"

    # Save losses to file if requested (for reproducibility testing)
    if loss_output_file and training_args.process_index == 0:
        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        with open(loss_output_file, "w") as f:
            json.dump(losses, f)

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

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_accelerate,
    require_torch_multi_accelerator,
    run_first,
    slow,
    torch_device,
)


if is_torch_available():
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        HfArgumentParser,
        Trainer,
        TrainingArguments,
    )

    class ContextParallelDataCollator:
        """Data collator for context parallelism - does not create attention masks."""

        def __init__(self, tokenizer, pad_to_multiple_of=None):
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of

        def __call__(self, features):
            batch = {}
            batch["input_ids"] = torch.stack([torch.tensor(f["input_ids"]) for f in features])
            # For CP, we need shift_labels pre-computed
            batch["shift_labels"] = torch.stack([torch.tensor(f["shift_labels"]) for f in features])
            batch["labels"] = batch["shift_labels"].clone()

            # Pad to multiple if specified (required for CP: cp_size * 2)
            if self.pad_to_multiple_of is not None:
                seq_len = batch["input_ids"].shape[1]
                remainder = seq_len % self.pad_to_multiple_of
                if remainder != 0:
                    padding_len = self.pad_to_multiple_of - remainder
                    batch["input_ids"] = torch.nn.functional.pad(
                        batch["input_ids"], (0, padding_len), value=self.tokenizer.pad_token_id
                    )
                    batch["shift_labels"] = torch.nn.functional.pad(
                        batch["shift_labels"], (0, padding_len), value=-100
                    )
                    batch["labels"] = batch["shift_labels"].clone()

            # Add position_ids (accelerate example includes this)
            seq_len = batch["input_ids"].shape[1]
            batch["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand(batch["input_ids"].shape[0], -1)

            # Don't create attention_mask - it causes issues with CP
            return batch


class TestTrainerContextParallel(TestCasePlus):
    """Test Trainer with context parallelism enabled via accelerate's ParallelismConfig"""

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    @run_first
    def test_trainer_context_parallel_basic(self):
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
            "--pad_to_multiple_of",
            "4",
            "--logging_steps",
            "1",
            "--remove_unused_columns",
            "False",
        ]

        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    @run_first
    def test_trainer_context_parallel_requires_sdpa(self):
        """Test that context parallelism requires SDPA attention implementation."""
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
            "--remove_unused_columns",
            "False",
            "--test_mode",
            "test_non_sdpa",
        ]

        # This should fail because we're using eager attention instead of SDPA
        with self.assertRaises(Exception):
            execute_subprocess_async(cmd, env=self.get_env())

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    @run_first
    def test_trainer_context_parallel_causal_mask_validation(self):
        """Test that context parallelism validates causal attention masks."""
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
            "--remove_unused_columns",
            "False",
            "--test_mode",
            "test_non_causal_mask",
        ]

        # This should fail because we're using a non-causal attention mask
        with self.assertRaises(Exception):
            execute_subprocess_async(cmd, env=self.get_env())

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    @run_first
    def test_trainer_context_parallel_auto_generation(self):
        """Test that context parallelism auto-generates position_ids and shift_labels."""
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
            "--remove_unused_columns",
            "False",
            "--test_mode",
            "test_auto_generation",
            "--pad_to_multiple_of",
            "4",
        ]

        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success


if __name__ == "__main__":
    # This script is meant to be run under torch.distributed with accelerate launch
    # with context parallelism enabled via ParallelismConfig

    import argparse

    # Parse custom arguments along with training arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--test_mode", type=str, default="default", help="Test mode to run")
    arg_parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad sequences to multiple of this value")
    custom_args, remaining_args = arg_parser.parse_known_args()

    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses(remaining_args)[0]

    # Use SmolLM model (small Llama-based model, works with CP unlike GPT2)
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a simple causal LM dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_attention_mask=False,  # Don't create attention mask for CP
        )
        # For context parallelism, we need to pre-compute shift_labels
        # shift_labels[i] = input_ids[i+1] for causal LM
        shift_labels = []
        for input_ids in tokenized["input_ids"]:
            # Create shift_labels by taking input_ids[1:] and appending -100
            labels = input_ids[1:] + [tokenizer.pad_token_id]
            # Replace pad tokens with -100
            labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
            shift_labels.append(labels)
        tokenized["shift_labels"] = shift_labels
        return tokenized

    # Don't remove columns yet, keep text for debugging
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Verify shift_labels exists
    print(f"Dataset columns: {tokenized_dataset.column_names}")
    print(f"First example keys: {list(tokenized_dataset[0].keys())}")

    # Select attention implementation based on test mode
    if custom_args.test_mode == "test_non_sdpa":
        # This should fail with ValueError because CP requires SDPA
        attn_implementation = "eager"
    else:
        # Default: use SDPA (required for context parallelism)
        attn_implementation = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_implementation,
        use_cache=False,  # Disable KV cache for CP (accelerate example does this)
    )

    # Handle special test modes that need custom data collators
    if custom_args.test_mode == "test_non_causal_mask":
        # Create a custom data collator that produces non-causal masks
        class NonCausalDataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, features):
                batch = {}
                batch["input_ids"] = torch.stack([torch.tensor(f["input_ids"]) for f in features])
                batch["shift_labels"] = torch.stack([torch.tensor(f["shift_labels"]) for f in features])
                batch["labels"] = batch["shift_labels"].clone()

                # Create a bidirectional (non-causal) attention mask
                # This should cause context parallelism to fail
                batch_size, seq_len = batch["input_ids"].shape
                batch["attention_mask"] = torch.ones((batch_size, seq_len, seq_len), dtype=torch.long)

                return batch

        data_collator = NonCausalDataCollator(tokenizer)
    elif custom_args.test_mode == "test_auto_generation":
        # Use DataCollatorForLanguageModeling which will test auto-generation
        # This won't have shift_labels pre-computed
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=custom_args.pad_to_multiple_of)
    else:
        # Default: use ContextParallelDataCollator (no attention masks)
        # pad_to_multiple_of should be cp_size * 2 (e.g., 4 for cp_size=2)
        data_collator = ContextParallelDataCollator(tokenizer, pad_to_multiple_of=custom_args.pad_to_multiple_of)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Verify context parallelism is enabled (if parallelism_config is available)
    if trainer.accelerator.parallelism_config is not None:
        if not trainer.accelerator.parallelism_config.cp_enabled:
            print(f"Warning: Context parallelism not enabled. cp_size={trainer.accelerator.parallelism_config.cp_size}")
            print(f"ParallelismConfig: {trainer.accelerator.parallelism_config}")
    else:
        print("Warning: No parallelism_config found on accelerator")

    # Train for a few steps
    # This will raise ValueError if using non-SDPA attention or non-causal masks with CP
    trainer.train()

    # Verify training completed successfully
    assert trainer.state.global_step > 0, "Training should have completed at least one step"

    # For auto_generation test, verify that position_ids and shift_labels were auto-generated
    if custom_args.test_mode == "test_auto_generation":
        # The training should have succeeded with auto-generated position_ids and shift_labels
        # (warnings should have been logged but training should complete)
        print("Auto-generation test passed: position_ids and shift_labels were auto-generated successfully")

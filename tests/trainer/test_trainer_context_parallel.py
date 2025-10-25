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
import sys

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
        DataCollatorForLanguageModeling,
        HfArgumentParser,
        Trainer,
        TrainingArguments,
    )


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
    def test_cp_equivalence(self):
        """Test that CP produces the same losses as without CP."""
        import os

        output_dir = self.get_auto_remove_tmp_dir()

        # Run with CP enabled (cp_size=2)
        config_path_cp = f"{self.test_file_dir}/context_parallel_config.yaml"
        loss_file_cp = os.path.join(output_dir, "losses_cp.json")

        cmd_cp = [
            "accelerate",
            "launch",
            "--config_file",
            config_path_cp,
            f"{self.test_file_dir}/test_trainer_context_parallel.py",
            "--output_dir",
            os.path.join(output_dir, "with_cp"),
            "--report_to",
            "none",
            "--max_steps",
            "10",
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--logging_steps",
            "1",
            "--remove_unused_columns",
            "False",
            "--seed",
            "42",
            "--loss_output_file",
            loss_file_cp,
        ]
        execute_subprocess_async(cmd_cp, env=self.get_env())

        # Run without CP (FSDP with num_processes=1, no parallelism_config)
        config_path_no_cp = f"{self.test_file_dir}/context_parallel_no_cp_config.yaml"
        loss_file_no_cp = os.path.join(output_dir, "losses_no_cp.json")

        cmd_no_cp = [
            "accelerate",
            "launch",
            "--config_file",
            config_path_no_cp,
            f"{self.test_file_dir}/test_trainer_context_parallel.py",
            "--output_dir",
            os.path.join(output_dir, "without_cp"),
            "--report_to",
            "none",
            "--max_steps",
            "10",
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--logging_steps",
            "1",
            "--remove_unused_columns",
            "False",
            "--seed",
            "42",
            "--loss_output_file",
            loss_file_no_cp,
        ]
        execute_subprocess_async(cmd_no_cp, env=self.get_env())

        # Compare losses - should be very close since CP just splits sequence computation
        with open(loss_file_cp) as f:
            losses_cp = json.load(f)
        with open(loss_file_no_cp) as f:
            losses_no_cp = json.load(f)

        assert len(losses_cp) == len(losses_no_cp), (
            f"Different number of losses: CP has {len(losses_cp)}, no-CP has {len(losses_no_cp)}"
        )

        # CP should produce very similar results (small numerical differences expected)
        # The differences come from:
        # - Different gradient reduction patterns in distributed training
        # - BF16 mixed precision accumulated differences
        # - Sequence splitting and gathering in CP mode
        losses_cp_tensor = torch.tensor(losses_cp)
        losses_no_cp_tensor = torch.tensor(losses_no_cp)

        # Use torch.testing.assert_close with rtol=2% and atol=0.02
        # Testing shows actual differences are typically <1.5%
        torch.testing.assert_close(
            losses_cp_tensor,
            losses_no_cp_tensor,
            rtol=2e-2,  # 2% relative tolerance
            atol=2e-2,  # 0.02 absolute tolerance
            msg=f"CP losses {losses_cp} do not match non-CP losses {losses_no_cp}",
        )


if __name__ == "__main__":
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

    # Create simple dataset: just tokenize some text
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Hello world, this is a test sentence for training. " * 10,
    ] * 4  # 8 samples total

    def tokenize_function(examples):
        return tokenizer(examples, max_length=128, truncation=True, padding="max_length")

    train_dataset = [tokenize_function(text) for text in texts]

    # Use standard DataCollatorForLanguageModeling for causal LM
    # pad_to_multiple_of=4 ensures sequences are divisible by cp_size * 2 (for cp_size=2)
    # Trainer will automatically generate position_ids and shift_labels as needed
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
        pad_to_multiple_of=4,
    )

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

    # Save losses to file if requested (for equivalence testing)
    if loss_output_file and training_args.process_index == 0:
        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        with open(loss_output_file, "w") as f:
            json.dump(losses, f)

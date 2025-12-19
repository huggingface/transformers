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
from pathlib import Path

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


class TestContextParallel(TestCasePlus):
    """Test Trainer with Torch context parallelism enabled via accelerate's ParallelismConfig."""

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    @run_first
    def test_cp_equivalence(self):
        """Test that CP produces the same losses as without CP."""

        # Shared setup
        world_size = 2
        script_path = __file__

        # Step 1: Run with CP enabled (cp_size=world_size)
        cp_yes_output_dir = Path(self.get_auto_remove_tmp_dir()).resolve()
        cp_yes_config_path = cp_yes_output_dir / "context_parallel_config.yaml"
        cp_yes_losses_path = cp_yes_output_dir / "cp_yes_losses.json"

        # Write config file inline (self-contained test)
        with open(cp_yes_config_path, "w") as f:
            f.write(
                f"""distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_version: 2
mixed_precision: bf16
num_processes: {world_size}
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 1
  parallelism_config_tp_size: 1
  parallelism_config_cp_size: {world_size}
  parallelism_config_cp_comm_strategy: alltoall
"""
            )

        cmd_cp_yes = f"""
            accelerate launch
            --config_file {cp_yes_config_path}
            {script_path}
            --output_dir {cp_yes_output_dir}
            --report_to none
            --max_steps 10
            --per_device_train_batch_size 1
            --gradient_accumulation_steps 1
            --logging_steps 1
            --remove_unused_columns False
            --seed 42
            --loss_output_file {cp_yes_losses_path}
        """.split()

        execute_subprocess_async(cmd_cp_yes, env=self.get_env())

        # Step 2: Run without CP (FSDP with num_processes=1, no parallelism_config)
        cp_no_output_dir = Path(self.get_auto_remove_tmp_dir()).resolve()
        cp_no_config_path = cp_no_output_dir / "context_parallel_config.yaml"
        cp_no_losses_path = cp_no_output_dir / "cp_no_losses.json"

        # Write config file inline (self-contained test)
        with open(cp_no_config_path, "w") as f:
            f.write(
                """distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_version: 2
mixed_precision: bf16
num_processes: 1
"""
            )

        cmd_cp_no = f"""
            accelerate launch
            --config_file {cp_no_config_path}
            {script_path}
            --output_dir {cp_no_output_dir}
            --report_to none
            --max_steps 10
            --per_device_train_batch_size 1
            --gradient_accumulation_steps 1
            --logging_steps 1
            --remove_unused_columns False
            --seed 42
            --loss_output_file {cp_no_losses_path}
        """.split()

        execute_subprocess_async(cmd_cp_no, env=self.get_env())

        # Compare losses - should be very close since CP just splits sequence computation
        with open(cp_yes_losses_path) as f:
            cp_yes_losses = json.load(f)
        with open(cp_no_losses_path) as f:
            cp_no_losses = json.load(f)

        assert len(cp_yes_losses) == len(cp_no_losses), (
            f"Different number of losses: CP has {len(cp_yes_losses)}, no-CP has {len(cp_no_losses)}"
        )

        # CP should produce very similar results (small numerical differences expected)
        # The differences come from:
        # - Different gradient reduction patterns in distributed training
        # - BF16 mixed precision accumulated differences
        # - Sequence splitting and gathering in CP mode
        cp_yes_losses_tensor = torch.tensor(cp_yes_losses)
        cp_no_losses_tensor = torch.tensor(cp_no_losses)

        # Use torch.testing.assert_close with rtol=2% and atol=0.02
        # Testing shows actual differences are typically <1.5%
        torch.testing.assert_close(
            cp_yes_losses_tensor,
            cp_no_losses_tensor,
            rtol=2e-2,  # 2% relative tolerance
            atol=2e-2,  # 0.02 absolute tolerance
            msg=f"CP losses {cp_yes_losses} do not match non-CP losses {cp_no_losses}",
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
        dtype=torch.float32,
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

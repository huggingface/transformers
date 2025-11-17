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
    read_json_file,
    require_accelerate,
    require_torch_multi_accelerator,
    slow,
    write_file,
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


class TestTrainerALSTUlyssesSP(TestCasePlus):
    """Test Trainer with ALST/Ulysses sequence parallelism enabled via accelerate's ParallelismConfig."""

    @require_torch_multi_accelerator
    @require_accelerate
    @slow
    def test_sp_equivalence(self):
        """Test that ALST/Ulysses sequence parallelism produces the same losses as without it."""

        # shared setup
        world_size = 2
        script_path = __file__  # self.test_file_dir} / "test_alst_ulysses_sp.py"
        ds_config_path = self.test_file_dir / "ds_config_zero2.json"

        # step 1. Run with SP enabled (sp_size=world_size)
        sp_yes_output_dir = self.get_auto_remove_tmp_dir(return_pathlib_obj=True)
        sp_yes_accelerate_config_path = sp_yes_output_dir / "context_parallel_config.yaml"
        sp_yes_losses_path = sp_yes_output_dir / "sp_yes_losses.json"
        write_file(
            sp_yes_accelerate_config_path,
            f"""
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: {ds_config_path}
machine_rank: 0
num_machines: 1
num_processes: {world_size}
parallelism_config:
  parallelism_config_sp_size: {world_size}
  parallelism_config_sp_backend: deepspeed
  parallelism_config_sp_seq_length_is_variable: true
  parallelism_config_sp_attn_implementation: sdpa
                   """,
        )

        cmd_sp = f"""
            accelerate launch
            --config_file {sp_yes_accelerate_config_path}
            {script_path}
            --output_dir {sp_yes_output_dir}
            --report_to none
            --max_steps 10
            --per_device_train_batch_size 1
            --gradient_accumulation_steps 1
            --logging_steps 1
            --remove_unused_columns False
            --seed 42
            --loss_output_file {sp_yes_losses_path}
        """.split()

        execute_subprocess_async(cmd_sp, env=self.get_env())

        # step 2. Run without SP enabled (sp_size=world_size)
        sp_no_output_dir = self.get_auto_remove_tmp_dir(return_pathlib_obj=True)
        sp_no_accelerate_config_path = sp_no_output_dir / "context_parallel_config.yaml"
        sp_no_losses_path = sp_no_output_dir / "sp_yes_losses.json"
        write_file(
            sp_no_accelerate_config_path,
            f"""
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: {ds_config_path}
machine_rank: 0
num_machines: 1
num_processes: {world_size}
                   """,
        )

        cmd_sp = f"""
            accelerate launch
            --config_file {sp_no_accelerate_config_path}
            {script_path}
            --output_dir {sp_no_output_dir}
            --report_to none
            --max_steps 10
            --per_device_train_batch_size 1
            --gradient_accumulation_steps 1
            --logging_steps 1
            --remove_unused_columns False
            --seed 42
            --loss_output_file {sp_no_losses_path}
        """.split()

        execute_subprocess_async(cmd_sp, env=self.get_env())

        # Compare losses - should be very close since SP just splits sequence computation
        sp_yes_losses = read_json_file(sp_yes_losses_path)
        sp_no_losses = read_json_file(sp_no_losses_path)

        assert len(sp_yes_losses) == len(sp_no_losses), (
            f"Different number of losses: SP has {len(sp_yes_losses)}, no-SP has {len(sp_no_losses)}"
        )

        # ALST/UlyssesSP should produce very similar results (small numerical differences expected)
        # The differences come from:
        # - Different gradient reduction patterns in distributed training
        # - BF16 mixed precision accumulated differences
        sp_yes_losses_tensor = torch.tensor(sp_yes_losses)
        sp_no_losses_tensor = torch.tensor(sp_no_losses)
        torch.testing.assert_close(
            sp_yes_losses_tensor,
            sp_no_losses_tensor,
            atol=2e-2,
            rtol=2e-5,
            msg=f"SP-enabled losses {sp_yes_losses} do not match SP-disabled losses {sp_no_losses}",
        )


if __name__ == "__main__":
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    # Parse custom arguments (not TrainingArguments parameters)
    loss_output_file = None

    if "--loss_output_file" in sys.argv:
        idx = sys.argv.index("--loss_output_file")
        loss_output_file = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",  # SP requires SDPA or FA
    )
    # fix the outdated testing model config
    model.generation_config.pad_token_id = 1

    # Create simple dataset: just tokenize some text
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Hello world, this is a test sentence for training. " * 10,
    ] * 4  # 8 samples total

    def tokenize_function(examples):
        return tokenizer(examples, max_length=128, truncation=True, padding="max_length")

    train_dataset = [tokenize_function(text) for text in texts]

    # Use standard DataCollatorForLanguageModeling for causal LM
    # pad_to_multiple_of=4 ensures sequences are divisible by sp_size * 2 (for sp_size=2)
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

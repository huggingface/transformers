# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import sys

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_accelerate,
    require_fp8,
    require_torch_multi_accelerator,
    run_first,
    torch_device,
)


if is_torch_available():
    import torch
    import torch.distributed
    import torch.utils.data

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EvalPrediction,
        GenerationConfig,
        HfArgumentParser,
        PreTrainedTokenizerBase,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    class DummyTextDataset(torch.utils.data.Dataset[str]):
        def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
            data = 4 * [
                "Hello world!",
                "The quick brown fox jumps over the lazy dog.",
            ]
            self.data = [
                {k: v.squeeze(0) for k, v in tokenizer(item, return_tensors="pt", return_attention_mask=True).items()}
                for item in data
            ]
            for item in self.data:
                item["labels"] = item["input_ids"]

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, i: int) -> str:
            return self.data[i]


class TestFSDPTrainer(TestCasePlus):
    @require_torch_multi_accelerator
    @require_accelerate
    @run_first
    def test_trainer(self):
        output_dir = self.get_auto_remove_tmp_dir()
        cmd = [
            "accelerate",
            "launch",
            "--use_fsdp",
            "--main_process_port",
            f"{get_torch_dist_unique_port()}",
            "--num_processes",
            f"{backend_device_count(torch_device)}",
            "--fsdp_transformer_layer_cls_to_wrap",
            "GPT2Block",
            f"{self.test_file_dir}/test_trainer_fsdp.py",
            "--output_dir",
            f"{output_dir}",
            "--report_to",
            "none",
        ]
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call


class TestFSDPTrainerFP8(TestCasePlus):
    @require_torch_multi_accelerator
    @require_accelerate
    @require_fp8
    @run_first
    def test_trainer(self):
        output_dir = self.get_auto_remove_tmp_dir()
        cmd = [
            "accelerate",
            "launch",
            "--use_fsdp",
            "--main_process_port",
            f"{get_torch_dist_unique_port()}",
            "--num_processes",
            f"{backend_device_count(torch_device)}",
            "--mixed_precision",
            "fp8",
            "--fsdp_transformer_layer_cls_to_wrap",
            "GPT2Block",
            f"{self.test_file_dir}/test_trainer_fsdp.py",
            "--output_dir",
            f"{output_dir}",
            "--report_to",
            "none",
        ]
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call


class TestFSDPTrainerWrap(TestCasePlus):
    @require_torch_multi_accelerator
    @require_accelerate
    @run_first
    def test_trainer(self):
        output_dir = self.get_auto_remove_tmp_dir()
        cmd = [
            "accelerate",
            "launch",
            "--use_fsdp",
            "--main_process_port",
            f"{get_torch_dist_unique_port()}",
            "--num_processes",
            f"{backend_device_count(torch_device)}",
            "--fsdp_transformer_layer_cls_to_wrap",
            "GPT2Block",
            f"{self.test_file_dir}/test_trainer_fsdp.py",
            "--output_dir",
            f"{output_dir}",
            "--report_to",
            "none",
            "--auto_find_batch_size",
            "True",
        ]
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call


class TestFSDPTrainerTorchCompile(TestCasePlus):
    @require_torch_multi_accelerator
    @require_accelerate
    @run_first
    def test_trainer(self):
        output_dir = self.get_auto_remove_tmp_dir()
        cmd = [
            "accelerate",
            "launch",
            "--use_fsdp",
            "--main_process_port",
            f"{get_torch_dist_unique_port()}",
            "--num_processes",
            f"{backend_device_count(torch_device)}",
            "--fsdp_transformer_layer_cls_to_wrap",
            "GPT2Block",
            f"{self.test_file_dir}/test_trainer_fsdp.py",
            "--torch_compile_mode",
            "default",
            "--output_dir",
            f"{output_dir}",
            "--report_to",
            "none",
        ]
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call


class TestFSDPTrainerWithParallelismConfig(TestCasePlus):
    """Test that FSDP with parallelism_config set does not break save_model.

    Regression test for https://github.com/huggingface/transformers/issues/43125

    When parallelism_config is set, save_model should still use the FSDP-specific
    save handler, not the generic parallelism_config handler that could cause issues.
    """

    @require_torch_multi_accelerator
    @require_accelerate
    @run_first
    def test_trainer_with_parallelism_config(self):
        world_size = backend_device_count(torch_device)
        output_dir = self.get_auto_remove_tmp_dir()
        script_path = __file__

        # Create accelerate config file with FSDP + parallelism_config
        # This is the setup that triggers the bug in issue #43125
        # Using FULL_STATE_DICT because that's where get_state_dict is called
        config_path = f"{output_dir}/fsdp_parallelism_config.yaml"
        with open(config_path, "w") as f:
            f.write(
                f"""distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_version: 2
  fsdp_transformer_layer_cls_to_wrap: GPT2Block
mixed_precision: "no"
num_processes: {world_size}
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 1
  parallelism_config_tp_size: 1
"""
            )

        # Run with parallelism_config set - this triggers the bug path
        cmd = f"""
            accelerate launch
            --config_file {config_path}
            {script_path}
            --output_dir {output_dir}
            --report_to none
            --test_parallelism_config_save
        """.split()

        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call


if __name__ == "__main__":
    # Parse custom arguments (not TrainingArguments parameters) before HfArgumentParser
    test_parallelism_config_save = "--test_parallelism_config_save" in sys.argv
    if test_parallelism_config_save:
        sys.argv.remove("--test_parallelism_config_save")

    parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.per_device_eval_batch_size = 1
    training_args.predict_with_generate = True
    training_args.generation_config = GenerationConfig(max_length=30)

    pretrained_model_name = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(torch.distributed.get_rank())
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)

    def compute_metrics(p: EvalPrediction) -> dict[str, bool]:
        return {"accuracy": (p.predictions == p.label_ids).mean()}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        eval_dataset=DummyTextDataset(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Test for https://github.com/huggingface/transformers/issues/43125
    # FSDP with parallelism_config set should not break save_model
    import os
    from unittest.mock import patch

    if test_parallelism_config_save:
        # The accelerator should have parallelism_config set from the config file
        pc = getattr(trainer.accelerator, "parallelism_config", None)

        # Verify parallelism_config is set (this is what triggers the bug)
        assert pc is not None, "accelerator.parallelism_config should be set"
        assert trainer.is_fsdp_enabled, "FSDP should be enabled"

        # Track whether get_state_dict was called (the fix ensures it IS called)
        # Use a list to avoid nonlocal issues in nested function
        get_state_dict_called = [False]
        original_get_state_dict = trainer.accelerator.get_state_dict

        def mock_get_state_dict(*args, **kwargs):
            get_state_dict_called[0] = True
            return original_get_state_dict(*args, **kwargs)

        # Patch get_state_dict to track if it's called
        with patch.object(trainer.accelerator, "get_state_dict", side_effect=mock_get_state_dict):
            # Save model - this should use FSDP's save handler, not the generic handler
            # The bug was that save_model would take the wrong code path when
            # accelerator.parallelism_config is set, bypassing FSDP's state dict handling
            save_dir = f"{training_args.output_dir}/checkpoint_parallelism_config"
            trainer.save_model(save_dir)

        # CRITICAL: Verify get_state_dict was called
        # The bug was that with parallelism_config set, save_model would skip
        # the FSDP state dict handling entirely
        assert get_state_dict_called[0], (
            "accelerator.get_state_dict should have been called. "
            "This means save_model took the wrong code path (bypassed FSDP handler)."
        )

        # Verify checkpoint directory was created (only on main rank)
        if training_args.should_save:
            assert os.path.exists(save_dir), f"Expected {save_dir} to exist"

            # Verify checkpoint contains expected files
            import glob

            checkpoint_files = glob.glob(os.path.join(save_dir, "*.bin")) + glob.glob(
                os.path.join(save_dir, "*.safetensors")
            )
            assert len(checkpoint_files) > 0, f"Expected checkpoint files in {save_dir}"

        print("SUCCESS: save_model with parallelism_config works correctly!")
    else:
        metrics = trainer.evaluate()

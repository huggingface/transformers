import os
from dataclasses import dataclass
from typing import Optional

import torch
from accelerate import ParallelismConfig
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.distributed import DistributedConfig
from transformers.utils import is_torch_neuron_available


@dataclass
class ScriptArguments:
    dataset_name: str
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    num_examples: int = 16  # tiny fixed subset to overfit on
    max_length: int = 1024


@dataclass
class ModelArguments:
    model_name_or_path: str
    model_revision: str = "main"
    trust_remote_code: bool = False


def main(script_args, training_args, model_args):
    if not torch.cuda.is_available() and is_torch_neuron_available(check_device=True):
        import torch_neuronx  # noqa: F401

    tp_size = int(os.environ.get("TP_SIZE", "1"))
    fsdp_size = int(os.environ.get("FSDP_SIZE", "1"))
    if tp_size > 1 and fsdp_size > 1:
        raise ValueError(
            f"TP_SIZE ({tp_size}) > 1 together with FSDP_SIZE ({fsdp_size}) > 1 is not supported: "
            "1D `DistributedConfig` only. Use one or the other."
        )
    if (tp_size > 1 or fsdp_size > 1) and training_args.fsdp:
        raise ValueError(
            f"TP_SIZE ({tp_size}) / FSDP_SIZE ({fsdp_size}) together with --fsdp is not supported: "
            "that flag configures Accelerate's own FSDP plugin, a separate mechanism from "
            "`distributed_config`. Use one or the other."
        )

    kwargs = {}
    if tp_size > 1:
        training_args.parallelism_config = ParallelismConfig(tp_size=tp_size)
        kwargs["distributed_config"] = DistributedConfig(tp_size=tp_size)
    elif fsdp_size > 1:
        kwargs["distributed_config"] = DistributedConfig(fsdp_size=fsdp_size)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------------------------
    # Dataset: a tiny fixed subset, tokenized once via the chat template.
    # ---------------------------------------------------------------------------
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_split)
    dataset = dataset.select(range(script_args.num_examples))

    def tokenize(example):
        input_ids = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=False,
            truncation=True,
            max_length=script_args.max_length,
        )
        return {"input_ids": input_ids}

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Check that the model can be saved and reloaded correctly, even when sharded.
    rank = int(os.environ.get("RANK", "0"))
    distributed_config = kwargs.get("distributed_config")

    trainer.save_model(training_args.output_dir)

    if rank == 0:
        original_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype)
        original_state_dict = original_model.state_dict()

        unsharded_model = AutoModelForCausalLM.from_pretrained(training_args.output_dir, torch_dtype=dtype)
        unsharded_state_dict = unsharded_model.state_dict()
        

        mismatches = []
        for key, expected_value in unsharded_state_dict.items():
            if key not in original_state_dict:
                mismatches.append(f"{key}: missing from saved checkpoint")
                continue
            try:
                torch.testing.assert_close(expected_value, original_state_dict[key], rtol=0, atol=0)
            except AssertionError as e:
                mismatches.append(f"{key}: {e}")
        if mismatches:
            raise AssertionError("Save correctness check failed:\n" + "\n".join(mismatches))

        print(
            f"Save correctness check passed: {len(unsharded_state_dict)} parameters "
            "match the unsharded checkpoint exactly."
        )


    # We can start training.
    # trainer.train()



if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelArguments))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    main(script_args, training_args, model_args)

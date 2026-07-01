# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from accelerate import ParallelismConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from peft import get_peft_model
from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser, get_peft_config


def main(script_args, training_args, model_args):
    if not dist.is_initialized():
        backend = dist.Backend.default_device_backend_map.get("neuron")
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    # ---------------------------------------------------------------------------
    # Tensor Parallelism
    # ---------------------------------------------------------------------------
    tp_size = int(os.environ.get("TP_SIZE", "1"))

    tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    }

    kwargs = {}
    if tp_size > 1:
        training_args.parallelism_config = ParallelismConfig(tp_size=tp_size)
        kwargs["tp_plan"] = tp_plan
        kwargs["tp_size"] = tp_size

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        trust_remote_code=True,
        # attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs,
    )

    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    if int(os.environ.get("COMPILE", "0")):
        print("Compiling model")
        model = torch.compile(model, backend="neuron")

    device = torch.device("neuron")
    model = model.to(device)
    model.train()

    # ---------------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading and packing dataset")
    raw = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    max_len = training_args.max_length or 1024

    all_tokens = []
    for example in raw[script_args.dataset_train_split]:
        if "messages" in example:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        elif "text" in example:
            text = example["text"]
        else:
            continue
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=True))
        all_tokens.append(tokenizer.eos_token_id)

    n_seqs = len(all_tokens) // (max_len + 1)
    packed = torch.tensor(all_tokens[: n_seqs * (max_len + 1)]).view(n_seqs, max_len + 1)
    log(f"Packed {n_seqs} sequences of length {max_len}")

    # Shard across data-parallel ranks
    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    rows_per_dp = len(packed) // dp_size
    local_data = packed[dp_rank * rows_per_dp : (dp_rank + 1) * rows_per_dp]

    # ---------------------------------------------------------------------------
    # Optimizer & training loop
    # ---------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_args.learning_rate,
        foreach=False,
    )

    batch_size = training_args.per_device_train_batch_size
    grad_accum = training_args.gradient_accumulation_steps
    max_steps = training_args.max_steps if training_args.max_steps > 0 else 100

    step = 0
    for i in range(0, rows_per_dp, batch_size * grad_accum):
        optimizer.zero_grad()
        for j in range(grad_accum):
            idx = i + j * batch_size
            if idx >= rows_per_dp:
                break
            batch = local_data[idx : idx + batch_size]
            inputs = batch[:, :-1].to(device)
            labels = batch[:, 1:].to(device)
            loss = model(inputs, labels=labels).loss
            (loss / grad_accum).backward()
        optimizer.step()
        step += 1
        if step % training_args.logging_steps == 0:
            log(f"step={step}/{max_steps} loss={loss.item():.4f}")
        if 0 < max_steps <= step:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    if training_args.max_steps < 0:
        print("max_steps not set, defaulting to 100 for profiling")
        training_args.max_steps = 100
    main(script_args, training_args, model_args)

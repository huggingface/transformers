# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Compare native Qwen3-MoE Trainer gradients across DDP world sizes, without downloads."""

import argparse
from pathlib import Path

import torch

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainingArguments, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()
    set_seed(17)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=1,
        max_position_embeddings=128,
        router_aux_loss_coef=0.01,
        output_router_logits=True,
        use_cache=False,
        pad_token_id=0,
    )
    model = Qwen3MoeForCausalLM(config)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            use_cpu=args.use_cpu,
            average_tokens_across_devices=True,
            ddp_find_unused_parameters=True,
            report_to="none",
        ),
    )
    accelerator = trainer.accelerator
    model = accelerator.prepare(model)

    # Repeat every sequence once per rank. Each rank then sees the same token
    # and routing distribution as the full batch used by the single-rank run.
    rows = [(torch.arange(9) * (3 + i) + 92 + i * 11) % 125 + 3 for i in range(4)]
    input_ids = torch.stack(rows).to(trainer.args.device)
    if accelerator.num_processes == 1:
        input_ids = input_ids.repeat_interleave(2, dim=0)
    batch = {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)}
    num_items = trainer._get_num_items_in_batch([batch], trainer.args.device)
    trainer.current_gradient_accumulation_steps = 1
    loss = trainer.training_step(model, batch, num_items_in_batch=num_items)
    loss = accelerator.reduce(loss, reduction="mean")

    if accelerator.is_main_process:
        gradients = {
            name: parameter.grad.detach().cpu()
            for name, parameter in accelerator.unwrap_model(model).named_parameters()
            if parameter.grad is not None
        }
        result = {"loss": loss.cpu(), "gradients": gradients}
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(result, Path(args.output_dir) / "result.pt")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()

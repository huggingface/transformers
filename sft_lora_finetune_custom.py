#!/usr/bin/env python3
"""
LoRA finetuning with Tensor Parallelism on AWS Trainium.
Supports TP sharding across Neuron cores for memory-efficient training.
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def build_optimizer(model, lr, weight_decay):
    """Build AdamW optimizer with weight decay for weights, no decay for biases/1D params."""
    params_decay = []
    params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Apply weight decay to 2D+ parameters (weights), not to biases or 1D params
        if param.ndim >= 2 and not name.endswith(".bias"):
            params_decay.append(param)
        else:
            params_no_decay.append(param)

    return torch.optim.AdamW(
        [{"params": params_decay, "weight_decay": weight_decay}, {"params": params_no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
        foreach=False,  # Required for TP (DTensor compatibility)
    )


def cosine_schedule(step, warmup_steps, total_steps):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def pack_dataset(tokenizer, dataset_name, max_seq_len):
    """Load dataset and pack into fixed-length sequences."""
    ds = load_dataset(dataset_name, split="train")

    all_tokens = []
    for row in ds:
        # Handle different dataset formats
        if "text" in row:
            text = row["text"]
        elif "messages" in row:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        elif "instruction" in row and "output" in row:
            # Alpaca-style format (instruction + optional input + output)
            instruction = row["instruction"]
            input_text = row.get("input", "")
            output = row["output"]

            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        else:
            raise ValueError(f"Unknown dataset format. Available keys: {row.keys()}")

        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)

    # Pack into (max_seq_len + 1) sized sequences for input/label pairs
    n_seqs = len(all_tokens) // (max_seq_len + 1)
    all_tokens = all_tokens[: n_seqs * (max_seq_len + 1)]

    packed = torch.tensor(all_tokens, dtype=torch.long)
    packed = packed.view(n_seqs, max_seq_len + 1)

    print(f"Packed {len(all_tokens):,} tokens into {packed.shape}")
    return packed


def clip_grad_norm(model, max_norm, tp_mesh=None):
    """Gradient clipping that handles both TP (DTensor) and non-TP cases."""
    if tp_mesh is None:
        # Standard gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    else:
        # TP case: handle DTensor gradients
        from torch.distributed.tensor import DTensor

        grads = [p.grad for p in model.parameters() if p.grad is not None]

        # Compute norm for each gradient (handle both DTensor and Tensor)
        norms = []
        for g in grads:
            if isinstance(g, DTensor):
                norms.append(g.to_local().norm())
            else:
                norms.append(g.norm())

        total_norm = torch.stack(norms).norm()

        # All-reduce norm across TP ranks if we have DTensors
        if any(isinstance(g, DTensor) for g in grads):
            dist.all_reduce(total_norm, group=tp_mesh.get_group())
            total_norm = total_norm / (tp_mesh.size() ** 0.5)

        # Clip gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grads:
                g.mul_(clip_coef)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    # Initialize distributed training
    backend = dist.Backend.default_device_backend_map.get("neuron")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("neuron")

    def log(msg):
        """Only rank 0 prints to avoid duplicate logs."""
        if rank == 0:
            print(msg, flush=True)

    # Setup Tensor Parallelism (TP)
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    if world_size % tp_size != 0:
        raise ValueError(f"world_size must be divisible by TP_SIZE")

    if tp_size > 1:
        # Create 2D mesh: (data parallel, tensor parallel)
        mesh = init_device_mesh("neuron", (world_size // tp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = mesh["tp"]
        dp_mesh = mesh["dp"]
        dp_size = dp_mesh.size()
    else:
        tp_mesh = None
        dp_mesh = None
        dp_size = world_size

    log(f"world_size={world_size} tp={tp_size} dp={dp_size}")

    # Load tokenizer
    log("Loading tokenizer and dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and pack dataset
    packed = pack_dataset(tokenizer, args.dataset_name, args.max_seq_length)

    # Load model
    log("Loading model")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Apply Tensor Parallelism (must be done BEFORE LoRA)
    if tp_mesh:
        log(f"Applying TP (tp_size={tp_size})")
        tp_plan = {
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }
        for layer in model.model.layers:
            parallelize_module(layer, tp_mesh, tp_plan)

    # Apply LoRA adapters
    log(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing (saves memory by recomputing activations)
    log("Enabling gradient checkpointing")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Move model to Neuron device
    model = model.to(device)

    # Compile model with Neuron backend (optimizes memory and performance)
    if args.compile:
        log("Compiling model")
        model = torch.compile(model, backend="neuron")

    model.train()

    # Setup optimizer and scheduler
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)

    # Calculate training steps
    rows_per_dp = packed.size(0) // dp_size
    steps_per_epoch = rows_per_dp // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, args.warmup_steps, total_steps)
    )

    global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * dp_size
    log(f"Training: {steps_per_epoch} steps/epoch, {total_steps} total, batch={global_batch}")

    # Shard dataset across data parallel ranks
    dp_rank = dp_mesh.get_local_rank() if dp_mesh else rank
    local_data = packed[dp_rank * rows_per_dp : (dp_rank + 1) * rows_per_dp]

    # Training loop
    step = 0
    for epoch in range(args.num_train_epochs):
        batch_stride = args.per_device_train_batch_size * args.gradient_accumulation_steps

        for i in range(0, rows_per_dp, batch_stride):
            t0 = time.perf_counter()
            optimizer.zero_grad()
            total_loss = 0.0
            total_tokens = 0

            # Gradient accumulation loop
            for micro in range(args.gradient_accumulation_steps):
                idx = i + micro * args.per_device_train_batch_size
                if idx >= rows_per_dp:
                    break

                # Get batch and split into inputs/labels
                batch = local_data[idx : idx + args.per_device_train_batch_size]
                inputs = batch[:, :-1].to(device)
                labels = batch[:, 1:].to(device)

                print(f"inputs: {inputs.shape}, labels: {labels.shape}")

                # Forward and backward
                loss = model(inputs, labels=labels).loss
                (loss / args.gradient_accumulation_steps).backward()

                total_loss += loss.item()
                total_tokens += inputs.numel()

            # Gradient clipping
            if args.max_grad_norm > 0:
                clip_grad_norm(model, args.max_grad_norm, tp_mesh)

            # Update weights
            optimizer.step()
            scheduler.step()
            step += 1

            # Logging
            if step % args.logging_steps == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = total_loss / args.gradient_accumulation_steps
                tok_per_sec = total_tokens / elapsed
                ms_per_step = elapsed * 1000
                log(f"step={step}/{total_steps} loss={avg_loss:.4f} tok/s={tok_per_sec:.0f} ms/step={ms_per_step:.1f}")

            # Checkpoint saving
            if args.save_steps > 0 and step % args.save_steps == 0 and rank == 0:
                save_path = f"{args.output_dir}/checkpoint-{step}"
                log(f"Saving checkpoint to {save_path}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

    # Save final model
    if rank == 0:
        log(f"Saving final model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    log("Training complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

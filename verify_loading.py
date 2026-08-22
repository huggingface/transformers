# Save/load roundtrip test for distributed models (TP, FSDP, TP+FSDP).
#
# Verifies that save_pretrained → from_pretrained preserves model weights by
# checking that the cross-entropy loss is identical before and after the roundtrip.
# This catches bugs in DTensor gather-on-save and shard-on-read paths.
#
# Usage:
#   python verify_loading.py --mode single_gpu
#   torchrun --nproc_per_node=2 verify_loading.py --mode fsdp
#   torchrun --nproc_per_node=2 verify_loading.py --mode tp
#   torchrun --nproc_per_node=4 verify_loading.py --mode tp_fsdp
#   MODEL=Qwen/Qwen3-0.6B torchrun --nproc_per_node=2 verify_loading.py --mode tp
import argparse
import os
import shutil

import torch
from torch.distributed.tensor import DTensor, Replicate

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["single_gpu", "fsdp", "tp", "tp_sp", "tp_fsdp", "tp_sp_fsdp"], required=True)
parser.add_argument("--model", type=str, default=None, help="Model ID (or set MODEL env var)")
args = parser.parse_args()

model_id = args.model or os.environ.get("MODEL") or os.environ.get("MODEL_ID") or "hf-internal-testing/tiny-random-MixtralForCausalLM"

if args.mode != "single_gpu":
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    rank = 0
    local_rank = 0
    torch.cuda.set_device(0)

configs = {
    "single_gpu": None,
    "fsdp": DistributedConfig(fsdp_size=2, fsdp_plan="auto"),
    "tp": DistributedConfig(tp_size=2, tp_plan="auto"),
    "tp_sp": DistributedConfig(tp_size=2, tp_plan="auto", enable_sequence_parallel=True),
    "tp_fsdp": DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=2, fsdp_plan="auto"),
    "tp_sp_fsdp": DistributedConfig(tp_size=2, tp_plan="auto", fsdp_size=2, fsdp_plan="auto", enable_sequence_parallel=True),
}

tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "The capital of France is Paris. The largest ocean is the Pacific."


def materialize_full_logits(logits: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, DTensor):
        with torch.no_grad():
            return logits.redistribute(placements=[Replicate()] * logits.device_mesh.ndim, async_op=False).to_local()
    return logits


def compute_loss(model):
    inputs = tokenizer(text, return_tensors="pt").to(f"cuda:{local_rank}")
    input_ids = inputs["input_ids"]
    # Pad sequence length to a multiple of tp_size so DTensor Shard(1) splits evenly
    # across ranks in SP mode. Always pad (even for non-TP modes) so that all modes
    # compute on the same input and losses are directly comparable.
    max_tp = max((c.tp_size if c is not None else 1) for c in configs.values())
    seq_len = input_ids.shape[1]
    if seq_len % max_tp != 0:
        pad_len = max_tp - (seq_len % max_tp)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.cat([input_ids, input_ids.new_full((1, pad_len), pad_token_id)], dim=1)
    labels = input_ids.clone()
    labels[:, seq_len:] = -100  # ignore padding in loss
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, position_ids=position_ids).logits
        logits = materialize_full_logits(logits)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="mean",
            ignore_index=-100,
        )
    return loss.item()


# --- Step 1: Load original model and compute loss ---
model = AutoModelForCausalLM.from_pretrained(model_id, distributed_config=configs[args.mode], dtype=torch.float32)
if args.mode == "single_gpu":
    model = model.to("cuda:0")

loss_before = compute_loss(model)
if rank == 0:
    print(f"{args.mode}: loss_before = {loss_before:.6f}")

# --- Step 2: Save to local dir (shared path across ranks) ---
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"verify_ckpt_{args.mode}")
if rank == 0:
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
if args.mode != "single_gpu":
    torch.distributed.barrier()
model.save_pretrained(save_dir, is_main_process=(rank == 0))
if rank == 0:
    print(f"{args.mode}: saved to {save_dir}")

# Ensure all ranks see the saved files before reloading
if args.mode != "single_gpu":
    torch.distributed.barrier()

del model
torch.cuda.empty_cache()

# --- Step 3: Reload from saved checkpoint and compute loss ---
model2 = AutoModelForCausalLM.from_pretrained(save_dir, distributed_config=configs[args.mode], dtype=torch.float32)
if args.mode == "single_gpu":
    model2 = model2.to("cuda:0")

loss_after = compute_loss(model2)
if rank == 0:
    print(f"{args.mode}: loss_after  = {loss_after:.6f}")

# --- Step 4: Compare ---
if rank == 0:
    diff = abs(loss_before - loss_after)
    print(f"{args.mode}: diff = {diff:.2e}")
    if diff < 1e-5:
        print("PASS: save/load roundtrip is lossless")
    else:
        print("FAIL: loss mismatch after save/load roundtrip!")

if args.mode != "single_gpu":
    torch.distributed.destroy_process_group()

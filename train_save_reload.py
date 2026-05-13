"""Minimal save/reload demo: FSDP + TP on Isotonic/TinyMixtral-4x248M-MoE.

    torchrun --nproc_per_node=4 train_save_reload.py                    # save+reload
    torchrun --nproc_per_node=4 train_save_reload.py --mode baseline    # straight N steps

The training loop deliberately overfits a single fixed sample so that, after
enough steps, the model memorizes it and `generate()` from a prefix produces
the rest of the sentence verbatim. To verify the save/reload round-trip is
lossless, run both modes and diff the loss / grad_norm logs *and* the final
generated token stream — they should all match step-for-step.
"""

import argparse
import os

import torch
from torch.distributed.tensor import DTensor

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import _replicate_dtensor, load_optimizer_distributed, save_optimizer_distributed


MODEL_NAME = "Isotonic/TinyMixtral-4x248M-MoE"
TOTAL_STEPS = 30
HALFWAY = TOTAL_STEPS // 2
LR = 1e-3
SEED = 42
BATCH_SIZE = 1

# A single passage long enough to tokenize to at least SEQ_LEN+1 tokens. The
# prompt below is a prefix of this; after overfitting, generation should
# reproduce the continuation verbatim.
OVERFIT_TEXT = (
    "In a quiet village nestled between rolling hills and a slow river, the "
    "autumn mornings arrived with mist that hung low over the fields and a sky "
    "that turned from grey to pale gold as the sun climbed."
)
GEN_PROMPT = "In a quiet village"


def run_phase(model, optimizer, batch_iter, local_rank, rank, start, stop):
    for step in range(start, stop):
        batch = next(batch_iter)
        input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
        labels = batch["labels"].to(f"cuda:{local_rank}")

        loss = model(input_ids, labels=labels).loss
        loss.backward()

        # Custom grad clip that tolerates DTensor grads with mixed placements:
        # _replicate_dtensor handles _StridedShard (which redistribute() can't).
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        local_grads = [
            _replicate_dtensor(g).to_local() if isinstance(g, DTensor) else g for g in grads
        ]
        total_norm = torch.nn.utils.get_total_norm(local_grads, norm_type=2.0)
        torch.nn.utils.clip_grads_with_norm_(grads, max_norm=1.0, total_norm=total_norm)

        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            # Single canonical line per step so `diff` between modes is mechanical.
            print(f"step {step:>3d} | loss {loss.item():.6f} | grad_norm {total_norm.item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["save_reload", "baseline"], default="save_reload")
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(SEED)

    distributed_config = DistributedConfig(
        tp_size=2,
        fsdp_size=2,
        tp_plan="auto",
        fsdp_plan="auto",
        enable_sequence_parallel=True,
    )

    # Each mode writes to its own top-level directory so the artifacts of one run
    # don't clobber the other and can be inspected side-by-side after the fact.
    save_dir = f"./checkpoints_{args.mode}"
    intermediate_dir = os.path.join(save_dir, "intermediate")

    if rank == 0:
        print(f"# mode = {args.mode} | save_dir = {save_dir}")

    # Build the initial model + optimizer.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        distributed_config=distributed_config,
        torch_dtype=torch.bfloat16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = tokenizer(OVERFIT_TEXT, return_tensors="pt").input_ids[0]
    fixed_batch = {
        "input_ids": ids.unsqueeze(0).to(f"cuda:{local_rank}"),
        "labels": ids.unsqueeze(0).to(f"cuda:{local_rank}"),
    }

    def fixed_iter():
        while True:
            yield fixed_batch

    batch_iter = fixed_iter()

    if args.mode == "baseline":
        run_phase(model, optimizer, batch_iter, local_rank, rank, 0, TOTAL_STEPS)
    else:
        # 1. Train first half.
        run_phase(model, optimizer, batch_iter, local_rank, rank, 0, HALFWAY)

        # 2. Save (model via DCP→HF-format consolidation; optimizer via DCP).
        model.save_pretrained(intermediate_dir, distributed_checkpoint=True)
        save_optimizer_distributed(model, optimizer, os.path.join(intermediate_dir, "optimizer"))
        if rank == 0:
            print(f"# saved intermediate to {intermediate_dir}")

        # 3. Tear down + reload from disk. Note: the dataloader iterator stays alive across
        #    the boundary so batch indices line up with the baseline run.
        del model, optimizer
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            intermediate_dir,
            distributed_config=distributed_config,
            torch_dtype=torch.bfloat16,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        load_optimizer_distributed(model, optimizer, os.path.join(intermediate_dir, "optimizer"))
        model.train()
        if rank == 0:
            print(f"# reloaded model + optimizer from {intermediate_dir}")

        # 4. Train second half.
        run_phase(model, optimizer, batch_iter, local_rank, rank, HALFWAY, TOTAL_STEPS)

    # Final save: canonical safetensors for the model + DCP for the optimizer.
    model.save_pretrained(save_dir)
    save_optimizer_distributed(model, optimizer, os.path.join(save_dir, "optimizer"))
    if rank == 0:
        print(f"# saved final model + optimizer to {save_dir}")

    del model, optimizer
    torch.cuda.empty_cache()

    gen_distributed_config = DistributedConfig(
        tp_size=4,
        tp_plan="auto",
        enable_sequence_parallel=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        distributed_config=gen_distributed_config,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    inputs = tokenizer(GEN_PROMPT, return_tensors="pt").to(f"cuda:{local_rank}")
    max_new = ids.numel() - inputs.input_ids.shape[-1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)

    if rank == 0:
        tokens = output_ids[0].tolist()
        expected = ids.tolist()
        print(f"# gen tokens: {tokens}")
        print(f"# exp tokens: {expected}")
        print(f"# gen text:   {tokenizer.decode(tokens, skip_special_tokens=True)!r}")
        print(f"# exp text:   {tokenizer.decode(expected, skip_special_tokens=True)!r}")
        assert tokens == expected, (
            f"generated tokens do not match OVERFIT_TEXT — "
            f"first mismatch at index {next((i for i, (g, e) in enumerate(zip(tokens, expected)) if g != e), min(len(tokens), len(expected)))}"
        )

    torch.distributed.destroy_process_group()

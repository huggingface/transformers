"""
torchrun --nproc_per_node=4 overfit_demo.py

The script overfit one sentence following the steps:
    - Train first half using FSDP=2+TP=2
    - Save the model and optimizer in distributed checkpoint
    - Reload the model and optimizer from the distributed checkpoint
    - Train the rest in TP=4 (change distributed config)
    - Save the model and optimizer in distributed checkpoint
    - Reload the model in a single safetensors file.
    - Do inference in TP=4  nd assert greedy generation reproduces the sentence verbatim
"""

import os

import torch
from torch.distributed.device_mesh import init_device_mesh

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import (
    clip_grad_norm,
    load_optimizer_distributed,
    save_optimizer_distributed,
)

NAME = "Isotonic/TinyMixtral-4x248M-MoE"
TEXT = "In a quiet village nestled between rolling hills and a slow river, the autumn mornings arrived with mist that hung low over the fields and a sky that turned from grey to pale gold as the sun climbed."
CKPT = "./checkpoints"
OPT = os.path.join(CKPT, "optimizer")
STEPS = 10
HALF = STEPS // 2

rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))

tokenizer = AutoTokenizer.from_pretrained(NAME)
ids = tokenizer(TEXT, return_tensors="pt").input_ids.to(f"cuda:{local_rank}")

# Train first half, distributed-save model + optimizer.
model = AutoModelForCausalLM.from_pretrained(
    NAME,
    distributed_config=DistributedConfig(
        tp_size=2,
        fsdp_size=2,
        tp_plan="auto",
        fsdp_plan="auto",
        enable_sequence_parallel=True,
    ),
    dtype=torch.bfloat16,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
for step in range(0, HALF):
    loss = model(ids, labels=ids).loss
    loss.backward()
    total_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    if rank == 0:
        print(f"step {step:>2} | loss {loss.item():.5f} grad norm {total_norm.item():.5f}")

model.save_pretrained(CKPT, distributed_checkpoint=True)
save_optimizer_distributed(model, optimizer, OPT)
del model, optimizer
torch.cuda.empty_cache()

# Reload model + optimizer from the distributed checkpoint, train the rest.
model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    distributed_config=DistributedConfig(tp_size=4, tp_plan="auto", enable_sequence_parallel=True),
    dtype=torch.bfloat16,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
load_optimizer_distributed(model, optimizer, OPT)

model.train()
for step in range(HALF, STEPS):
    loss = model(ids, labels=ids).loss
    loss.backward()
    total_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    if rank == 0:
        print(f"step {step:>2} | loss {loss.item():.5f} grad norm {total_norm.item():.5f}")

# INFERENCE in TP=4
model.save_pretrained(CKPT)
save_optimizer_distributed(model, optimizer, OPT + "_tp4")
del model, optimizer
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    CKPT,
    distributed_config=DistributedConfig(tp_size=4, tp_plan="auto"),
    dtype=torch.bfloat16,
)
model.eval()
prompt = tokenizer("In a quiet village", return_tensors="pt").to(f"cuda:{local_rank}")
out = model.generate(**prompt, max_new_tokens=ids.shape[-1] - prompt.input_ids.shape[-1], do_sample=False)

got, want = out[0].tolist(), ids[0].tolist()
if rank == 0:
    print(f"generated: {tokenizer.decode(got, skip_special_tokens=True)!r}")
    print(f"expected: {tokenizer.decode(want, skip_special_tokens=True)!r}")
assert got == want, (
    f"generation mismatch at index {next((i for i, (g, e) in enumerate(zip(got, want)) if g != e), -1)}"
)

torch.distributed.destroy_process_group()

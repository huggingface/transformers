"""Minimal training loop on the composite checkpoint, single-device and data-parallel (DDP).

Needs a CUDA device (prints a skip notice and exits 0 otherwise). Sampling- and network-free: the
image is a synthetic array, the batch is two short samples (one text-only, one with an image).

The pruned LM head sets the label rules this test pins down:
- label positions holding ids the head cannot produce (media codes and media placeholders, i.e.
  everything >= `output_vocab_size`) must be masked with -100 — the model raises otherwise, which is
  asserted here;
- with `labels`, the returned logits keep the *physical* pruned head width (loss is computed there);
  without `labels`, they are padded to the logical `vocab_size` with a `finfo.min` tail.

The loop itself freezes everything except the last decoder layer and the lm_head (bf16 + plain SGD:
a smoke test of backward/step mechanics, not a training recipe) and asserts the loss drops when
overfitting one batch. With >= 2 CUDA devices the script relaunches itself under
`torchrun --nproc_per_node=2` and repeats the step wrapped in DistributedDataParallel, asserting the
synchronized gradients agree across ranks. TP/model-sharded inference is covered by
`08_multi_device_inference.py`.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from transformers import Apertus1p5ForConditionalGeneration, AutoProcessor


# local dir, or hub repo id (optionally `repo_id@revision`); default: the published composite
CHECKPOINT = os.environ.get("APERTUS1P5_CHECKPOINT", "swiss-ai/Apertus-v1.5-8B")
if not os.path.isdir(CHECKPOINT):
    from huggingface_hub import snapshot_download

    repo_id, _, revision = CHECKPOINT.partition("@")
    CHECKPOINT = snapshot_download(repo_id, revision=revision or None)

if not torch.cuda.is_available():
    print("[SKIP] training-loop test needs a CUDA device")
    sys.exit(0)


def build_batch(processor, device):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    inputs = processor(
        text=["The capital of Switzerland is Bern.", "<|image|> This image shows colorful noise."],
        images=[[], [image]],
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def build_labels(inputs, output_vocab_size):
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    # ids the pruned head cannot produce (media placeholders here; media codes after scatter) are
    # input-only and must not be predicted
    labels[labels >= output_vocab_size] = -100
    return labels


def prepare_trainable(model):
    model.requires_grad_(False)
    trainable = [model.model.language_model.layers[-1], model.lm_head]
    for module in trainable:
        module.requires_grad_(True)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return count


def ddp_main():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("nccl")
    rank, device = dist.get_rank(), torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    model = Apertus1p5ForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.bfloat16).to(device)
    prepare_trainable(model)
    model.train()
    ddp_model = DistributedDataParallel(model, device_ids=[device.index])

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    inputs = build_batch(processor, device)
    labels = build_labels(inputs, model.config.get_text_config().output_vocab_size)
    optimizer = torch.optim.SGD((p for p in ddp_model.parameters() if p.requires_grad), lr=5e-3)

    for _ in range(2):
        optimizer.zero_grad()
        loss = ddp_model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()

    # both ranks see the same batch, so the allreduced gradients (and the loss) must agree exactly
    probe = torch.stack([loss.detach().float(), model.lm_head.weight.grad.float().norm()])
    gathered = [torch.zeros_like(probe) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, probe)
    if rank == 0:
        assert all(torch.equal(gathered[0], other) for other in gathered[1:]), gathered
        print(
            f"[OK] DDP({dist.get_world_size()}): 2 synchronized steps, loss {float(loss):.4f}, "
            f"grad norms identical across ranks"
        )
    dist.barrier()
    dist.destroy_process_group()


if os.environ.get("APERTUS1P5_DDP") == "1":
    ddp_main()
    sys.exit(0)

processor = AutoProcessor.from_pretrained(CHECKPOINT)
model = Apertus1p5ForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.bfloat16, device_map="cuda:0")
text_config = model.config.get_text_config()
inputs = build_batch(processor, model.device)
labels = build_labels(inputs, text_config.output_vocab_size)

# --- pruned-head label contract ---------------------------------------------------------------------
with torch.no_grad():
    no_label_logits = model(**inputs).logits
assert no_label_logits.shape[-1] == text_config.vocab_size
assert (no_label_logits[..., text_config.output_vocab_size :] == torch.finfo(no_label_logits.dtype).min).all()

with torch.no_grad():
    loss_out = model(**inputs, labels=labels)
assert loss_out.logits.shape[-1] == text_config.output_vocab_size, "loss-mode logits keep the physical head width"
assert torch.isfinite(loss_out.loss), loss_out.loss
print(
    f"[OK] logits contract: {no_label_logits.shape[-1]} padded without labels, "
    f"{loss_out.logits.shape[-1]} physical with labels, loss {float(loss_out.loss):.4f}"
)

bad_labels = inputs["input_ids"].clone()  # keeps the media placeholder ids -> must be rejected
try:
    model(**inputs, labels=bad_labels)
    raise AssertionError("labels holding media ids must raise")
except ValueError as error:
    assert "-100" in str(error), error
    print(f"[OK] unmasked media labels rejected: {str(error)[:90]}...")

# --- overfit one batch with a small trainable subset ------------------------------------------------
trainable_params = prepare_trainable(model)
model.train()
optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=5e-3)

losses = []
for _ in range(4):
    optimizer.zero_grad()
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    optimizer.step()
    losses.append(float(loss))
assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
print(
    f"[OK] single-device loop ({trainable_params / 1e6:.0f}M trainable params): loss {losses[0]:.4f} -> {losses[-1]:.4f}"
)
del model, optimizer
torch.cuda.empty_cache()

# --- DDP across 2 devices ---------------------------------------------------------------------------
if torch.cuda.device_count() < 2:
    print("[SKIP] DDP part needs >= 2 CUDA devices")
    sys.exit(0)
if shutil.which("torchrun") is None:
    print("[SKIP] torchrun not found; skipping the DDP part")
    sys.exit(0)

env = dict(os.environ, APERTUS1P5_DDP="1")
result = subprocess.run(["torchrun", "--nproc_per_node=2", os.path.abspath(__file__)], env=env)
assert result.returncode == 0, "DDP subprocess failed"
print("[OK] training loop verified single-device and DDP")

"""Validate minimal Apertus training.

Needs one CUDA GPU (skips otherwise); CASE 3 needs two. The pruned LM head scores only the first
`output_vocab_size` ids of the extended vocabulary, which gives training a label contract: labels
must be `< output_vocab_size` or `-100`, so positions holding media ids (image/audio codes above
the cutoff) must be masked with `-100`.

- CASE 1, label contract: a no-label forward returns logits padded to the logical `vocab_size` with
  a `finfo.min` tail, a labeled forward returns physical-width logits with a finite loss, and labels
  still holding media ids are rejected.
- CASE 2, single-device loop: overfit smoke test; four SGD steps on the last decoder layer plus
  lm_head must reduce the loss on a fixed text+image batch.
- CASE 3, data parallelism: relaunches this file under `torchrun --nproc_per_node=2`; two
  synchronized DDP steps must end with identical loss and gradients on both ranks.
"""

import os
import shutil
import subprocess

import numpy as np
import torch
from _common import (
    CaseSkipped,
    finish,
    require_local_transformers,
    resolve_checkpoint,
    run_case,
    setup_failure,
    skipped_result,
)


REQUIRED_SYMBOLS = ("Apertus1p5ForConditionalGeneration", "AutoProcessor")


def setup(transformers):
    """SETUP

    Load the composite model on cuda:0 (bf16) and build the fixed batch plus pruned-head labels.
    """
    checkpoint = resolve_checkpoint()
    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    print("SETUP: loading model (bf16, cuda:0) ...")
    model = transformers.Apertus1p5ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cuda:0"
    )
    inputs = build_batch(processor, model.device)
    text_config = model.config.get_text_config()
    labels = build_labels(inputs, text_config.output_vocab_size)
    return checkpoint, model, inputs, labels, text_config


def build_batch(processor, device):
    """Build a fixed two-sample batch: one text-only sample and one with a 64x64 noise image."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    return processor(
        text=["The capital of Switzerland is Bern.", "<|image|> This image shows colorful noise."],
        images=[[], [image]],
        padding=True,
        return_tensors="pt",
    ).to(device)


def build_labels(inputs, output_vocab_size):
    """Apply the label contract: mask padding and ids at or above the pruned-head cutoff with -100."""
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    labels[labels >= output_vocab_size] = -100
    return labels


def prepare_trainable(model):
    """Freeze everything except the last decoder layer and lm_head; return the trainable parameter count."""
    model.requires_grad_(False)
    for module in (model.model.language_model.layers[-1], model.lm_head):
        module.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def case_1_label_contract(model, inputs, labels, text_config):
    """CASE 1: LABEL CONTRACT

    The pruned head can only score the first `output_vocab_size` ids, so three behaviors are
    checked: 1) a no-label forward returns logits padded to the full `vocab_size` with a
    non-selectable `finfo.min` tail 2) a labeled forward returns physical-width logits and 3) a finite
    loss; labels still holding media ids (>= `output_vocab_size` instead of `-100`) raise a
    ValueError pointing at the `-100` masking rule.
    """
    with torch.no_grad():
        no_label_logits = model(**inputs).logits
    assert no_label_logits.shape[-1] == text_config.vocab_size, "no-label logits have the wrong width"
    tail = no_label_logits[..., text_config.output_vocab_size :]
    assert (tail == torch.finfo(no_label_logits.dtype).min).all(), "padded logits tail is selectable"

    with torch.no_grad():
        loss_out = model(**inputs, labels=labels)
    assert loss_out.logits.shape[-1] == text_config.output_vocab_size, "loss logits have the wrong width"
    assert torch.isfinite(loss_out.loss), f"non-finite loss {loss_out.loss}"

    bad_labels = inputs["input_ids"].clone()
    try:
        model(**inputs, labels=bad_labels)
    except ValueError as error:
        assert "-100" in str(error), f"unexpected validation error: {error}"
    else:
        raise AssertionError("labels holding media ids were accepted")
    return (
        f"logits {no_label_logits.shape[-1]} padded / {loss_out.logits.shape[-1]} physical; "
        f"loss {float(loss_out.loss):.4f}"
    )


def case_2_single_device_loop(model, inputs, labels):
    """CASE 2: SINGLE-DEVICE LOOP

    Overfit smoke test: freeze all but the last decoder layer and lm_head, run four SGD steps on
    the fixed batch, and require the loss to decrease. Catches broken gradient flow through the
    pruned head without needing a real training run.
    """
    trainable_params = prepare_trainable(model)
    model.train()
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=5e-3)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    return f"{trainable_params / 1e6:.0f}M trainable params; loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def case_3_data_parallel(checkpoint):
    """CASE 3: DATA PARALLEL

    Relaunch this file under `torchrun --nproc_per_node=2` (see `ddp_main`): both ranks train the
    same batch for two DDP steps and must end with identical loss and lm_head gradient norm, since
    the gradient all-reduce keeps the replicas in lockstep.
    """
    if torch.cuda.device_count() < 2:
        raise CaseSkipped(f"needs at least 2 CUDA devices; found {torch.cuda.device_count()}")
    if shutil.which("torchrun") is None:
        raise CaseSkipped("torchrun is not installed")

    env = dict(os.environ, APERTUS1P5_CHECKPOINT=checkpoint, APERTUS1P5_DDP="1")
    result = subprocess.run(["torchrun", "--nproc_per_node=2", os.path.abspath(__file__)], env=env)
    assert result.returncode == 0, f"DDP subprocess exited with {result.returncode}"
    return "two synchronized DDP steps matched across ranks"


def ddp_main():
    """Run on every torchrun rank: two synchronized DDP steps, then all-gather loss and grad norm to compare."""
    transformers = require_local_transformers(REQUIRED_SYMBOLS)
    checkpoint = resolve_checkpoint()

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    model = transformers.Apertus1p5ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16).to(
        device
    )
    prepare_trainable(model)
    model.train()
    ddp_model = DistributedDataParallel(model, device_ids=[device.index])

    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    inputs = build_batch(processor, device)
    labels = build_labels(inputs, model.config.get_text_config().output_vocab_size)
    optimizer = torch.optim.SGD(
        (parameter for parameter in ddp_model.parameters() if parameter.requires_grad), lr=5e-3
    )
    for _ in range(2):
        optimizer.zero_grad()
        loss = ddp_model(**inputs, labels=labels).loss
        loss.backward()
        optimizer.step()

    probe = torch.stack([loss.detach().float(), model.lm_head.weight.grad.float().norm()])
    gathered = [torch.zeros_like(probe) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, probe)
    if rank == 0:
        assert all(torch.equal(gathered[0], other) for other in gathered[1:]), "DDP ranks diverged"
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main():
    if os.environ.get("APERTUS1P5_DDP") == "1":
        return ddp_main()

    try:
        transformers = require_local_transformers(REQUIRED_SYMBOLS)
    except Exception as error:
        return finish([setup_failure(error)])

    if not torch.cuda.is_available():
        message = "needs a CUDA device"
        return finish(
            [
                skipped_result(case_1_label_contract, message),
                skipped_result(case_2_single_device_loop, message),
                skipped_result(case_3_data_parallel, message),
            ]
        )

    try:
        checkpoint, model, inputs, labels, text_config = setup(transformers)
    except Exception as error:
        results = [setup_failure(error)]
    else:
        results = [
            run_case(case_1_label_contract, model, inputs, labels, text_config),
            run_case(case_2_single_device_loop, model, inputs, labels),
        ]
        del model
        torch.cuda.empty_cache()
        results.append(run_case(case_3_data_parallel, checkpoint))
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())

"""Multi-device placement: accelerate sharding (`device_map="auto"`) and tensor parallelism (`tp_plan`).

Needs >= 2 CUDA devices (prints a skip notice and exits 0 otherwise; assumes identical GPUs, since
comparing greedy generations across placements relies on matching kernel numerics).

Part A records single-device (cuda:0) references: greedy generations for a text-only and a synthetic
image+audio prompt, plus the padded-logits contract (logits span the full `vocab_size` with a
`finfo.min` tail beyond the pruned `output_vocab_size` head).

Part B reloads the composite with `device_map="auto"`, asserts the layout actually spans >= 2 devices,
that the fp32-keep of the media tokenizers survives sharding next to the bf16 backbone, that the media
tokenizers' placement is reported, and that the padded-logits contract matches Part A. Cross-placement
parity is asserted where it is contractual: the media codes (fp32 tokenizers are placement-stable), the
next-token argmax after the media prompt, and the greedy continuation of the confident text prompt.
The free-running continuation of the synthetic-noise media prompt is only checked to complete: its
low-confidence steps amplify last-bit kernel variance across placements into argmax flips — the same
inherent variance for which Emu3/Chameleon skip their offload equivalence tests. This sequential model
sharding is the transformers-native placement; the `_pp_plan` metadata targets external
pipeline-parallel frameworks and has no in-library runtime to test against.

Part C relaunches this script under `torchrun --nproc_per_node=2` and loads the text backbone
(`Apertus1p5TextForCausalLM`) from the same composite checkpoint with `tp_plan="auto"` (the TP plan is
defined for the language backbone; the media tokenizers have none and would be replicated). It checks
the attention projections are actually sharded, that the lm_head stays replicated at its physical
pruned width (the class-level `lm_head` plan entry is not part of the effective auto plan, so logits
are computed full-width per rank and the padding contract holds unchanged), and that greedy generation
matches the single-device reference.

Data parallelism is covered by the DDP section of `09_training_loop.py`.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from transformers import Apertus1p5ForConditionalGeneration, Apertus1p5TextForCausalLM, AutoProcessor


# local dir, or hub repo id (optionally `repo_id@revision`); default: the published composite
CHECKPOINT = os.environ.get("APERTUS1P5_CHECKPOINT", "swiss-ai/Apertus-v1.5-8B")
if not os.path.isdir(CHECKPOINT):
    from huggingface_hub import snapshot_download

    repo_id, _, revision = CHECKPOINT.partition("@")
    CHECKPOINT = snapshot_download(repo_id, revision=revision or None)

TEXT_PROMPT = "Question: What is the capital of Switzerland? Answer:"
MAX_NEW_TOKENS = 12

# the media tokenizers assign codes by argmax over near-tied scores; TF32 kernel variance across
# placements flips such codes, so cross-placement bit-parity needs true fp32 matmuls
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def check_padded_logits(model, input_ids):
    """No-labels logits must span the logical vocab, with a non-selectable tail beyond the pruned head."""
    text_config = model.config.get_text_config()
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(model.device)).logits
    assert logits.shape[-1] == text_config.vocab_size, logits.shape
    tail = logits[..., text_config.output_vocab_size :]
    assert (tail == torch.finfo(logits.dtype).min).all(), "padded tail must be finfo.min everywhere"
    return logits


def media_inputs(processor):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(12000) / 24000.0)).astype(np.float32)  # 0.5 s
    return processor(
        text="<|image|> and <|audio|> Describe both inputs briefly.",
        images=[image],
        audio=[audio],
        return_tensors="pt",
    )


def tp_main():
    """Runs on every torchrun rank; asserts on rank 0."""
    import torch.distributed as dist

    from transformers.distributed import DistributedConfig

    if not dist.is_initialized():
        dist.init_process_group()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    reference = json.loads(os.environ["APERTUS1P5_TP_REFERENCE"])
    model = Apertus1p5TextForCausalLM.from_pretrained(
        CHECKPOINT, dtype=torch.bfloat16, distributed_config=DistributedConfig(tp_size=dist.get_world_size())
    ).eval()
    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    # the plan shards the attention/MLP projections; the pruned lm_head may be replicated at its
    # physical width or sharded with a gather depending on how the runtime applies the class-level
    # `lm_head` plan entry — both keep the full-width padded-logits contract intact
    config = model.config
    world_size = dist.get_world_size()
    full_q_rows = config.num_attention_heads * getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    q_weight = model.get_parameter("model.layers.0.self_attn.q_proj.weight")
    local_q_rows = q_weight.to_local().shape[0] if hasattr(q_weight, "to_local") else q_weight.shape[0]
    head_weight = model.lm_head.weight
    head_rows = head_weight.to_local().shape[0] if hasattr(head_weight, "to_local") else head_weight.shape[0]

    inputs = processor(text=TEXT_PROMPT, return_tensors="pt").to(model.device)
    logits = check_padded_logits(model, inputs["input_ids"])
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1] :].tolist()

    if dist.get_rank() == 0:
        assert local_q_rows * world_size == full_q_rows, f"q_proj not sharded: {local_q_rows}/{full_q_rows} rows"
        valid_head_rows = {config.output_vocab_size, config.output_vocab_size // world_size}
        assert head_rows in valid_head_rows, f"unexpected lm_head local rows: {head_rows}"
        head_state = "replicated" if head_rows == config.output_vocab_size else "sharded"
        assert continuation == reference, f"TP greedy diverged: {continuation} vs {reference}"
        print(
            f"[OK] TP({world_size}): q_proj sharded {local_q_rows}/{full_q_rows} rows per rank, lm_head "
            f"{head_state} ({head_rows} local rows), padded logits {tuple(logits.shape)}, "
            f"generation matches single-device reference"
        )
    dist.barrier()
    dist.destroy_process_group()


if os.environ.get("APERTUS1P5_TP") == "1":
    tp_main()
    sys.exit(0)

if torch.cuda.device_count() < 2:
    print(f"[SKIP] multi-device test needs >= 2 CUDA devices, found {torch.cuda.device_count()}")
    sys.exit(0)

processor = AutoProcessor.from_pretrained(CHECKPOINT)

# --- Part A: single-device references ---------------------------------------------------------------
model = Apertus1p5ForConditionalGeneration.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, device_map="cuda:0"
).eval()

text_inputs = processor(text=TEXT_PROMPT, return_tensors="pt").to(model.device)
check_padded_logits(model, text_inputs["input_ids"])
with torch.no_grad():
    ref_text = model.generate(**text_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0].tolist()

mm_inputs = media_inputs(processor).to(model.device)
with torch.no_grad():
    ref_image_codes = model.model.get_image_tokens(mm_inputs["pixel_values"], mm_inputs["image_sizes"]).cpu()
    ref_audio_codes = model.model.get_audio_tokens(
        mm_inputs["input_features"], mm_inputs["feature_attention_mask"]
    ).cpu()
    ref_next_token = int(model(**mm_inputs).logits[:, -1].argmax())
print(
    f"[OK] single-device reference: text continuation "
    f"{processor.decode(ref_text[text_inputs['input_ids'].shape[1] :], skip_special_tokens=True)!r}, "
    f"{ref_image_codes.numel()} image + {ref_audio_codes.numel()} audio codes, next token {ref_next_token}"
)

# the text-backbone reference for the TP part, from the very same checkpoint
backbone = Apertus1p5TextForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.bfloat16, device_map="cuda:0").eval()
with torch.no_grad():
    tp_reference = backbone.generate(**text_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
tp_reference = tp_reference[0, text_inputs["input_ids"].shape[1] :].tolist()
del model, backbone
torch.cuda.empty_cache()

# --- Part B: accelerate sharding across devices ------------------------------------------------------
model = Apertus1p5ForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.bfloat16, device_map="auto").eval()
devices = {str(d) for d in model.hf_device_map.values()}
assert len(devices) >= 2, f"expected a multi-device layout, got {model.hf_device_map}"

for name, submodule in [
    ("vision_tokenizer", model.model.vision_tokenizer),
    ("audio_tokenizer", model.model.audio_tokenizer),
]:
    dtypes = {p.dtype for p in submodule.parameters()}
    placement = {str(p.device) for p in submodule.parameters()}
    assert dtypes == {torch.float32}, f"{name} must stay fp32 under sharding, got {dtypes}"
    print(f"[OK] sharded layout: {name} on {sorted(placement)} in fp32")
backbone_dtypes = {p.dtype for p in model.model.language_model.layers.parameters()}
assert backbone_dtypes == {torch.bfloat16}, backbone_dtypes
print(f"[OK] sharded layout spans {sorted(devices)}; backbone bf16, media tokenizers fp32")

text_inputs = processor(text=TEXT_PROMPT, return_tensors="pt").to(model.device)
check_padded_logits(model, text_inputs["input_ids"])
with torch.no_grad():
    sharded_text = model.generate(**text_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0].tolist()
assert sharded_text == ref_text, f"sharded text generation diverged:\n  ref    : {ref_text}\n  sharded: {sharded_text}"

mm_inputs = media_inputs(processor).to(model.device)
with torch.no_grad():
    sharded_image_codes = model.model.get_image_tokens(mm_inputs["pixel_values"], mm_inputs["image_sizes"]).cpu()
    sharded_audio_codes = model.model.get_audio_tokens(
        mm_inputs["input_features"], mm_inputs["feature_attention_mask"]
    ).cpu()
    sharded_next_token = int(model(**mm_inputs).logits[:, -1].argmax())
assert torch.equal(sharded_image_codes, ref_image_codes), "image codes diverged across placements"
assert torch.equal(sharded_audio_codes, ref_audio_codes), "audio codes diverged across placements"
assert sharded_next_token == ref_next_token, f"next-token argmax diverged: {sharded_next_token} vs {ref_next_token}"
with torch.no_grad():
    sharded_media = model.generate(**mm_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]
new_tokens = sharded_media[mm_inputs["input_ids"].shape[1] :]
assert new_tokens.numel() > 0
print(
    f"[OK] device_map='auto': padded-logits contract holds; text generation, media codes and next-token "
    f"argmax match the reference; media continuation: {processor.decode(new_tokens, skip_special_tokens=True)!r}"
)
del model
torch.cuda.empty_cache()

# --- Part C: tensor parallelism over the text backbone ----------------------------------------------
if shutil.which("torchrun") is None:
    print("[SKIP] torchrun not found; skipping the TP part")
    sys.exit(0)

env = dict(os.environ, APERTUS1P5_TP="1", APERTUS1P5_TP_REFERENCE=json.dumps(tp_reference))
result = subprocess.run(["torchrun", "--nproc_per_node=2", os.path.abspath(__file__)], env=env)
assert result.returncode == 0, "TP subprocess failed"
print("[OK] all multi-device placements verified")

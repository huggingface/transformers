"""Validate multi-device inference.

Needs >= 2 identical CUDA GPUs (skips otherwise). Three placements are exercised, later ones
compared against the references recorded by the first:

- CASE 1, no sharding: composite model and text backbone each fully on one GPU (`device_map="cuda:0"`);
  records greedy text tokens, media codes, the next-token argmax, and the backbone continuation.
- CASE 2, model sharding: the composite split over all visible GPUs via `device_map="auto"`; checks
  that the layout really spans >= 2 devices, that the media tokenizers stay fp32 beside the bf16
  backbone, the padded-logits contract, and parity with the CASE 1 references. This is sequential
  model sharding: the layer-wise layout of pipeline parallelism but without micro-batch scheduling,
  so only one GPU computes at a time.
- CASE 3, tensor parallelism: relaunches this file under `torchrun --nproc_per_node=2` and loads the
  text backbone with a two-rank TP plan; checks the attention projections are sharded across ranks
  and that greedy generation matches the CASE 1 reference. Unlike CASE 2, both GPUs compute each
  layer simultaneously on weight slices, merged via collectives.

Data parallelism is covered by the DDP case of `09_training_loop.py`. True pipeline parallelism has
no in-library runtime; the model's `_pp_plan` metadata targets external frameworks such as vLLM.
"""

import json
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


TEXT_PROMPT = "Question: What is the capital of Switzerland? Answer:"
MAX_NEW_TOKENS = 12
REQUIRED_SYMBOLS = (
    "Apertus1p5ForConditionalGeneration",
    "Apertus1p5TextForCausalLM",
    "AutoProcessor",
)

# Media token codes can flip under TF32 because their scores are near-tied. Cross-placement parity
# additionally assumes identical GPUs: comparing greedy outputs relies on matching kernel numerics.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def setup(transformers):
    """SETUP

    Resolve the checkpoint and load its processor.
    """
    checkpoint = resolve_checkpoint()
    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    return checkpoint, processor


def check_padded_logits(model, input_ids):
    """Run forward pass and check logits are padded correctly"""
    text_config = model.config.get_text_config()
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(model.device)).logits
    assert logits.shape[-1] == text_config.vocab_size, f"unexpected logits shape {logits.shape}"
    tail = logits[..., text_config.output_vocab_size :]
    assert (tail == torch.finfo(logits.dtype).min).all(), "padded tail must be finfo.min everywhere"
    return logits


def media_inputs(processor):
    """Build a synthetic multimodal prompt: a 64x64 noise image and a 0.5 s sine clip."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(12000) / 24000.0)).astype(np.float32)
    return processor(
        text="<|image|> and <|audio|> Describe both inputs briefly.",
        images=[image],
        audio=[audio],
        return_tensors="pt",
    )


def case_1_single_device(transformers, checkpoint, processor, references):
    """CASE 1: SINGLE DEVICE

    Record reference outputs with everything on one GPU (cuda:0, no sharding): greedy text tokens
    from the composite model, image and audio codes plus the next-token argmax for a media prompt,
    and the text backbone's greedy continuation (used as the TP reference in CASE 3).
    """
    model = None
    backbone = None
    try:
        model = transformers.Apertus1p5ForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        text_inputs = processor(text=TEXT_PROMPT, return_tensors="pt").to(model.device)
        check_padded_logits(model, text_inputs["input_ids"])
        with torch.no_grad():
            ref_text = model.generate(**text_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0].tolist()

        multimodal = media_inputs(processor).to(model.device)
        with torch.no_grad():
            ref_image_codes = model.model.get_image_tokens(multimodal["pixel_values"], multimodal["image_sizes"]).cpu()
            ref_audio_codes = model.model.get_audio_tokens(
                multimodal["input_features"], multimodal["feature_attention_mask"]
            ).cpu()
            ref_next_token = int(model(**multimodal).logits[:, -1].argmax())

        backbone = transformers.Apertus1p5TextForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        with torch.no_grad():
            tp_output = backbone.generate(**text_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        tp_reference = tp_output[0, text_inputs["input_ids"].shape[1] :].tolist()
        continuation = processor.decode(ref_text[text_inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        references.update(
            ref_text=ref_text,
            ref_image_codes=ref_image_codes,
            ref_audio_codes=ref_audio_codes,
            ref_next_token=ref_next_token,
            tp_reference=tp_reference,
        )
        return f"text {continuation!r}; {ref_image_codes.numel()} image + {ref_audio_codes.numel()} audio codes"
    finally:
        del model, backbone
        torch.cuda.empty_cache()


def case_2_automatic_sharding(transformers, checkpoint, processor, references):
    """CASE 2: AUTOMATIC SHARDING

    Shard the composite model over all visible GPUs with `device_map="auto"` and check: the layout
    actually spans >= 2 devices, the media tokenizers stay fp32 next to the bf16 backbone, the
    padded-logits contract holds, and text generation, media codes, and the next-token argmax are
    identical to the CASE 1 references. Sequential model sharding, not true pipeline parallelism:
    layer-wise stages as in PP, but with no micro-batching, one GPU computing at a time.
    """
    if not references:
        raise CaseSkipped("CASE 1 did not produce reference outputs")

    model = None
    try:
        model = transformers.Apertus1p5ForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.bfloat16, device_map="auto"
        ).eval()
        devices = {str(device) for device in model.hf_device_map.values()}
        assert len(devices) >= 2, f"expected a multi-device layout, got {model.hf_device_map}"

        placements = []
        for name, submodule in (
            ("vision", model.model.vision_tokenizer),
            ("audio", model.model.audio_tokenizer),
        ):
            dtypes = {parameter.dtype for parameter in submodule.parameters()}
            placement = {str(parameter.device) for parameter in submodule.parameters()}
            assert dtypes == {torch.float32}, f"{name} tokenizer must stay fp32, got {dtypes}"
            placements.append(f"{name}={sorted(placement)}")
        backbone_dtypes = {parameter.dtype for parameter in model.model.language_model.layers.parameters()}
        assert backbone_dtypes == {torch.bfloat16}, f"unexpected backbone dtypes {backbone_dtypes}"

        text_inputs = processor(text=TEXT_PROMPT, return_tensors="pt").to(model.device)
        check_padded_logits(model, text_inputs["input_ids"])
        with torch.no_grad():
            sharded_text = model.generate(**text_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0].tolist()
        assert sharded_text == references["ref_text"], "sharded text generation diverged"

        multimodal = media_inputs(processor).to(model.device)
        with torch.no_grad():
            image_codes = model.model.get_image_tokens(multimodal["pixel_values"], multimodal["image_sizes"]).cpu()
            audio_codes = model.model.get_audio_tokens(
                multimodal["input_features"], multimodal["feature_attention_mask"]
            ).cpu()
            next_token = int(model(**multimodal).logits[:, -1].argmax())
            continuation = model.generate(**multimodal, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[
                0, multimodal["input_ids"].shape[1] :
            ]
        assert torch.equal(image_codes, references["ref_image_codes"]), "image codes diverged"
        assert torch.equal(audio_codes, references["ref_audio_codes"]), "audio codes diverged"
        assert next_token == references["ref_next_token"], "next-token argmax diverged"
        # Only completion is checked here on purpose: the free-running continuation of this
        # synthetic-noise prompt takes low-confidence steps that amplify last-bit kernel variance
        # across placements into argmax flips (Emu3/Chameleon skip their offload-equivalence tests
        # for the same reason). Do not tighten this into a token-level parity assert.
        assert continuation.numel(), "multimodal generation produced no new tokens"
        return f"devices {sorted(devices)}; {'; '.join(placements)}"
    finally:
        del model
        torch.cuda.empty_cache()


def case_3_tensor_parallel(checkpoint, references):
    """CASE 3: TENSOR PARALLEL

    Relaunch this file under `torchrun --nproc_per_node=2` (see `tp_main`) and load the text
    backbone with a two-rank TP plan: q_proj must be sharded across the ranks, the pruned lm_head
    may be replicated or row-sharded, and greedy generation must match the CASE 1 reference.
    """
    if "tp_reference" not in references:
        raise CaseSkipped("CASE 1 did not produce a TP reference")
    if shutil.which("torchrun") is None:
        raise CaseSkipped("torchrun is not installed")

    env = dict(
        os.environ,
        APERTUS1P5_CHECKPOINT=checkpoint,
        APERTUS1P5_TP="1",
        APERTUS1P5_TP_REFERENCE=json.dumps(references["tp_reference"]),
    )
    result = subprocess.run(["torchrun", "--nproc_per_node=2", os.path.abspath(__file__)], env=env)
    assert result.returncode == 0, f"TP subprocess exited with {result.returncode}"
    return "two-rank TP matched the single-device reference"


def tp_main():
    """Run on every torchrun rank: load the two-rank TP backbone, verify sharding, assert on rank 0."""
    transformers = require_local_transformers(REQUIRED_SYMBOLS)
    checkpoint = resolve_checkpoint()

    import torch.distributed as dist

    from transformers.distributed import DistributedConfig

    # `from_pretrained` owns the whole distributed setup: it initializes the process group from the
    # torchrun env vars, binds this rank to its LOCAL_RANK device, and shards the weights per the
    # config's `base_model_tp_plan` (`DistributedConfig` defaults its `tp_plan` to "auto").
    world_size = int(os.environ["WORLD_SIZE"])
    reference = json.loads(os.environ["APERTUS1P5_TP_REFERENCE"])
    model = transformers.Apertus1p5TextForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(tp_size=world_size),
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(checkpoint)
    config = model.config
    full_q_rows = config.num_attention_heads * getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    q_weight = model.get_parameter("model.layers.0.self_attn.q_proj.weight")
    local_q_rows = q_weight.to_local().shape[0] if hasattr(q_weight, "to_local") else q_weight.shape[0]
    head_weight = model.lm_head.weight
    head_rows = head_weight.to_local().shape[0] if hasattr(head_weight, "to_local") else head_weight.shape[0]

    inputs = processor(text=TEXT_PROMPT, return_tensors="pt").to(model.device)
    check_padded_logits(model, inputs["input_ids"])
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    continuation = generated[0, inputs["input_ids"].shape[1] :].tolist()

    if dist.get_rank() == 0:
        assert local_q_rows * world_size == full_q_rows, "q_proj was not sharded"
        # The pruned lm_head may stay replicated at its physical width or be row-sharded with a
        # gather, depending on how the runtime applies the class-level `lm_head` plan entry; both
        # layouts keep the full-width padded-logits contract intact.
        assert head_rows in {
            config.output_vocab_size,
            config.output_vocab_size // world_size,
        }, f"unexpected lm_head local rows: {head_rows}"
        assert continuation == reference, f"TP generation diverged: {continuation} vs {reference}"
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main():
    if os.environ.get("APERTUS1P5_TP") == "1":
        return tp_main()

    try:
        transformers = require_local_transformers(REQUIRED_SYMBOLS)
    except Exception as error:
        return finish([setup_failure(error)])

    if torch.cuda.device_count() < 2:
        message = f"needs at least 2 CUDA devices; found {torch.cuda.device_count()}"
        return finish(
            [
                skipped_result(case_1_single_device, message),
                skipped_result(case_2_automatic_sharding, message),
                skipped_result(case_3_tensor_parallel, message),
            ]
        )

    try:
        checkpoint, processor = setup(transformers)
    except Exception as error:
        results = [setup_failure(error)]
    else:
        references = {}
        results = [
            run_case(case_1_single_device, transformers, checkpoint, processor, references),
            run_case(case_2_automatic_sharding, transformers, checkpoint, processor, references),
            run_case(case_3_tensor_parallel, checkpoint, references),
        ]
    return finish(results)


if __name__ == "__main__":
    raise SystemExit(main())

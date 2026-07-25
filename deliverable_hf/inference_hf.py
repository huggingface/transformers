"""Simple HF inference for the Onyx model: chat-formatted generation with optional
image / video.

Nothing downstream is reimplemented here. Tokenization + media-token expansion +
image/video preprocessing all go through OnyxProcessor (via AutoProcessor), which
owns OnyxImageProcessor (images) and OnyxVideoProcessor (video frame sampling +
per-group timestamps, torchcodec / training-faithful) -- the same preprocessing
code validate_nll_hf.py drives. Generation uses the model's native generate()
(OnyxForCausalLM.prepare_inputs_for_generation forwards pixel_values on the
prefill step, and OnyxModel derives the vision mask and moves/casts the pixel
tensors itself). This script only builds the prompt, calls those, and prints --
mirroring validate_nll_hf.py, minus the NLL scoring.

Usage:
    # Text-only
    CUDA_VISIBLE_DEVICES=0 python hf/inference_hf.py \
        --hf_dir ~/models/onyx-hf \
        --prompt "The meaning of life is"

    # With image (use <img> as placeholder)
    CUDA_VISIBLE_DEVICES=0 python hf/inference_hf.py \
        --hf_dir ~/models/onyx-hf \
        --image photo.jpg \
        --prompt "Describe this image: <img>"

    # With video (use <video> as placeholder; requires torchcodec + ffmpeg)
    CUDA_VISIBLE_DEVICES=0 python hf/inference_hf.py \
        --hf_dir ~/models/onyx-hf \
        --video clip.mp4 \
        --prompt "Describe this video: <video>"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

# Canned response emitted in place of the model output when the streaming safety
# probe flags the response as unsafe.
_SAFETY_REFUSAL = "I'm sorry, but I can't help with that request."

# Provisional max-EWMA logit threshold used when the caller does not pass
# --safety_threshold. The probe ships with an UNCALIBRATED threshold (0.0 in
# safety_probe_metadata.json), which is too low to gate on -- even safe
# responses score positive, so 0.0 would refuse nearly everything.
#
# This is a STOPGAP, NOT a calibrated operating point. It is per-checkpoint and
# does not transfer across probes/betas. For rc1__v07e (beta 0.15), the 100
# SAFE reference docs (reference_probe_rc1_v07e_wildguardmix_test_100.pt) span
# max-EWMA -790..+1499 (p50 -168, p95 +771); we have NO unsafe scores at this
# beta, so the upper edge of the separating band is unknown. 1500 sits just
# above the observed safe max, but the real operating point MUST be calibrated
# by the probe team on a labeled (safe+unsafe) eval set before this gates.
_PROVISIONAL_SAFETY_THRESHOLD = 1500.0


def _load_safety_probe(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a linear safety probe (weight [1, L*H], bias [1]) from a local .pt.

    Self-contained: plain local file, weights_only load (no pickle-RCE surface),
    no internal deps -- safe to ship in the HF deliverable.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt
    if isinstance(ckpt, dict) and not ("weight" in ckpt and "bias" in ckpt):
        for key in ("probe_state_dict", "model_state_dict"):
            inner = ckpt.get(key)
            if isinstance(inner, dict) and "weight" in inner and "bias" in inner:
                state = inner
                break
    if not (isinstance(state, dict) and "weight" in state and "bias" in state):
        raise ValueError(
            f"Safety probe {path} must contain weight/bias "
            "(optionally under probe_state_dict/model_state_dict)."
        )
    return state["weight"].float(), state["bias"].float()


def _default_probe_threshold(hf_dir: str, fallback: float) -> float:
    """Threshold to use when --safety_threshold is not passed.

    Reads safety_probe_metadata.json, but ONLY trusts a strictly-positive
    value -- the shipped metadata carries an uncalibrated 0.0, which would
    refuse nearly everything. A non-positive/missing metadata threshold falls
    back to the provisional stopgap.
    """
    meta_path = Path(hf_dir) / "safety_probe_metadata.json"
    if not meta_path.is_file():
        return fallback
    try:
        meta = json.loads(meta_path.read_text())
        probes = meta.get("probes", {})
        for spec in probes.values():
            thr = spec.get("safety_probe_threshold")
            if thr is not None and float(thr) > 0.0:
                return float(thr)
    except (OSError, ValueError, TypeError):
        pass
    return fallback


def _get_decoder_layers(model: object) -> object:
    """Return the text decoder's ModuleList of layers, across Onyx architectures.

    The safety probe concatenates per-layer hidden states, so it must reach the
    text decoder layers regardless of how the checkpoint's architecture nests them:

      - Modular arch (target, ``OnyxForConditionalGeneration``):
            model.model (OnyxModel) -> .language_model (OnyxTextModel) -> .layers
      - Text-only causal arch (``OnyxForCausalLM``):
            model.model (OnyxTextModel) -> .layers

    Probe order over layers must match training (layer 0..N-1), which .layers preserves.
    """
    for attr_path in (
        ("model", "language_model", "layers"),  # OnyxForConditionalGeneration (modular target)
        ("model", "layers"),                    # OnyxForCausalLM (text-only)
        ("language_model", "layers"),           # defensive: bare OnyxModel
    ):
        obj = model
        ok = True
        for attr in attr_path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and obj is not None:
            return obj
    raise AttributeError(
        "Could not locate the text decoder layers on the model. Tried "
        "model.model.language_model.layers, model.model.layers, "
        "model.language_model.layers. The safety probe needs per-layer hidden states."
    )


def _max_ewma_response_logit(
    output_ids: torch.Tensor,
    resp_start: int,
    *,
    model: object,
    weight: torch.Tensor,
    bias: torch.Tensor,
    beta: float,
    pixel_values: object = None,
    attention_mask: torch.Tensor | None = None,
) -> float:
    """Score the response with the streaming probe: max_t EWMA(z_t) over raw logits.

    One teacher-forced forward pass over the full sequence (causal attention makes
    it identical to streaming generation) captures every decoder layer's output
    (post-block residual). For each response token, concatenate per-layer hidden
    states in layer order -> psi, z = psi @ W^T + b, then aggregate with the max
    of the EWMA (z~_0 = z_0; z~_t = beta*z_t + (1-beta)*z~_{t-1}). Raw logits only
    -- no sigmoid on the decision path (avoids float underflow).

    pixel_values/attention_mask MUST be forwarded when the prompt was multimodal:
    output_ids still contains image/video sentinel tokens, so a text-only re-run
    would produce hidden states that differ from what generate() actually used.
    """
    layers = _get_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            captured[idx] = (out[0] if isinstance(out, (tuple, list)) else out).detach()

        return hook

    handles = []
    try:
        for i in range(len(layers)):
            handles.append(layers[i].register_forward_hook(make_hook(i)))
        forward_kwargs: dict[str, object] = {}
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            model(output_ids, **forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    seq_len = output_ids.shape[1]
    if seq_len <= resp_start:
        return float("-inf")
    resp_pos = list(range(resp_start, seq_len))
    psi = torch.cat(
        [captured[i][0, resp_pos, :] for i in range(len(layers))], dim=-1
    ).float()
    logits = (
        (psi @ weight.t().to(psi.device)).squeeze(-1) + bias.to(psi.device)
    ).tolist()

    current = float(logits[0])
    best = current
    for logit in logits[1:]:
        current = beta * float(logit) + (1.0 - beta) * current
        best = max(best, current)
    return best


def build_user_content(
    prompt: str, has_image: bool, has_video: bool
) -> str | list[dict]:
    """Turn a prompt with an <img>/<video> placeholder into chat-template content.

    Text-only -> the raw string. With one media placeholder -> a list of parts
    ({"type": "text"|"image"|"video"}) in document order, so the media lands at the
    placeholder's position; the chat template renders the image/video part as the
    single sentinel that OnyxProcessor expands.
    """
    if not (has_image or has_video):
        return prompt
    placeholder, media_type = ("<img>", "image") if has_image else ("<video>", "video")
    parts: list[dict] = []
    chunks = prompt.split(placeholder)
    for i, chunk in enumerate(chunks):
        if chunk:
            parts.append({"type": "text", "text": chunk})
        if i < len(chunks) - 1:
            parts.append({"type": media_type})
    return parts


def main():
    parser = argparse.ArgumentParser(description="HF inference for Onyx")
    parser.add_argument("--hf_dir", required=True, help="Path to HF model directory")
    parser.add_argument(
        "--prompt",
        default="The meaning of life is",
        help="Text prompt (use <img> / <video> for vision placeholders)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt; pass '' to omit (a default placeholder is used).",
    )
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument(
        "--video", default=None, help="Path to video file (use <video> placeholder)"
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    parser.add_argument(
        "--device", default="auto", help="Device map: 'auto', 'cuda:0', etc."
    )
    parser.add_argument(
        "--safety_probe",
        default=None,
        help="Path to a linear safety-probe .pt (weight [1, L*H], bias [1]). When "
        "set, the response is scored and replaced with a canned refusal if the "
        "probe flags it as unsafe.",
    )
    parser.add_argument(
        "--safety_beta",
        type=float,
        default=0.15,
        help="EWMA decay for the safety probe (per-checkpoint; 0.15 for "
        "rc1__v07e, 0.3 for the older rl_v1).",
    )
    parser.add_argument(
        "--safety_threshold",
        type=float,
        default=None,
        help="Max-EWMA logit threshold; response is refused when the score meets "
        "or exceeds it. Defaults to a strictly-positive value in "
        "safety_probe_metadata.json, else a provisional stopgap (uncalibrated; "
        "see _PROVISIONAL_SAFETY_THRESHOLD).",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.hf_dir}")
    t0 = time.time()
    # Modular Onyx ships as OnyxForConditionalGeneration (image-text-to-text); the
    # older text-only export is OnyxForCausalLM. Try the modular loader first, then
    # fall back so this script works against either deliverable.
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.hf_dir, **load_kwargs)
    except (ValueError, KeyError, OSError):
        model = AutoModelForCausalLM.from_pretrained(args.hf_dir, **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.hf_dir, trust_remote_code=True)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    has_image = args.image is not None and "<img>" in args.prompt
    has_video = args.video is not None and "<video>" in args.prompt

    # Read media through OnyxProcessor's shipped sub-processors: images as PIL;
    # video as a path that OnyxProcessor decodes + groups via OnyxVideoProcessor
    # (torchcodec, training-faithful) -- the same path validate_nll_hf.py uses.
    images = videos = None
    if has_image:
        images = [Image.open(args.image).convert("RGB")]
        print(f"Image: {args.image} ({images[0].width}x{images[0].height})")
    elif has_video:
        videos = [args.video]
        print(f"Video: {args.video}")

    # One user turn: the chat template emits BOS + the trailing assistant
    # generation prompt and renders each media part as a sentinel, which
    # processor(...) expands into the full span and returns pixel_values for.
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append(
        {
            "role": "user",
            "content": build_user_content(args.prompt, has_image, has_video),
        }
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    # pixel_values is a list of variable-size tensors; pass it straight through --
    # the vision encoder moves + casts each tensor to its own device/dtype.
    pixel_values = inputs.get("pixel_values")

    print(f"Input: {input_ids.shape[1]} tokens")
    print("Generating...")
    do_sample = args.temperature > 0
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = args.temperature

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **gen_kwargs,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0, input_ids.shape[1] :]
    n_tokens = len(generated_ids)

    # The generation prompt ends at "<|start|>assistant", so the model emits the
    # chat channel header (e.g. " to=user") as plain text before "<|message|>".
    # skip_special_tokens drops "<|message|>"/"<|eot|>" but not that header, so
    # slice past the last "<|message|>" to keep only the final answer channel.
    message_id = processor.tokenizer.convert_tokens_to_ids("<|message|>")
    gen_list = generated_ids.tolist()
    if message_id in gen_list:
        content_ids = generated_ids[len(gen_list) - gen_list[::-1].index(message_id) :]
    else:
        content_ids = generated_ids
    text_out = processor.tokenizer.decode(content_ids, skip_special_tokens=True)

    # Streaming safety probe: score the generated response and refuse if the
    # probe flags it. Non-streaming here (we score the full response after
    # generate), so the refusal cleanly replaces the whole output.
    probe_score = None
    if args.safety_probe is not None:
        weight, bias = _load_safety_probe(args.safety_probe)
        threshold = (
            args.safety_threshold
            if args.safety_threshold is not None
            else _default_probe_threshold(
                args.hf_dir, fallback=_PROVISIONAL_SAFETY_THRESHOLD
            )
        )
        probe_score = _max_ewma_response_logit(
            output_ids,
            input_ids.shape[1],
            model=model,
            weight=weight,
            bias=bias,
            beta=args.safety_beta,
            # Forward the vision inputs so multimodal prompts rescore against the
            # same features generate() used. output_ids is a single unpadded
            # sequence, so the scoring-pass mask is all ones over its length.
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(output_ids),
        )
        if probe_score >= threshold:
            text_out = _SAFETY_REFUSAL

    print(f"\n{'=' * 60}")
    print(f"PROMPT: {args.prompt}")
    print(f"{'=' * 60}")
    if probe_score is not None:
        print(f"SAFETY PROBE: max_ewma={probe_score:.2f} (threshold={threshold:.2f})")
    print(f"OUTPUT: {text_out}")
    print(f"{'=' * 60}")
    print(
        f"Generated {n_tokens} tokens in {elapsed:.2f}s ({n_tokens / elapsed:.1f} tok/s)"
    )


if __name__ == "__main__":
    main()

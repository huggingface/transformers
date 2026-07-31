# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""
Assemble a self-contained Apertus 1.5 checkpoint from three converted sources:

1. An Apertus 1.5 causal-LM backbone with the enlarged input vocabulary. Released checkpoints use a pruned,
   text-only LM head whose physical width is recorded by `output_vocab_size`.
2. The encode-only EMU3.5 vision tokenizer, converted from `BAAI/Emu3.5-VisionTokenizer` into
   `Apertus1p5VisionTokenizerModel` format.
3. A WavTokenizer checkpoint produced by `convert_wavtokenizer_checkpoint.py`.

Weights are mapped into `Apertus1p5ForConditionalGeneration` as follows:

- backbone `model.X` -> `model.language_model.X`; `lm_head.weight` remains at the top level
- vision `X` -> `model.vision_tokenizer.X`
- audio `X` -> `model.audio_tokenizer.X`

For a tied backbone without `lm_head.weight`, the tie setting is copied to the composite config. The output
contains all three weight sets in source-grouped safetensor shards, the merged config, and the processor stack
derived from the backbone tokenizer. `--processor_only` refreshes only that stack, while `--verify` checks loading,
dtypes, token mappings, text generation, and processor-driven image/audio forwards.

Sources may be local directories or Hub identifiers in `repo_id` or `repo_id@revision` form.

Example:
    python src/transformers/models/apertus1p5/convert_apertus1p5_weights_to_hf.py \
        --apertus_checkpoint apertus-ai/Apertus-v1.5-8B-integration@refs/pr/1 \
        --vision_tokenizer_checkpoint /path/to/apertus1p5-visionvq-hf \
        --audio_tokenizer_checkpoint /path/to/wavtokenizer-large-unify-40token-hf \
        --output_dir /path/to/Apertus-1.5-8B-composite --verify
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from transformers import (
    Apertus1p5Config,
    Apertus1p5ForConditionalGeneration,
    Apertus1p5ImageProcessor,
    Apertus1p5Processor,
    AutoProcessor,
    AutoTokenizer,
    WavTokenizerFeatureExtractor,
    logging,
)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# The named media special-token attributes are expected to come from the backbone tokenizer itself
# (apertus-omni-tokenizer emits them in tokenizer_config.json's `extra_special_tokens`); the converter
# only warns when a legacy backbone lacks them, and the processor falls back to its built-in defaults.
NAMED_SPECIAL_TOKEN_ATTRIBUTES = (
    "image_token",
    "audio_token",
    "boi_token",
    "eoi_token",
    "image_wrapper_token",
    "eol_token",
    "boa_token",
    "eoa_token",
)

# Marker of the chat template's list-of-content-blocks user-message branch (the standard Transformers
# message format); templates without it only accept string or `{"parts": [...]}` user content.
_TEMPLATE_LIST_CONTENT_MARKER = "message.content is not string and message.content is not mapping"


def _has_valid_logits_layout(logits: torch.Tensor, output_vocab_size: int, vocab_size: int) -> bool:
    """Check that physical logits are finite and the padded input-only tail is non-generatable."""
    if logits.shape[-1] != vocab_size:
        return False
    tail = logits[..., output_vocab_size:]
    return bool(
        torch.isfinite(logits[..., :output_vocab_size]).all() and (tail == torch.finfo(logits.dtype).min).all()
    )


def _check_output_is_not_a_source(output_dir: str, *source_dirs: str) -> None:
    """Reject an output directory aliasing a source: the conversion overwrites `config.json` and deletes
    canonical weight files in `output_dir`, which would destroy the source checkpoint."""
    output_real = os.path.realpath(output_dir)
    for source in source_dirs:
        if os.path.realpath(source) == output_real:
            raise ValueError(
                f"`--output_dir` resolves to the same directory as the source checkpoint {source!r}; "
                "choose a separate output directory."
            )


def build_processor(apertus_checkpoint: str, audio_tokenizer_config: dict) -> Apertus1p5Processor:
    """Build the composite processor, warning when the backbone tokenizer lacks expected multimodal metadata."""
    tokenizer = AutoTokenizer.from_pretrained(apertus_checkpoint)
    missing = [name for name in NAMED_SPECIAL_TOKEN_ATTRIBUTES if getattr(tokenizer, name, None) is None]
    if missing:
        logger.warning(
            f"The backbone tokenizer does not declare the named media special tokens {missing} "
            "(it predates the apertus-omni-tokenizer `extra_special_tokens` emission). The processor "
            "falls back to its built-in defaults, but the converted tokenizer will not expose these "
            "named attributes."
        )
    # the tokenizer is taken as-is from the backbone; the converter only checks, never modifies
    if tokenizer.padding_side != "left":
        logger.warning(
            f"The backbone tokenizer uses padding_side={tokenizer.padding_side!r}; batched generation "
            "requires left padding. Fix the source tokenizer (apertus-omni-tokenizer pins left padding)."
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is not None and _TEMPLATE_LIST_CONTENT_MARKER not in chat_template:
        logger.warning(
            "The backbone chat template does not accept list-of-content-blocks user messages, so standard "
            "multimodal chat calls and media auto-loading will fail. Use a backbone with the updated "
            "apertus-omni-tokenizer template."
        )
    hop_length = int(math.prod(audio_tokenizer_config.get("upsampling_ratios", (6, 5, 5, 4))))
    feature_extractor = WavTokenizerFeatureExtractor(
        sampling_rate=audio_tokenizer_config.get("sampling_rate", 24000), hop_length=hop_length
    )
    return Apertus1p5Processor(
        image_processor=Apertus1p5ImageProcessor(),
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )


def build_config(
    apertus_checkpoint: str, vision_tokenizer_checkpoint: str, audio_tokenizer_checkpoint: str
) -> Apertus1p5Config:
    """Merge the three source configurations into an Apertus 1.5 composite configuration."""
    with open(os.path.join(apertus_checkpoint, "config.json")) as f:
        text_config = json.load(f)
    text_config.pop("architectures", None)
    text_config.pop("transformers_version", None)
    # the 1.5 backbone supersedes plain apertus; its config understands `output_vocab_size` (pruned LM head)
    if text_config.get("model_type", "apertus") == "apertus":
        text_config["model_type"] = "apertus1p5_text"
    with open(os.path.join(vision_tokenizer_checkpoint, "config.json")) as f:
        vision_tokenizer_config = json.load(f)
    with open(os.path.join(audio_tokenizer_checkpoint, "config.json")) as f:
        audio_tokenizer_config = json.load(f)
    # token ids and offsets: the Apertus1p5Config defaults are the verified values of the Apertus 1.5 tokenizer.
    # tie_word_embeddings must live on the composite's top-level config: it gates the lm_head <-> embed_tokens
    # tie, and tied backbones ship no `lm_head.weight` tensor.
    config = Apertus1p5Config(
        text_config=text_config,
        vision_tokenizer_config=vision_tokenizer_config,
        audio_tokenizer_config=audio_tokenizer_config,
        tie_word_embeddings=bool(text_config.get("tie_word_embeddings", False)),
    )
    # `model.save_pretrained` would stamp this from the model class; the converter streams shards without
    # instantiating the model, so it must set the entrypoint itself for `AutoModel` resolution
    config.architectures = [Apertus1p5ForConditionalGeneration.__name__]
    return config


def resolve_checkpoint_dir(path_or_repo_id: str) -> str:
    """Resolve a local directory or download a Hub checkpoint (`repo_id` or `repo_id@revision`) to the cache."""
    if os.path.isdir(path_or_repo_id):
        return path_or_repo_id
    repo_id, _, revision = path_or_repo_id.partition("@")
    logger.info(f"downloading {repo_id}" + (f" (revision {revision})" if revision else "") + " from the hub")
    return snapshot_download(repo_id, revision=revision or None)


def iter_source_shards(checkpoint_dir: str):
    """Yield (shard_filename, state_dict) for a single- or multi-shard safetensors checkpoint."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        for shard in sorted(set(index["weight_map"].values())):
            yield shard, load_file(os.path.join(checkpoint_dir, shard))
    else:
        yield "model.safetensors", load_file(os.path.join(checkpoint_dir, "model.safetensors"))


def _check_fp32_tokenizer_source(source: str, state_dict: dict[str, torch.Tensor]) -> None:
    """Half-precision tokenizer sources are already degraded (code assignment is a precision-sensitive argmax)
    and would silently pass `--verify`, whose fp32 check runs after the load-time upcast."""
    for key, value in state_dict.items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(
                f"The {source} source stores `{key}` in {value.dtype}; tokenizer weights must be stored in "
                "float32 (half-precision weights flip discrete codes)."
            )


def remapped_sources(apertus_checkpoint: str, vision_tokenizer_checkpoint: str, audio_tokenizer_checkpoint: str):
    """Yield (source_name, shard_name, remapped_state_dict) for all three weight sources."""
    for shard, state_dict in iter_source_shards(apertus_checkpoint):
        remapped = {}
        for key, value in state_dict.items():
            if key == "lm_head.weight":
                remapped[key] = value
            elif key.startswith("model."):
                remapped["model.language_model." + key[len("model.") :]] = value
            else:
                raise ValueError(f"Unexpected key in the Apertus backbone checkpoint: {key}")
        yield "apertus", shard, remapped
    for shard, state_dict in iter_source_shards(vision_tokenizer_checkpoint):
        _check_fp32_tokenizer_source("vision tokenizer", state_dict)
        yield "vision_tokenizer", shard, {f"model.vision_tokenizer.{key}": value for key, value in state_dict.items()}
    for shard, state_dict in iter_source_shards(audio_tokenizer_checkpoint):
        _check_fp32_tokenizer_source("audio tokenizer", state_dict)
        yield "wavtokenizer", shard, {f"model.audio_tokenizer.{key}": value for key, value in state_dict.items()}


def verify(composite_dir: str, max_new_tokens: int = 12):
    """Load a composite checkpoint and run configuration, dtype, processor, generation, and modality smoke checks."""
    failed_checks = []

    model, loading_info = Apertus1p5ForConditionalGeneration.from_pretrained(
        composite_dir, dtype=torch.bfloat16, output_loading_info=True
    )
    model = model.eval()
    config = model.config
    loading_problems = {kind: keys for kind, keys in loading_info.items() if keys}
    if loading_problems:
        raise RuntimeError(f"The composite checkpoint did not load cleanly: {loading_problems}")
    print("[PASS] load: no missing/unexpected/mismatched keys")

    # --- architectures: the AutoModel entrypoint the converter stamps into the config -----------------------
    architectures_ok = config.architectures == ["Apertus1p5ForConditionalGeneration"]
    if not architectures_ok:
        failed_checks.append("architectures")
    print(f"[{'PASS' if architectures_ok else 'FAIL'}] config architectures: {config.architectures}")

    dtypes = {
        "language_model": next(model.model.language_model.parameters()).dtype,
        "vision_tokenizer": next(model.model.vision_tokenizer.parameters()).dtype,
        "audio_tokenizer": next(model.model.audio_tokenizer.parameters()).dtype,
        "lm_head": model.lm_head.weight.dtype,
    }
    tokenizers_fp32 = dtypes["vision_tokenizer"] == torch.float32 and dtypes["audio_tokenizer"] == torch.float32
    if not tokenizers_fp32:
        failed_checks.append("dtypes")
    print(f"[{'PASS' if tokenizers_fp32 else 'FAIL'}] dtypes (tokenizers must stay fp32 in a bf16 load): {dtypes}")

    # --- LM head size: Apertus 1.5 checkpoints are expected to ship a pruned, text-only head ----------------
    output_vocab_size = getattr(config.text_config, "output_vocab_size", None)
    expected_head = output_vocab_size or config.text_config.vocab_size
    head_ok = model.lm_head.weight.shape[0] == expected_head
    if not head_ok:
        failed_checks.append("lm_head size")
    print(
        f"[{'PASS' if head_ok else 'FAIL'}] lm_head rows: {model.lm_head.weight.shape[0]} "
        f"(expected {expected_head}; vocab_size {config.text_config.vocab_size})"
    )
    if output_vocab_size is None:
        print("[WARN] the composite has an UNPRUNED LM head; expected a pruned backbone (`output_vocab_size`)")

    tokenizer = AutoTokenizer.from_pretrained(composite_dir)

    # --- processor round trip: component classes and token-id agreement with the config ---------------------
    processor = AutoProcessor.from_pretrained(composite_dir)
    component_classes = (
        type(processor).__name__,
        type(processor.image_processor).__name__,
        type(processor.feature_extractor).__name__,
    )
    # the two placeholder ids must match the model config; the six structure tokens have no config ids and
    # only need to resolve through the real vocabulary (not to unk)
    token_id_pairs = [
        (processor.image_token, config.image_token_id),
        (processor.audio_token, config.audio_token_id),
    ]
    token_ids_ok = all(tokenizer.convert_tokens_to_ids(token) == expected for token, expected in token_id_pairs)
    structure_tokens = (
        processor.boi_token,
        processor.eoi_token,
        processor.image_wrapper_token,
        processor.eol_token,
        processor.boa_token,
        processor.eoa_token,
    )
    structure_ids = [tokenizer.convert_tokens_to_ids(token) for token in structure_tokens]
    structure_ok = all(token_id is not None and token_id != tokenizer.unk_token_id for token_id in structure_ids)
    processor_ok = component_classes == (
        "Apertus1p5Processor",
        "Apertus1p5ImageProcessor",
        "WavTokenizerFeatureExtractor",
    )
    if not (processor_ok and token_ids_ok and structure_ok):
        failed_checks.append("processor")
    print(
        f"[{'PASS' if processor_ok and token_ids_ok and structure_ok else 'FAIL'}] processor: components "
        f"{component_classes}, placeholder ids match the config: {token_ids_ok}, structure tokens resolve: "
        f"{structure_ok}"
    )

    # --- image offset cross-check against the real vocabulary -----------------------------------------------
    torch.manual_seed(0)
    image = torch.rand(1, 3, 256, 256) * 2 - 1  # [-1, 1], multiple of 16
    with torch.no_grad():
        image_ids = model.model.get_image_tokens(image, torch.tensor([[256, 256]]))
    in_range = bool(
        (image_ids >= config.image_token_offset).all()
        and (image_ids < config.image_token_offset + config.vision_tokenizer_config.codebook_size).all()
    )
    code = int(image_ids[0]) - config.image_token_offset
    token_str = tokenizer.convert_ids_to_tokens(int(image_ids[0]))
    image_ok = in_range and image_ids.numel() == 256 and token_str == f"<|visual token {code}|>"
    if not image_ok:
        failed_checks.append("image")
    print(
        f"[{'PASS' if image_ok else 'FAIL'}] image: {image_ids.numel()} "
        f"codes (expected 256), range ok: {in_range}, id {int(image_ids[0])} -> {token_str!r}"
    )

    # --- audio offset cross-check ----------------------------------------------------------------------------
    t = torch.arange(24000) / 24000.0
    sine = (0.5 * torch.sin(2 * torch.pi * 440.0 * t))[None, None, :]
    with torch.no_grad():
        audio_ids = model.model.get_audio_tokens(sine, torch.ones(1, 24000, dtype=torch.long))
    in_range = bool(
        (audio_ids >= config.audio_token_offset).all()
        and (audio_ids < config.audio_token_offset + config.audio_tokenizer_config.codebook_size).all()
    )
    code = int(audio_ids[0]) - config.audio_token_offset
    token_str = tokenizer.convert_ids_to_tokens(int(audio_ids[0]))
    audio_ok = in_range and audio_ids.numel() == 40 and token_str == f"<|audio token {code}|>"
    if not audio_ok:
        failed_checks.append("audio")
    print(
        f"[{'PASS' if audio_ok else 'FAIL'}] "
        f"audio: {audio_ids.numel()} codes (expected 40), range ok: {in_range}, id {int(audio_ids[0])} -> {token_str!r}"
    )

    # --- short greedy text generation ------------------------------------------------------------------------
    messages = [{"role": "user", "content": "What is the capital of Switzerland? Answer in one word."}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = generated[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    ids_in_range = int(new_ids.max()) < expected_head
    if not ids_in_range:
        failed_checks.append("generated ids")
    print(f"[{'PASS' if ids_in_range else 'FAIL'}] text generation (ids < {expected_head}): {completion!r}")

    # --- processor-driven multimodal forwards --------------------------------------------------------------
    np_image = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    inputs = processor(text="<|image|>Describe the image.", images=[np_image], return_tensors="pt")
    decoded = tokenizer.decode(inputs["input_ids"][0])
    header_ok = "<|img_start|>16*16<|img_token_start|>" in decoded and decoded.count("<|image|>") == 256
    with torch.no_grad():
        logits = model(**inputs).logits
    image_forward_ok = header_ok and _has_valid_logits_layout(logits, expected_head, config.text_config.vocab_size)
    if not image_forward_ok:
        failed_checks.append("image forward")
    print(
        f"[{'PASS' if image_forward_ok else 'FAIL'}] processor image forward: header+counts ok: {header_ok}, "
        f"logits {tuple(logits.shape)}, finite physical prefix and finfo.min padded tail"
    )

    inputs = processor(text="<|audio|>What is said?", audio=[sine[0, 0].numpy()], return_tensors="pt")
    decoded = tokenizer.decode(inputs["input_ids"][0])
    audio_layout_ok = decoded.count("<|audio|>") == 40 and "<|audio_start|>" in decoded
    with torch.no_grad():
        logits = model(**inputs).logits
    audio_forward_ok = audio_layout_ok and _has_valid_logits_layout(
        logits, expected_head, config.text_config.vocab_size
    )
    if not audio_forward_ok:
        failed_checks.append("audio forward")
    print(
        f"[{'PASS' if audio_forward_ok else 'FAIL'}] processor audio forward: layout ok: {audio_layout_ok}, "
        f"logits {tuple(logits.shape)}, finite physical prefix and finfo.min padded tail"
    )

    if failed_checks:
        raise RuntimeError(f"Composite verification failed: {failed_checks}")


def convert(
    apertus_checkpoint: str, vision_tokenizer_checkpoint: str, audio_tokenizer_checkpoint: str, output_dir: str
):
    """Remap and write all three weight sources together with their composite config and processor."""
    _check_output_is_not_a_source(
        output_dir, apertus_checkpoint, vision_tokenizer_checkpoint, audio_tokenizer_checkpoint
    )
    os.makedirs(output_dir, exist_ok=True)
    config = build_config(apertus_checkpoint, vision_tokenizer_checkpoint, audio_tokenizer_checkpoint)
    config.save_pretrained(output_dir)

    weight_map, total_size = {}, 0
    for source, shard, remapped in remapped_sources(
        apertus_checkpoint, vision_tokenizer_checkpoint, audio_tokenizer_checkpoint
    ):
        # always prefix the source name: a bare `model.safetensors` would shadow the index at load time
        out_shard = f"model-{source}-{shard}"
        save_file(remapped, os.path.join(output_dir, out_shard), metadata={"format": "pt"})
        for key, value in remapped.items():
            weight_map[key] = out_shard
            total_size += value.numel() * value.element_size()
        logger.info(f"wrote {out_shard}: {len(remapped)} tensors")

    if not config.tie_word_embeddings and "lm_head.weight" not in weight_map:
        raise ValueError(
            "The Apertus backbone checkpoint has no `lm_head.weight` but does not use tied word embeddings; the "
            "composite would load with a randomly initialized LM head."
        )

    # Canonical unsharded weights take precedence over an index in `from_pretrained`; remove stale entrypoints from
    # earlier conversions while leaving unrelated files and unreferenced shards untouched.
    for filename in (SAFE_WEIGHTS_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME):
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"removed stale weight file {path}")

    with open(os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    write_processor(apertus_checkpoint, audio_tokenizer_checkpoint, output_dir)
    logger.info(f"composite checkpoint written to {output_dir}")


def write_processor(apertus_checkpoint: str, audio_tokenizer_checkpoint: str, output_dir: str):
    """Write the tokenizer, image processor, audio feature extractor, and unified processor configuration."""
    _check_output_is_not_a_source(output_dir, apertus_checkpoint, audio_tokenizer_checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(audio_tokenizer_checkpoint, "config.json")) as f:
        audio_tokenizer_config = json.load(f)
    processor = build_processor(apertus_checkpoint, audio_tokenizer_config)
    processor.save_pretrained(output_dir)
    generation_config = os.path.join(apertus_checkpoint, "generation_config.json")
    output_generation_config = os.path.join(output_dir, "generation_config.json")
    if os.path.exists(generation_config):
        shutil.copy(generation_config, output_generation_config)
    elif os.path.isfile(output_generation_config):
        # an earlier conversion may have copied one from a different backbone source
        os.remove(output_generation_config)
        logger.info("removed stale generation_config.json (the backbone source ships none)")
    logger.info(f"processor, tokenizer and chat template written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble and optionally verify an Apertus 1.5 composite checkpoint.")
    parser.add_argument(
        "--apertus_checkpoint",
        help="Apertus 1.5 causal-LM backbone: local directory or Hub `repo_id[@revision]` "
        "(not read with --skip_convert)",
    )
    parser.add_argument(
        "--vision_tokenizer_checkpoint",
        help="Converted Apertus1p5VisionTokenizerModel: local directory or Hub `repo_id[@revision]` "
        "(not read with --processor_only or --skip_convert)",
    )
    parser.add_argument(
        "--audio_tokenizer_checkpoint",
        help="Converted WavTokenizerModel: local directory or Hub `repo_id[@revision]` (not read with --skip_convert)",
    )
    parser.add_argument("--output_dir", required=True, help="Directory in which to write or find the composite")
    parser.add_argument(
        "--verify", action="store_true", help="Load the composite from --output_dir and run smoke checks"
    )
    parser.add_argument(
        "--skip_convert", action="store_true", help="Skip weight conversion and reuse the composite in --output_dir"
    )
    parser.add_argument(
        "--processor_only",
        action="store_true",
        help="Only (re)write the processor stack into --output_dir, without re-sharding the weights",
    )
    args = parser.parse_args()

    if args.skip_convert and not args.processor_only and not args.verify:
        parser.error("--skip_convert without --verify does nothing; pass --verify or drop --skip_convert")
    if (args.processor_only or not args.skip_convert) and (
        args.apertus_checkpoint is None or args.audio_tokenizer_checkpoint is None
    ):
        parser.error("--apertus_checkpoint and --audio_tokenizer_checkpoint are required unless --skip_convert is set")

    # resolve only the sources the selected mode actually reads, so e.g. --skip_convert --verify downloads nothing
    if args.processor_only:
        write_processor(
            resolve_checkpoint_dir(args.apertus_checkpoint),
            resolve_checkpoint_dir(args.audio_tokenizer_checkpoint),
            args.output_dir,
        )
    elif not args.skip_convert:
        if args.vision_tokenizer_checkpoint is None:
            parser.error("--vision_tokenizer_checkpoint is required unless --processor_only or --skip_convert is set")
        convert(
            resolve_checkpoint_dir(args.apertus_checkpoint),
            resolve_checkpoint_dir(args.vision_tokenizer_checkpoint),
            resolve_checkpoint_dir(args.audio_tokenizer_checkpoint),
            args.output_dir,
        )
    if args.verify:
        verify(args.output_dir)

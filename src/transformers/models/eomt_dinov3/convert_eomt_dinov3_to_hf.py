"""Conversion script for EoMT-DINOv3 checkpoints.

To convert one of the official checkpoints directly from the Hugging Face Hub you can run:

```bash
HF_TOKEN=your_token_here \
python -m transformers.models.eomt_dinov3.convert_eomt_dinov3_to_hf \
    --model-id tue-mps/coco_panoptic_eomt_large_640_dinov3 \
    --output-dir /tmp/eomt_converted \
    --verify \
    --original-repo-path /tmp/eomt
```

Make sure the token used above has been granted access to the gated DINOv3 weights.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from transformers import EomtDinov3Config, EomtDinov3ForUniversalSegmentation, EomtImageProcessorFast


CAT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


DEFAULT_BACKBONE_REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_IMAGE_SIZE = 640


class CheckpointSpec(NamedTuple):
    """Metadata describing how to convert an official EoMT-DINOv3 checkpoint."""

    model_id: str
    backbone_repo_id: str
    image_size: int


CHECKPOINT_CATALOG: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        model_id="tue-mps/coco_panoptic_eomt_small_640_dinov3",
        backbone_repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        image_size=640,
    ),
    CheckpointSpec(
        model_id="tue-mps/coco_panoptic_eomt_base_640_dinov3",
        backbone_repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        image_size=640,
    ),
    CheckpointSpec(
        model_id="tue-mps/coco_panoptic_eomt_large_640_dinov3",
        backbone_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size=640,
    ),
    CheckpointSpec(
        model_id="tue-mps/coco_panoptic_eomt_large_1280_dinov3",
        backbone_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size=1280,
    ),
    CheckpointSpec(
        model_id="tue-mps/ade20k_semantic_eomt_large_512_dinov3",
        backbone_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size=512,
    ),
    CheckpointSpec(
        model_id="tue-mps/coco_instance_eomt_large_640_dinov3",
        backbone_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size=640,
    ),
    CheckpointSpec(
        model_id="tue-mps/coco_instance_eomt_large_1280_dinov3",
        backbone_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size=1280,
    ),
    CheckpointSpec(
        model_id="tue-mps/ade_semantic_eomt_large_512_dinov3",
        backbone_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        image_size=512,
    ),
)


def _build_checkpoint_index() -> dict[str, CheckpointSpec]:
    index: dict[str, CheckpointSpec] = {}

    for spec in CHECKPOINT_CATALOG:
        keys = {spec.model_id.lower(), spec.model_id.split("/", maxsplit=1)[-1].lower()}
        for key in keys:
            index[key] = spec

    return index


CHECKPOINT_SPECS = _build_checkpoint_index()


def resolve_checkpoint_spec(model_id: str) -> CheckpointSpec:
    key = model_id.lower()
    if key not in CHECKPOINT_SPECS:
        available = ", ".join(sorted(spec.model_id for spec in CHECKPOINT_CATALOG))
        raise ValueError(f"Unknown checkpoint '{model_id}'. Available options: {available}.")
    return CHECKPOINT_SPECS[key]


def print_checkpoint_catalog() -> None:
    print("Supported checkpoints:")
    for spec in CHECKPOINT_CATALOG:
        print(f"- {spec.model_id} (image_size={spec.image_size}, backbone={spec.backbone_repo_id})")


DELTA_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"^network\.encoder\.backbone\.patch_embed\.cls_token$", "embeddings.cls_token"),
    (r"^network\.encoder\.backbone\.patch_embed\.register_tokens$", "embeddings.register_tokens"),
    (
        r"^network\.encoder\.backbone\.patch_embed\.patch_embeddings\.",
        "embeddings.patch_embeddings.",
    ),
    (r"^network\.encoder\.backbone\.blocks\.(\d+)\.", r"layers.\1."),
    (r"^network\.encoder\.backbone\.norm\.", "layernorm."),
    (r"^network\.q\.", "query."),
    (r"^network\.class_head\.", "class_predictor."),
    (r"^network\.mask_head\.0\.", "mask_head.fc1."),
    (r"^network\.mask_head\.2\.", "mask_head.fc2."),
    (r"^network\.mask_head\.4\.", "mask_head.fc3."),
    (r"^network\.upscale\.(\d+)\.conv1\.", r"upscale_block.block.\1.conv1."),
    (r"^network\.upscale\.(\d+)\.conv2\.", r"upscale_block.block.\1.conv2."),
    (r"^network\.upscale\.(\d+)\.norm\.", r"upscale_block.block.\1.layernorm2d."),
    (r"^network\.attn_mask_probs$", "attn_mask_probs"),
    (r"^criterion\.", "criterion."),
)

SKIP_KEYS = {
    "network.encoder.pixel_mean",
    "network.encoder.pixel_std",
}


def _rename_delta_key(key: str) -> tuple[str | None, bool]:
    if key in SKIP_KEYS:
        return None, False

    for pattern, replacement in DELTA_KEY_REPLACEMENTS:
        if re.match(pattern, key):
            new_key = re.sub(pattern, replacement, key)
            return new_key, key.startswith("network.encoder.backbone")

    if key.startswith("network.encoder.backbone"):
        raise KeyError(f"Unhandled backbone key: {key}")

    return None, False


def convert_delta_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], set[str]]:
    converted: dict[str, torch.Tensor] = {}
    backbone_keys: set[str] = set()

    for key, value in state_dict.items():
        new_key, is_backbone = _rename_delta_key(key)
        if new_key is None:
            continue

        converted[new_key] = value
        if is_backbone:
            backbone_keys.add(new_key)

    return converted, backbone_keys


def map_dinov3_state_to_eomt(base_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}

    for key, tensor in base_state_dict.items():
        if key.startswith("layer."):
            new_key = key.replace("layer.", "layers.", 1)
        elif key == "norm.weight":
            new_key = "layernorm.weight"
        elif key == "norm.bias":
            new_key = "layernorm.bias"
        else:
            new_key = key

        if new_key == "embeddings.mask_token":
            continue

        mapped[new_key] = tensor

    return mapped


def merge_backbone_weights(
    base_backbone: dict[str, torch.Tensor],
    delta_backbone: dict[str, torch.Tensor],
    backbone_delta_keys: Iterable[str],
) -> dict[str, torch.Tensor]:
    merged = dict(base_backbone)

    for key, value in delta_backbone.items():
        if key in backbone_delta_keys:
            merged[key] = merged[key] + value
        else:
            merged[key] = value

    return merged


def build_eomt_config(
    *,
    base_config: dict[str, object],
    delta_state_dict: dict[str, torch.Tensor],
    image_size: int,
) -> EomtDinov3Config:
    num_queries = delta_state_dict["network.q.weight"].shape[0]
    num_blocks = delta_state_dict["network.attn_mask_probs"].numel()
    num_upscale_blocks = len({int(key.split(".")[2]) for key in delta_state_dict if key.startswith("network.upscale")})
    num_register_tokens = delta_state_dict["network.encoder.backbone.patch_embed.register_tokens"].shape[1]
    num_labels = delta_state_dict["network.class_head.weight"].shape[0] - 1

    config = EomtDinov3Config(
        hidden_size=base_config["hidden_size"],
        num_hidden_layers=base_config["num_hidden_layers"],
        num_attention_heads=base_config["num_attention_heads"],
        intermediate_size=base_config["intermediate_size"],
        hidden_act=base_config["hidden_act"],
        hidden_dropout_prob=0.0,
        initializer_range=base_config["initializer_range"],
        layer_norm_eps=base_config["layer_norm_eps"],
        image_size=image_size,
        patch_size=base_config["patch_size"],
        num_channels=base_config.get("num_channels", 3),
        layerscale_value=base_config.get("layerscale_value", 1.0),
        drop_path_rate=base_config.get("drop_path_rate", 0.0),
        attention_dropout=base_config.get("attention_dropout", 0.0),
        num_upscale_blocks=num_upscale_blocks,
        num_blocks=num_blocks,
        num_queries=num_queries,
        num_register_tokens=num_register_tokens,
        rope_parameters={"rope_theta": base_config.get("rope_theta", 100.0)},
        query_bias=base_config.get("query_bias", True),
        key_bias=base_config.get("key_bias", False),
        value_bias=base_config.get("value_bias", True),
        proj_bias=base_config.get("proj_bias", True),
        mlp_bias=base_config.get("mlp_bias", True),
        use_gated_mlp=base_config.get("use_gated_mlp", False),
        pos_embed_shift=base_config.get("pos_embed_shift"),
        pos_embed_jitter=base_config.get("pos_embed_jitter"),
        pos_embed_rescale=base_config.get("pos_embed_rescale"),
        num_labels=num_labels,
    )

    return config


def convert_checkpoint(
    *,
    delta_state_dict: dict[str, torch.Tensor],
    backbone_repo_id: str,
    image_size: int,
) -> tuple[EomtDinov3Config, dict[str, torch.Tensor]]:
    # load model.safetensors
    filepath = hf_hub_download(backbone_repo_id, filename="model.safetensors")
    base_state_dict = load_file(filepath)

    # load config.json
    filepath = hf_hub_download(backbone_repo_id, filename="config.json")
    with open(filepath, "r") as f:
        base_config = json.load(f)

    mapped_base = map_dinov3_state_to_eomt(base_state_dict)
    converted_delta, backbone_delta_keys = convert_delta_state_dict(delta_state_dict)
    merged_state_dict = merge_backbone_weights(mapped_base, converted_delta, backbone_delta_keys)

    config = build_eomt_config(
        base_config=base_config,
        delta_state_dict=delta_state_dict,
        image_size=image_size,
    )

    return config, merged_state_dict


def ensure_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def convert_model(
    *,
    model_id: str,
    output_dir: Path,
    verify: bool,
    original_repo_path: Path | None,
    push_to_hub: bool = False,
) -> None:
    # resolve checkpoint spec
    spec = resolve_checkpoint_spec(model_id)
    backbone_repo_id = spec.backbone_repo_id
    image_size = spec.image_size

    # load delta state
    delta_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    raw_delta_state = torch.load(delta_path, map_location="cpu")
    delta_state_dict = ensure_state_dict(raw_delta_state)

    # convert checkpoint
    config, merged_state_dict = convert_checkpoint(
        delta_state_dict=delta_state_dict,
        backbone_repo_id=backbone_repo_id,
        image_size=image_size,
    )

    with init_empty_weights():
        model = EomtDinov3ForUniversalSegmentation(config)

    model.load_state_dict(merged_state_dict, strict=True, assign=True)

    processor = EomtImageProcessorFast(
        size={"shortest_edge": image_size, "longest_edge": image_size},
        do_split_image=False,
        do_pad=True,
    )

    if verify:
        verify_conversion(
            hf_model=model,
            processor=processor,
            delta_state_dict=delta_state_dict,
            backbone_repo_id=backbone_repo_id,
            image_size=image_size,
            original_repo_path=original_repo_path,
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    if push_to_hub:
        # Extract model name from model_id (e.g. "tue-mps/coco_panoptic_eomt_large_640_dinov3" -> "eomt-dinov3-coco-panoptic-large-640")
        base_name = model_id.split("/")[-1]  # e.g. "coco_panoptic_eomt_large_640_dinov3"
        parts = base_name.replace("_dinov3", "").split("_")  # ["coco", "panoptic", "eomt", "large", "640"]
        # Reorder to: eomt-dinov3-{task}-{dataset}-{size}-{resolution}
        if "eomt" in parts:
            parts.remove("eomt")
        repo_name = "eomt-dinov3-" + "-".join(parts)
        model.push_to_hub(repo_id=f"nielsr/{repo_name}")
        processor.push_to_hub(repo_id=f"nielsr/{repo_name}")


def _prepare_image(processor: EomtImageProcessorFast) -> torch.Tensor:
    image = Image.open(requests.get(CAT_URL, stream=True).raw).convert("RGB")
    inputs = processor(images=image, do_normalize=False, return_tensors="pt")
    return inputs.pixel_values


def _load_original_model(
    *,
    original_repo_path: Path,
    backbone_repo_id: str,
    image_size: int,
    num_labels: int,
    num_queries: int,
    num_blocks: int,
    delta_state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    sys.path.insert(0, str(original_repo_path))

    from models.eomt import EoMT
    from models.vit import ViT

    encoder = ViT((image_size, image_size), backbone_name=backbone_repo_id, ckpt_path=None)
    model = EoMT(encoder=encoder, num_classes=num_labels, num_q=num_queries, num_blocks=num_blocks)

    state_dict = model.state_dict()
    for key, value in delta_state_dict.items():
        if key in SKIP_KEYS or key.startswith("criterion"):
            continue

        target_key = key
        if key.startswith("network."):
            target_key = key.replace("network.", "", 1)

        if target_key.startswith("encoder.backbone"):
            state_dict[target_key] = state_dict[target_key] + value
        elif target_key in state_dict:
            state_dict[target_key] = value

    model.load_state_dict(state_dict)
    model.eval()
    return model


class BackboneVerificationOutputs(NamedTuple):
    patch_embeddings: torch.Tensor
    rope_embeddings: tuple[torch.Tensor, torch.Tensor]
    hidden_states: list[torch.Tensor]
    mask_logits: list[torch.Tensor]
    class_logits: list[torch.Tensor]
    sequence_output: torch.Tensor


def _collect_original_backbone_states(model, pixel_values: torch.Tensor) -> BackboneVerificationOutputs:
    backbone = model.encoder.backbone
    hidden_states = (pixel_values - model.encoder.pixel_mean) / model.encoder.pixel_std

    rope = None
    if hasattr(backbone, "rope_embeddings"):
        rope = backbone.rope_embeddings(hidden_states)

    hidden_states = backbone.patch_embed(hidden_states)
    patch_embeddings = hidden_states.detach().clone()
    if rope is None:
        raise ValueError("Original model is missing rope embeddings")
    outputs = []
    mask_logits_list = []
    class_logits_list = []
    attn_mask = None

    for idx, block in enumerate(backbone.blocks):
        if idx == len(backbone.blocks) - model.num_blocks:
            query = model.q.weight[None, :, :].expand(hidden_states.shape[0], -1, -1)
            hidden_states = torch.cat((query, hidden_states), dim=1)

        if idx >= len(backbone.blocks) - model.num_blocks:
            norm_hidden_states = backbone.norm(hidden_states)
            mask_logits, class_logits = model._predict(norm_hidden_states)
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)
            attn_mask = model._attn_mask(hidden_states, mask_logits, idx)

        attn_module = block.attention if hasattr(block, "attention") else block.attn
        attn_output = model._attn(attn_module, block.norm1(hidden_states), attn_mask, rope=rope)
        if hasattr(block, "layer_scale1"):
            hidden_states = hidden_states + block.layer_scale1(attn_output)
        else:
            hidden_states = hidden_states + block.ls1(attn_output)

        mlp_output = block.mlp(block.norm2(hidden_states))
        if hasattr(block, "layer_scale2"):
            hidden_states = hidden_states + block.layer_scale2(mlp_output)
        else:
            hidden_states = hidden_states + block.ls2(mlp_output)

        outputs.append(hidden_states)

    sequence_output = backbone.norm(hidden_states)
    mask_logits, class_logits = model._predict(sequence_output)
    mask_logits_list.append(mask_logits)
    class_logits_list.append(class_logits)

    return BackboneVerificationOutputs(
        patch_embeddings=patch_embeddings,
        rope_embeddings=rope,
        hidden_states=outputs,
        mask_logits=mask_logits_list,
        class_logits=class_logits_list,
        sequence_output=sequence_output,
    )


def _collect_hf_backbone_states(
    model: EomtDinov3ForUniversalSegmentation, pixel_values: torch.Tensor
) -> BackboneVerificationOutputs:
    position_embeddings = model.rope_embeddings(pixel_values)
    hidden_states = model.embeddings(pixel_values)
    patch_embeddings = hidden_states.detach().clone()

    outputs = []
    mask_logits_list = []
    class_logits_list = []
    attention_mask = None

    for idx, layer_module in enumerate(model.layers):
        if idx == model.num_hidden_layers - model.config.num_blocks:
            query = model.query.weight[None, :, :].expand(hidden_states.shape[0], -1, -1).to(hidden_states.device)
            hidden_states = torch.cat((query, hidden_states), dim=1)

        if idx >= model.num_hidden_layers - model.config.num_blocks:
            norm_hidden_states = model.layernorm(hidden_states)
            mask_logits, class_logits = model.predict(norm_hidden_states)
            mask_logits_list.append(mask_logits)
            class_logits_list.append(class_logits)

            probs_index = idx - model.num_hidden_layers + model.config.num_blocks
            if model.training or model.attn_mask_probs[probs_index] > 0:
                attention_mask = torch.ones(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    hidden_states.shape[1],
                    device=hidden_states.device,
                    dtype=torch.bool,
                )

                interpolated_logits = torch.nn.functional.interpolate(
                    mask_logits,
                    size=model.grid_size,
                    mode="bilinear",
                ).view(mask_logits.size(0), mask_logits.size(1), -1)

                num_query_tokens = model.config.num_queries
                encoder_start_tokens = num_query_tokens + model.embeddings.num_prefix_tokens
                attention_mask[:, :num_query_tokens, encoder_start_tokens:] = interpolated_logits > 0

                attention_mask = model._disable_attention_mask(
                    attention_mask,
                    prob=model.attn_mask_probs[probs_index],
                    num_query_tokens=num_query_tokens,
                    encoder_start_tokens=encoder_start_tokens,
                    device=hidden_states.device,
                )

                attention_mask = attention_mask[:, None, ...].expand(-1, model.config.num_attention_heads, -1, -1)

                bool_attention_mask = attention_mask
                attention_mask = attention_mask.float().masked_fill(~bool_attention_mask, -1e9)
                if attention_mask.dtype != hidden_states.dtype:
                    attention_mask = attention_mask.to(dtype=hidden_states.dtype)

        hidden_states = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        outputs.append(hidden_states)

    sequence_output = model.layernorm(hidden_states)
    mask_logits, class_logits = model.predict(sequence_output)
    mask_logits_list.append(mask_logits)
    class_logits_list.append(class_logits)

    return BackboneVerificationOutputs(
        patch_embeddings=patch_embeddings,
        rope_embeddings=position_embeddings,
        hidden_states=outputs,
        mask_logits=mask_logits_list,
        class_logits=class_logits_list,
        sequence_output=sequence_output,
    )


def _assert_allclose(reference: Iterable[torch.Tensor], actual: Iterable[torch.Tensor], message: str) -> None:
    for idx, (ref_tensor, act_tensor) in enumerate(zip(reference, actual)):
        if not torch.allclose(ref_tensor, act_tensor, atol=1e-4, rtol=1e-4):
            raise ValueError(f"Mismatch in {message} at index {idx}")


def verify_conversion(
    *,
    hf_model: EomtDinov3ForUniversalSegmentation,
    processor: EomtImageProcessorFast,
    delta_state_dict: dict[str, torch.Tensor],
    backbone_repo_id: str,
    image_size: int,
    original_repo_path: Path | None,
) -> None:
    if original_repo_path is None:
        raise ValueError("Original repository path is required for verification")

    torch.manual_seed(0)

    pixel_values = _prepare_image(processor)

    image_mean = torch.tensor(processor.image_mean, dtype=pixel_values.dtype, device=pixel_values.device)[
        None, :, None, None
    ]
    image_std = torch.tensor(processor.image_std, dtype=pixel_values.dtype, device=pixel_values.device)[
        None, :, None, None
    ]
    normalized_pixel_values = (pixel_values - image_mean) / image_std

    original_model = _load_original_model(
        original_repo_path=original_repo_path,
        backbone_repo_id=backbone_repo_id,
        image_size=image_size,
        num_labels=hf_model.config.num_labels,
        num_queries=hf_model.config.num_queries,
        num_blocks=hf_model.config.num_blocks,
        delta_state_dict=delta_state_dict,
    )

    hf_model.eval()

    with torch.no_grad():
        orig_outputs = _collect_original_backbone_states(original_model, pixel_values)
        hf_outputs = _collect_hf_backbone_states(hf_model, normalized_pixel_values)

    patch_abs_diff = (orig_outputs.patch_embeddings - hf_outputs.patch_embeddings).abs()
    print(f"Patch embedding max abs diff: {patch_abs_diff.max().item():.6e}")

    rope_abs_diffs = [(orig - hf).abs() for orig, hf in zip(orig_outputs.rope_embeddings, hf_outputs.rope_embeddings)]
    rope_max_diffs = [diff.max().item() for diff in rope_abs_diffs]
    print(
        "RoPE embedding max abs diff: "
        + ", ".join(f"component_{idx}={value:.6e}" for idx, value in enumerate(rope_max_diffs))
    )

    if not torch.allclose(orig_outputs.patch_embeddings, hf_outputs.patch_embeddings, atol=1e-4, rtol=1e-4):
        raise ValueError("Mismatch in patch embeddings")

    _assert_allclose(orig_outputs.rope_embeddings, hf_outputs.rope_embeddings, "rope embeddings")
    _assert_allclose(orig_outputs.hidden_states, hf_outputs.hidden_states, "backbone hidden states")
    _assert_allclose(orig_outputs.mask_logits, hf_outputs.mask_logits, "mask logits")
    _assert_allclose(orig_outputs.class_logits, hf_outputs.class_logits, "class logits")
    print("Looks good!")

    if not torch.allclose(orig_outputs.sequence_output, hf_outputs.sequence_output, atol=1e-4, rtol=1e-4):
        raise ValueError("Mismatch in final sequence output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert EoMT-DINOv3 checkpoints to 🤗 Transformers format")
    parser.add_argument("--list-models", action="store_true", help="List supported checkpoint names and exit")
    parser.add_argument(
        "--model-id",
        help="Name of an official EoMT-DINOv3 checkpoint to download and convert",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, help="Directory to save the converted model")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--original-repo-path", type=Path, default=None, help="Path to the original EoMT repository")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Whether to push the converted model to the Hugging Face Hub."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        print_checkpoint_catalog()
        return

    convert_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        verify=args.verify,
        original_repo_path=args.original_repo_path,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()

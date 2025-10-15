import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from transformers.models.nanochat import NanoGPTConfig


LOGGER = logging.get_logger(__name__)


def infer_kv_heads(config: NanoGPTConfig, state_dict: dict[str, torch.Tensor]) -> int:
    key_weight = state_dict.get("transformer.h.0.attn.c_k.weight")
    if key_weight is None:
        return config.num_key_value_heads
    rows = key_weight.shape[0]
    head_dim = config.hidden_size // config.num_attention_heads
    if rows % head_dim != 0:
        return config.num_key_value_heads
    inferred = rows // head_dim
    LOGGER.info("Inferred %s key_value heads from checkpoint", inferred)
    return max(inferred, 1)


def convert_layer(old_prefix: str, new_prefix: str) -> dict[str, str]:
    return {
        f"{old_prefix}.attn.c_q.weight": f"{new_prefix}.self_attn.q_proj.weight",
        f"{old_prefix}.attn.c_k.weight": f"{new_prefix}.self_attn.k_proj.weight",
        f"{old_prefix}.attn.c_v.weight": f"{new_prefix}.self_attn.v_proj.weight",
        f"{old_prefix}.attn.c_proj.weight": f"{new_prefix}.self_attn.o_proj.weight",
        f"{old_prefix}.mlp.c_fc.weight": f"{new_prefix}.mlp.fc.weight",
        f"{old_prefix}.mlp.c_proj.weight": f"{new_prefix}.mlp.proj.weight",
    }


def convert_checkpoint(source_dir: Path, dest_dir: Path) -> tuple[NanoGPTConfig, OrderedDict[str, torch.Tensor]]:
    config = NanoGPTConfig.from_pretrained(source_dir)
    LOGGER.info("Loaded config hidden_size=%s num_layers=%s", config.hidden_size, config.num_hidden_layers)

    old_state = torch.load(source_dir / "pytorch_model.bin", map_location="cpu")
    inferred_kv = infer_kv_heads(config, old_state)
    config.num_key_value_heads = inferred_kv
    if config.num_attention_heads % config.num_key_value_heads != 0:
        LOGGER.info("Adjusting num_attention_heads from %s to %s", config.num_attention_heads, config.num_key_value_heads)
        config.num_attention_heads = config.num_key_value_heads

    new_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    rename_map: dict[str, str] = {}

    def assign(old_key: str, new_key: str) -> None:
        tensor = old_state.get(old_key)
        if tensor is None:
            LOGGER.debug("Missing key %s in source checkpoint", old_key)
            return
        new_state[new_key] = tensor.clone()
        rename_map[old_key] = new_key

    assign("transformer.wte.weight", "model.embed_tokens.weight")
    assign("lm_head.weight", "lm_head.weight")

    for layer_idx in range(config.num_hidden_layers):
        old_prefix = f"transformer.h.{layer_idx}"
        new_prefix = f"model.layers.{layer_idx}"
        mapping = convert_layer(old_prefix, new_prefix)
        for old_key, new_key in mapping.items():
            assign(old_key, new_key)

    missing = [key for key in old_state.keys() if key not in rename_map]
    if missing:
        LOGGER.info("Skipped %d legacy entries that have no equivalent in the shared implementation", len(missing))

    dest_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(dest_dir)
    torch.save(new_state, dest_dir / "pytorch_model.bin")

    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = source_dir / filename
        if src.exists():
            (dest_dir / filename).write_bytes(src.read_bytes())
    return config, new_state


def run_test(dest_dir: Path, prompt: str, max_new_tokens: int) -> None:
    LOGGER.info("Running quick generation test with prompt: %s", prompt)
    tokenizer = AutoTokenizer.from_pretrained(dest_dir)
    model = AutoModelForCausalLM.from_pretrained(dest_dir)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    LOGGER.info("Generated text: %s", generated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NanoGPT checkpoints to the shared HF implementation")
    parser.add_argument("--source", type=Path, required=True, help="Path to the legacy checkpoint directory")
    parser.add_argument("--dest", type=Path, required=True, help="Output directory for the converted checkpoint")
    parser.add_argument("--test-prompt", type=str, default=None, help="Optional prompt for a quick generation test")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate during the test")
    return parser.parse_args()


def main():
    args = parse_args()
    config, _ = convert_checkpoint(args.source, args.dest)
    LOGGER.info("Converted checkpoint saved to %s (layers=%d kv_heads=%d)", args.dest, config.num_hidden_layers, config.num_key_value_heads)
    if args.test_prompt:
        run_test(args.dest, args.test_prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()

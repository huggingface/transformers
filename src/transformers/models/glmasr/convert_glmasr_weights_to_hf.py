import argparse
import re

import torch
from safetensors.torch import load_file

from transformers import (
    GlmAsrConfig,
    GlmAsrForConditionalGeneration,
    GlmAsrProcessor,
    TokenizersBackend,
    WhisperFeatureExtractor,
)
from transformers.utils.hub import cached_file


chat_template = """{%- macro to_text(content) -%}
{%- if content is string -%}
{{- content -}}
{%- elif content is iterable and content is not mapping -%}
{%- for item in content -%}
{%- if item is mapping and item.type == 'text' and item.text is defined -%}
{{- item.text -}}
{%- elif item is mapping and (item.type == 'audio' or 'audio' in item) -%}
<|begin_of_audio|><|pad|><|end_of_audio|><|user|>
{% elif item is string -%}
{{- item -}}
{%- endif -%}
{%- endfor -%}
{%- else -%}
{{- content -}}
{%- endif -%}
{%- endmacro -%}
{%- for m in messages -%}
{%- if m.role == 'system' -%}
<|system|>
{{ to_text(m.content) | trim }}
{%- elif m.role == 'user' -%}
<|user|>
{{ to_text(m.content) | trim }}
{%- elif m.role == 'assistant' -%}
<|assistant|>
{{ to_text(m.content) | trim }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>
{% endif -%}"""

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^model\.norm\.weight$":                                      r"language_model.model.norm.weight",
    r"^model\.(.*)$":                                              r"language_model.model.\1",
    r"^lm_head\.(.*)$":                                            r"language_model.lm_head.\1",
    r"^audio_encoder\.adapting\.0\.(.*)$":                         r"multi_modal_projector.linear_1.\1",
    r"^audio_encoder\.adapting\.2\.(.*)$":                         r"multi_modal_projector.linear_2.\1",
    r"^audio_encoder\.proj\.(weight|bias)$":                       r"multi_modal_projector.\1",
    r"^audio_encoder\.whisper\.(.*)$":                             r"audio_tower.\1",
    r"^audio_encoder\.layer_norm\.(.*)$":                          r"audio_tower.norm.\1",
    r"^audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.(.*)$":    r"audio_tower.layers.\1.self_attn.o_proj.\2",
    r"^audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.(.*)$":   r"audio_tower.layers.\1.input_layernorm.\2",
    r"^audio_tower\.layers\.(\d+)\.final_layer_norm\.(.*)$":       r"audio_tower.layers.\1.post_attention_layernorm.\2",
    r"^audio_tower\.layers\.(\d+)\.(fc[12])\.(.*)$":               r"audio_tower.layers.\1.mlp.\2.\3",
}
# fmt: on


def permute_rope(tensor, config):
    # IMPORTANT: the original checkpoint applies partial rope (half dimension) in the interleaved manner
    # since we use a different rope implementation, we want to permute the order like:
    # original order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    # permuted order: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

    if tensor.dim() == 2:
        dim1, dim2 = tensor.shape
    else:
        dim1 = tensor.shape[0]

    n_heads = config.audio_config.num_attention_heads
    head_dim = config.audio_config.head_dim
    rope_dim = dim1 // 2

    rope_indices = torch.arange(rope_dim)
    rope_indices = rope_indices.view(n_heads, rope_dim // n_heads // 2, 2)
    rope_indices = rope_indices.transpose(1, 2)
    rope_indices = rope_indices.reshape(n_heads, -1)

    non_rope_start = head_dim // 2
    non_rope_indices = torch.arange(non_rope_start, head_dim, dtype=torch.long)
    non_rope_indices = non_rope_indices.expand(n_heads, -1)

    head_offsets = torch.arange(n_heads, dtype=torch.long)[:, None] * (head_dim // 2)
    non_rope_indices = non_rope_indices + head_offsets.expand(n_heads, head_dim // 2)

    combined_indices = torch.cat([rope_indices, non_rope_indices], dim=1)
    global_head_offsets = torch.arange(n_heads, dtype=torch.long)[:, None] * (head_dim // 2)
    combined_indices = combined_indices + global_head_offsets.expand(n_heads, head_dim)

    permutation_indices = combined_indices.reshape(-1)
    tensor = tensor[permutation_indices]

    return tensor


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def main():
    parser = argparse.ArgumentParser(description="Convert GLM-ASR model weights to Hugging Face format")
    parser.add_argument(
        "--input_path_or_repo",
        type=str,
        default="zai-org/GLM-ASR-Nano-2512",
        help="Path to input model file or Hugging Face repository ID",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="91967eab799804ab256a3819a085b92378906eb2",
        help="Revision of the input repository",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save the converted model and processor",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Repository ID to push the model and processor to Hub (if not provided, won't push)",
    )

    args = parser.parse_args()

    path = cached_file(args.input_path_or_repo, "model.safetensors", revision=args.revision)
    state_dict = load_file(path)

    config = GlmAsrConfig()
    model = GlmAsrForConditionalGeneration(config)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = convert_key(k, ORIGINAL_TO_CONVERTED_KEY_MAPPING)

        # those are not used
        if new_key in [
            "audio_encoder.audio_bos_eos_token.weight",  # already present in the emb
            "audio_tower.embed_positions.weight",
            "multi_modal_projector.bias",
            "multi_modal_projector.weight",
        ]:
            continue

        if "audio_tower" in new_key and ("q_proj" in new_key or "k_proj" in new_key):
            v = permute_rope(v, config)

        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True, assign=True)

    feature_extractor = WhisperFeatureExtractor(feature_size=128)
    tokenizer = TokenizersBackend.from_pretrained(args.input_path_or_repo, revision=args.revision)
    tokenizer.pad_token = tokenizer.eos_token

    processor = GlmAsrProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer, chat_template=chat_template)

    if args.output_dir:
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()

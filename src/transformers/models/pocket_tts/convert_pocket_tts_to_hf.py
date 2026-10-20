# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Converts a Pocket TTS checkpoint (`kyutai/pocket-tts`, one language folder) to the transformers format."""

import argparse
import json
import re
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file
from tokenizers import AddedToken, Tokenizer

from transformers import (
    AutoTokenizer,
    DynamicCache,
    EncodecFeatureExtractor,
    GenerationConfig,
    PocketTTSAudioTokenizerConfig,
    PocketTTSAudioTokenizerModel,
    PocketTTSConfig,
    PocketTTSFeatureExtractor,
    PocketTTSForConditionalGeneration,
    PocketTTSProcessor,
)


ORIGINAL_REPO = "kyutai/pocket-tts"
# where the original configs read the tokenizers from
TOKENIZER_REPO = "kyutai/pocket-tts-without-voice-cloning"
TOKENIZER_REVISION = "00eac05ed3d16bdc3f6b5d598874019c34a89214"
AUDIO_PROMPT_TOKEN = "<|audio_prompt|>"
AUDIO_LATENT_TOKEN = "<|audio_latent|>"
AUDIO_PROMPT_BOS_TOKEN = "<|audio_prompt_bos|>"
AUDIO_BOS_TOKEN = "<|audio_bos|>"

# The values of the original's `pocket_tts/config/<checkpoint>.yaml` that differ between checkpoints (the codec and the
# flow head width are shared). `weights_revision` / `tokenizer` pin the files the original config points to, and `text`
# holds the text settings of the language (`replace_characters`, `remove_semicolons`), which the tokenizer applies.
CHECKPOINTS = {
    "english": {
        "weights_revision": "7f364c05d1be3faf2b234cdbe4cbb68a026d4218",
        "tokenizer": (TOKENIZER_REPO, "languages/english/tokenizer.json", TOKENIZER_REVISION),
        "num_hidden_layers": 6,
        "flow_type": "lsd",
        "text": {},
    },
    "english_drifting_26-09": {
        "weights_revision": "75cfe24e3142233bb81887fed3482e9652bb4a6e",
        # the drifting checkpoint reuses the tokenizer of english_2026-09
        "tokenizer": (TOKENIZER_REPO, "languages/english_2026-09/tokenizer.json", TOKENIZER_REVISION),
        "num_hidden_layers": 6,
        "flow_type": "drifting",
        "text": {},
    },
}
# Original tensor name (regex) -> transformers name. Applied in order, first match wins. Fused `in_proj` weights are split
# into q/k/v with the q/k rows permuted for rotate-half RoPE (the original rotates interleaved pairs), and the text
# embedding table gets the placeholder rows, both below.
STATE_DICT_MAPPING = [
    (r"^flow_lm\.speaker_proj_weight$", "model.audio_prompt_proj.weight"),
    (r"^flow_lm\.input_linear\.weight$", "model.audio_latent_proj.weight"),
    (
        r"^flow_lm\.transformer\.layers\.(\d+)\.self_attn\.out_proj\.weight$",
        r"model.layers.\1.self_attn.o_proj.weight",
    ),
    (r"^flow_lm\.transformer\.layers\.(\d+)\.linear1\.weight$", r"model.layers.\1.mlp.fc1.weight"),
    (r"^flow_lm\.transformer\.layers\.(\d+)\.linear2\.weight$", r"model.layers.\1.mlp.fc2.weight"),
    (r"^flow_lm\.transformer\.layers\.(\d+)\.norm1\.(weight|bias)$", r"model.layers.\1.input_layernorm.\2"),
    (r"^flow_lm\.transformer\.layers\.(\d+)\.norm2\.(weight|bias)$", r"model.layers.\1.post_attention_layernorm.\2"),
    (r"^flow_lm\.out_norm\.(weight|bias)$", r"model.norm.\1"),
    (r"^flow_lm\.out_eos\.(weight|bias)$", r"eos_head.\1"),
    (r"^flow_lm\.bos_emb$", "model.bos_latent"),
    (r"^flow_lm\.emb_mean$", "latent_mean"),
    (r"^flow_lm\.emb_std$", "latent_std"),
    (r"^flow_lm\.flow_net\.time_embed\.(\d+)\.freqs$", r"flow_head.timestep_embeddings.\1.freqs"),
    (r"^flow_lm\.flow_net\.time_embed\.(\d+)\.mlp\.0\.(weight|bias)$", r"flow_head.timestep_embeddings.\1.fc1.\2"),
    (r"^flow_lm\.flow_net\.time_embed\.(\d+)\.mlp\.2\.(weight|bias)$", r"flow_head.timestep_embeddings.\1.fc2.\2"),
    (r"^flow_lm\.flow_net\.time_embed\.(\d+)\.mlp\.3\.alpha$", r"flow_head.timestep_embeddings.\1.norm.weight"),
    (r"^flow_lm\.flow_net\.cond_embed\.(weight|bias)$", r"flow_head.cond_proj.\1"),
    (r"^flow_lm\.flow_net\.input_proj\.(weight|bias)$", r"flow_head.input_proj.\1"),
    (r"^flow_lm\.flow_net\.res_blocks\.(\d+)\.in_ln\.(weight|bias)$", r"flow_head.layers.\1.norm.\2"),
    (r"^flow_lm\.flow_net\.res_blocks\.(\d+)\.mlp\.0\.(weight|bias)$", r"flow_head.layers.\1.fc1.\2"),
    (r"^flow_lm\.flow_net\.res_blocks\.(\d+)\.mlp\.2\.(weight|bias)$", r"flow_head.layers.\1.fc2.\2"),
    (
        r"^flow_lm\.flow_net\.res_blocks\.(\d+)\.adaLN_modulation\.1\.(weight|bias)$",
        r"flow_head.layers.\1.adaln_proj.\2",
    ),
    (r"^flow_lm\.flow_net\.final_layer\.linear\.(weight|bias)$", r"flow_head.final_layer.proj.\1"),
    (r"^flow_lm\.flow_net\.final_layer\.adaLN_modulation\.1\.(weight|bias)$", r"flow_head.final_layer.adaln_proj.\1"),
    # codec, encoder side (the audio tokenizer, a model of its own)
    (r"^mimi\.downsample\.conv\.conv\.weight$", "audio_tokenizer.encoder.downsample.conv.weight"),
    (
        r"^mimi\.encoder\.model\.(\d+)\.conv\.(weight|bias)$",
        r"audio_tokenizer.encoder.encoder_seanet.layers.\1.conv.\2",
    ),
    (
        r"^mimi\.encoder\.model\.(\d+)\.block\.(\d+)\.conv\.(weight|bias)$",
        r"audio_tokenizer.encoder.encoder_seanet.layers.\1.block.\2.conv.\3",
    ),
    # codec, decoder side; `.conv` / `.convtr` both become the `.conv` of the wrapper
    (r"^mimi\.quantizer\.output_proj\.weight$", "audio_tokenizer.latent_proj.weight"),
    (r"^mimi\.upsample\.convtr\.convtr\.weight$", "audio_tokenizer.decoder.upsample.conv.weight"),
    (
        r"^mimi\.decoder\.model\.(\d+)\.(?:conv|convtr)\.(weight|bias)$",
        r"audio_tokenizer.decoder.decoder_seanet.layers.\1.conv.\2",
    ),
    (
        r"^mimi\.decoder\.model\.(\d+)\.block\.(\d+)\.conv\.(weight|bias)$",
        r"audio_tokenizer.decoder.decoder_seanet.layers.\1.block.\2.conv.\3",
    ),
]
# The codec transformers, encoder side and decoder side.
CODEC_TRANSFORMERS = {
    "encoder_transformer": "audio_tokenizer.encoder.encoder_transformer",
    "decoder_transformer": "audio_tokenizer.decoder.decoder_transformer",
}
CODEC_TRANSFORMER_MAPPING = [
    (r"self_attn\.out_proj\.weight", "self_attn.o_proj.weight"),
    (r"linear1\.weight", "mlp.fc1.weight"),
    (r"linear2\.weight", "mlp.fc2.weight"),
    (r"norm1\.(weight|bias)", r"input_layernorm.\1"),
    (r"norm2\.(weight|bias)", r"post_attention_layernorm.\1"),
    (r"layer_scale_1\.scale", "self_attn_layer_scale.scale"),
    (r"layer_scale_2\.scale", "mlp_layer_scale.scale"),
]
TEXT_EMBEDDINGS = "flow_lm.conditioner.embed.weight"
VOICE_BOS = "flow_lm.bos_before_voice"


# The chat template lays out the prompt: the voice BOS and the voice placeholder (expanded by the processor to one token
# per voice frame) when the user turn has an `audio` item (a recording, or an empty item for precomputed latents), the
# text and, as the generation prompt, the first audio position (`<|audio_bos|>`, the BOS latent). Without an `audio`
# item, the voice is the cache of its prefix (`past_key_values`, the shipped voices). The tokenizer prepares the text the
# way the original's `prepare_text_prompt` does, with the language's settings. With an `assistant` turn holding the
# target audio (training), the voice placeholder is only there when the user turn has a voice, the text may be empty
# (classifier-free guidance dropout), and the target is the first audio position and one latent placeholder, which the
# processor expands to the target's frames.
CHAT_TEMPLATE = """
{%- set training = messages | length == 2 -%}
{%- if messages | length > 2 or messages[0]['role'] != 'user' or (training and messages[1]['role'] != 'assistant') -%}
    {{- raise_exception('Pocket TTS takes one `user` message per conversation, followed by the `assistant` target audio to train on.') -}}
{%- endif -%}
{%- if training -%}
    {%- set target = messages[1]['content'] -%}
    {%- if target is string or target | length != 1 or target[0]['type'] != 'audio' -%}
        {{- raise_exception('The `assistant` turn holds one `audio` item, the target.') -}}
    {%- endif -%}
{%- endif -%}
{%- set content = messages[0]['content'] -%}
{%- if content is string -%}
    {%- set texts = [content] -%}
    {%- set num_audio = 0 -%}
{%- else -%}
    {%- set texts = content | selectattr('type', 'equalto', 'text') | map(attribute='text') | list -%}
    {%- set num_audio = content | selectattr('type', 'equalto', 'audio') | list | length -%}
{%- endif -%}
{%- if texts | length > 1 or (texts | length == 0 and not training) or num_audio > 1 -%}
    {{- raise_exception('The `user` turn holds one `text` item and at most one `audio` item (the voice).') -}}
{%- endif -%}
{%- set text = (texts[0] if texts else '') | trim -%}
{%- if training -%}
    {{- '<|audio_prompt_bos|>' ~ ('<|audio_prompt|>' if num_audio else '') ~ text ~ '<|audio_bos|><|audio_prompt|>' -}}
{%- else -%}
    {%- if not text -%}
        {{- raise_exception('Text prompt cannot be empty') -}}
    {%- endif -%}
    {#- A voice given as the cache of its prefix (the voice BOS and frames) leaves the text alone in the prompt. -#}
    {{- ('<|audio_prompt_bos|><|audio_prompt|>' if num_audio else '') ~ text -}}
    {%- if add_generation_prompt -%}
        {{- '<|audio_bos|>' -}}
    {%- endif -%}
{%- endif -%}
"""


def permute_for_rope(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reorders the rows of a q/k projection from interleaved RoPE pairs to the rotate-half layout."""
    out_dim, in_dim = weight.shape
    return weight.view(num_heads, out_dim // num_heads // 2, 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim)


def build_audio_tokenizer_config() -> PocketTTSAudioTokenizerConfig:
    # The codec of every released checkpoint (identical weights): `mimi` section of the original configs. The Python
    # original attends to 250 keys including the current one and uses the tanh approximation of GELU.
    return PocketTTSAudioTokenizerConfig(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, sliding_window=250
    )


def build_config(checkpoint: dict) -> PocketTTSConfig:
    return PocketTTSConfig(
        flow_head_config={"hidden_size": 512, "num_hidden_layers": 6, "flow_type": checkpoint["flow_type"]},
        # n_bins (4000) + 1 embedding rows in the checkpoint; row 4000 is the audio-prompt placeholder (never read), then
        # the audio-latent placeholder, the voice BOS and the first audio position (never read either)
        vocab_size=4004,
        hidden_size=1024,
        num_hidden_layers=checkpoint["num_hidden_layers"],
        num_attention_heads=16,
        intermediate_size=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        dtype="float32",
    )


def convert_state_dict(
    original: dict[str, torch.Tensor], config: PocketTTSConfig, audio_tokenizer_config: PocketTTSAudioTokenizerConfig
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Returns the state dicts of the TTS model and of the audio tokenizer."""
    converted, unmatched = {}, []
    for name, tensor in original.items():
        tensor = tensor.float()
        if name in (TEXT_EMBEDDINGS, VOICE_BOS):
            continue
        if (match := re.match(r"^flow_lm\.transformer\.layers\.(\d+)\.self_attn\.in_proj\.weight$", name)) or (
            match := re.match(
                r"^mimi\.(encoder|decoder)_transformer\.transformer\.layers\.(\d+)\.self_attn\.in_proj\.weight$", name
            )
        ):
            if name.startswith("flow_lm"):
                prefix, num_heads = f"model.layers.{match.group(1)}.self_attn", config.num_attention_heads
            else:
                transformer = CODEC_TRANSFORMERS[f"{match.group(1)}_transformer"]
                prefix = f"{transformer}.layers.{match.group(2)}.self_attn"
                num_heads = audio_tokenizer_config.num_attention_heads
            query, key, value = tensor.chunk(3, dim=0)
            converted[f"{prefix}.q_proj.weight"] = permute_for_rope(query, num_heads)
            converted[f"{prefix}.k_proj.weight"] = permute_for_rope(key, num_heads)
            converted[f"{prefix}.v_proj.weight"] = value
            continue
        if match := re.match(r"^mimi\.(encoder|decoder)_transformer\.transformer\.layers\.(\d+)\.(.+)$", name):
            for pattern, replacement in CODEC_TRANSFORMER_MAPPING:
                if re.fullmatch(pattern, match.group(3)):
                    suffix = re.sub(pattern, replacement, match.group(3))
                    converted[
                        f"{CODEC_TRANSFORMERS[match.group(1) + '_transformer']}.layers.{match.group(2)}.{suffix}"
                    ] = tensor
                    break
            else:
                unmatched.append(name)
            continue
        for pattern, replacement in STATE_DICT_MAPPING:
            if re.match(pattern, name):
                converted[re.sub(pattern, replacement, name)] = tensor
                break
        else:
            unmatched.append(name)
    if unmatched:
        raise ValueError(f"Unmapped tensors in the checkpoint: {unmatched}")

    # Text embeddings (4001 rows), then the audio-latent placeholder (never read, zero), the learnt vector the original
    # prepends to the voice prompt, which the processor places as `<|audio_prompt_bos|>`, and the first audio position
    # (never read, zero: the model embeds the BOS latent there).
    text_embeddings = original[TEXT_EMBEDDINGS].float()
    placeholder = text_embeddings.new_zeros(1, text_embeddings.shape[1])
    voice_bos = original[VOICE_BOS].float().reshape(1, -1)
    converted["model.embed_tokens.weight"] = torch.cat([text_embeddings, placeholder, voice_bos, placeholder], dim=0)
    codec_prefix = "audio_tokenizer."
    audio_tokenizer = {
        name[len(codec_prefix) :]: tensor for name, tensor in converted.items() if name.startswith(codec_prefix)
    }
    model = {name: tensor for name, tensor in converted.items() if not name.startswith(codec_prefix)}
    return model, audio_tokenizer


def convert_tokenizer(checkpoint: dict, output_dir: Path):
    repo_id, filename, revision = checkpoint["tokenizer"]
    tokenizer = Tokenizer.from_file(hf_hub_download(repo_id, filename, revision=revision))
    # Appended in this order, they get the ids 4000, 4001, 4002 and 4003 of the config.
    tokenizer.add_special_tokens(
        [
            AddedToken(AUDIO_PROMPT_TOKEN, special=True, normalized=False),
            AddedToken(AUDIO_LATENT_TOKEN, special=True, normalized=False),
            AddedToken(AUDIO_PROMPT_BOS_TOKEN, special=True, normalized=False),
            AddedToken(AUDIO_BOS_TOKEN, special=True, normalized=False),
        ]
    )
    tokenizer.save(str(output_dir / "tokenizer.json"))
    tokenizer_config = {
        "tokenizer_class": "PocketTTSTokenizer",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        # Call defaults of the processor (TRF019) travel with the tokenizer; the Unigram model has no post-processor,
        # so `add_special_tokens` adds nothing and needs no default.
        "padding": True,
        "padding_side": "left",
        "model_input_names": ["input_ids", "attention_mask"],
        "clean_up_tokenization_spaces": False,
        "model_specific_special_tokens": {
            "audio_prompt_token": AUDIO_PROMPT_TOKEN,
            "audio_latent_token": AUDIO_LATENT_TOKEN,
            "audio_prompt_bos_token": AUDIO_PROMPT_BOS_TOKEN,
            "audio_bos_token": AUDIO_BOS_TOKEN,
        },
        **checkpoint["text"],
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    return AutoTokenizer.from_pretrained(output_dir)


def convert_voices(checkpoint_name: str, checkpoint: dict, config: PocketTTSConfig, output_dir: Path):
    """
    The checkpoint's predefined voices, shipped by Kyutai as the KV cache of the voice prefix (the voice BOS and the voice
    frames) of every layer, saved in the transformers cache layout: `layers.{i}.keys` / `layers.{i}.values` of shape
    `(1, num_heads, num_positions, head_dim)`, the keys moved from interleaved RoPE pairs to rotate-half like the weights.
    """
    (output_dir / "voices").mkdir(exist_ok=True)
    folder = f"languages/{checkpoint_name}/embeddings/"
    files = HfApi().list_repo_files(ORIGINAL_REPO, revision=checkpoint["weights_revision"])
    for filename in sorted(f for f in files if f.startswith(folder) and f.endswith(".safetensors")):
        state = load_file(hf_hub_download(ORIGINAL_REPO, filename, revision=checkpoint["weights_revision"]))
        tensors = {}
        for layer in range(config.num_hidden_layers):
            prefix = f"transformer.layers.{layer}.self_attn/"
            cache = state[prefix + "cache"].float()  # (2, batch, positions, heads, head_dim): keys, values
            if int(state[prefix + "pad"]) != 0 or int(state[prefix + "offset"]) != cache.shape[2]:
                raise ValueError(f"{filename}: unexpected cache state at layer {layer}")
            keys, values = cache[0].transpose(1, 2), cache[1].transpose(1, 2)
            head_dim = keys.shape[-1]
            keys = keys.unflatten(-1, (head_dim // 2, 2)).transpose(-1, -2).flatten(-2)
            tensors[f"layers.{layer}.keys"] = keys.contiguous()
            tensors[f"layers.{layer}.values"] = values.contiguous()
        save_file(tensors, output_dir / "voices" / filename.split("/")[-1])


def load_voice(path: Path, config: PocketTTSConfig) -> DynamicCache:
    state = load_file(path)
    layers = [(state[f"layers.{i}.keys"], state[f"layers.{i}.values"]) for i in range(config.num_hidden_layers)]
    return DynamicCache(layers, config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="english")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--audio_tokenizer_repo",
        type=str,
        default="eustlb/pocket-tts-audio-tokenizer",
        help="Repo id (or local path) of the audio tokenizer, which the processor points to",
    )
    parser.add_argument(
        "--audio_tokenizer_output_dir",
        type=Path,
        default=None,
        help="Where to save the audio tokenizer, shared by all",
    )
    parser.add_argument("--push_to_hub", type=str, default=None, help="Repo id to upload the converted checkpoint to")
    parser.add_argument("--push_audio_tokenizer", action="store_true", help="Upload the audio tokenizer too")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS[args.checkpoint]

    config = build_config(checkpoint)
    audio_tokenizer_config = build_audio_tokenizer_config()
    weights = hf_hub_download(
        ORIGINAL_REPO, f"languages/{args.checkpoint}/model.safetensors", revision=checkpoint["weights_revision"]
    )
    state_dict, audio_tokenizer_state_dict = convert_state_dict(load_file(weights), config, audio_tokenizer_config)

    model = PocketTTSForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # the training-only state (loss weighting, latent statistics counter) starts at its initialisation
    missing = [
        key for key in missing if not any(re.search(pattern, key) for pattern in model._keys_to_ignore_on_load_missing)
    ]
    if missing or unexpected:
        raise ValueError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    model.save_pretrained(args.output_dir)

    # The audio tokenizer is a model of its own, held by the processor and shared by every language checkpoint.
    audio_tokenizer = PocketTTSAudioTokenizerModel(audio_tokenizer_config)
    missing, unexpected = audio_tokenizer.load_state_dict(audio_tokenizer_state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Audio tokenizer missing keys: {missing}\nUnexpected keys: {unexpected}")
    audio_tokenizer.eval()
    if args.audio_tokenizer_output_dir is not None:
        audio_tokenizer.save_pretrained(args.audio_tokenizer_output_dir)
        # Mimi's feature extractor, to use the audio tokenizer on its own (the processor's adds the voice-prompt pause):
        # chunks of one latent frame without overlap zero-pad the audio to a multiple of `frame_size`, as the original does
        EncodecFeatureExtractor(
            feature_size=1,
            sampling_rate=24000,
            chunk_length_s=audio_tokenizer_config.frame_size / 24000,
            overlap=0.0,
        ).save_pretrained(args.audio_tokenizer_output_dir)
    # what the processor's `processor_config.json` points to
    audio_tokenizer.name_or_path = args.audio_tokenizer_repo

    tokenizer = convert_tokenizer(checkpoint, args.output_dir)
    processor = PocketTTSProcessor(
        PocketTTSFeatureExtractor(),
        tokenizer,
        audio_tokenizer=audio_tokenizer,
        chat_template=CHAT_TEMPLATE.strip() + "\n",
    )
    processor.save_pretrained(args.output_dir)
    # The original's defaults: sampling with temperature 0.3 (the noise variance of the flow sampler), one LSD step, and
    # an end-of-speech logit threshold of -4 ignored on the first 6 frames. `generate` returns the padded latents with
    # their mask, which `processor.decode` takes as they are.
    GenerationConfig(
        return_dict_in_generate=True,
        do_sample=True,
        temperature=0.3,
        pad_token_id=config.pad_token_id,
        num_flow_steps=1,
        eos_threshold=-4.0,
        min_frames_before_eos=6,
    ).save_pretrained(args.output_dir)
    convert_voices(args.checkpoint, checkpoint, config, args.output_dir)

    # Reload from disk, fail on any loading issue, and print one cheap sanity value.
    model, loading_info = PocketTTSForConditionalGeneration.from_pretrained(args.output_dir, output_loading_info=True)
    if any(loading_info.values()):
        raise ValueError(f"Loading issues: {loading_info}")
    processor = PocketTTSProcessor.from_pretrained(args.output_dir)
    # the voice is the cache of its prefix: the prompt is the text only
    voice = load_voice(args.output_dir / "voices" / "alba.safetensors", model.config)
    text = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs."
    conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, add_generation_prompt=True)
    with torch.no_grad():
        output = model.generate(**inputs, past_key_values=voice, do_sample=False)
    num_frames = output.audio_latents_mask.sum().item()
    print(f"Sanity: {num_frames} frames generated (the original, english: 68 frames).")

    if args.push_to_hub:
        api = HfApi()
        api.create_repo(args.push_to_hub, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(args.output_dir), repo_id=args.push_to_hub)
    if args.push_audio_tokenizer:
        if args.audio_tokenizer_output_dir is None:
            raise ValueError("`--push_audio_tokenizer` needs `--audio_tokenizer_output_dir`.")
        api = HfApi()
        api.create_repo(args.audio_tokenizer_repo, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(args.audio_tokenizer_output_dir), repo_id=args.audio_tokenizer_repo)


if __name__ == "__main__":
    main()

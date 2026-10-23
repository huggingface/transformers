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
"""
Adds the transformers format to a Pocket TTS checkpoint folder (`kyutai/pocket-tts`, `languages/<checkpoint>/`), next to the
runtime's files: its weights (`model.safetensors`, renamed at load by `conversion_mapping.py`), voices (`embeddings/`) and
`tokenizer.json` are read as they are. Converts the codec, shared by every checkpoint, to an audio tokenizer of its own.
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from tokenizers import AddedToken, Regex, Tokenizer, normalizers

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
# holds the text settings of the language (`replace_characters`, `remove_semicolons`), which the normalizer applies.
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
# The codec's tensors (regex) -> the audio tokenizer's names. Applied in order, first match wins. Fused `in_proj` weights are
# split into q/k/v with the q/k rows permuted for rotate-half RoPE (the original rotates interleaved pairs), below.
CODEC_MAPPING = [
    # encoder side
    (r"^mimi\.downsample\.conv\.conv\.weight$", "encoder.downsample.conv.weight"),
    (
        r"^mimi\.encoder\.model\.(\d+)\.conv\.(weight|bias)$",
        r"encoder.encoder_seanet.layers.\1.conv.\2",
    ),
    (
        r"^mimi\.encoder\.model\.(\d+)\.block\.(\d+)\.conv\.(weight|bias)$",
        r"encoder.encoder_seanet.layers.\1.block.\2.conv.\3",
    ),
    # decoder side; `.conv` / `.convtr` both become the `.conv` of the wrapper
    (r"^mimi\.quantizer\.output_proj\.weight$", "latent_proj.weight"),
    (r"^mimi\.upsample\.convtr\.convtr\.weight$", "decoder.upsample.conv.weight"),
    (
        r"^mimi\.decoder\.model\.(\d+)\.(?:conv|convtr)\.(weight|bias)$",
        r"decoder.decoder_seanet.layers.\1.conv.\2",
    ),
    (
        r"^mimi\.decoder\.model\.(\d+)\.block\.(\d+)\.conv\.(weight|bias)$",
        r"decoder.decoder_seanet.layers.\1.block.\2.conv.\3",
    ),
]
# The codec transformers, encoder side and decoder side.
CODEC_TRANSFORMERS = {
    "encoder_transformer": "encoder.encoder_transformer",
    "decoder_transformer": "decoder.decoder_transformer",
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
# The placeholders reuse rows of the original's text table it never reads: its padding row and the `<s>` / `</s>` rows (it
# tokenizes without them). `<|audio_prompt_bos|>` is the row `conversion_mapping.py` appends: the learnt vector the
# original prepends to the voice prompt.
RESERVED_TOKENS = {"<s>": AUDIO_LATENT_TOKEN, "</s>": AUDIO_BOS_TOKEN}
APPENDED_TOKENS = [AUDIO_PROMPT_TOKEN, AUDIO_PROMPT_BOS_TOKEN]
# Next to the runtime's `tokenizer.json`, transformers reads a versioned tokenizer file (`fast_tokenizer_files`).
TRANSFORMERS_TOKENIZER_FILE = "tokenizer.5.0.0.json"
TRANSFORMERS_FILES = ["config.json", "generation_config.json", "processor_config.json", "chat_template.jinja"]
TRANSFORMERS_FILES += [TRANSFORMERS_TOKENIZER_FILE, "tokenizer_config.json"]


# The chat template lays out the prompt: the voice BOS and the voice placeholder (expanded by the processor to one token
# per voice frame) when the user turn has an `audio` item (a recording, or an empty item for precomputed latents), the
# text and, as the generation prompt, the first audio position (`<|audio_bos|>`, the BOS latent). Without an `audio`
# item, the voice is the cache of its prefix (`past_key_values`, the shipped voices). The normalizer of the tokenizer
# prepares the text the way the original's `prepare_text_prompt` does (`build_normalizer`). With an `assistant` turn
# holding the target audio (training), the voice placeholder is only there when the user turn has a voice, the text may be empty
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
        hidden_size=1024,
        num_hidden_layers=checkpoint["num_hidden_layers"],
        num_attention_heads=16,
        intermediate_size=4096,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        dtype="float32",
    )


def convert_audio_tokenizer_state_dict(
    original: dict[str, torch.Tensor], audio_tokenizer_config: PocketTTSAudioTokenizerConfig
) -> dict[str, torch.Tensor]:
    """The codec's tensors (`mimi.`) as the audio tokenizer's state dict."""
    converted, unmatched = {}, []
    num_heads = audio_tokenizer_config.num_attention_heads
    for name, tensor in original.items():
        if not name.startswith("mimi."):
            continue
        tensor = tensor.float()
        if match := re.match(r"^mimi\.(encoder|decoder)_transformer\.transformer\.layers\.(\d+)\.(.+)$", name):
            prefix = f"{CODEC_TRANSFORMERS[match.group(1) + '_transformer']}.layers.{match.group(2)}"
            if match.group(3) == "self_attn.in_proj.weight":
                query, key, value = tensor.chunk(3, dim=0)
                converted[f"{prefix}.self_attn.q_proj.weight"] = permute_for_rope(query, num_heads)
                converted[f"{prefix}.self_attn.k_proj.weight"] = permute_for_rope(key, num_heads)
                converted[f"{prefix}.self_attn.v_proj.weight"] = value
                continue
            for pattern, replacement in CODEC_TRANSFORMER_MAPPING:
                if re.fullmatch(pattern, match.group(3)):
                    converted[f"{prefix}.{re.sub(pattern, replacement, match.group(3))}"] = tensor
                    break
            else:
                unmatched.append(name)
            continue
        for pattern, replacement in CODEC_MAPPING:
            if re.match(pattern, name):
                converted[re.sub(pattern, replacement, name)] = tensor
                break
        else:
            unmatched.append(name)
    if unmatched:
        raise ValueError(f"Unmapped codec tensors in the checkpoint: {unmatched}")
    return converted


SENTENCE_END_MARKS = ".!?…"
WEAK_PUNCTUATION = r",;:\-–—"
CLOSING_MARKS = r"\"'”’)\]» "


def build_normalizer(
    letters: str, replace_characters: dict[str, str] | None = None, remove_semicolons: bool = False
) -> list:
    """
    The original's `prepare_text_prompt`, with the language's settings, as tokenizer normalizers: strip, apply
    `replace_characters` (then collapse the whitespace, and drop a comma, semicolon or colon right after a sentence mark),
    turn line breaks into spaces, replace semicolons with commas, upper-case the first character (one of `letters`, the
    lowercase letters of the vocabulary) and end the text with sentence-final punctuation.
    """
    steps = [normalizers.Strip()]
    if replace_characters:
        steps += [normalizers.Replace(source, target) for source, target in replace_characters.items()]
        # deleted quotes leave '"Hi?", she said' as 'Hi?, she said': keep the sentence mark only
        steps += [
            normalizers.Replace(Regex(r"\s+"), " "),
            normalizers.Strip(),
            normalizers.Replace(Regex(rf"(?<=[{SENTENCE_END_MARKS}])\s*[,;:]"), ""),
        ]
    steps += [normalizers.Replace("\n", " "), normalizers.Replace("\r", " "), normalizers.Replace("  ", " ")]
    if remove_semicolons:
        steps.append(normalizers.Replace(";", ","))
    # a normalizer cannot upper-case: one step per lowercase letter of the vocabulary
    steps += [normalizers.Replace(Regex(f"^{letter}"), letter.upper()) for letter in letters]
    # The text ends with sentence-final punctuation, then its closing marks: a trailing comma, colon or dash becomes a
    # period, any other ending gets one (after up to 8 closing marks: the lookbehind has a fixed length).
    last_char = rf"[^{SENTENCE_END_MARKS}{WEAK_PUNCTUATION}{CLOSING_MARKS}]"
    ending = "|".join(f"{last_char}[{CLOSING_MARKS}]{{{n}}}" for n in range(9))
    steps += [
        normalizers.Replace(Regex(rf"[{WEAK_PUNCTUATION} ]*[{WEAK_PUNCTUATION}] *(?=[{CLOSING_MARKS}]*$)"), "."),
        normalizers.Replace(Regex(f"(?<={ending})$"), "."),
    ]
    return steps


def convert_tokenizer(checkpoint: dict, output_dir: Path):
    """
    The transformers tokenizer: the runtime's `tokenizer.json` with the placeholders on the rows the original never reads
    (and the appended voice BOS), and the runtime's text preparation as its normalizer, and the tokenizer config.
    """
    repo_id, filename, revision = checkpoint["tokenizer"]
    tokenizer = json.loads(Path(hf_hub_download(repo_id, filename, revision=revision)).read_text(encoding="utf-8"))
    for token in tokenizer["added_tokens"]:
        token["content"] = RESERVED_TOKENS.get(token["content"], token["content"])
    for piece in tokenizer["model"]["vocab"]:
        piece[0] = RESERVED_TOKENS.get(piece[0], piece[0])
    backend = Tokenizer.from_str(json.dumps(tokenizer))
    # the text preparation, then the original's normalizer (the word-start marker): the last step, which the processor
    # keeps alone for training transcripts, as the original's training tokenizes them as they are
    letters = "".join(sorted(c for c in {c for piece in backend.get_vocab() for c in piece} if c.upper() != c))
    backend.normalizer = normalizers.Sequence([*build_normalizer(letters, **checkpoint["text"]), backend.normalizer])
    # appended in this order, they get the ids 4000 (the original's padding row) and 4001 (the appended voice BOS)
    backend.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in APPENDED_TOKENS])
    backend.save(str(output_dir / TRANSFORMERS_TOKENIZER_FILE))
    tokenizer_config = {
        "tokenizer_class": "TokenizersBackend",
        "fast_tokenizer_files": [TRANSFORMERS_TOKENIZER_FILE],
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        # Call defaults of the processor (TRF019) travel with the tokenizer; the Unigram model has no post-processor,
        # so `add_special_tokens` adds nothing and needs no default.
        "padding": True,
        "padding_side": "left",
        "return_tensors": "pt",
        "model_input_names": ["input_ids", "attention_mask"],
        "clean_up_tokenization_spaces": False,
        "model_specific_special_tokens": {
            "audio_prompt_token": AUDIO_PROMPT_TOKEN,
            "audio_latent_token": AUDIO_LATENT_TOKEN,
            "audio_prompt_bos_token": AUDIO_PROMPT_BOS_TOKEN,
            "audio_bos_token": AUDIO_BOS_TOKEN,
        },
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    return AutoTokenizer.from_pretrained(output_dir)


def load_voice(path: Path, config: PocketTTSConfig) -> DynamicCache:
    """
    One of the runtime's voice states (`embeddings/<voice>.safetensors`, also what `pocket-tts export-voice` writes): the
    KV cache of the voice prefix (the voice BOS and the voice frames) of every layer, keys moved from interleaved RoPE
    pairs to rotate-half like the weights.
    """
    state = load_file(path)
    layers = []
    for i in range(config.num_hidden_layers):
        # (2, 1, length, num_heads, head_dim) -> keys and values, each (1, num_heads, length, head_dim)
        keys, values = state[f"transformer.layers.{i}.self_attn/cache"].float().transpose(2, 3)
        keys = keys.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
        layers.append((keys, values))
    return DynamicCache(layers, config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="english")
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Gets the checkpoint's weights and the transformers files"
    )
    parser.add_argument(
        "--audio_tokenizer_repo",
        type=str,
        default="eustlb/pocket-tts-mimi",
        help="Repo id (or local path) of the audio tokenizer, which the processor points to",
    )
    parser.add_argument(
        "--audio_tokenizer_output_dir",
        type=Path,
        default=None,
        help="Where to save the audio tokenizer, shared by all",
    )
    parser.add_argument("--push_to_hub", type=str, default=None, help="Repo id to add the transformers files to")
    parser.add_argument("--create_pr", action="store_true", help="Open a pull request instead of committing")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS[args.checkpoint]
    folder = f"languages/{args.checkpoint}"

    # The weights stay in the runtime's layout (copied for the loading check below, not uploaded).
    weights = hf_hub_download(ORIGINAL_REPO, f"{folder}/model.safetensors", revision=checkpoint["weights_revision"])
    shutil.copy(weights, args.output_dir / "model.safetensors")
    config = build_config(checkpoint)
    config.save_pretrained(args.output_dir)

    # The audio tokenizer is a model of its own, held by the processor and shared by every checkpoint.
    audio_tokenizer_config = build_audio_tokenizer_config()
    audio_tokenizer = PocketTTSAudioTokenizerModel(audio_tokenizer_config)
    audio_tokenizer_state_dict = convert_audio_tokenizer_state_dict(load_file(weights), audio_tokenizer_config)
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

    processor = PocketTTSProcessor(
        PocketTTSFeatureExtractor(),
        convert_tokenizer(checkpoint, args.output_dir),
        audio_tokenizer=audio_tokenizer,
        chat_template=CHAT_TEMPLATE.strip() + "\n",
    )
    # only the processor's own files: the tokenizer's stay as written above
    with tempfile.TemporaryDirectory() as processor_dir:
        processor.save_pretrained(processor_dir)
        for name in ("processor_config.json", "chat_template.jinja"):
            shutil.copy(Path(processor_dir) / name, args.output_dir / name)
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

    # Reload from the folder (the runtime's weights through the conversion mapping), fail on any loading issue, and
    # print one cheap sanity value, with one of the runtime's voices.
    model, loading_info = PocketTTSForConditionalGeneration.from_pretrained(args.output_dir, output_loading_info=True)
    if any(loading_info.values()):
        raise ValueError(f"Loading issues: {loading_info}")
    if args.audio_tokenizer_output_dir is not None:
        audio_tokenizer.name_or_path = str(args.audio_tokenizer_output_dir)
    processor = PocketTTSProcessor(
        PocketTTSFeatureExtractor(),
        AutoTokenizer.from_pretrained(args.output_dir),
        audio_tokenizer=audio_tokenizer,
        chat_template=CHAT_TEMPLATE.strip() + "\n",
    )
    voice_file = hf_hub_download(
        ORIGINAL_REPO, f"{folder}/embeddings/alba.safetensors", revision=checkpoint["weights_revision"]
    )
    voice = load_voice(Path(voice_file), model.config)
    text = "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs."
    conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, add_generation_prompt=True)
    with torch.no_grad():
        output = model.generate(**inputs, past_key_values=voice, do_sample=False)
    num_frames = output.audio_latents_mask.sum().item()
    print(f"Sanity: {num_frames} frames generated (the original, english: 68 frames).")

    if args.push_to_hub:
        HfApi().upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.push_to_hub,
            path_in_repo=folder,
            allow_patterns=TRANSFORMERS_FILES,
            commit_message=f"transformers 🤗: {args.checkpoint}",
            create_pr=args.create_pr,
        )


if __name__ == "__main__":
    main()

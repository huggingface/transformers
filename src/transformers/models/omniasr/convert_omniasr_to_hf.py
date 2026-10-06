# Copyright 2026 The HuggingFace Inc. team.
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


import argparse
import os
import re
import urllib.request

import torch
from fairseq2.models.llama import LLaMAConfig
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.config import Wav2Vec2Config
from fairseq2.runtime.config_registry import get_config
from fairseq2.runtime.dependency import get_dependency_resolver
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.config import ModelType, Wav2Vec2LlamaConfig

from transformers import (
    LlamaConfig,
    OmniASRAudioConfig,
    OmniASRConfig,
    OmniASRFeatureExtractor,
    OmniASRForConditionalGeneration,
    OmniASRProcessor,
    TokenizersBackend,
    logging,
)
from transformers.convert_slow_tokenizer import OmniASRConverter


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# fmt: off
ENCODER_KEY_MAPPING = {
    r"^encoder_frontend\.feature_extractor\.layers\.":      "{encoder}subsampling.conv_layers.",
    r"^encoder_frontend\.post_extract_layer_norm\.":         "{encoder}subsampling.layer_norm.",
    r"^encoder_frontend\.model_dim_proj\.":                  "{encoder}subsampling.projection.",
    r"^encoder_frontend\.pos_encoder\.conv\.":               "{encoder}encode_positions.conv.",
    r"^encoder\.layer_norm\.":                               "{encoder}layer_norm.",
    r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.":     r"{encoder}layers.\1.layer_norm.",
    r"^encoder\.layers\.(\d+)\.self_attn\.output_proj\.":   r"{encoder}layers.\1.attention.out_proj.",
    r"^encoder\.layers\.(\d+)\.self_attn\.":                r"{encoder}layers.\1.attention.",
    r"^encoder\.layers\.(\d+)\.ffn_layer_norm\.":           r"{encoder}layers.\1.final_layer_norm.",
    r"^encoder\.layers\.(\d+)\.ffn\.inner_proj\.":          r"{encoder}layers.\1.feed_forward.intermediate_dense.",
    r"^encoder\.layers\.(\d+)\.ffn\.output_proj\.":         r"{encoder}layers.\1.feed_forward.output_dense.",
}

LLM_KEY_MAPPING = {
    r"^final_proj\.":                                         "lm_head.",
    r"^encoder_proj\.":                                       "model.multi_modal_projector.",
    r"^text_frontend\.":                                      "model.language_model.embed_tokens.",
    r"^llama_decoder\.layer_norm\.":                          "model.language_model.norm.",
    r"^llama_decoder\.layers\.(\d+)\.self_attn_layer_norm\.":  r"model.language_model.layers.\1.input_layernorm.",
    r"^llama_decoder\.layers\.(\d+)\.self_attn\.output_proj\.": r"model.language_model.layers.\1.self_attn.o_proj.",
    r"^llama_decoder\.layers\.(\d+)\.self_attn\.":             r"model.language_model.layers.\1.self_attn.",
    r"^llama_decoder\.layers\.(\d+)\.ffn_layer_norm\.":        r"model.language_model.layers.\1.post_attention_layernorm.",
    r"^llama_decoder\.layers\.(\d+)\.ffn\.gate_proj\.":        r"model.language_model.layers.\1.mlp.gate_proj.",
    r"^llama_decoder\.layers\.(\d+)\.ffn\.inner_proj\.":       r"model.language_model.layers.\1.mlp.up_proj.",
    r"^llama_decoder\.layers\.(\d+)\.ffn\.output_proj\.":      r"model.language_model.layers.\1.mlp.down_proj.",
}

# Rows reserved right after the tokenizer's vocabulary, in the order the tokenizer declares them: `<extra_id_0>`
# (the LID marker) and `<extra_id_1>` (audio placeholder). The language tokens follow them, see `language_tokens`.
NUM_RESERVED_TOKENS = 2

# The language-agnostic mode the original model reaches by looking up row 0 of its language embedding table.
LANGUAGE_AGNOSTIC = "auto"

# The decoder prompt `audio | lid_marker | language | bos`, from which the transcription is decoded.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
        "{%- if message['role'] == 'user' -%}"
            "{%- set ns = namespace(language='" + LANGUAGE_AGNOSTIC + "') -%}"
            "{%- for item in message['content'] -%}"
                "{%- if item['type'] == 'audio' -%}<extra_id_1>"
                "{%- elif item['type'] == 'language' -%}{%- set ns.language = item['language'] | lower -%}"
                "{%- endif -%}"
            "{%- endfor -%}"
            "<extra_id_0><|lang:{{ ns.language }}|><s>"
        "{%- elif message['role'] == 'assistant' -%}"
            "{%- for item in message['content'] if item['type'] == 'text' -%}{{ item['text'] }}{%- endfor -%}</s>"
        "{%- endif -%}"
    "{%- endfor -%}"
)
# fmt: on


def get_encoder_key_mapping(encoder_prefix):
    """The mapping of the speech encoder's weights, which sit under `encoder_prefix` in the Transformers model."""
    return {
        pattern: replacement.replace("{encoder}", encoder_prefix)
        for pattern, replacement in ENCODER_KEY_MAPPING.items()
    }


def convert_state_dict(state_dict, key_mapping):
    """Rename the keys of `state_dict` with the first pattern of `key_mapping` each one matches."""
    converted = {}
    sources = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in key_mapping.items():
            new_key, num_matches = re.subn(pattern, replacement, key)
            if num_matches:
                break
        if new_key in converted:
            raise ValueError(
                f"Key collision while renaming: both `{sources[new_key]}` and `{key}` map to `{new_key}`."
            )
        converted[new_key] = value
        sources[new_key] = key
    return converted


def load_state_dict(hf_model, state_dict):
    """Load the converted `state_dict` onto `hf_model`, failing loudly on any key left over."""
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = set({k for k in extra_keys if "num_updates" not in k})  # filter unnecessary param
    if len(extra_keys) != 0:
        raise ValueError(f"{len(extra_keys)} extra keys found: {extra_keys}")
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(missing_keys) != 0:
        raise ValueError(f"{len(missing_keys)} missing keys found: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=True)
    n_params = param_count(hf_model)

    logger.info(f"model loaded: {round(n_params / 1e6, 1)}M params")

    hf_model.eval()
    return hf_model


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters())


def _get_pos_conv(hf_model):
    """Locate the positional convolution, whichever OmniASR class wraps the speech encoder."""
    for name, module in hf_model.named_modules():
        if name.endswith("encode_positions"):
            return module.conv
    raise ValueError(f"Could not find `encode_positions` in {hf_model.__class__.__name__}.")


def apply_weight_norm(hf_model):
    torch.nn.utils.weight_norm(_get_pos_conv(hf_model), name="weight", dim=2)


def remove_weight_norm(hf_model):
    torch.nn.utils.remove_weight_norm(_get_pos_conv(hf_model), name="weight")


def get_device_and_dtype(bfloat16):
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available, conversion will be done on CPU but it is STRONGLY recommended to use GPU for proper removal of weight norm."
        )
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    dtype = torch.bfloat16 if bfloat16 else torch.float32
    return device, dtype


def get_original_encoder_config(model_card, target_vocab_size):
    """The original config of the speech encoder (with its CTC head) of `model_card`."""
    resolver = get_dependency_resolver()
    if "300m" in model_card.lower():
        encoder_config_name = "large_lv60k"
    elif "1b" in model_card.lower():
        encoder_config_name = "1b"
    elif "3b" in model_card.lower():
        encoder_config_name = "3b"
    elif "7b" in model_card.lower():
        encoder_config_name = "7b"
    else:
        raise ValueError(f"Unsupported size, got {model_card}")

    original_config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
    original_config.encoder_config = get_config(resolver, Wav2Vec2Config, encoder_config_name).encoder_config

    original_config.encoder_config.dropout_p = 0.0
    original_config.encoder_config.attn_dropout_p = 0.0
    original_config.encoder_config.ffn_inner_dropout_p = 0.1
    original_config.encoder_config.layer_drop_p = 0.1

    original_config.use_masking = False
    original_config.max_temporal_mask_prob = 0.0
    original_config.max_spatial_mask_prob = 0.0
    original_config.target_vocab_size = target_vocab_size
    return original_config


def convert_audio_config(original_config):
    """The `OmniASRAudioConfig` of the speech encoder described by `original_config`."""
    conv_dim, conv_kernel, conv_stride = zip(*original_config.encoder_config.feature_extractor_layer_descs)
    if not original_config.encoder_config.feature_extractor_layer_norm_convs:
        raise ValueError(
            "OmniASR only implements layer-normed feature extractor convolutions, but the original config has "
            "`feature_extractor_layer_norm_convs=False`."
        )
    return OmniASRAudioConfig(
        hidden_size=original_config.encoder_config.model_dim,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        conv_bias=original_config.encoder_config.feature_extractor_bias,
        attention_dropout=original_config.encoder_config.attn_dropout_p,
        num_hidden_layers=original_config.encoder_config.num_encoder_layers,
        num_attention_heads=original_config.encoder_config.num_encoder_attn_heads,
        num_conv_pos_embeddings=original_config.encoder_config.pos_conv_kernel_size,
        num_conv_pos_embedding_groups=original_config.encoder_config.num_pos_conv_groups,
        hidden_dropout=original_config.encoder_config.ffn_inner_dropout_p,
        activation_dropout=original_config.encoder_config.ffn_inner_dropout_p,
        intermediate_size=original_config.encoder_config.ffn_inner_dim,
        layerdrop=original_config.encoder_config.layer_drop_p,
    )


def get_feature_extractor():
    return OmniASRFeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=True,
    )


def download_tokenizer(model_card):
    """Download the SentencePiece model of `model_card`, and return its local path."""
    # Release v1: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v1.yaml
    # Release v2: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/cards/models/rc_models_v2.yaml
    if "v2" in model_card:
        tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model"
    elif model_card in ["omniASR_LLM_7B"]:
        tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model"
    else:
        tokenizer_url = "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model"
    tokenizer_path = os.path.join(os.getcwd(), os.path.basename(tokenizer_url))
    urllib.request.urlretrieve(tokenizer_url, tokenizer_path)
    return tokenizer_path


def convert_tokenizer(tokenizer_path, padding_side, tokenizer_class=TokenizersBackend):
    converter = OmniASRConverter(tokenizer_path)
    trainer_spec = converter.proto.trainer_spec
    # No special token is appended: the chat template (ALM) and the processor (CTC labels) add them themselves.
    return tokenizer_class(
        tokenizer_object=converter.converted(),
        bos_token=trainer_spec.bos_piece,
        eos_token=trainer_spec.eos_piece,
        unk_token=trainer_spec.unk_piece,
        pad_token=trainer_spec.pad_piece,
        clean_up_tokenization_spaces=False,
        padding_side=padding_side,
    )


def _convert_model(original_model, hf_model, verbose=False):
    """Rename the original state dict onto `hf_model` and load it, failing loudly on any key left over."""

    state_dict = original_model.state_dict()
    print("Number of keys in original model :", len(state_dict))
    print("Number of keys in HF model       : ", len(hf_model.state_dict()))

    # The speech encoder sits under `model.audio_tower`, following Voxtral's naming.
    key_mapping = {**get_encoder_key_mapping("model.audio_tower."), **LLM_KEY_MAPPING}
    state_dict = convert_state_dict(state_dict, key_mapping)

    # Rearrange Q/K projection weights for RoPE compatibility (interleaved -> half-split)
    # Based on convert_pe_audio_video_to_hf.py and convert_perception_lm_weights_to_hf.py
    num_heads = 8
    num_key_value_heads = 8
    head_dim = 512
    for k in list(state_dict.keys()):
        # Only the decoder's Q/K weights: the encoder has no RoPE
        if "language_model.layers" in k and ".self_attn.q_proj.weight" in k:
            weight = state_dict[k]
            dim1, dim2 = weight.shape
            state_dict[k] = weight.view(num_heads, head_dim // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
            if verbose:
                print(f"Permuted {k} for RoPE: {weight.shape} -> {state_dict[k].shape}")
        elif "language_model.layers" in k and ".self_attn.k_proj.weight" in k:
            weight = state_dict[k]
            dim1, dim2 = weight.shape
            state_dict[k] = (
                weight.view(num_key_value_heads, head_dim // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
            )
            if verbose:
                print(f"Permuted {k} for RoPE: {weight.shape} -> {state_dict[k].shape}")

    # Fold the original's separate language embedding table into the input embeddings, one token per language.
    embed_key = "model.language_model.embed_tokens.weight"
    lang_embeddings = state_dict.pop("lang_embeddings.weight", None)
    if lang_embeddings is not None:
        text_embeddings = state_dict[embed_key]
        num_text_rows = hf_model.config.text_config.vocab_size - lang_embeddings.shape[0]
        if text_embeddings.shape[0] < num_text_rows:
            text_embeddings = torch.cat(
                [
                    text_embeddings,
                    torch.zeros_like(text_embeddings[:1]).repeat(num_text_rows - len(text_embeddings), 1),
                ]
            )
        state_dict[embed_key] = torch.cat([text_embeddings[:num_text_rows], lang_embeddings])
        logger.info(f"Folded {list(lang_embeddings.shape)} language embeddings into {embed_key}")

    # Zero-pad the rows of the tokens added after the original vocabulary (also in `suppress_tokens`).
    for key in ("lm_head.weight", embed_key):
        if key not in state_dict or key not in hf_model.state_dict():
            continue
        src_shape = state_dict[key].shape
        tgt_shape = hf_model.state_dict()[key].shape
        if src_shape[0] < tgt_shape[0]:
            padding = torch.zeros(
                tgt_shape[0] - src_shape[0],
                src_shape[1],
                dtype=state_dict[key].dtype,
                device=state_dict[key].device,
            )
            state_dict[key] = torch.cat([state_dict[key], padding], dim=0)
            logger.info(f"Padded {key} from {list(src_shape)} to {list(state_dict[key].shape)}")

    return load_state_dict(hf_model, state_dict)


@torch.no_grad()
def convert_omniasr_checkpoint(model_card, repo_id=None, bfloat16=False):
    device, dtype = get_device_and_dtype(bfloat16)

    # Only the LLM (with language ID) variants are supported, not the streaming ("Unlimited") and zero-shot LLM
    # variants, whose prompts use other special tokens.
    if model_card is None or "LLM" not in model_card:
        raise ValueError(f"Only the LLM variants are supported, got `model_card={model_card!r}`.")
    if "unlimited" in model_card.lower() or "zs" in model_card.lower():
        raise ValueError(f"The streaming and zero-shot LLM variants are not supported, got {model_card!r}.")

    # 1) Load original model
    pipeline = ASRInferencePipeline(model_card=model_card, device=device, dtype=dtype)
    original_model = pipeline.model
    original_tokenizer = pipeline.tokenizer
    original_config = get_original_encoder_config(model_card, original_tokenizer.vocab_info.size)

    # v2: https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/config.py#L257
    # v1: https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/config.py#L229
    # v2 and v1 are same except for vocab size which we can get programmatically
    llama_config = LLaMAConfig(
        model_dim=4096,
        max_seq_len=8192,
        vocab_size=original_tokenizer.vocab_info.size,
        pad_idx=1,
        num_layers=12,
        num_attn_heads=8,
        num_key_value_heads=8,
        ffn_inner_dim=4096,
        rope_theta=10_000.0,
        dropout_p=0.1,
    )
    original_config_llm = Wav2Vec2LlamaConfig(wav2vec2_asr_config=original_config, llama_config=llama_config)
    original_config_llm.lang_embeddings_p = 0.5
    original_config_llm.n_special_tokens = 1
    original_config_llm.model_type = ModelType.LLM_ASR_LID

    # 2) Initialize Transformers model
    # -- Compute intermediate_size according to original: https://github.com/facebookresearch/fairseq2/blob/7f06d6f4f5d497eec02b1a238d2071eb5dc48df3/src/fairseq2/models/transformer/ffn.py#L298-L300
    intermediate_size = original_config_llm.llama_config.ffn_inner_dim
    inner_dim_scale = original_config_llm.llama_config.ffn_inner_dim_scale
    inner_dim_to_multiple = original_config_llm.llama_config.ffn_inner_dim_multiple_of
    if inner_dim_scale != 1.0:
        intermediate_size = int(intermediate_size * inner_dim_scale)
    if inner_dim_to_multiple != 1:
        intermediate_size = inner_dim_to_multiple * (
            (intermediate_size + inner_dim_to_multiple - 1) // inner_dim_to_multiple
        )
    # One token per row of the original language embedding table, row 0 being the language-agnostic mode.
    num_language_embeddings = len(original_model.lang_embeddings.weight)
    language_tokens = [None] * num_language_embeddings
    language_tokens[0] = f"<|lang:{LANGUAGE_AGNOSTIC}|>"
    for code, index in original_model.lang_mapping.items():
        language_tokens[index] = f"<|lang:{code}|>"
    if None in language_tokens:
        raise ValueError(
            f"`lang_mapping` leaves row {language_tokens.index(None)} of the {num_language_embeddings}-row "
            "language embedding table unnamed, so it cannot be given a token."
        )

    # Frame stacking is only used by the zero-shot variant, which is not supported.
    if original_config_llm.encoder_stacking != 1:
        raise ValueError(f"`encoder_stacking` must be 1, got {original_config_llm.encoder_stacking}.")

    text_config = LlamaConfig(
        vocab_size=original_config_llm.llama_config.vocab_size + NUM_RESERVED_TOKENS + num_language_embeddings,
        hidden_size=original_config_llm.llama_config.model_dim,
        intermediate_size=intermediate_size,
        max_position_embeddings=original_config_llm.llama_config.max_seq_len,
        num_hidden_layers=original_config_llm.llama_config.num_layers,
        num_attention_heads=original_config_llm.llama_config.num_attn_heads,
        num_key_value_heads=original_config_llm.llama_config.num_key_value_heads,
        tie_word_embeddings=original_config_llm.llama_config.tied_embeddings,
        rms_norm_eps=1e-5,
    )

    # The reserved rows sit right after the base vocabulary, in the order the tokenizer declares them:
    # `<extra_id_0>` is the LID marker the original model already used, and `<extra_id_1>` is the
    # placeholder that `OmniASRProcessor` writes into `input_ids` for the audio frames. The language
    # tokens follow, one per row of the original language embedding table.
    lid_marker_id = original_config_llm.llama_config.vocab_size
    config = OmniASRConfig(
        audio_config=convert_audio_config(original_config),
        text_config=text_config,
        bos_token_id=original_config_llm.bos_idx,
        pad_token_id=original_config_llm.pad_idx,
        eos_token_id=original_config_llm.eos_idx,
        audio_token_id=lid_marker_id + 1,
    )
    hf_model = OmniASRForConditionalGeneration(config)
    # None of the tokens past the tokenizer's vocabulary is a valid target: the audio placeholder stands for an
    # embedding that is scattered in, and the LID marker and language tokens only ever open a prompt. Their
    # `lm_head` rows are zeroed rather than trained, so they have to be suppressed rather than left to score.
    hf_model.generation_config.suppress_tokens = list(range(lid_marker_id, config.text_config.vocab_size))
    hf_model.to(device).to(dtype)

    # 3) Convert weights
    apply_weight_norm(hf_model)
    print(f"Total parameters (original): {param_count(original_model)}")
    print(f"Total parameters (HF)      : {param_count(hf_model)}")

    hf_model = _convert_model(original_model, hf_model)
    remove_weight_norm(hf_model)

    # 4) Prepare processor (feature extraction and tokenizer)
    tokenizer_path = download_tokenizer(model_card)
    # The prompts are left-padded, so that decoding continues from each prompt's closing BOS for the whole batch.
    tokenizer = convert_tokenizer(tokenizer_path, padding_side="left")
    # The added tokens take the ids that follow the vocabulary, in this order, so they line up with the rows
    # `config` reserves and with the language embeddings folded into the input embeddings above.
    tokenizer.add_special_tokens(
        {"extra_special_tokens": [f"<extra_id_{i}>" for i in range(NUM_RESERVED_TOKENS)] + language_tokens}
    )

    # The chat template writes the decoder prompt, and the convolution geometry counts how many audio placeholders
    # it holds.
    processor = OmniASRProcessor(
        feature_extractor=get_feature_extractor(),
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
        conv_kernel=list(config.audio_config.conv_kernel),
        conv_stride=list(config.audio_config.conv_stride),
    )
    prompt_ids = [config.audio_token_id, lid_marker_id, config.bos_token_id]
    if processor.tokenizer.convert_tokens_to_ids(["<extra_id_1>", "<extra_id_0>", "<s>"]) != prompt_ids:
        raise ValueError(f"The chat template's prompt tokens do not match the config's ids {prompt_ids}.")

    # 5) Upload to hub
    if repo_id:
        logger.info("Pushing model to the Hub ...")
        hf_model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)

    # 6) Cleanup
    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)


"""
Setup
```
pip install omnilingual-asr
python -m pip install --upgrade huggingface_hub
# install torch, torchvision, torchaudio, fairseq
```

Example conversion:
```python
python src/transformers/models/omniasr/convert_omniasr_to_hf.py \
    --model_card omniASR_LLM_300M_v2 \
    --repo_id bezzam/omniasr-llm-300m-v2
```

See here for available models: https://github.com/facebookresearch/omnilingual-asr?tab=readme-ov-file#model-architectures
Original model checkpoints are saved under: ~/.cache/fairseq2/assets/
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_card", default=None, type=str, help="Name of original model in omnilingual-asr")
    parser.add_argument("--repo_id", default=None, type=str, help="The repository ID for pushing the model to the Hub")
    parser.add_argument("--bfloat16", action="store_true", help="Whether to do bfloat16, otherwise default is float32")
    # Original defaults to bfloat16: https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/pipeline.py#L157
    args = parser.parse_args()

    convert_omniasr_checkpoint(args.model_card, args.repo_id, args.bfloat16)

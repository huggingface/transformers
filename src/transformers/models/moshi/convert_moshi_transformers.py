# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Moshi checkpoints."""

import argparse

import safetensors
import sentencepiece
import torch

from transformers import (
    AutoFeatureExtractor,
    GenerationConfig,
    MimiModel,  # initial audio encoder
    MoshiConfig,
    MoshiForConditionalGeneration,
    PreTrainedTokenizerFast,
    logging,
)
from transformers.convert_slow_tokenizer import MoshiConverter


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.mimi")


def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


convert_list = [
    # GENERAL
    ("out_norm", "decoder.model.norm"),
    ("depformer_emb", "depth_decoder.emb"),
    ("depformer_text_emb", "depth_decoder.text_emb"),
    ("text_emb", "decoder.model.emb"),
    ("emb", "embed_tokens"),
    ("text_linear", "decoder.lm_head"),
    ("depformer", "depth_decoder"),
    ("transformer", "decoder.model"),
    # TRANSFORMERS PART
    ("gating.linear_in", "mlp.fc1"),
    ("gating.linear_out", "mlp.fc2"),
    ("self_attn.out_proj", "self_attn.o_proj.linear"),
    ("norm1", "input_layernorm"),
    ("norm2", "post_attention_layernorm"),
    ("layer_scale_1", "self_attn_layer_scale"),
    ("layer_scale_2", "mlp_layer_scale"),
    ("alpha", "weight"),
]


def _preprocess_state_dict(state_dict, config):
    # Moshi original weights are using a gating mechanism

    # pattern for depth transformer:
    # stack(gating.{i}.linear_in)->mlp.fc1
    # stack(gating.{i}.linear_out)->mlp.fc2

    for layer_idx in range(config.depth_decoder_config.num_hidden_layers):
        linear_layers_in = [
            state_dict.pop(f"depformer.layers.{layer_idx}.gating.{i}.linear_in.weight")
            for i in range(config.num_codebooks)
        ]
        linear_layers_out = [
            state_dict.pop(f"depformer.layers.{layer_idx}.gating.{i}.linear_out.weight")
            for i in range(config.num_codebooks)
        ]

        state_dict[f"depth_decoder.layers.{layer_idx}.mlp.fc1.weight"] = torch.stack(linear_layers_in)
        state_dict[f"depth_decoder.layers.{layer_idx}.mlp.fc2.weight"] = torch.stack(linear_layers_out)

    input_projections = []
    lm_heads = []
    for codebook_idx in range(config.num_codebooks):
        input_projections.append(state_dict.pop(f"depformer_in.{codebook_idx}.weight"))
        lm_heads.append(state_dict.pop(f"linears.{codebook_idx}.weight"))

    state_dict["depth_decoder.input_projections.weight"] = torch.stack(input_projections, dim=0)
    state_dict["depth_decoder.lm_heads.weight"] = torch.stack(lm_heads, dim=0)

    return state_dict


def _convert_model(
    state_dict,
    hf_model,
    convert_list,
    device,
    config,
    unwanted_prefix=None,
):
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_heads = int(config.hidden_size // config.head_dim)
    num_key_value_heads = config.num_key_value_heads
    key_value_head_dim = config.num_key_value_heads * head_dim

    state_dict = _preprocess_state_dict(state_dict, config)

    # permute for sliced rotary
    def permute(w, n_heads, dim1=hidden_size, dim2=hidden_size):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for k, v in list(state_dict.items()):
        if "audio_encoder" not in k:
            new_k = k if unwanted_prefix is None else k[len(unwanted_prefix) :]
            for old_layer_name, new_layer_name in convert_list:
                if old_layer_name in new_k:
                    new_k = new_k.replace(old_layer_name, new_layer_name)

            if "alpha" in k:
                state_dict[k] = state_dict[k].squeeze()

            if "in_proj_weight" in new_k:
                # split qkv into query key and value
                mixed_qkv = state_dict.pop(k)
                if "depth_decoder" in new_k:
                    mixed_qkv = mixed_qkv.view(config.num_codebooks, -1, mixed_qkv.shape[-1])

                    qkv_dim = mixed_qkv.size(1) // 3

                    query_layer = mixed_qkv[:, :qkv_dim]
                    key_layer = mixed_qkv[:, qkv_dim : qkv_dim * 2]
                    value_layer = mixed_qkv[:, qkv_dim * 2 :]
                    state_dict[new_k.replace("in_proj_weight", "q_proj.linear.weight")] = query_layer
                    state_dict[new_k.replace("in_proj_weight", "k_proj.linear.weight")] = key_layer

                else:
                    qkv_dim = mixed_qkv.size(0) // 3

                    query_layer = mixed_qkv[:qkv_dim]
                    key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
                    value_layer = mixed_qkv[qkv_dim * 2 :]
                    state_dict[new_k.replace("in_proj_weight", "q_proj.linear.weight")] = permute(
                        query_layer, num_heads, hidden_size, hidden_size
                    )
                    state_dict[new_k.replace("in_proj_weight", "k_proj.linear.weight")] = permute(
                        key_layer, num_key_value_heads, key_value_head_dim, hidden_size
                    )

                state_dict[new_k.replace("in_proj_weight", "v_proj.linear.weight")] = value_layer
            elif "o_proj" in new_k and "depth_decoder" in new_k:
                output_layer = state_dict.pop(k)
                state_dict[new_k] = output_layer.view(config.num_codebooks, -1, output_layer.shape[-1])
            else:
                state_dict[new_k] = state_dict.pop(k)

    # Do the last one by hand
    state_dict["depth_decoder.text_embed_tokens.weight"] = state_dict.pop(
        "depth_decoder.decoder.model.embed_tokens.weight"
    )

    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=True)
    n_params = param_count(hf_model)

    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    mimi_repo_id,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    device = _grab_best_device()

    mimi_model = MimiModel.from_pretrained(mimi_repo_id, torch_dtype=torch.bfloat16)

    if config_path is not None:
        config = MoshiConfig.from_pretrained(config_path)
    else:
        audio_encoder_config = mimi_model.config
        config = MoshiConfig.from_audio_encoder_config(audio_encoder_config)

    model = MoshiForConditionalGeneration(config).to(torch.bfloat16)

    depth_decoder_generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_k=250,
        min_length=config.num_codebooks + 1,
        max_length=config.num_codebooks + 1,
        cache_implementation="sliding_window",
    )

    generation_config = GenerationConfig(
        do_sample=True,
        temp=0.7,
        top_k=25,
        cache_implementation="sliding_window",
        pad_token_id=config.vocab_size,
        bos_token_id=config.vocab_size,
    )
    generation_config.depth_decoder_config = depth_decoder_generation_config.to_diff_dict()

    model.generation_config = generation_config

    original_checkpoint = safetensors.torch.load_file(checkpoint_path)
    if "best_state" in original_checkpoint:
        # we might have a training state saved, in which case discard the yaml results and just retain the weights
        original_checkpoint = original_checkpoint["best_state"]

    audio_checkpoint = mimi_model.state_dict()
    original_checkpoint.update({f"audio_encoder.{key}": value for (key, value) in audio_checkpoint.items()})

    model = _convert_model(original_checkpoint, model, convert_list, device, config)

    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument(
        "--tokenizer_vocab_path", required=False, default=None, type=str, help="Path to original tokenizer vocab file"
    )
    parser.add_argument("--mimi_repo_id", required=True, default=None, type=str, help="Repository id to HF Mimi.")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()

    # convert tokenizer
    if args.tokenizer_vocab_path:
        original_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer_vocab_path)
        tokenizer = MoshiConverter(args.tokenizer_vocab_path).converted()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            chat_template=None,
            unk_token="<unk>",
            model_input_names=["input_ids", "attention_mask"],
            clean_up_tokenization_spaces=False,
            bos_token_id=original_tokenizer.bos_id(),
            eos_token_id=original_tokenizer.eos_id(),
            pad_token_id=original_tokenizer.pad_id(),
        )

        tokenizer.save_pretrained(args.pytorch_dump_folder_path)

        if args.push_to_hub:
            print("Pushing the tokenizer to the hub...")
            tokenizer.push_to_hub(args.push_to_hub)

    # upload feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.mimi_repo_id)
    feature_extractor.save_pretrained(args.pytorch_dump_folder_path)

    if args.push_to_hub:
        print("Pushing the feature extractor to the hub...")
        feature_extractor.push_to_hub(args.push_to_hub)

    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.mimi_repo_id,
        args.config_path,
        args.push_to_hub,
    )

# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Xcodec2 checkpoints."""

import argparse

import safetensors
import torch
import json

from transformers import (
    AutoConfig,
    DacFeatureExtractor,
    Xcodec2Config,
    Xcodec2Model,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.xcodec2")


def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def _convert_model(state_dict, hf_model):
    tensors = {}

    for k in state_dict.keys():
        # replace beta with bias and handle special cases
        if "generator.backbone.transformers" in k:
            if "c_attn" in k:
                # split into 3 tensors
                c_attn_weight = state_dict[k]
                W_q, W_k, W_v = torch.chunk(c_attn_weight, chunks=3, dim=0)

                n_heads = hf_model.config.num_attention_heads
                W_q = permute_for_rope(W_q, n_heads, W_q.shape[0], W_q.shape[1])
                W_k = permute_for_rope(W_k, n_heads, W_k.shape[0], W_k.shape[1])
                k_mod = k.replace("att.", "self_attn.")
                tensors[k_mod.replace("c_attn", "q_proj")] = W_q
                tensors[k_mod.replace("c_attn", "k_proj")] = W_k
                tensors[k_mod.replace("c_attn", "v_proj")] = W_v
            elif "c_proj" in k:
                tensors[k.replace(".att.c_proj", ".self_attn.o_proj")] = state_dict[k]
            elif "att_norm" in k:
                tensors[k.replace("att_norm", "input_layernorm")] = state_dict[k]
            elif "ffn_norm" in k:
                tensors[k.replace("ffn_norm", "post_attention_layernorm")] = state_dict[k]
            else:
                new_k = k.replace("beta", "bias")
                tensors[new_k] = state_dict[k]
        # change weight_g to parametrizations.weight.original0 and weight_v to parametrizations.weight.original1
        elif "weight_g" in k:
            tensors[k.replace("weight_g", "parametrizations.weight.original0")] = state_dict[k]
        elif "weight_v" in k:
            tensors[k.replace("weight_v", "parametrizations.weight.original1")] = state_dict[k]
        else:
            new_k = k.replace("beta", "bias")
            tensors[new_k] = state_dict[k]
    state_dict = tensors
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=True)
    n_params = param_count(hf_model)

    logger.info(f"model loaded: {round(n_params / 1e6, 1)}M params")

    del state_dict

    return hf_model


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path=None,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    # load json
    if config_path is not None:
        with open(config_path, "r") as f:
            model_config = json.load(f)
        # NOTE: `semantic_hidden_size` not needed as inside `semantic_model_config.hidden_size` (below)
        # semantic_hidden_size = model_config["semantic_hidden_size"]
        encoder_hidden_size = model_config["codec_encoder_hidden_size"]
        decoder_hidden_size = model_config["codec_decoder_hidden_size"]
        use_vocos = model_config["use_vocos"]
    else:
        # default to https://huggingface.co/HKUSTAudio/xcodec2/blob/main/config.json
        encoder_hidden_size = 1024
        decoder_hidden_size = 1024
        use_vocos = True

    # create config
    # -- use hardcoded semantic model: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py#L19
    semantic_model_config = AutoConfig.from_pretrained("facebook/w2v-bert-2.0", output_hidden_states=True)

    config = Xcodec2Config(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        use_vocos=use_vocos,
        semantic_model_config=semantic_model_config
    )

    # create model
    if not torch.cuda.is_available():
        raise ValueError("Run this script on a machine with a GPU for weight norm layers to be correctly copied.")
    torch_device = "cuda"
    model = Xcodec2Model(config).to(torch_device).eval()

    original_checkpoint = safetensors.torch.load_file(checkpoint_path)
    if "best_state" in original_checkpoint:
        # we might have a training state saved, in which case discard the yaml results and just retain the weights
        original_checkpoint = original_checkpoint["best_state"]
    model = _convert_model(original_checkpoint, model)

    # create feature extractor
    feature_extractor = DacFeatureExtractor(
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
    )

    # save and upload
    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)

"""
While there is training code on GitHub: https://github.com/zhenye234/X-Codec-2.0
Modeling code on the Hub is used by xcodec2 (pip) package: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py

Given their example usage: https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py
It seems best to use the config and weights from the Hub.

Conversion example usage:
```
# download model and config
wget https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/config.json -P /raid/eric/xcodec2_original
wget https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/model.safetensors -P /raid/eric/xcodec2_original

# run conversion
python src/transformers/models/xcodec2/convert_xcodec2_checkpoint_to_pytorch.py \
    --checkpoint_path /raid/eric/xcodec2_original/model.safetensors \
    --config_path /raid/eric/xcodec2_original/config.json \
    --push_to_hub bezzam/xcodec2
```

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )

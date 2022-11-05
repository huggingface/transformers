# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert VideoMAE checkpoints from the original repository: https://github.com/MCG-NJU/VideoMAE"""

import argparse
import json

import numpy as np
import torch

import gdown
from huggingface_hub import hf_hub_download
from transformers import (
    VideoMAEConfig,
    VideoMAEFeatureExtractor,
    VideoMAEForPreTraining,
    VideoMAEForVideoClassification,
)


def get_videomae_config(model_name):
    config = VideoMAEConfig()

    if "large" in model_name:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.decoder_num_hidden_layers = 12
        config.decoder_num_attention_heads = 8
        config.decoder_hidden_size = 512
        config.decoder_intermediate_size = 2048

    if "finetuned" not in model_name:
        config.use_mean_pooling = False

    if "finetuned" in model_name:
        repo_id = "huggingface/label-files"
        if "kinetics" in model_name:
            config.num_labels = 400
            filename = "kinetics400-id2label.json"
        elif "ssv2" in model_name:
            config.num_labels = 174
            filename = "something-something-v2-id2label.json"
        else:
            raise ValueError("Model name should either contain 'kinetics' or 'ssv2' in case it's fine-tuned.")
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name):
    if "encoder." in name:
        name = name.replace("encoder.", "")
    if "cls_token" in name:
        name = name.replace("cls_token", "videomae.embeddings.cls_token")
    if "decoder_pos_embed" in name:
        name = name.replace("decoder_pos_embed", "decoder.decoder_pos_embed")
    if "pos_embed" in name and "decoder" not in name:
        name = name.replace("pos_embed", "videomae.embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "videomae.embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "videomae.embeddings.norm")
    if "decoder.blocks" in name:
        name = name.replace("decoder.blocks", "decoder.decoder_layers")
    if "blocks" in name:
        name = name.replace("blocks", "videomae.encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name and "bias" not in name:
        name = name.replace("attn", "attention.self")
    if "attn" in name:
        name = name.replace("attn", "attention.attention")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "decoder_embed" in name:
        name = name.replace("decoder_embed", "decoder.decoder_embed")
    if "decoder_norm" in name:
        name = name.replace("decoder_norm", "decoder.decoder_norm")
    if "decoder_pred" in name:
        name = name.replace("decoder_pred", "decoder.decoder_pred")
    if "norm.weight" in name and "decoder" not in name and "fc" not in name:
        name = name.replace("norm.weight", "videomae.layernorm.weight")
    if "norm.bias" in name and "decoder" not in name and "fc" not in name:
        name = name.replace("norm.bias", "videomae.layernorm.bias")
    if "head" in name and "decoder" not in name:
        name = name.replace("head", "classifier")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key.startswith("encoder."):
            key = key.replace("encoder.", "")

        if "qkv" in key:
            key_split = key.split(".")
            if key.startswith("decoder.blocks"):
                dim = config.decoder_hidden_size
                layer_num = int(key_split[2])
                prefix = "decoder.decoder_layers."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                dim = config.hidden_size
                layer_num = int(key_split[1])
                prefix = "videomae.encoder.layer."
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# We will verify our results on a video of eating spaghetti
# Frame indices used: [164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


def convert_videomae_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    config = get_videomae_config(model_name)

    if "finetuned" in model_name:
        model = VideoMAEForVideoClassification(config)
    else:
        model = VideoMAEForPreTraining(config)

    # download original checkpoint, hosted on Google Drive
    output = "pytorch_model.bin"
    gdown.cached_download(checkpoint_url, output, quiet=False)
    files = torch.load(output, map_location="cpu")
    if "model" in files:
        state_dict = files["model"]
    else:
        state_dict = files["module"]
    new_state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(new_state_dict)
    model.eval()

    # verify model on basic input
    feature_extractor = VideoMAEFeatureExtractor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    video = prepare_video()
    inputs = feature_extractor(video, return_tensors="pt")

    if "finetuned" not in model_name:
        local_path = hf_hub_download(repo_id="hf-internal-testing/bool-masked-pos", filename="bool_masked_pos.pt")
        inputs["bool_masked_pos"] = torch.load(local_path)

    outputs = model(**inputs)
    logits = outputs.logits

    model_names = [
        # Kinetics-400 checkpoints (short = pretrained only for 800 epochs instead of 1600)
        "videomae-base-short",
        "videomae-base-short-finetuned-kinetics",
        "videomae-base",
        "videomae-base-finetuned-kinetics",
        "videomae-large",
        "videomae-large-finetuned-kinetics",
        # Something-Something-v2 checkpoints (short = pretrained only for 800 epochs instead of 2400)
        "videomae-base-short-ssv2",
        "videomae-base-short-finetuned-ssv2",
        "videomae-base-ssv2",
        "videomae-base-finetuned-ssv2",
    ]

    # NOTE: logits were tested with image_mean and image_std equal to [0.5, 0.5, 0.5] and [0.5, 0.5, 0.5]
    if model_name == "videomae-base":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7739, 0.7968, 0.7089], [0.6701, 0.7487, 0.6209], [0.4287, 0.5158, 0.4773]])
    elif model_name == "videomae-base-short":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7994, 0.9612, 0.8508], [0.7401, 0.8958, 0.8302], [0.5862, 0.7468, 0.7325]])
        # we verified the loss both for normalized and unnormalized targets for this one
        expected_loss = torch.tensor([0.5142]) if config.norm_pix_loss else torch.tensor([0.6469])
    elif model_name == "videomae-large":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7149, 0.7997, 0.6966], [0.6768, 0.7869, 0.6948], [0.5139, 0.6221, 0.5605]])
    elif model_name == "videomae-large-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.0771, 0.0011, -0.3625])
    elif model_name == "videomae-base-short-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.6588, 0.0990, -0.2493])
    elif model_name == "videomae-base-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.3669, -0.0688, -0.2421])
    elif model_name == "videomae-base-short-ssv2":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.4712, 0.5296, 0.5786], [0.2278, 0.2729, 0.4026], [0.0352, 0.0730, 0.2506]])
    elif model_name == "videomae-base-short-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-0.0537, -0.1539, -0.3266])
    elif model_name == "videomae-base-ssv2":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.8131, 0.8727, 0.8546], [0.7366, 0.9377, 0.8870], [0.5935, 0.8874, 0.8564]])
    elif model_name == "videomae-base-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0.1961, -0.8337, -0.6389])
    else:
        raise ValueError(f"Model name not supported. Should be one of {model_names}")

    # verify logits
    assert logits.shape == expected_shape
    if "finetuned" in model_name:
        assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    else:
        print("Logits:", logits[0, :3, :3])
        assert torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4)
    print("Logits ok!")

    # verify loss, if applicable
    if model_name == "videomae-base-short":
        loss = outputs.loss
        assert torch.allclose(loss, expected_loss, atol=1e-4)
        print("Loss ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and feature extractor to {pytorch_dump_folder_path}")
        feature_extractor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(model_name, organization="nielsr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=1tEhLyskjb755TJ65ptsrafUG2llSwQE1&amp;export=download&amp;confirm=t&amp;uuid=aa3276eb-fb7e-482a-adec-dc7171df14c4",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/Users/nielsrogge/Documents/VideoMAE/Test",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--model_name", default="videomae-base", type=str, help="Name of the model.")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_videomae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub)

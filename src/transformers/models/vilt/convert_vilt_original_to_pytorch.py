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
"""Convert ViLT checkpoints from the original Github repository."""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import (
    BertTokenizer,
    ViltConfig,
    ViltFeatureExtractor,
    ViltForImageAndTextRetrieval,
    ViltForImagesAndTextClassification,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, vqa_model=False, nlvr_model=False, irtr_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"transformer.blocks.{i}.norm1.weight", f"vilt.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm1.bias", f"vilt.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.weight", f"vilt.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.bias", f"vilt.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"transformer.blocks.{i}.norm2.weight", f"vilt.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm2.bias", f"vilt.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.mlp.fc1.weight", f"vilt.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc1.bias", f"vilt.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.weight", f"vilt.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.bias", f"vilt.encoder.layer.{i}.output.dense.bias"))

    # embeddings
    rename_keys.extend(
        [
            # text embeddings
            ("text_embeddings.word_embeddings.weight", "vilt.embeddings.text_embeddings.word_embeddings.weight"),
            (
                "text_embeddings.position_embeddings.weight",
                "vilt.embeddings.text_embeddings.position_embeddings.weight",
            ),
            ("text_embeddings.position_ids", "vilt.embeddings.text_embeddings.position_ids"),
            (
                "text_embeddings.token_type_embeddings.weight",
                "vilt.embeddings.text_embeddings.token_type_embeddings.weight",
            ),
            ("text_embeddings.LayerNorm.weight", "vilt.embeddings.text_embeddings.LayerNorm.weight"),
            ("text_embeddings.LayerNorm.bias", "vilt.embeddings.text_embeddings.LayerNorm.bias"),
            # patch embeddings
            ("transformer.cls_token", "vilt.embeddings.cls_token"),
            ("transformer.patch_embed.proj.weight", "vilt.embeddings.patch_embeddings.projection.weight"),
            ("transformer.patch_embed.proj.bias", "vilt.embeddings.patch_embeddings.projection.bias"),
            ("transformer.pos_embed", "vilt.embeddings.position_embeddings"),
            # token type embeddings
            ("token_type_embeddings.weight", "vilt.embeddings.token_type_embeddings.weight"),
        ]
    )

    # final layernorm + pooler
    rename_keys.extend(
        [
            ("transformer.norm.weight", "vilt.layernorm.weight"),
            ("transformer.norm.bias", "vilt.layernorm.bias"),
            ("pooler.dense.weight", "vilt.pooler.dense.weight"),
            ("pooler.dense.bias", "vilt.pooler.dense.bias"),
        ]
    )

    # classifier head(s)
    if vqa_model:
        # classification head
        rename_keys.extend(
            [
                ("vqa_classifier.0.weight", "classifier.0.weight"),
                ("vqa_classifier.0.bias", "classifier.0.bias"),
                ("vqa_classifier.1.weight", "classifier.1.weight"),
                ("vqa_classifier.1.bias", "classifier.1.bias"),
                ("vqa_classifier.3.weight", "classifier.3.weight"),
                ("vqa_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    elif nlvr_model:
        # classification head
        rename_keys.extend(
            [
                ("nlvr2_classifier.0.weight", "classifier.0.weight"),
                ("nlvr2_classifier.0.bias", "classifier.0.bias"),
                ("nlvr2_classifier.1.weight", "classifier.1.weight"),
                ("nlvr2_classifier.1.bias", "classifier.1.bias"),
                ("nlvr2_classifier.3.weight", "classifier.3.weight"),
                ("nlvr2_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    else:
        pass

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        prefix = "vilt."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


@torch.no_grad()
def convert_vilt_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ViLT structure.
    """

    # define configuration and initialize HuggingFace model
    config = ViltConfig(image_size=384, patch_size=32, tie_word_embeddings=False)
    mlm_model = False
    vqa_model = False
    nlvr_model = False
    irtr_model = False
    if "vqa" in checkpoint_url:
        vqa_model = True
        config.num_labels = 3129
        repo_id = "datasets/huggingface/label-files"
        filename = "vqa2-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        model = ViltForQuestionAnswering(config)
    elif "nlvr" in checkpoint_url:
        nlvr_model = True
        config.num_labels = 2
        config.id2label = {0: "False", 1: "True"}
        config.label2id = {v: k for k, v in config.id2label.items()}
        config.modality_type_vocab_size = 3
        model = ViltForImagesAndTextClassification(config)
    elif "irtr" in checkpoint_url:
        irtr_model = True
        model = ViltForImageAndTextRetrieval(config)
    elif "mlm_itm" in checkpoint_url:
        mlm_model = True
        model = ViltForMaskedLM(config)
    else:
        raise ValueError("Unknown model type")

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]
    rename_keys = create_rename_keys(config, vqa_model, nlvr_model, irtr_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)
    if mlm_model or irtr_model:
        ignore_keys = ["itm_score.fc.weight", "itm_score.fc.bias"]
        for k in ignore_keys:
            state_dict.pop(k, None)

    # load state dict into HuggingFace model
    model.eval()
    if mlm_model:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert missing_keys == ["mlm_score.decoder.bias"]
    else:
        model.load_state_dict(state_dict)

    # Define processor
    feature_extractor = ViltFeatureExtractor(size=384)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = ViltProcessor(feature_extractor, tokenizer)

    # Forward pass on example inputs (image + text)
    if nlvr_model:
        image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
        image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
        text = "The left image contains twice the number of dogs as the right image, and at least two dogs in total are standing."
        encoding_1 = processor(image1, text, return_tensors="pt")
        encoding_2 = processor(image2, text, return_tensors="pt")
        outputs = model(
            input_ids=encoding_1.input_ids,
            pixel_values=encoding_1.pixel_values,
            pixel_values_2=encoding_2.pixel_values,
        )
    else:
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        if mlm_model:
            text = "a bunch of [MASK] laying on a [MASK]."
        else:
            text = "How many cats are there?"
        encoding = processor(image, text, return_tensors="pt")
        outputs = model(**encoding)

    # Verify outputs
    if mlm_model:
        expected_shape = torch.Size([1, 11, 30522])
        expected_slice = torch.tensor([-12.5061, -12.5123, -12.5174])
        assert outputs.logits.shape == expected_shape
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=1e-4)

        # verify masked token prediction equals "cats"
        predicted_id = outputs.logits[0, 4, :].argmax(-1).item()
        assert tokenizer.decode([predicted_id]) == "cats"
    elif vqa_model:
        expected_shape = torch.Size([1, 3129])
        expected_slice = torch.tensor([-15.9495, -18.1472, -10.3041])
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
        assert outputs.logits.shape == expected_shape
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=1e-4)

        # verify vqa prediction equals "2"
        predicted_idx = outputs.logits.argmax(-1).item()
        assert model.config.id2label[predicted_idx] == "2"
    elif nlvr_model:
        expected_shape = torch.Size([1, 2])
        expected_slice = torch.tensor([-2.8721, 2.1291])
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
        assert outputs.logits.shape == expected_shape

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model and processor to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vilt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)

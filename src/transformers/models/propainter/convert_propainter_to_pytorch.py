# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert ViT and non-distilled DeiT checkpoints from the timm library."""


import argparse
import json
from pathlib import Path

import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DeiTImageProcessor, ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    ## OPTICAL FLOW
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Norm1.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Norm1.bias"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.bias"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv2.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv2.bias"))
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        for j in range(layers):
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Conv1.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Conv1.bias"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Conv2.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Conv2.bias"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Norm1.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Norm1.bias"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Norm2.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.FeatureNet.Conv1.layer{j}.{i}.Norm2.bias"))


    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Norm1.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Norm1.bias"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.bias"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv2.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv2.bias"))
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        for j in range(layers):
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Conv1.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Conv1.bias"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Conv2.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Conv2.bias"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Norm1.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Norm1.bias"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Norm2.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.ContextNet.Conv1.layer{j}.{i}.Norm2.bias"))

    for block in ["Conv_c1","Conv_c2","Conv_f1","Conv_f2","Conv_"]:
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.Encoder.{block}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.Encoder.{block}.bias"))

    for block in ["Conv_z1","Conv_r2","Conv_q1","Conv_z2","Conv_r2","Conv_q2"]:
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.GRU.{block}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.GRU.{block}.bias"))

    for block in ["Conv1","Conv2"]:
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.FlowHead.{block}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.FlowHead.{block}.bias"))

    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.Mask.0.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.Mask.0.bias"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.Mask.2.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.Mask.2.bias"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"OpticalFlow.UpdateBlock.FlowHead.{block}.bias"))


    ## RecurentFlowCompleteNet
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.Downsample.0.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.Downsample.0.bias"))

    for i in range(0,config.encoder_depth,2):#7
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.Encoder.0.{i}.Conv1.0.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.Encoder.0.{i}.Conv1.0.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.Encoder.0.{i}.Conv2.0.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.Encoder.0.{i}.Conv2.0.bias"))
    
    for i in range(0, config.middilation_depth,2):#5
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.MidDilation.{i}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.MidDilation.{i}.bias"))

    for block in ["backward_1","forward_1"]
        for i in range(0,config.convoffset_depth,2):#6
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.feat_prop_module.deform_align.{block}.ConvOffset.{i}.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.feat_prop_module.deform_align.{block}.ConvOffset.{i}.bias"))

            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.feat_prop_module.deform_align.{block}.ConvOffset.{i}.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.feat_prop_module.deform_align.{block}.ConvOffset.{i}.bias"))


        for i in range(0,config.backbone_depth,2):#2
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.feat_prop_module.backbone.{block}.{i}.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.feat_prop_module.backbone.{block}.{i}.bias"))

            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.feat_prop_module.backbone.{block}.{i}.weight"))
            rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.feat_prop_module.backbone.{block}.{i}.bias"))

    for i in range(0,config.fuse_depth,2):
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.feat_prop_module.fuse.{i}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.feat_prop_module.fuse.{i}.bias"))

        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.feat_prop_module.fuse.{i}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.feat_prop_module.fuse.{i}.bias"))

    for block in ["Decoder1","Decoder2","Upsample"]:
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.{block}.0.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.{block}.0.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.{block}.2.Conv.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.{block}.2.Conv.bias"))

    for block in ["edge_projection","edge_layer_1","edge_layer_2"]:
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.EdgeDetector.{block}.0.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.EdgeDetector.{block}.0.bias"))

    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.EdgeDetector.edge_out.0.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"FlowComplete.EdgeDetector.edge_out.0.bias"))
    
    ##ProPainter
    for i in range(0,config.encoder_depth,2):#17
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.encoder.Layers.{i}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.encoder.Layers.{i}.bias"))

    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.decoder.4.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.decoder.4.bias"))
    for i in [2,6]:#6
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.decoder.{i}.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.decoder.{i}.bias"))

    for block in ["sc","ss"]:
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.{block}.embedding.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.{block}.embedding.bias"))

    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.sc.bias_conv.weight"))
    rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.sc.bias_conv.bias"))

    
    for i in range(config.transformer_block):#7
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.key.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.key.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.query.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.query.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.value.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.value.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.proj.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.proj.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.pool_layer.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.attention.pool_layer.bias"))

        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.norm1.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.norm1.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.norm2.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.norm2.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.mlp.fc1.0.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.mlp.fc1.0.bias"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.mlp.fc2.1.weight"))
        rename_keys.append((f"OpticalFlow.FeatureNet.Norm1.weight", f"InPainting.transformers.transformer.{i}.mlp.fc2.1.bias"))

# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_propainter_checkpoint(vit_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ProPainter structure.
    """

    # define default ViT configuration
    config = ProPainterConfig()
    base_model = False


    # load original model from timm
    timm_model = timm.create_model(vit_name, pretrained=True)
    timm_model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = timm_model.state_dict()
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    # load HuggingFace model
    if vit_name[-5:] == "in21k":
        model = ViTModel(config).eval()
    else:
        model = ViTForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by ViTImageProcessor/DeiTImageProcessor
    if "deit" in vit_name:
        image_processor = DeiTImageProcessor(size=config.image_size)
    else:
        image_processor = ViTImageProcessor(size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)

    if base_model:
        timm_pooled_output = timm_model.forward_features(pixel_values)
        assert timm_pooled_output.shape == outputs.pooler_output.shape
        assert torch.allclose(timm_pooled_output, outputs.pooler_output, atol=1e-3)
    else:
        timm_logits = timm_model(pixel_values)
        assert timm_logits.shape == outputs.logits.shape
        assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {vit_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--vit_name",
        default="vit_base_patch16_224",
        type=str,
        help="Name of the ViT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vit_checkpoint(args.vit_name, args.pytorch_dump_folder_path)

# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Convert RT Detr V2 checkpoints with Timm backbone"""

import argparse
import json
import re
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import RTDetrImageProcessor, RtDetrV2Config, RtDetrV2ForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_rt_detr_v2_config(model_name: str) -> RtDetrV2Config:
    config = RtDetrV2Config()

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    if model_name == "rtdetr_v2_r18vd":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.depths = [2, 2, 2, 2]
        config.backbone_config.layer_type = "basic"
        config.encoder_in_channels = [128, 256, 512]
        config.hidden_expansion = 0.5
        config.decoder_layers = 3
    elif model_name == "rtdetr_v2_r34vd":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.depths = [3, 4, 6, 3]
        config.backbone_config.layer_type = "basic"
        config.encoder_in_channels = [128, 256, 512]
        config.hidden_expansion = 0.5
        config.decoder_layers = 4
    # TODO: check this not working
    elif model_name == "rtdetr_v2_r50vd_m":
        config.hidden_expansion = 0.5
    elif model_name == "rtdetr_v2_r50vd":
        pass
    elif model_name == "rtdetr_v2_r101vd":
        config.backbone_config.depths = [3, 4, 23, 3]
        config.encoder_ffn_dim = 2048
        config.encoder_hidden_dim = 384
        config.decoder_in_channels = [384, 384, 384]

    return config


# Define a mapping from original keys to converted keys using regex
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"backbone.conv1.conv1_1.conv.weight": r"model.backbone.model.embedder.embedder.0.convolution.weight",
    r"backbone.conv1.conv1_1.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.embedder.0.normalization.\1",
    r"backbone.conv1.conv1_2.conv.weight": r"model.backbone.model.embedder.embedder.1.convolution.weight",
    r"backbone.conv1.conv1_2.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.embedder.1.normalization.\1",
    r"backbone.conv1.conv1_3.conv.weight": r"model.backbone.model.embedder.embedder.2.convolution.weight",
    r"backbone.conv1.conv1_3.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.embedder.embedder.2.normalization.\1",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2a.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.0.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2a.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.0.normalization.\3",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2b.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.1.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2b.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.1.normalization.\3",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2c.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.2.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.(\d+).branch2c.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.\2.layer.2.normalization.\3",
    r"encoder.encoder.(\d+).layers.0.self_attn.out_proj.weight": r"model.encoder.encoder.\1.layers.0.self_attn.out_proj.weight",
    r"encoder.encoder.(\d+).layers.0.self_attn.out_proj.bias": r"model.encoder.encoder.\1.layers.0.self_attn.out_proj.bias",
    r"encoder.encoder.(\d+).layers.0.linear1.weight": r"model.encoder.encoder.\1.layers.0.fc1.weight",
    r"encoder.encoder.(\d+).layers.0.linear1.bias": r"model.encoder.encoder.\1.layers.0.fc1.bias",
    r"encoder.encoder.(\d+).layers.0.linear2.weight": r"model.encoder.encoder.\1.layers.0.fc2.weight",
    r"encoder.encoder.(\d+).layers.0.linear2.bias": r"model.encoder.encoder.\1.layers.0.fc2.bias",
    r"encoder.encoder.(\d+).layers.0.norm1.weight": r"model.encoder.encoder.\1.layers.0.self_attn_layer_norm.weight",
    r"encoder.encoder.(\d+).layers.0.norm1.bias": r"model.encoder.encoder.\1.layers.0.self_attn_layer_norm.bias",
    r"encoder.encoder.(\d+).layers.0.norm2.weight": r"model.encoder.encoder.\1.layers.0.final_layer_norm.weight",
    r"encoder.encoder.(\d+).layers.0.norm2.bias": r"model.encoder.encoder.\1.layers.0.final_layer_norm.bias",
    r"encoder.input_proj.(\d+).conv.weight": r"model.encoder_input_proj.\1.0.weight",
    r"encoder.input_proj.(\d+).norm.(.*)": r"model.encoder_input_proj.\1.1.\2",
    r"encoder.fpn_blocks.(\d+).conv(\d+).conv.weight": r"model.encoder.fpn_blocks.\1.conv\2.conv.weight",
    # r"encoder.fpn_blocks.(\d+).conv(\d+).norm.(.*)": r"model.encoder.fpn_blocks.\1.conv\2.norm.\3",
    r"encoder.fpn_blocks.(\d+).conv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.fpn_blocks.\1.conv\2.norm.\3",
    r"encoder.lateral_convs.(\d+).conv.weight": r"model.encoder.lateral_convs.\1.conv.weight",
    r"encoder.lateral_convs.(\d+).norm.(.*)": r"model.encoder.lateral_convs.\1.norm.\2",
    r"encoder.fpn_blocks.(\d+).bottlenecks.(\d+).conv(\d+).conv.weight": r"model.encoder.fpn_blocks.\1.bottlenecks.\2.conv\3.conv.weight",
    r"encoder.fpn_blocks.(\d+).bottlenecks.(\d+).conv(\d+).norm.(\w+)": r"model.encoder.fpn_blocks.\1.bottlenecks.\2.conv\3.norm.\4",
    r"encoder.pan_blocks.(\d+).conv(\d+).conv.weight": r"model.encoder.pan_blocks.\1.conv\2.conv.weight",
    r"encoder.pan_blocks.(\d+).conv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.conv\2.norm.\3",
    r"encoder.pan_blocks.(\d+).bottlenecks.(\d+).conv(\d+).conv.weight": r"model.encoder.pan_blocks.\1.bottlenecks.\2.conv\3.conv.weight",
    r"encoder.pan_blocks.(\d+).bottlenecks.(\d+).conv(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.pan_blocks.\1.bottlenecks.\2.conv\3.norm.\4",
    r"encoder.downsample_convs.(\d+).conv.weight": r"model.encoder.downsample_convs.\1.conv.weight",
    r"encoder.downsample_convs.(\d+).norm.(weight|bias|running_mean|running_var)": r"model.encoder.downsample_convs.\1.norm.\2",
    r"decoder.decoder.layers.(\d+).self_attn.out_proj.weight": r"model.decoder.layers.\1.self_attn.out_proj.weight",
    r"decoder.decoder.layers.(\d+).self_attn.out_proj.bias": r"model.decoder.layers.\1.self_attn.out_proj.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.sampling_offsets.weight": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.sampling_offsets.bias": r"model.decoder.layers.\1.encoder_attn.sampling_offsets.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.attention_weights.weight": r"model.decoder.layers.\1.encoder_attn.attention_weights.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.attention_weights.bias": r"model.decoder.layers.\1.encoder_attn.attention_weights.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.value_proj.weight": r"model.decoder.layers.\1.encoder_attn.value_proj.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.value_proj.bias": r"model.decoder.layers.\1.encoder_attn.value_proj.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.output_proj.weight": r"model.decoder.layers.\1.encoder_attn.output_proj.weight",
    r"decoder.decoder.layers.(\d+).cross_attn.output_proj.bias": r"model.decoder.layers.\1.encoder_attn.output_proj.bias",
    r"decoder.decoder.layers.(\d+).norm1.weight": r"model.decoder.layers.\1.self_attn_layer_norm.weight",
    r"decoder.decoder.layers.(\d+).norm1.bias": r"model.decoder.layers.\1.self_attn_layer_norm.bias",
    r"decoder.decoder.layers.(\d+).norm2.weight": r"model.decoder.layers.\1.encoder_attn_layer_norm.weight",
    r"decoder.decoder.layers.(\d+).norm2.bias": r"model.decoder.layers.\1.encoder_attn_layer_norm.bias",
    r"decoder.decoder.layers.(\d+).linear1.weight": r"model.decoder.layers.\1.fc1.weight",
    r"decoder.decoder.layers.(\d+).linear1.bias": r"model.decoder.layers.\1.fc1.bias",
    r"decoder.decoder.layers.(\d+).linear2.weight": r"model.decoder.layers.\1.fc2.weight",
    r"decoder.decoder.layers.(\d+).linear2.bias": r"model.decoder.layers.\1.fc2.bias",
    r"decoder.decoder.layers.(\d+).norm3.weight": r"model.decoder.layers.\1.final_layer_norm.weight",
    r"decoder.decoder.layers.(\d+).norm3.bias": r"model.decoder.layers.\1.final_layer_norm.bias",
    r"decoder.decoder.layers.(\d+).cross_attn.num_points_scale": r"model.decoder.layers.\1.encoder_attn.n_points_scale",
    r"decoder.dec_score_head.(\d+).weight": r"model.decoder.class_embed.\1.weight",
    r"decoder.dec_score_head.(\d+).bias": r"model.decoder.class_embed.\1.bias",
    r"decoder.dec_bbox_head.(\d+).layers.(\d+).(weight|bias)": r"model.decoder.bbox_embed.\1.layers.\2.\3",
    r"decoder.denoising_class_embed.weight": r"model.denoising_class_embed.weight",
    r"decoder.query_pos_head.layers.0.weight": r"model.decoder.query_pos_head.layers.0.weight",
    r"decoder.query_pos_head.layers.0.bias": r"model.decoder.query_pos_head.layers.0.bias",
    r"decoder.query_pos_head.layers.1.weight": r"model.decoder.query_pos_head.layers.1.weight",
    r"decoder.query_pos_head.layers.1.bias": r"model.decoder.query_pos_head.layers.1.bias",
    r"decoder.enc_output.proj.weight": r"model.enc_output.0.weight",
    r"decoder.enc_output.proj.bias": r"model.enc_output.0.bias",
    r"decoder.enc_output.norm.weight": r"model.enc_output.1.weight",
    r"decoder.enc_output.norm.bias": r"model.enc_output.1.bias",
    r"decoder.enc_score_head.weight": r"model.enc_score_head.weight",
    r"decoder.enc_score_head.bias": r"model.enc_score_head.bias",
    r"decoder.enc_bbox_head.layers.(\d+).(weight|bias)": r"model.enc_bbox_head.layers.\1.\2",
    r"backbone.res_layers.0.blocks.0.short.conv.weight": r"model.backbone.model.encoder.stages.0.layers.0.shortcut.convolution.weight",
    r"backbone.res_layers.0.blocks.0.short.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.0.layers.0.shortcut.normalization.\1",
    r"backbone.res_layers.(\d+).blocks.0.short.conv.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.0.short.conv.norm.(\w+)": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.normalization.\2",
    # Mapping for subsequent blocks in other stages
    r"backbone.res_layers.(\d+).blocks.0.short.conv.weight": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.convolution.weight",
    r"backbone.res_layers.(\d+).blocks.0.short.norm.(weight|bias|running_mean|running_var)": r"model.backbone.model.encoder.stages.\1.layers.0.shortcut.1.normalization.\2",
    r"decoder.input_proj.(\d+).conv.weight": r"model.decoder_input_proj.\1.0.weight",
    r"decoder.input_proj.(\d+).norm.(.*)": r"model.decoder_input_proj.\1.1.\2",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    # Use the mapping to rename keys
    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict_keys.keys()):
            new_key = re.sub(original_key, converted_key, key)
            if new_key != key:
                state_dict_keys[new_key] = state_dict_keys.pop(key)

    return state_dict_keys


def read_in_q_k_v(state_dict, config):
    prefix = ""
    encoder_hidden_dim = config.encoder_hidden_dim

    # first: transformer encoder
    for i in range(config.encoder_layers):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
            :encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
            encoder_hidden_dim : 2 * encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
            encoder_hidden_dim : 2 * encoder_hidden_dim
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
            -encoder_hidden_dim:, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def write_model_and_image_processor(model_name, output_dir, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our RTDETR structure.
    """

    # load default config
    config = get_rt_detr_v2_config(model_name)

    # load original model from torch hub
    model_name_to_checkpoint_url = {
        "rtdetr_v2_r18vd": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "rtdetr_v2_r34vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth",
        "rtdetr_v2_r50vd_m": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth",
        "rtdetr_v2_r50vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth",
        "rtdetr_v2_r101vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
    }
    logger.info(f"Converting model {model_name}...")
    state_dict = torch.hub.load_state_dict_from_url(model_name_to_checkpoint_url[model_name], map_location="cpu")[
        "ema"
    ]["module"]
    # rename keys
    state_dict = convert_old_keys_to_new_keys(state_dict)
    for key in state_dict.copy().keys():
        if key.endswith("num_batches_tracked"):
            del state_dict[key]
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    for key in state_dict.copy().keys():
        if key.endswith("num_batches_tracked"):
            del state_dict[key]
        # for two_stage
        if "bbox_embed" in key or ("class_embed" in key and "denoising_" not in key):
            state_dict[key.split("model.decoder.")[-1]] = state_dict[key]

    # no need in ckpt
    del state_dict["decoder.anchors"]
    del state_dict["decoder.valid_mask"]
    # finally, create HuggingFace model and load state dict
    model = RtDetrV2ForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # load image processor
    image_processor = RTDetrImageProcessor()

    # prepare image
    img = prepare_img()

    # preprocess image
    transformations = transforms.Compose(
        [
            transforms.Resize([640, 640], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    original_pixel_values = transformations(img).unsqueeze(0)  # insert batch dimension

    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    assert torch.allclose(original_pixel_values, pixel_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    # Pass image by the model
    outputs = model(pixel_values)

    if model_name == "rtdetr_v2_r18vd":
        expected_slice_logits = torch.tensor(
            [[-3.7045, -5.1913, -6.1787], [-4.0106, -9.3450, -5.2043], [-4.1287, -4.7463, -5.8634]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2582, 0.5497, 0.4764], [0.1684, 0.1985, 0.2120], [0.7665, 0.4146, 0.4669]]
        )
    elif model_name == "rtdetr_v2_r34vd":
        expected_slice_logits = torch.tensor(
            [[-4.6108, -5.9453, -3.8505], [-3.8702, -6.1136, -5.5677], [-3.7790, -6.4538, -5.9449]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.1691, 0.1984, 0.2118], [0.2594, 0.5506, 0.4736], [0.7669, 0.4136, 0.4654]]
        )
    elif model_name == "rtdetr_v2_r50vd_m":
        expected_slice_logits = torch.tensor(
            [[-2.7453, -5.4595, -7.3702], [-3.1858, -5.3803, -7.9838], [-5.0293, -7.0083, -4.2888]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.7711, 0.4135, 0.4577], [0.2570, 0.5480, 0.4755], [0.1694, 0.1992, 0.2127]]
        )
    elif model_name == "rtdetr_v2_r50vd":
        expected_slice_logits = torch.tensor(
            [[-4.7881, -4.6754, -6.1624], [-5.4441, -6.6486, -4.3840], [-3.5455, -4.9318, -6.3544]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2588, 0.5487, 0.4747], [0.5497, 0.2760, 0.0573], [0.7688, 0.4133, 0.4634]]
        )
    elif model_name == "rtdetr_v2_r101vd":
        expected_slice_logits = torch.tensor(
            [[-4.6162, -4.9189, -4.6656], [-4.4701, -4.4997, -4.9659], [-5.6641, -7.9000, -5.0725]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.7707, 0.4124, 0.4585], [0.2589, 0.5492, 0.4735], [0.1688, 0.1993, 0.2108]]
        )
    else:
        raise ValueError(f"Unknown rt_detr_v2_name: {model_name}")
    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits.to(outputs.logits.device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes.to(outputs.pred_boxes.device), atol=1e-3)

    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {output_dir}")
        model.save_pretrained(output_dir)
        print(f"Saving image processor to {output_dir}")
        image_processor.save_pretrained(output_dir)

    if push_to_hub:
        # Upload model, image processor and config to the hub
        logger.info("Uploading PyTorch model and image processor to the hub...")
        config.push_to_hub(
            repo_id=repo_id,
            commit_message="Add config from convert_rt_detr_v2_original_pytorch_checkpoint_to_pytorch.py",
        )
        model.push_to_hub(
            repo_id=repo_id,
            commit_message="Add model from convert_rt_detr_v2_original_pytorch_checkpoint_to_pytorch.py",
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_rt_detr_v2_original_pytorch_checkpoint_to_pytorch.py",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="rtdetr_v2_r18vd",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument("--output_dir", default=None, type=str, help="Location to write HF model and image processor")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    write_model_and_image_processor(args.model_name, args.output_dir, args.push_to_hub, args.repo_id)

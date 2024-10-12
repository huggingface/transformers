# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert DAB-DETR checkpoints."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path
import gc
import re

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DabDetrConfig, DabDetrForObjectDetection, DabDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    # for dab-DETR, also convert reference point head and query scale MLP
    r"input_proj.weight":                                                       r"input_projection.weight",
    r"input_proj.bias":                                                         r"input_projection.bias",
    r"refpoint_embed.weight":                                                   r"query_refpoint_embeddings.weight",
    r"class_embed.weight":                                                      r"class_embed.weight",
    r"class_embed.bias":                                                        r"class_embed.bias",
    # negative lookbehind because of the overlap
    r"(?<!transformer\.decoder\.)bbox_embed.layers.(\d+).weight":               r"bbox_predictor.layers.\1.weight",
    r"(?<!transformer\.decoder\.)bbox_embed.layers.(\d+).bias":                 r"bbox_predictor.layers.\1.bias",
    r"transformer.encoder.query_scale.layers.(\d+).weight":                     r"encoder.query_scale.layers.\1.weight",
    r"transformer.encoder.query_scale.layers.(\d+).bias":                       r"encoder.query_scale.layers.\1.bias",
    r"transformer.decoder.bbox_embed.layers.(\d+).weight":                      r"decoder.bbox_embed.layers.\1.weight",
    r"transformer.decoder.bbox_embed.layers.(\d+).bias":                        r"decoder.bbox_embed.layers.\1.bias",
    r"transformer.decoder.norm.weight":                                         r"decoder.layernorm.weight",
    r"transformer.decoder.norm.bias":                                           r"decoder.layernorm.bias",
    r"transformer.decoder.ref_point_head.layers.(\d+).weight":                  r"decoder.ref_point_head.layers.\1.weight",
    r"transformer.decoder.ref_point_head.layers.(\d+).bias":                    r"decoder.ref_point_head.layers.\1.bias",
    r"transformer.decoder.ref_anchor_head.layers.(\d+).weight":                 r"decoder.ref_anchor_head.layers.\1.weight",
    r"transformer.decoder.ref_anchor_head.layers.(\d+).bias":                   r"decoder.ref_anchor_head.layers.\1.bias",
    r"transformer.decoder.query_scale.layers.(\d+).weight":                     r"decoder.query_scale.layers.\1.weight",
    r"transformer.decoder.query_scale.layers.(\d+).bias":                       r"decoder.query_scale.layers.\1.bias",
    r"transformer.decoder.layers.0.ca_qpos_proj.weight":                        r"decoder.layers.0.layer.1.cross_attn_query_pos_proj.weight",
    r"transformer.decoder.layers.0.ca_qpos_proj.bias":                          r"decoder.layers.0.layer.1.cross_attn_query_pos_proj.bias",
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms + activation function
    # output projection
    r"transformer.encoder.layers.(\d+).self_attn.out_proj.weight":              r"encoder.layers.\1.self_attn.out_proj.weight",
    r"transformer.encoder.layers.(\d+).self_attn.out_proj.bias":                r"encoder.layers.\1.self_attn.out_proj.bias",
    # FFN layer
    # FFN 1
    r"transformer.encoder.layers.(\d+).linear1.weight":                         r"encoder.layers.\1.fc1.weight",
    r"transformer.encoder.layers.(\d+).linear1.bias":                           r"encoder.layers.\1.fc1.bias",
    # FFN 2
    r"transformer.encoder.layers.(\d+).linear2.weight":                         r"encoder.layers.\1.fc2.weight",
    r"transformer.encoder.layers.(\d+).linear2.bias":                           r"encoder.layers.\1.fc2.bias",
    # normalization layers
    # nm1
    r"transformer.encoder.layers.(\d+).norm1.weight":                           r"encoder.layers.\1.self_attn_layer_norm.weight",
    r"transformer.encoder.layers.(\d+).norm1.bias":                             r"encoder.layers.\1.self_attn_layer_norm.bias",
    # nm2
    r"transformer.encoder.layers.(\d+).norm2.weight":                           r"encoder.layers.\1.final_layer_norm.weight",
    r"transformer.encoder.layers.(\d+).norm2.bias":                             r"encoder.layers.\1.final_layer_norm.bias",
    # activation function weight
    r"transformer.encoder.layers.(\d+).activation.weight":                      r"encoder.layers.\1.activation_fn.weight",

    #########################################################################################################################################
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms + activiation function weight
    r"transformer.decoder.layers.(\d+).self_attn.out_proj.weight":              r"decoder.layers.\1.layer.0.self_attn.output_projection.weight",
    r"transformer.decoder.layers.(\d+).self_attn.out_proj.bias":                r"decoder.layers.\1.layer.0.self_attn.output_projection.bias",
    r"transformer.decoder.layers.(\d+).cross_attn.out_proj.weight":             r"decoder.layers.\1.layer.1.cross_attn.output_projection.weight",
    r"transformer.decoder.layers.(\d+).cross_attn.out_proj.bias":               r"decoder.layers.\1.layer.1.cross_attn.output_projection.bias",
    # FFN 1
    r"transformer.decoder.layers.(\d+).linear1.weight":                         r"decoder.layers.\1.layer.2.fc1.weight",
    r"transformer.decoder.layers.(\d+).linear1.bias":                           r"decoder.layers.\1.layer.2.fc1.bias",
    # FFN 2
    r"transformer.decoder.layers.(\d+).linear2.weight":                         r"decoder.layers.\1.layer.2.fc2.weight",
    r"transformer.decoder.layers.(\d+).linear2.bias":                           r"decoder.layers.\1.layer.2.fc2.bias",
    # nm1
    r"transformer.decoder.layers.(\d+).norm1.weight":                           r"decoder.layers.\1.layer.0.self_attn_layer_norm.weight",
    r"transformer.decoder.layers.(\d+).norm1.bias":                             r"decoder.layers.\1.layer.0.self_attn_layer_norm.bias",
    # nm2
    r"transformer.decoder.layers.(\d+).norm2.weight":                           r"decoder.layers.\1.layer.1.cross_attn_layer_norm.weight",
    r"transformer.decoder.layers.(\d+).norm2.bias":                             r"decoder.layers.\1.layer.1.cross_attn_layer_norm.bias",
    # nm3
    r"transformer.decoder.layers.(\d+).norm3.weight":                           r"decoder.layers.\1.layer.2.final_layer_norm.weight",
    r"transformer.decoder.layers.(\d+).norm3.bias":                             r"decoder.layers.\1.layer.2.final_layer_norm.bias",
    # activation function weight
    r"transformer.decoder.layers.(\d+).activation.weight":                      r"decoder.layers.\1.layer.2.activation_fn.weight",
    # q, k, v projections in self-attention in decoder
    r"transformer.decoder.layers.(\d+).sa_qcontent_proj.weight":                r"decoder.layers.\1.layer.0.self_attn_query_content_proj.weight",
    r"transformer.decoder.layers.(\d+).sa_kcontent_proj.weight":                r"decoder.layers.\1.layer.0.self_attn_key_content_proj.weight",
    r"transformer.decoder.layers.(\d+).sa_qpos_proj.weight":                    r"decoder.layers.\1.layer.0.self_attn_query_pos_proj.weight",
    r"transformer.decoder.layers.(\d+).sa_kpos_proj.weight":                    r"decoder.layers.\1.layer.0.self_attn_key_pos_proj.weight",
    r"transformer.decoder.layers.(\d+).sa_v_proj.weight":                       r"decoder.layers.\1.layer.0.self_attn_value_proj.weight",
    # q, k, v projections in cross-attention in decoder
    r"transformer.decoder.layers.(\d+).ca_qcontent_proj.weight":                r"decoder.layers.\1.layer.1.cross_attn_query_content_proj.weight",
    r"transformer.decoder.layers.(\d+).ca_kcontent_proj.weight":                r"decoder.layers.\1.layer.1.cross_attn_key_content_proj.weight",
    r"transformer.decoder.layers.(\d+).ca_kpos_proj.weight":                    r"decoder.layers.\1.layer.1.cross_attn_key_pos_proj.weight",
    r"transformer.decoder.layers.(\d+).ca_v_proj.weight":                       r"decoder.layers.\1.layer.1.cross_attn_value_proj.weight",
    r"transformer.decoder.layers.(\d+).ca_qpos_sine_proj.weight":               r"decoder.layers.\1.layer.1.cross_attn_query_pos_sine_proj.weight",
    # q, k, v biases in self-attention in decoder
    r"transformer.decoder.layers.(\d+).sa_qcontent_proj.bias":                  r"decoder.layers.\1.layer.0.self_attn_query_content_proj.bias",
    r"transformer.decoder.layers.(\d+).sa_kcontent_proj.bias":                  r"decoder.layers.\1.layer.0.self_attn_key_content_proj.bias",
    r"transformer.decoder.layers.(\d+).sa_qpos_proj.bias":                      r"decoder.layers.\1.layer.0.self_attn_query_pos_proj.bias",
    r"transformer.decoder.layers.(\d+).sa_kpos_proj.bias":                      r"decoder.layers.\1.layer.0.self_attn_key_pos_proj.bias",
    r"transformer.decoder.layers.(\d+).sa_v_proj.bias":                         r"decoder.layers.\1.layer.0.self_attn_value_proj.bias",
    # q, k, v biases in cross-attention in decoder
    r"transformer.decoder.layers.(\d+).ca_qcontent_proj.bias":                  r"decoder.layers.\1.layer.1.cross_attn_query_content_proj.bias",
    r"transformer.decoder.layers.(\d+).ca_kcontent_proj.bias":                  r"decoder.layers.\1.layer.1.cross_attn_key_content_proj.bias",
    r"transformer.decoder.layers.(\d+).ca_kpos_proj.bias":                      r"decoder.layers.\1.layer.1.cross_attn_key_pos_proj.bias",
    r"transformer.decoder.layers.(\d+).ca_v_proj.bias":                         r"decoder.layers.\1.layer.1.cross_attn_value_proj.bias",
    r"transformer.decoder.layers.(\d+).ca_qpos_sine_proj.bias":                 r"decoder.layers.\1.layer.1.cross_attn_query_pos_sine_proj.bias",
}

def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict

# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_dab_detr_checkpoint(model_name, pretrained_model_weights_path, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DAB-DETR structure.
    """

    # load modified config. Why? After loading the default config, the backbone kwargs are already set.
    if "dc5" in model_name:
        config = DabDetrConfig(dilation=True)
    else:
        # load default config
        config = DabDetrConfig()

    # set other attributes
    if "dab-detr-resnet-50-dc5" == model_name:
        config.temperature_height = 10
        config.temperature_width = 10
    if "fixxy" in model_name:
        config.random_refpoint_xy = True
    if "pat3" in model_name:
        config.num_patterns = 3
        # only when the number of patterns (num_patterns parameter in config) are more than 0 like r50-pat3 or r50dc5-pat3
        ORIGINAL_TO_CONVERTED_KEY_MAPPING.update({r"transformer.patterns.weight": r"patterns.weight"})

    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # load image processor
    format = "coco_detection"
    image_processor = DabDetrImageProcessor(format=format)

    # prepare image
    img = prepare_img()
    encoding = image_processor(images=[img], return_tensors="pt")

    logger.info(f"Converting model {model_name}...")
    # load original model from local path
    loaded = torch.load(pretrained_model_weights_path, map_location=torch.device("cpu"))["model"]
    # Renaming the original model state dictionary to HF compatibile
    all_keys = list(loaded.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)
    state_dict = {}
    for key in all_keys:
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model._backbone")
            state_dict[new_key] = loaded[key]
        # Q, K, V encoder values mapping
        elif re.search("self_attn.in_proj_(weight|bias)", key):
            # Dynamically find the layer number
            pattern = r'layers\.(\d+)\.self_attn\.in_proj_(weight|bias)'
            match = re.search(pattern, key)
            if match:
                layer_num = match.group(1)
            else:
                raise ValueError(f"Pattern not found in key: {key}")

            in_proj_value = loaded.pop(key)
            if "weight" in key:
                state_dict[f"encoder.layers.{layer_num}.self_attn.q_proj.weight"] = in_proj_value[:256, :]
                state_dict[f"encoder.layers.{layer_num}.self_attn.k_proj.weight"] = in_proj_value[256:512, :]
                state_dict[f"encoder.layers.{layer_num}.self_attn.v_proj.weight"] = in_proj_value[-256:, :]
            elif "bias" in key:
                state_dict[f"encoder.layers.{layer_num}.self_attn.q_proj.bias"] = in_proj_value[:256]
                state_dict[f"encoder.layers.{layer_num}.self_attn.k_proj.bias"] = in_proj_value[256:512]
                state_dict[f"encoder.layers.{layer_num}.self_attn.v_proj.bias"] = in_proj_value[-256:]
        else:
            new_key = new_keys[key]
            state_dict[new_key] = loaded[key]

    del loaded
    gc.collect()
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val

    # Expected logits and pred_boxes results of each model
    if model_name == "dab-detr-resnet-50":
        expected_slice_logits = torch.tensor(
            [[-10.1765, -5.5243, -8.9324], [-9.8138, -5.6721, -7.5161], [-10.3054, -5.6081, -8.5931]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.3708, 0.3000, 0.2753], [0.5211, 0.6125, 0.9495], [0.2897, 0.6730, 0.5459]]
        )
        logits_atol = 3e-4
        boxes_atol = 1e-4
    elif model_name == "dab-detr-resnet-50-pat3":
        expected_slice_logits = torch.tensor(
            [[-10.1069, -6.7068, -8.5944], [-9.4003, -7.3787, -8.7304], [-9.5858, -6.1514, -8.4744]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5834, 0.6162, 0.2534], [0.6670, 0.2703, 0.1468], [0.5532, 0.1936, 0.0411]]
        )
        logits_atol = 1e-4
        boxes_atol = 1e-4
    elif model_name == "dab-detr-resnet-50-dc5":
        expected_slice_logits = torch.tensor(
            [[-9.9054, -6.0638, -7.8630], [-9.9112, -5.2952, -7.8175], [-9.8720, -5.3681, -7.7094]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.4077, 0.3644, 0.2689], [0.4429, 0.6903, 0.8238], [0.5188, 0.7933, 0.9989]]
        )
        logits_atol = 3e-3
        boxes_atol = 1e-3
    elif model_name == "dab-detr-resnet-50-dc5-pat3":
        expected_slice_logits = torch.tensor(
            [[-11.2264, -5.4028, -8.9815], [-10.8721, -6.0637, -9.1898], [-10.8535, -6.8360, -9.4203]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.8532, 0.5143, 0.1799], [0.6903, 0.3749, 0.3506], [0.5275, 0.2726, 0.0535]]
        )
        logits_atol = 1e-4
        boxes_atol = 1e-4
    elif model_name == "dab-detr-resnet-50-dc5-fixxy":
        expected_slice_logits = torch.tensor(
            [[-9.9362, -5.8105, -8.4952], [-9.6947, -4.9066, -8.3175], [-8.6919, -3.6328, -8.8972]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.4420, 0.3688, 0.2510], [0.5112, 0.7156, 0.9774], [0.4985, 0.4967, 0.9990]]
        )
        logits_atol = 5e-4
        boxes_atol = 1e-3

    # finally, create HuggingFace model and load state dict
    model = DabDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    if push_to_hub:
        model.push_to_hub(repo_id=model_name, commit_message="Add new model")
    model.eval()
    # verify our conversion
    outputs = model(**encoding)

    # "model.decoder.layers.0.self_attn.self_attn_query_content_proj.weight",
    # "model.decoder.layers.0.layer.0.self_attn_query_content_proj.weight"

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=logits_atol)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=boxes_atol)
    print('s')
    # Save model and image processor
    # logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    # Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # model.save_pretrained(pytorch_dump_folder_path)
    # image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="dab-detr-resnet-50",
        type=str,
        help="Name of the DAB_DETR model you'd like to convert.",
    )
    parser.add_argument(
        "--pretrained_model_weights_path",
        default="/Users/davidhajdu/Desktop/all_weights/R50/checkpoint.pth",
        type=str,
        help="The path of the original model weights like: Users/username/Desktop/checkpoint.pth",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="DAB_DETR", type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=True, type=bool, help="Whether to upload the converted weights to the HuggingFace model profile. Default is set to false."
    )
    args = parser.parse_args()
    convert_dab_detr_checkpoint(args.model_name, args.pretrained_model_weights_path, args.pytorch_dump_folder_path, args.push_to_hub)

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

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    DABDETRConfig,
    DABDETRForObjectDetection,
    DABDETRImageProcessor,
    DABDETRModel
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, HF name on the right)
rename_keys = []
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms + activation function
    # output projection
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.bias", f"encoder.layers.{i}.self_attn.out_proj.bias")
    )
    # FFN layer
    # FFN 1
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))
    # FFN 2
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))
    # normalization layers
    # nm1
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.weight", f"encoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias"))
    # nm2
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
    # activation function weight
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.activation.weight", f"encoder.layers.{i}.activation_fn.weight")
    )
    #########################################################################################################################################
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms + activiation function weight
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.output_projection.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.output_projection.bias")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.weight",
            f"decoder.layers.{i}.cross_attn.output_projection.weight",
        )
    )
    # activation function weight
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.activation.weight", f"decoder.layers.{i}.activation_fn.weight")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.cross_attn.out_proj.bias",
            f"decoder.layers.{i}.cross_attn.output_projection.bias",
        )
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.cross_attn_layer_norm.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.cross_attn_layer_norm.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

    # q, k, v projections in self/cross-attention in decoder for DAB-DETR
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.weight", f"decoder.layers.{i}.self_attn_query_content_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.weight", f"decoder.layers.{i}.self_attn_key_content_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qpos_proj.weight", f"decoder.layers.{i}.self_attn_query_pos_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kpos_proj.weight", f"decoder.layers.{i}.self_attn_key_pos_proj.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.weight", f"decoder.layers.{i}.self_attn_value_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.weight", f"decoder.layers.{i}.cross_attn_query_content_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.weight", f"decoder.layers.{i}.cross_attn_key_content_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kpos_proj.weight", f"decoder.layers.{i}.cross_attn_key_pos_proj.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.weight", f"decoder.layers.{i}.cross_attn_value_proj.weight"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.weight", f"decoder.layers.{i}.cross_attn_query_pos_sine_proj.weight")
    )

    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_qcontent_proj.bias", f"decoder.layers.{i}.self_attn_query_content_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.sa_kcontent_proj.bias", f"decoder.layers.{i}.self_attn_key_content_proj.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_qpos_proj.bias", f"decoder.layers.{i}.self_attn_query_pos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_kpos_proj.bias", f"decoder.layers.{i}.self_attn_key_pos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.sa_v_proj.bias", f"decoder.layers.{i}.self_attn_value_proj.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qcontent_proj.bias", f"decoder.layers.{i}.cross_attn_query_content_proj.bias")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_kcontent_proj.bias", f"decoder.layers.{i}.cross_attn_key_content_proj.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_kpos_proj.bias", f"decoder.layers.{i}.cross_attn_key_pos_proj.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.ca_v_proj.bias", f"decoder.layers.{i}.cross_attn_value_proj.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.ca_qpos_sine_proj.bias", f"decoder.layers.{i}.cross_attn_query_pos_sine_proj.bias")
    )

# convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
# for dab-DETR, also convert reference point head and query scale MLP
rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        ("refpoint_embed.weight", "query_refpoint_embeddings.weight"),
        ("class_embed.weight", "class_embed.weight"),
        ("class_embed.bias", "class_embed.bias"),
        ("bbox_embed.layers.0.weight", "bbox_predictor.layers.0.weight"),
        ("bbox_embed.layers.0.bias", "bbox_predictor.layers.0.bias"),
        ("bbox_embed.layers.1.weight", "bbox_predictor.layers.1.weight"),
        ("bbox_embed.layers.1.bias", "bbox_predictor.layers.1.bias"),
        ("bbox_embed.layers.2.weight", "bbox_predictor.layers.2.weight"),
        ("bbox_embed.layers.2.bias", "bbox_predictor.layers.2.bias"),
        ("transformer.encoder.query_scale.layers.0.weight", "encoder.query_scale.layers.0.weight"),
        ("transformer.encoder.query_scale.layers.0.bias", "encoder.query_scale.layers.0.bias"),
        ("transformer.encoder.query_scale.layers.1.weight", "encoder.query_scale.layers.1.weight"),
        ("transformer.encoder.query_scale.layers.1.bias", "encoder.query_scale.layers.1.bias"),
        ("transformer.decoder.bbox_embed.layers.0.weight", "decoder.bbox_embed.layers.0.weight"),
        ("transformer.decoder.bbox_embed.layers.0.bias", "decoder.bbox_embed.layers.0.bias"),
        ("transformer.decoder.bbox_embed.layers.1.weight", "decoder.bbox_embed.layers.1.weight"),
        ("transformer.decoder.bbox_embed.layers.1.bias", "decoder.bbox_embed.layers.1.bias"),
        ("transformer.decoder.bbox_embed.layers.2.weight", "decoder.bbox_embed.layers.2.weight"),
        ("transformer.decoder.bbox_embed.layers.2.bias", "decoder.bbox_embed.layers.2.bias"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
        ("transformer.decoder.ref_point_head.layers.0.weight", "decoder.ref_point_head.layers.0.weight"),
        ("transformer.decoder.ref_point_head.layers.0.bias", "decoder.ref_point_head.layers.0.bias"),
        ("transformer.decoder.ref_point_head.layers.1.weight", "decoder.ref_point_head.layers.1.weight"),
        ("transformer.decoder.ref_point_head.layers.1.bias", "decoder.ref_point_head.layers.1.bias"),
        ("transformer.decoder.ref_anchor_head.layers.0.weight", "decoder.ref_anchor_head.layers.0.weight"),
        ("transformer.decoder.ref_anchor_head.layers.0.bias", "decoder.ref_anchor_head.layers.0.bias"),
        ("transformer.decoder.ref_anchor_head.layers.1.weight", "decoder.ref_anchor_head.layers.1.weight"),
        ("transformer.decoder.ref_anchor_head.layers.1.bias", "decoder.ref_anchor_head.layers.1.bias"),
        ("transformer.decoder.query_scale.layers.0.weight", "decoder.query_scale.layers.0.weight"),
        ("transformer.decoder.query_scale.layers.0.bias", "decoder.query_scale.layers.0.bias"),
        ("transformer.decoder.query_scale.layers.1.weight", "decoder.query_scale.layers.1.weight"),
        ("transformer.decoder.query_scale.layers.1.bias", "decoder.query_scale.layers.1.bias"),
        ("transformer.decoder.layers.0.ca_qpos_proj.weight", "decoder.layers.0.cross_attn_query_pos_proj.weight"),
        ("transformer.decoder.layers.0.ca_qpos_proj.bias", "decoder.layers.0.cross_attn_query_pos_proj.bias"),
    ]
)


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model._backbone")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "dab_detr."

    # first: transformer encoder
    for i in range(6):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_dab_detr_checkpoint(model_name, pretrained_model_weights_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our DAB-DETR structure.
    """

    # load default config
    config = DABDETRConfig()
    # set backbone and dilation attributes
    if "resnet101" in model_name:
        config.backbone = "resnet101"
    if "dc5" in model_name:
        config.dilation = True

    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # load image processor
    format = "coco_detection"
    image_processor = DABDETRImageProcessor(format=format)

    # prepare image
    img = prepare_img()
    encoding = image_processor(images=[img], return_tensors="pt")

    logger.info(f"Converting model {model_name}...")

    # load original model from torch hub
    state_dict = torch.load(pretrained_model_weights_path, map_location=torch.device("cpu"))["model"]
    # rename keys
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val

    expected_slice_logits = torch.tensor(
        [[-10.1765, -5.5243, -8.9324], [-9.8138, -5.6721, -7.5161], [-10.3054, -5.6081, -8.5931]]
    )
    expected_slice_boxes = torch.tensor([[0.3708, 0.3000, 0.2753], [0.5211, 0.6125, 0.9495], [0.2897, 0.6730, 0.5459]])
    # finally, create HuggingFace model and load state dict
    model = DABDETRForObjectDetection(config)
    model.load_state_dict(state_dict)
    # model.push_to_hub(repo_id=model_name, organization="davidhajdu", commit_message="Add model")
    model.eval()
    # verify our conversion
    z = {'output_hidden_states': True}
    outputs = model(**encoding, **z)
    outputs2 = model(**encoding, **z, return_dict=False)
    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=3e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4)

    # Save model and image processor
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)


from typing import Dict, List, Tuple
from transformers import DABDETRConfig, ResNetConfig
import math
import random
import copy

torch_device = torch.device('cpu')

global_rng = random.Random()
def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()

class DABDETRModelTester:
    def __init__(
        self,
        batch_size=8,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=12,
        num_channels=3,
        min_size=200,
        max_size=200,
        n_targets=8,
        num_labels=91,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.n_targets = n_targets
        self.num_labels = num_labels

        # we also set the expected seq length for both encoder and decoder
        self.encoder_seq_length = math.ceil(self.min_size / 32) * math.ceil(self.max_size / 32)
        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.min_size, self.max_size])

        pixel_mask = torch.ones([self.batch_size, self.min_size, self.max_size], device=torch_device)

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(
                    high=self.num_labels, size=(self.n_targets,), device=torch_device
                )
                target["boxes"] = torch.rand(self.n_targets, 4, device=torch_device)
                target["masks"] = torch.rand(self.n_targets, self.min_size, self.max_size, device=torch_device)
                labels.append(target)

        config = self.get_config()
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        resnet_config = ResNetConfig(
            num_channels=3,
            embeddings_size=10,
            hidden_sizes=[10, 20, 30, 40],
            depths=[1, 1, 2, 1],
            hidden_act="relu",
            num_labels=3,
            out_features=["stage2", "stage3", "stage4"],
            out_indices=[2, 3, 4],
        )
        return DABDETRConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            use_timm_backbone=False,
            backbone_config=resnet_config,
            backbone=None,
            use_pretrained_backbone=False,
        )
    
    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    
def _prepare_for_class_(inputs_dict):
        inputs_dict = copy.deepcopy(inputs_dict)

        return inputs_dict

# special case for head models
def _prepare_for_class(model_tester, inputs_dict, model_class, return_labels=False):
    inputs_dict = _prepare_for_class_(inputs_dict)

    if return_labels:
        if model_class.__name__ in ["DABDETRForObjectDetection"]:
            labels = []
            for i in range(model_tester.batch_size):
                target = {}
                target["class_labels"] = torch.ones(
                    size=(model_tester.n_targets,), device=torch_device, dtype=torch.long
                )
                target["boxes"] = torch.ones(
                    model_tester.n_targets, 4, device=torch_device, dtype=torch.float
                )
                target["masks"] = torch.ones(
                    model_tester.n_targets,
                    model_tester.min_size,
                    model_tester.max_size,
                    device=torch_device,
                    dtype=torch.float,
                )
                labels.append(target)
            inputs_dict["labels"] = labels

    return inputs_dict

def _mock_init_weights(self, module):
    for name, param in module.named_parameters(recurse=False):
        # Use the first letter of the name to get a value and go from a <> -13 to z <> 12
        value = ord(name[0].lower()) - 110
        param.data.fill_(value)
import os
import tempfile
def _mock_all_init_weights(self):

    import transformers.modeling_utils

    if transformers.modeling_utils._init_weights:
        for module in self.modules():
            module._is_hf_initialized = False
        # Initialize weights
        self.apply(self._initialize_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

def test_save_load_fast_init_to_base(model_tester):
        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()
        
        # make a copy of model class to not break future tests
        # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class

        class CopyClass(DABDETRModel):
            pass

        base_class_copy = CopyClass

        # make sure that all keys are expected for test
        base_class_copy._keys_to_ignore_on_load_missing = []

        # make init deterministic, but make sure that
        # non-initialized weights throw errors nevertheless
        base_class_copy._init_weights = _mock_init_weights
        base_class_copy.init_weights = _mock_all_init_weights

        model = DABDETRModel(config)
        state_dict = model.state_dict()

        # this will often delete a single weight of a multi-weight module
        # to test an edge case
        random_key_to_del = random.choice(list(state_dict.keys()))
        del state_dict[random_key_to_del]

        # check that certain keys didn't get saved with the model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.config.save_pretrained(tmpdirname)
            torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

            model_fast_init = base_class_copy.from_pretrained(tmpdirname)
            model_slow_init = base_class_copy.from_pretrained(tmpdirname, _fast_init=False)

            for key in model_fast_init.state_dict().keys():
                if isinstance(model_slow_init.state_dict()[key], torch.BoolTensor):
                    max_diff = torch.max(
                        model_slow_init.state_dict()[key] ^ model_fast_init.state_dict()[key]
                    ).item()
                else:
                    max_diff = torch.max(
                        torch.abs(model_slow_init.state_dict()[key] - model_fast_init.state_dict()[key])
                    ).item()
                # assertLessEqual(max_diff, 1e-3, msg=f"{key} not identical")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--model_name",
    #     default="dab-detr-resnet-50",
    #     type=str,
    #     help="Name of the DAB_DETR model you'd like to convert.",
    # )
    # parser.add_argument(
    #     "--pretrained_model_weights_path",
    #     default="/Users/davidhajdu/Desktop/dab_detr_r50.pth",
    #     type=str,
    #     help="The path of the original model weights like: Users/username/Desktop/dab_detr_r50.pth",
    # )
    # parser.add_argument(
    #     "--pytorch_dump_folder_path", default="DAB_DETR", type=str, help="Path to the folder to output PyTorch model."
    # )
    # args = parser.parse_args()
    # convert_dab_detr_checkpoint(args.model_name, args.pretrained_model_weights_path, args.pytorch_dump_folder_path)

    model_tester = DABDETRModelTester()
    test_save_load_fast_init_to_base(model_tester)
    # m = {'Z':10, 'a':2, 'D':5}

    # v = tuple(k for k in m.keys())

    # print('zzzz')

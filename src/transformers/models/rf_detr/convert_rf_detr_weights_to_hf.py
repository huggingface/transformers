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
"""Convert RF Detr checkpoints to Hugging Face Transformers format."""

import argparse
import json
import re
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import (
    AutoConfig,
    RFDetrConfig,
    RFDetrDinov2WithRegistersConfig,
    RFDetrForObjectDetection,
    RTDetrImageProcessor,
    RTDetrImageProcessorFast,
)
from transformers.utils import logging


torch.set_printoptions(precision=6, sci_mode=False)


def custom_repr(self):
    # return f"{tuple(self.shape)} {self.flatten()[-10:].tolist()} {original_repr(self)}"
    return f"{tuple(self.shape)} {self.flatten()[-3:].tolist()}"


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_rt_detr_v2_config(model_name: str) -> RFDetrConfig:
    if model_name in ["rf-detr-base", "rf-detr-base-2"]:
        dinov2_size = "small"
    elif model_name == "rf-detr-large":
        dinov2_size = "base"

    base_backbone_model_name = f"facebook/dinov2-with-registers-{dinov2_size}"
    num_register_tokens = 0
    out_indices = [2, 5, 8, 11]
    base_backbone = AutoConfig.from_pretrained(
        base_backbone_model_name,
        num_register_tokens=num_register_tokens,
        out_indices=out_indices,
    )

    num_windows = 4
    backbone_config = RFDetrDinov2WithRegistersConfig(
        **base_backbone.to_dict(),
        num_windows=num_windows,
    )

    scale_factors = [2.0, 0.5]
    d_model = 384
    decoder_self_attention_heads = 12
    decoder_cross_attention_heads = 24
    num_labels = 91
    config = RFDetrConfig(
        backbone_config=backbone_config,
        scale_factors=scale_factors,
        d_model=d_model,
        decoder_self_attention_heads=decoder_self_attention_heads,
        decoder_cross_attention_heads=decoder_cross_attention_heads,
        num_labels=num_labels,
    )

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    if model_name in ["rf-detr-base", "rf-detr-base-2"]:
        pass
        # config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        # config.backbone_config.depths = [2, 2, 2, 2]
        # config.backbone_config.layer_type = "basic"
        # config.encoder_in_channels = [128, 256, 512]
        # config.hidden_expansion = 0.5
        # config.decoder_layers = 3
    elif model_name == "rf-detr-large":
        pass
        # config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        # config.backbone_config.depths = [3, 4, 6, 3]
        # config.backbone_config.layer_type = "basic"
        # config.encoder_in_channels = [128, 256, 512]
        # config.hidden_expansion = 0.5
        # config.decoder_layers = 4

    return config


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"backbone.0.encoder.encoder": r"model.backbone.conv_encoder.model",
    r"backbone.0.projector.stages_sampling.(\d+).(\d+).(\d+).(weight|bias)": r"model.backbone.conv_encoder.projector.scale_layers.\1.sampling_layers.\2.layers.\3.\4",
    r"backbone.0.projector.stages_sampling.(\d+).(\d+).(\d+).conv": r"model.backbone.conv_encoder.projector.scale_layers.\1.sampling_layers.\2.layers.\3.conv",
    r"backbone.0.projector.stages_sampling.(\d+).(\d+).(\d+).bn": r"model.backbone.conv_encoder.projector.scale_layers.\1.sampling_layers.\2.layers.\3.norm",
    r"backbone.0.projector.stages.(\d+).0.cv1.conv": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.conv1.conv",
    r"backbone.0.projector.stages.(\d+).0.cv1.bn": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.conv1.norm",
    r"backbone.0.projector.stages.(\d+).0.cv2.conv": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.conv2.conv",
    r"backbone.0.projector.stages.(\d+).0.cv2.bn": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.conv2.norm",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.conv": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.bottlenecks.\2.conv1.conv",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv1.bn": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.bottlenecks.\2.conv1.norm",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.conv": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.bottlenecks.\2.conv2.conv",
    r"backbone.0.projector.stages.(\d+).0.m.(\d+).cv2.bn": r"model.backbone.conv_encoder.projector.scale_layers.\1.stage_layer.bottlenecks.\2.conv2.norm",
    r"backbone.0.projector.stages.(\d+).1": r"model.backbone.conv_encoder.projector.scale_layers.\1.layer_norm",
    r"transformer.decoder.layers.(\d+).self_attn.out_proj": r"model.decoder.layers.\1.self_attn.out_proj",
    r"transformer.decoder.layers.(\d+).norm1": r"model.decoder.layers.\1.self_attn_layer_norm",
    r"transformer.decoder.layers.(\d+).cross_attn.sampling_offsets": r"model.decoder.layers.\1.encoder_attn.sampling_offsets",
    r"transformer.decoder.layers.(\d+).cross_attn.attention_weights": r"model.decoder.layers.\1.encoder_attn.attention_weights",
    r"transformer.decoder.layers.(\d+).cross_attn.value_proj": r"model.decoder.layers.\1.encoder_attn.value_proj",
    r"transformer.decoder.layers.(\d+).cross_attn.output_proj": r"model.decoder.layers.\1.encoder_attn.output_proj",
    r"transformer.decoder.layers.(\d+).norm2": r"model.decoder.layers.\1.encoder_attn_layer_norm",
    r"transformer.decoder.layers.(\d+).linear1": r"model.decoder.layers.\1.fc1",
    r"transformer.decoder.layers.(\d+).linear2": r"model.decoder.layers.\1.fc2",
    r"transformer.decoder.layers.(\d+).norm3": r"model.decoder.layers.\1.final_layer_norm",
    r"transformer.decoder.norm": r"model.decoder.norm",
    r"transformer.decoder.ref_point_head": r"model.decoder.reference_points_head",
    r"refpoint_embed": r"model.reference_point_embeddings",
    r"transformer.enc_output": r"model.enc_output",
    r"transformer.enc_output_norm": r"model.enc_output_norm",
    r"transformer.enc_out_bbox_embed": r"model.enc_out_bbox_embed",
    r"transformer.enc_out_class_embed": r"model.enc_out_class_embed",
    r"query_feat": r"model.query_position_embeddings",
}


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    # Use the mapping to rename keys
    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict_keys.keys()):
            new_key = re.sub(original_key, converted_key, key)
            if new_key != key:
                state_dict_keys[new_key] = state_dict_keys.pop(key)

    return state_dict_keys


def read_in_q_k_v(state_dict, config: RFDetrConfig):
    prefix = "transformer.decoder.layers"
    decoder_hidden_dim = config.d_model

    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:decoder_hidden_dim, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:decoder_hidden_dim]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            decoder_hidden_dim : 2 * decoder_hidden_dim, :
        ]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[
            decoder_hidden_dim : 2 * decoder_hidden_dim
        ]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-decoder_hidden_dim:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-decoder_hidden_dim:]


def copy_weights(state_dict, config):
    for key, value in dict(state_dict.items()).items():
        if key.startswith("bbox_embed"):
            new_key = f"model.decoder.{key}"
            state_dict[new_key] = value
        if key.startswith("class_embed"):
            new_key = f"model.decoder.{key}"
            state_dict[new_key] = value


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
        "rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        # below is a less converged model that may be better for finetuning but worse for inference
        "rf-detr-base-2": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
        "rf-detr-large": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
    }
    logger.info(f"Converting model {model_name}...")
    state_dict = torch.hub.load_state_dict_from_url(model_name_to_checkpoint_url[model_name], map_location="cpu")[
        "model"
    ]
    original_state_dict = state_dict.copy()
    # rename keys
    state_dict = convert_old_keys_to_new_keys(state_dict)
    for key in state_dict.copy().keys():
        if key.startswith("query_feat"):
            del state_dict[key]

    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config)
    # certain weights are copied from the RFDetrForObjectDetection to the RFDetrDecoder
    copy_weights(state_dict, config)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    for key in state_dict.copy().keys():
        if key.endswith("num_batches_tracked"):
            del state_dict[key]

    # finally, create HuggingFace model and load state dict
    model = RFDetrForObjectDetection(config)
    target_state_dict = model.state_dict()
    model.load_state_dict(state_dict)
    loaded_state_dict = model.state_dict()
    model.eval()

    # load image processor
    image_processor = RTDetrImageProcessorFast(size={"height": 560, "width": 560}, do_normalize=True)

    # prepare image
    img = prepare_img()

    # preprocess image
    transformations = transforms.Compose(
        [
            transforms.Resize([560, 560], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    original_pixel_values = transformations(img).unsqueeze(0)  # insert batch dimension

    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    # Pass image by the model
    with torch.no_grad():
        outputs = model(pixel_values)

    if model_name == "rf-detr-base":
        expected_slice_logits = torch.tensor(
            [[-3.7045, -5.1913, -6.1787], [-4.0106, -9.3450, -5.2043], [-4.1287, -4.7463, -5.8634]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2582, 0.5497, 0.4764], [0.1684, 0.1985, 0.2120], [0.7665, 0.4146, 0.4669]]
        )
    elif model_name == "rf-detr-base-2":
        expected_slice_logits = torch.tensor(
            [[-4.6108, -5.9453, -3.8505], [-3.8702, -6.1136, -5.5677], [-3.7790, -6.4538, -5.9449]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.1691, 0.1984, 0.2118], [0.2594, 0.5506, 0.4736], [0.7669, 0.4136, 0.4654]]
        )
    elif model_name == "rf-detr-large":
        expected_slice_logits = torch.tensor(
            [[-4.7881, -4.6754, -6.1624], [-5.4441, -6.6486, -4.3840], [-3.5455, -4.9318, -6.3544]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2588, 0.5487, 0.4747], [0.5497, 0.2760, 0.0573], [0.7688, 0.4133, 0.4634]]
        )
    else:
        raise ValueError(f"Unknown rf_detr_name: {model_name}")
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
        default="rf-detr-large",
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

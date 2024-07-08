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
"""Convert OmDet-Turbo checkpoints from the original repository.

URL: https://github.com/IDEA-Research/GroundingDINO"""

import argparse

import requests
import torch
from PIL import Image
from torchvision import transforms as T

from transformers import (
    AutoTokenizer,
    OmDetTurboConfig,
    OmDetTurboForObjectDetection,
    OmDetTurboImageProcessor,
    OmDetTurboProcessor,
    SwinConfig,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_omdet_turbo_config(model_name):
    if "tiny" in model_name:
        window_size = 7
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
        image_size = 224
    elif "base" in model_name:
        window_size = 12
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        image_size = 384
    else:
        raise ValueError("Model not supported, only supports base and large variants")

    backbone_config = SwinConfig(
        window_size=window_size,
        image_size=image_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        out_indices=[2, 3, 4],
    )

    config = OmDetTurboConfig(backbone_config=backbone_config)

    return config


def create_rename_keys(state_dict, config):
    rename_keys = []
    # fmt: off
    ########################################## VISION BACKBONE - START
    # patch embedding layer
    rename_keys.append(("backbone.0.patch_embed.proj.weight",
                        "model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.0.patch_embed.proj.bias",
                        "model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.0.patch_embed.norm.weight",
                        "model.backbone.conv_encoder.model.embeddings.norm.weight"))
    rename_keys.append(("backbone.0.patch_embed.norm.bias",
                        "model.backbone.conv_encoder.model.embeddings.norm.bias"))

    for layer, depth in enumerate(config.backbone_config.depths):
        for block in range(depth):
            # layernorms
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.norm1.weight",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.layernorm_before.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.norm1.bias",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.layernorm_before.bias"))

            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.norm2.weight",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.layernorm_after.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.norm2.bias",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.layernorm_after.bias"))
            # attention
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.attn.relative_position_bias_table",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.attn.proj.weight",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.attn.proj.bias",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.output.dense.bias"))
            # intermediate
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.mlp.fc1.weight",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.mlp.fc1.bias",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.intermediate.dense.bias"))

            # output
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.mlp.fc2.weight",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.output.dense.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.mlp.fc2.bias",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.output.dense.bias"))

        # downsample
        if layer!=len(config.backbone_config.depths)-1:
            rename_keys.append((f"backbone.0.layers.{layer}.downsample.reduction.weight",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.downsample.norm.weight",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.downsample.norm.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.downsample.norm.bias",
                                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.downsample.norm.bias"))

    for out_indice in config.backbone_config.out_indices:
        # OmDet-Turbo implementation of out_indices isn't aligned with transformers
        rename_keys.append((f"backbone.0.norm{out_indice-1}.weight",
                        f"model.backbone.conv_encoder.model.hidden_states_norms.stage{out_indice}.weight"))
        rename_keys.append((f"backbone.0.norm{out_indice-1}.bias",
                        f"model.backbone.conv_encoder.model.hidden_states_norms.stage{out_indice}.bias"))

    ########################################## VISION BACKBONE - END

    ########################################## ENCODER - START
    deformable_key_mappings = {
        'self_attn.sampling_offsets.weight': 'deformable_layer.self_attn.sampling_offsets.weight',
        'self_attn.sampling_offsets.bias': 'deformable_layer.self_attn.sampling_offsets.bias',
        'self_attn.attention_weights.weight': 'deformable_layer.self_attn.attention_weights.weight',
        'self_attn.attention_weights.bias': 'deformable_layer.self_attn.attention_weights.bias',
        'self_attn.value_proj.weight': 'deformable_layer.self_attn.value_proj.weight',
        'self_attn.value_proj.bias': 'deformable_layer.self_attn.value_proj.bias',
        'self_attn.output_proj.weight': 'deformable_layer.self_attn.output_proj.weight',
        'self_attn.output_proj.bias': 'deformable_layer.self_attn.output_proj.bias',
        'norm1.weight': 'deformable_layer.self_attn_layer_norm.weight',
        'norm1.bias': 'deformable_layer.self_attn_layer_norm.bias',
        'linear1.weight': 'deformable_layer.fc1.weight',
        'linear1.bias': 'deformable_layer.fc1.bias',
        'linear2.weight': 'deformable_layer.fc2.weight',
        'linear2.bias': 'deformable_layer.fc2.bias',
        'norm2.weight': 'deformable_layer.final_layer_norm.weight',
        'norm2.bias': 'deformable_layer.final_layer_norm.bias',
    }
    text_enhancer_key_mappings = {
        'self_attn.in_proj_weight': 'text_enhancer_layer.self_attn.in_proj_weight',
        'self_attn.in_proj_bias': 'text_enhancer_layer.self_attn.in_proj_bias',
        'self_attn.out_proj.weight': 'text_enhancer_layer.self_attn.out_proj.weight',
        'self_attn.out_proj.bias': 'text_enhancer_layer.self_attn.out_proj.bias',
        'linear1.weight': 'text_enhancer_layer.fc1.weight',
        'linear1.bias': 'text_enhancer_layer.fc1.bias',
        'linear2.weight': 'text_enhancer_layer.fc2.weight',
        'linear2.bias': 'text_enhancer_layer.fc2.bias',
        'norm1.weight': 'text_enhancer_layer.layer_norm_before.weight',
        'norm1.bias': 'text_enhancer_layer.layer_norm_before.bias',
        'norm2.weight': 'text_enhancer_layer.layer_norm_after.weight',
        'norm2.bias': 'text_enhancer_layer.layer_norm_after.bias',
    }
    fusion_key_mappings = {
        'gamma_v': 'fusion_layer.vision_param',
        'gamma_l': 'fusion_layer.text_param',
        'layer_norm_v.weight': 'fusion_layer.layer_norm_vision.weight',
        'layer_norm_v.bias': 'fusion_layer.layer_norm_vision.bias',
        'layer_norm_l.weight': 'fusion_layer.layer_norm_text.weight',
        'layer_norm_l.bias': 'fusion_layer.layer_norm_text.bias',
        'attn.v_proj.weight': 'fusion_layer.attn.vision_proj.weight',
        'attn.v_proj.bias': 'fusion_layer.attn.vision_proj.bias',
        'attn.l_proj.weight': 'fusion_layer.attn.text_proj.weight',
        'attn.l_proj.bias': 'fusion_layer.attn.text_proj.bias',
        'attn.values_v_proj.weight': 'fusion_layer.attn.values_vision_proj.weight',
        'attn.values_v_proj.bias': 'fusion_layer.attn.values_vision_proj.bias',
        'attn.values_l_proj.weight': 'fusion_layer.attn.values_text_proj.weight',
        'attn.values_l_proj.bias': 'fusion_layer.attn.values_text_proj.bias',
        'attn.out_v_proj.weight': 'fusion_layer.attn.out_vision_proj.weight',
        'attn.out_v_proj.bias': 'fusion_layer.attn.out_vision_proj.bias',
        'attn.out_l_proj.weight': 'fusion_layer.attn.out_text_proj.weight',
        'attn.out_l_proj.bias': 'fusion_layer.attn.out_text_proj.bias',
    }
    for layer in range(config.encoder_layers):
        # deformable
        for src, dest in deformable_key_mappings.items():
            rename_keys.append((f"transformer.encoder.layers.{layer}.{src}",
                                f"model.encoder.layers.{layer}.{dest}"))
        # text enhance
        for src, dest in text_enhancer_key_mappings.items():
            rename_keys.append((f"transformer.encoder.text_layers.{layer}.{src}",
                                f"model.encoder.layers.{layer}.{dest}"))
        # fusion layers
        for src, dest in fusion_key_mappings.items():
            rename_keys.append((f"transformer.encoder.fusion_layers.{layer}.{src}",
                                f"model.encoder.layers.{layer}.{dest}"))
    ########################################## ENCODER - END

    ########################################## DECODER - START
    key_mappings_decoder = {
        'cross_attn.sampling_offsets.weight': 'encoder_attn.sampling_offsets.weight',
        'cross_attn.sampling_offsets.bias': 'encoder_attn.sampling_offsets.bias',
        'cross_attn.attention_weights.weight': 'encoder_attn.attention_weights.weight',
        'cross_attn.attention_weights.bias': 'encoder_attn.attention_weights.bias',
        'cross_attn.value_proj.weight': 'encoder_attn.value_proj.weight',
        'cross_attn.value_proj.bias': 'encoder_attn.value_proj.bias',
        'cross_attn.output_proj.weight': 'encoder_attn.output_proj.weight',
        'cross_attn.output_proj.bias': 'encoder_attn.output_proj.bias',
        'norm1.weight': 'encoder_attn_layer_norm.weight',
        'norm1.bias': 'encoder_attn_layer_norm.bias',
        'ca_text.in_proj_weight': 'encoder_attn_text.in_proj_weight',
        'ca_text.in_proj_bias': 'encoder_attn_text.in_proj_bias',
        'ca_text.out_proj.weight': 'encoder_attn_text.out_proj.weight',
        'ca_text.out_proj.bias': 'encoder_attn_text.out_proj.bias',
        'catext_norm.weight': 'encoder_attn_text_layer_norm.weight',
        'catext_norm.bias': 'encoder_attn_text_layer_norm.bias',
        'self_attn.in_proj_weight': 'self_attn.in_proj_weight',
        'self_attn.in_proj_bias': 'self_attn.in_proj_bias',
        'self_attn.out_proj.weight': 'self_attn.out_proj.weight',
        'self_attn.out_proj.bias': 'self_attn.out_proj.bias',
        'norm2.weight': 'self_attn_layer_norm.weight',
        'norm2.bias': 'self_attn_layer_norm.bias',
        'linear1.weight': 'fc1.weight',
        'linear1.bias': 'fc1.bias',
        'linear2.weight': 'fc2.weight',
        'linear2.bias': 'fc2.bias',
        'norm3.weight': 'final_layer_norm.weight',
        'norm3.bias': 'final_layer_norm.bias',
    }
    for layer_num in range(config.decoder_layers):
        source_prefix_decoder = f'transformer.decoder.layers.{layer_num}.'
        target_prefix_decoder = f'model.decoder.layers.{layer_num}.'

        for source_name, target_name in key_mappings_decoder.items():
            rename_keys.append((source_prefix_decoder + source_name,
                               target_prefix_decoder + target_name))
    ########################################## DECODER - END

    ########################################## Additional - START
    for layer_name, params in state_dict.items():
        #### TEXT BACKBONE
        if "bert" in layer_name:
            rename_keys.append((layer_name, layer_name.replace("bert", "model.text_backbone")))
        #### INPUT PROJ - PROJECT OUTPUT FEATURES FROM VISION BACKBONE
        if "input_proj" in layer_name:
            rename_keys.append((layer_name, layer_name.replace("input_proj", "model.input_proj_vision")))
        #### INPUT PROJ - PROJECT OUTPUT FEATURES FROM TEXT BACKBONE
        if "feat_map" in layer_name:
            rename_keys.append((layer_name, layer_name.replace("feat_map", "model.text_projection")))
        #### DECODER REFERENCE POINT HEAD
        if "transformer.decoder.ref_point_head" in layer_name:
            rename_keys.append((layer_name, layer_name.replace("transformer.decoder.ref_point_head",
                                                               "model.decoder.reference_points_head")))
        #### DECODER BBOX EMBED
        if "transformer.decoder.bbox_embed" in layer_name:
            rename_keys.append((layer_name, layer_name.replace("transformer.decoder.bbox_embed",
                                                               "model.decoder.bbox_embed")))
        if "transformer.enc_output" in layer_name:
            rename_keys.append((layer_name, layer_name.replace("transformer", "model")))

        if "transformer.enc_out_bbox_embed" in layer_name:
            rename_keys.append((layer_name, layer_name.replace("transformer.enc_out_bbox_embed",
                                                               "model.encoder_output_bbox_embed")))

    rename_keys.append(("transformer.level_embed", "model.level_embed"))
    rename_keys.append(("transformer.decoder.norm.weight", "model.decoder.layer_norm.weight"))
    rename_keys.append(("transformer.decoder.norm.bias", "model.decoder.layer_norm.bias"))
    rename_keys.append(("transformer.tgt_embed.weight", "model.query_position_embeddings.weight"))
    ########################################## Additional - END

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v_encoder(state_dict, config):
    ########################################## VISION BACKBONE - START
    embed_dim = config.backbone_config.embed_dim
    for layer, depth in enumerate(config.backbone_config.depths):
        hidden_size = embed_dim * 2**layer
        for block in range(depth):
            # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"backbone.0.layers.{layer}.blocks.{block}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.0.layers.{layer}.blocks.{block}.attn.qkv.bias")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[
                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.query.weight"
            ] = in_proj_weight[:hidden_size, :]
            state_dict[
                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.query.bias"
            ] = in_proj_bias[:hidden_size]

            state_dict[
                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.key.weight"
            ] = in_proj_weight[hidden_size : hidden_size * 2, :]
            state_dict[
                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.key.bias"
            ] = in_proj_bias[hidden_size : hidden_size * 2]

            state_dict[
                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.value.weight"
            ] = in_proj_weight[-hidden_size:, :]
            state_dict[
                f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.self.value.bias"
            ] = in_proj_bias[-hidden_size:]
    ########################################## VISION BACKBONE - END


def read_in_q_k_v_text_enhancer(state_dict, config):
    hidden_size = config.hidden_size
    for idx in range(config.encoder_layers):
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.query.weight"] = in_proj_weight[
            :hidden_size, :
        ]
        state_dict[f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.query.bias"] = in_proj_bias[:hidden_size]

        state_dict[f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.key.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]

        state_dict[f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.value.weight"] = in_proj_weight[
            -hidden_size:, :
        ]
        state_dict[f"model.encoder.layers.{idx}.text_enhancer_layer.self_attn.value.bias"] = in_proj_bias[
            -hidden_size:
        ]


def read_in_q_k_v_decoder(state_dict, config):
    hidden_size = config.hidden_size
    for idx in range(config.decoder_layers):
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"model.decoder.layers.{idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"model.decoder.layers.{idx}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{idx}.self_attn.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{idx}.self_attn.query.bias"] = in_proj_bias[:hidden_size]

        state_dict[f"model.decoder.layers.{idx}.self_attn.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"model.decoder.layers.{idx}.self_attn.key.bias"] = in_proj_bias[hidden_size : hidden_size * 2]

        state_dict[f"model.decoder.layers.{idx}.self_attn.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{idx}.self_attn.value.bias"] = in_proj_bias[-hidden_size:]

        # read in weights + bias of cross-attention
        in_proj_weight = state_dict.pop(f"model.decoder.layers.{idx}.encoder_attn_text.in_proj_weight")
        in_proj_bias = state_dict.pop(f"model.decoder.layers.{idx}.encoder_attn_text.in_proj_bias")

        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{idx}.encoder_attn_text.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{idx}.encoder_attn_text.query.bias"] = in_proj_bias[:hidden_size]

        state_dict[f"model.decoder.layers.{idx}.encoder_attn_text.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"model.decoder.layers.{idx}.encoder_attn_text.key.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]

        state_dict[f"model.decoder.layers.{idx}.encoder_attn_text.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{idx}.encoder_attn_text.value.bias"] = in_proj_bias[-hidden_size:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


@torch.no_grad()
def convert_omdet_turbo_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    push_to_hub = args.push_to_hub
    verify_logits = args.verify_logits

    checkpoint_mapping = {
        "omdet-turbo-tiny": "https://huggingface.co/ShilongLiu/OmDetTurbo/resolve/main/groundingdino_swint_ogc.pth",
        "omdet-turbo-base": "https://huggingface.co/ShilongLiu/OmDetTurbo/resolve/main/groundingdino_swinb_cogcoor.pth",
    }
    # Define default OmDetTurbo configuation
    config = get_omdet_turbo_config(model_name)

    # Load original checkpoint
    checkpoint_url = checkpoint_mapping[model_name]
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]
    original_state_dict = {k.replace("module.", ""): v for k, v in original_state_dict.items()}

    for name, param in original_state_dict.items():
        print(name, param.shape)

    # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(original_state_dict, config)

    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)
    read_in_q_k_v_encoder(new_state_dict, config)
    read_in_q_k_v_text_enhancer(new_state_dict, config)
    read_in_q_k_v_decoder(new_state_dict, config)

    # Load HF model
    model = OmDetTurboForObjectDetection(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Load and process test image
    image = prepare_img()
    transforms = T.Compose([T.Resize(size=800, max_size=1333), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    original_pixel_values = transforms(image).unsqueeze(0)

    image_processor = OmDetTurboImageProcessor()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = OmDetTurboProcessor(image_processor=image_processor, tokenizer=tokenizer)

    text = "a cat"
    inputs = processor(images=image, text=preprocess_caption(text), return_tensors="pt")

    assert torch.allclose(original_pixel_values, inputs.pixel_values, atol=1e-4)

    if verify_logits:
        # Running forward
        with torch.no_grad():
            outputs = model(**inputs)

        print(outputs.logits[0, :3, :3])

        expected_slice = torch.tensor(
            [[-4.8913, -0.1900, -0.2161], [-4.9653, -0.3719, -0.3950], [-5.9599, -3.3765, -3.3104]]
        )

        assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4)
        print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"EduardoPacheco/{model_name}")
        processor.push_to_hub(f"EduardoPacheco/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="omdet-turbo-tiny",
        type=str,
        choices=["omdet-turbo-tiny", "omdet-turbo-base"],
        help="Name of the OmDetTurbo model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    parser.add_argument(
        "--verify_logits", action="store_false", help="Whether or not to verify logits after conversion."
    )

    args = parser.parse_args()
    convert_omdet_turbo_checkpoint(args)

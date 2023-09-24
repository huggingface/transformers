# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert GroundingDINO SimMIM checkpoints from the original repository.

URL:
https://github.com/microsoft/GroundingDINO-Transformer/blob/main/MODELHUB.md#simmim-pretrained-grounding_dino-v1-models"""

import argparse

import requests
import torch
from PIL import Image
from torchvision import transforms as T

from transformers import AutoTokenizer, GroundingDINOConfig, GroundingDINOForObjectDetection


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_grounding_dino_config(model_name):
    config = GroundingDINOConfig()

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

    config.backbone_config.window_size = window_size
    config.backbone_config.image_size = image_size
    config.backbone_config.embed_dim = embed_dim
    config.backbone_config.depths = depths
    config.backbone_config.num_heads = num_heads
    config.backbone_config.out_indices = [2, 3, 4]

    return config


def create_rename_keys(state_dict, config):
    rename_keys = []
    # fmt: off
    #TODO names might change after modifing GroundingDINOModel class
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
            # rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.attn.relative_position_index",
            #                     f"encoder.layers.{layer}.blocks.{block}.attention.relative_position_index"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.attn.proj.weight",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.0.layers.{layer}.blocks.{block}.attn.proj.bias",
                            f"model.backbone.conv_encoder.model.encoder.layers.{layer}.blocks.{block}.attention.output.dense.bias"))
            # intermidiate
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
        # Grounding DINO implementation of out_indices isn't aligned with transformers
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
        'gamma_v': 'fusion_layer.gamma_v',
        'gamma_l': 'fusion_layer.gamma_l',
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

    #TODO convert head
    ########################################## HEAD - START
    ########################################## HEAD - END

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
            rename_keys.append((layer_name, layer_name.replace("feat_map", "model.input_proj_text")))
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
def read_in_q_k_v(state_dict, config):
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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def text_processor(text: str, config):
    def preprocess_caption(caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list) -> list:
        """Generate attention mask between each pair of special tokens
        Args:
            input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
            special_tokens_mask (list): special tokens mask.
        Returns:
            torch.Tensor: attention mask between each special tokens.
        """
        input_ids = tokenized["input_ids"]
        bs, num_token = input_ids.shape
        # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
        special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
        for special_token in special_tokens_list:
            special_tokens_mask |= input_ids == special_token

        # idxs: each row is a list of indices of special tokens
        idxs = torch.nonzero(special_tokens_mask)

        # generate attention mask and positional ids
        attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
        position_ids = torch.zeros((bs, num_token), device=input_ids.device)
        cate_to_token_mask_list = [[] for _ in range(bs)]
        previous_col = 0
        for i in range(idxs.shape[0]):
            row, col = idxs[i]
            if (col == 0) or (col == num_token - 1):
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
                position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                    0, col - previous_col, device=input_ids.device
                )
                c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
                c2t_maski[previous_col + 1 : col] = True
                cate_to_token_mask_list[row].append(c2t_maski)
            previous_col = col

        cate_to_token_mask_list = [
            torch.stack(cate_to_token_mask_listi, dim=0) for cate_to_token_mask_listi in cate_to_token_mask_list
        ]

        # # padding mask
        # padding_mask = tokenized['attention_mask']
        # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

        return attention_mask, position_ids.to(torch.long)

    tokenizer = AutoTokenizer.from_pretrained(config.text_backbone_config._name_or_path)
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    text = preprocess_caption(text)
    tokenized = tokenizer([text], padding="longest", return_tensors="pt")
    text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens
    )

    max_text_len = config.max_text_len
    sub_sentence_present = config.sub_sentence_present
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    # extract text embeddings
    if sub_sentence_present:
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids

    return tokenized_for_encoder, tokenized.attention_mask.bool()


@torch.no_grad()
def convert_grounding_dino_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str = None, push_to_hub: bool = False
):
    # Define default GroundingDINO configuation
    config = get_grounding_dino_config(model_name)

    # Load original checkpoint
    original_state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Rename keys
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(original_state_dict, config)

    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)
    read_in_q_k_v(new_state_dict, config)

    # Load HF implementation with default config and converted state dict
    model = GroundingDINOForObjectDetection(config).eval()
    model.load_state_dict(new_state_dict, strict=False)

    # Load and process test image
    image = prepare_img()
    text = "a cat"
    image_processor = T.Compose(
        [T.Resize(size=800, max_size=1333), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    image_inputs = image_processor(image)
    text_inputs, text_token_mask = text_processor(text, config)

    # Running forward
    model(
        pixel_values=image_inputs.unsqueeze(0),
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        token_type_ids=text_inputs["token_type_ids"],
        text_token_mask=text_token_mask,
        text_self_attention_masks=text_inputs["attention_mask"],
        position_ids=text_inputs["position_ids"],
    )

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and image processor for {model_name} to hub")
        model.push_to_hub(f"microsoft/{model_name}")
        image_processor.push_to_hub(f"microsoft/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="grounding-dino-tiny",
        type=str,
        choices=["grounding-dino-tiny", "grounding-dino-base"],
        help="Name of the GroundingDINO model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/home/eduardo/Desktop/Projects/GroundingDINO/weights/grounding_dino_tiny_clean.pth",
        type=str,
        help="Path to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_grounding_dino_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )

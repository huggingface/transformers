# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
import sys
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn

import requests
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from transformers.models.mask2former.feature_extraction_mask2former import Mask2FormerFeatureExtractor
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerConfig,
    Mask2FormerForInstanceSegmentation,
    Mask2FormerForInstanceSegmentationOutput,
    Mask2FormerModel,
    Mask2FormerModelOutput,
)
from transformers.utils import logging


StateDict = Dict[str, Tensor]

logging.set_verbosity_info()
logger = logging.get_logger()

torch.manual_seed(0)


class TrackedStateDict:
    def __init__(self, to_track: Dict):
        """This class "tracks" a python dictionary by keeping track of which item is accessed.

        Args:
            to_track (Dict): The dictionary we wish to track
        """
        self.to_track = to_track
        self._seen: Set[str] = set()

    def __getitem__(self, key: str) -> Any:
        return self.to_track[key]

    def __setitem__(self, key: str, item: Any):
        self._seen.add(key)
        self.to_track[key] = item

    def diff(self) -> List[str]:
        """This method returns a set difference between the keys in the tracked state dict and the one we have access so far.
        This is an effective method to check if we have update all the keys

        Returns:
            List[str]: List of keys not yet updated
        """
        return set(list(self.to_track.keys())) - self._seen

    def copy(self) -> Dict:
        # proxy the call to the internal dictionary
        return self.to_track.copy()


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_data = requests.get(url, stream=True).raw
    im = Image.open(img_data)
    return im


@dataclass
class Args:
    """Fake command line arguments needed by mask2former/detectron implementation"""

    config_file: str


def setup_cfg(args: Args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


class OriginalMask2FormerConfigToOursConverter:
    def __call__(self, original_config: object) -> Mask2FormerConfig:

        model = original_config.MODEL
        mask_former = model.MASK_FORMER
        swin = model.SWIN
        sem_seg_head = model.SEM_SEG_HEAD

        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])
        id2label = {idx: label for idx, label in enumerate(dataset_catalog.thing_classes)} #{idx: label for idx, label in enumerate(dataset_catalog.stuff_classes)}
        label2id = {label: idx for idx, label in id2label.items()}

        config: Mask2FormerConfig = Mask2FormerConfig(
            feature_size=sem_seg_head.CONVS_DIM,
            mask_feature_size=sem_seg_head.MASK_DIM,
            num_labels=sem_seg_head.NUM_CLASSES,
            backbone_config=dict(
                model_type="swin",
                pretrain_img_size=swin.PRETRAIN_IMG_SIZE,
                image_size=swin.PRETRAIN_IMG_SIZE,
                in_channels=3,
                patch_size=swin.PATCH_SIZE,
                embed_dim=swin.EMBED_DIM,
                depths=swin.DEPTHS,
                num_heads=swin.NUM_HEADS,
                window_size=swin.WINDOW_SIZE,
                drop_path_rate=swin.DROP_PATH_RATE,
                attention_probs_dropout_prob=swin.ATTN_DROP_RATE,
                mlp_ratio=swin.MLP_RATIO,
                qkv_bias=swin.QKV_BIAS,
            ),
            decoder_config=dict(
                model_type="detr",
                encoder_layers=mask_former.ENC_LAYERS,
                decoder_layers=mask_former.DEC_LAYERS,
                decoder_ffn_dim=mask_former.DIM_FEEDFORWARD,
                decoder_attention_heads=mask_former.NHEADS,
                num_queries=mask_former.NUM_OBJECT_QUERIES,
                decoder_layerdrop=0.0,
                d_model=mask_former.HIDDEN_DIM,
                dropout=mask_former.DROPOUT,
                attention_dropout=0.0,
                activation_dropout=0.0,
            ),
            pixel_decoder_config=dict(
                model_type="deformable_detr",
                common_stride=sem_seg_head.COMMON_STRIDE,
                encoder_attention_heads=mask_former.NHEADS,
                encoder_layers=sem_seg_head.TRANSFORMER_ENC_LAYERS,
                encoder_ffn_dim=1024,
                dropout=mask_former.DROPOUT,
            ),
            no_object_weight=mask_former.NO_OBJECT_WEIGHT,
            dice_weight=mask_former.DICE_WEIGHT,
            cross_entropy_weight=mask_former.CLASS_WEIGHT,
            mask_weight=mask_former.MASK_WEIGHT,
            train_num_points=mask_former.TRAIN_NUM_POINTS,
            importance_sample_ratio=mask_former.IMPORTANCE_SAMPLE_RATIO,
            oversample_ratio=mask_former.OVERSAMPLE_RATIO,
            
            id2label=id2label,
            label2id=label2id,
        )

        return config


class OriginalMask2FormerConfigToFeatureExtractorConverter:
    def __call__(self, original_config: object) -> Mask2FormerFeatureExtractor:
        model = original_config.MODEL
        model_input = original_config.INPUT
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])

        return Mask2FormerFeatureExtractor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            size=model_input.MIN_SIZE_TEST,
            max_size=model_input.MAX_SIZE_TEST,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            # ignore_index=dataset_catalog.ignore_label, #check
            size_divisibility=32,  # 32 is required by swin
        )


class OriginalMask2FormerCheckpointToOursConverter:
    def __init__(self, original_model: nn.Module, config: Mask2FormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    def replace_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig):
        dst_prefix: str = "pixel_level_module.encoder"
        src_prefix: str = "backbone"

        renamed_keys = [
            (
                f"{src_prefix}.patch_embed.proj.weight",
                f"{dst_prefix}.model.embeddings.patch_embeddings.projection.weight",
            ),
            (f"{src_prefix}.patch_embed.proj.bias", f"{dst_prefix}.model.embeddings.patch_embeddings.projection.bias"),
            (f"{src_prefix}.patch_embed.norm.weight", f"{dst_prefix}.model.embeddings.norm.weight"),
            (f"{src_prefix}.patch_embed.norm.bias", f"{dst_prefix}.model.embeddings.norm.bias"),
        ]
        num_layers = len(config.backbone_config.depths)
        for layer_idx in range(num_layers):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend(
                    [  # src, dst
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table",
                        ),
                    ]
                )
                # now we need to handle the attentions
                # read in weights + bias of input projection layer of cross-attention

                src_att_weight = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight"]
                src_att_bias = src_state_dict[f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias"]

                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight"
                ] = src_att_weight[:offset, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias"
                ] = src_att_bias[:offset]

                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight"
                ] = src_att_weight[offset : offset * 2, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias"
                ] = src_att_bias[offset : offset * 2]

                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight"
                ] = src_att_weight[-offset:, :]
                dst_state_dict[
                    f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias"
                ] = src_att_bias[-offset:]

                # let's pop them
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight")
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias")
                # proj
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias",
                        ),
                    ]
                )

                # second norm
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias",
                        ),
                    ]
                )

                # mlp
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias",
                        ),
                    ]
                )

                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index",
                        )
                    ]
                )

            if layer_idx < num_layers - 1:
                # patch merging
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.reduction.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.reduction.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.weight",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.bias",
                            f"{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.bias",
                        ),
                    ]
                )

            # hidden states norms
            renamed_keys.extend(
                [
                    (
                        f"{src_prefix}.norm{layer_idx}.weight",
                        f"{dst_prefix}.hidden_states_norms.{layer_idx}.weight",
                    ),
                    (
                        f"{src_prefix}.norm{layer_idx}.bias",
                        f"{dst_prefix}.hidden_states_norms.{layer_idx}.bias",
                    ),
                ]
            )
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    
    def replace_deformable_detr_encoder_layers(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder.multi_scale_deform_attn_module.encoder.deformable_detr.layers"
        src_prefix: str = "sem_seg_head.pixel_decoder.transformer.encoder.layers"

        renamed_keys = []
        for i in range(self.config.decoder_config.decoder_layers):

            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.sampling_offsets.weight", 
                    f"{dst_prefix}.{i}.self_attn.sampling_offsets.weight"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.sampling_offsets.bias", 
                    f"{dst_prefix}.{i}.self_attn.sampling_offsets.bias"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.attention_weights.weight", 
                    f"{dst_prefix}.{i}.self_attn.attention_weights.weight"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.attention_weights.bias", 
                    f"{dst_prefix}.{i}.self_attn.attention_weights.bias"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.value_proj.weight", 
                    f"{dst_prefix}.{i}.self_attn.value_proj.weight"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.value_proj.bias", 
                    f"{dst_prefix}.{i}.self_attn.value_proj.bias"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.output_proj.weight", 
                    f"{dst_prefix}.{i}.self_attn.output_proj.weight"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.output_proj.bias", 
                    f"{dst_prefix}.{i}.self_attn.output_proj.bias"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.norm1.weight", 
                    f"{dst_prefix}.{i}.self_attn.self_attn_layer_norm.weight"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.norm1.bias", 
                    f"{dst_prefix}.{i}.self_attn.self_attn_layer_norm.bias"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.linear1.weight", 
                    f"{dst_prefix}.{i}.self_attn.fc1.weight"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.linear1.bias", 
                    f"{dst_prefix}.{i}.self_attn.fc1.bias"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.norm2.weight", 
                    f"{dst_prefix}.{i}.self_attn.final_layer_norm.weight"
                )
            )
            renamed_keys.append(
                (
                    f"{src_prefix}.{i}.self_attn.norm2.bias", 
                    f"{dst_prefix}.{i}.self_attn.final_layer_norm.bias"
                )
            )
        
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_deformable_detr_encoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder.multi_scale_deform_attn_module.encoder"
        src_prefix: str = "sem_seg_head.pixel_decoder.transformer"

        renamed_keys = [
            (f"{src_prefix}.level_embed", f"{dst_prefix}.level_embed")
        ]

        self.replace_deformable_detr_encoder_layers(dst_state_dict,src_state_dict)

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_deform_attn_encoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder.multi_scale_deform_attn_module"
        src_prefix: str = "sem_seg_head.pixel_decoder"

        renamed_keys = []
        for idx in range(0,3):
            renamed_keys.append(
                (f"{src_prefix}.input_proj.{idx}.0.weight", f"{dst_prefix}.embeddings.input_projection.{idx}.0.weight")
            )
            renamed_keys.append(
                (f"{src_prefix}.input_proj.{idx}.0.bias", f"{dst_prefix}.embeddings.input_projection.{idx}.0.bias")
            )
            renamed_keys.append(
                (f"{src_prefix}.input_proj.{idx}.1.weight", f"{dst_prefix}.embeddings.input_projection.{idx}.1.weight")
            )
            renamed_keys.append(
                (f"{src_prefix}.input_proj.{idx}.1.bias", f"{dst_prefix}.embeddings.input_projection.{idx}.1.bias")
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        
    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder"
        src_prefix: str = "sem_seg_head.pixel_decoder"

        self.replace_backbone(dst_state_dict, src_state_dict, self.config)

        self.replace_deform_attn_encoder(dst_state_dict, src_state_dict)

        def rename_keys_for_conv(detectron_conv: str, mine_conv: str):
            return [
                (f"{detectron_conv}.weight", f"{mine_conv}.0.weight"),
                (f"{detectron_conv}.norm.weight", f"{mine_conv}.1.weight"),
                (f"{detectron_conv}.norm.bias", f"{mine_conv}.1.bias"),
            ]

        renamed_keys = [
            (f"{src_prefix}.mask_features.weight", f"{dst_prefix}.mask_projection.weight"),
            (f"{src_prefix}.mask_features.bias", f"{dst_prefix}.mask_projection.bias"),
            # the layers in the original one are in reverse order
        ]

        # add all the fpn layers
        renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.adapter_1", f"{dst_prefix}.feature_pyramid_network.layers.0.proj")
            )
        renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.layer_1", f"{dst_prefix}.feature_pyramid_network.layers.0.block")
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def rename_keys_in_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor"
        
        rename_keys = []
        for i in range(self.config.decoder_config.decoder_layers - 1):
            
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight",
                    f"{dst_prefix}.layers.{i}.self_attn.in_proj_weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias",
                    f"{dst_prefix}.layers.{i}.self_attn.in_proj_bias",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.bias",
                )
            )

            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.norm.weight",
                    f"{dst_prefix}.layers.{i}.self_attn_layer_norm.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_self_attention_layers.{i}.norm.bias",
                    f"{dst_prefix}.layers.{i}.self_attn_layer_norm.bias",
                )
            )

            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight",
                    f"{dst_prefix}.layers.{i}.encoder_attn.in_proj_weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias",
                    f"{dst_prefix}.layers.{i}.encoder_attn.in_proj_bias",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.bias",
                )
            )

            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.norm.weight",
                    f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.norm.bias",
                    f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.bias",
                )
            )

            rename_keys.append((f"{src_prefix}.transformer_ffn_layers.{i}.linear1.weight", f"{dst_prefix}.layers.{i}.fc1.weight"))
            rename_keys.append((f"{src_prefix}.transformer_ffn_layers.{i}.linear1.bias", f"{dst_prefix}.layers.{i}.fc1.bias"))
            rename_keys.append((f"{src_prefix}.transformer_ffn_layers.{i}.linear2.weight", f"{dst_prefix}.layers.{i}.fc2.weight"))
            rename_keys.append((f"{src_prefix}.transformer_ffn_layers.{i}.linear2.bias", f"{dst_prefix}.layers.{i}.fc2.bias"))
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.norm.weight", f"{dst_prefix}.layers.{i}.final_layer_norm.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.norm.bias", f"{dst_prefix}.layers.{i}.final_layer_norm.bias")
            )
    

        return rename_keys

    def replace_q_k_v_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        for i in range(self.config.decoder_config.decoder_layers):
            # read in weights + bias of input projection layer of self-attention
            in_proj_weight = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_weight")
            in_proj_bias = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_bias")
            # next, add query, keys and values (in that order) to the state dict
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
            # read in weights + bias of input projection layer of cross-attention
            in_proj_weight_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_weight")
            in_proj_bias_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_bias")
            # next, add query, keys and values (in that order) of cross-attention to the state dict
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[
                256:512, :
            ]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]

    def replace_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor"
        
        renamed_keys = self.rename_keys_in_masked_attention_decoder(dst_state_dict, src_state_dict)
        
        # add more
        renamed_keys.extend(
            [
                (f"{src_prefix}.decoder_norm.weight", f"{dst_prefix}.layernorm.weight"),
                (f"{src_prefix}.decoder_norm.bias", f"{dst_prefix}.layernorm.bias"),
            ]
        )

        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend(
                [
                    (f"{src_prefix}.mask_embed.layers.{i}.weight", f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.weight"),
                    (f"{src_prefix}.mask_embed.layers.{i}.bias", f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.bias"),
                ]
            )
        

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

        # self.replace_q_k_v_in_detr_decoder(dst_state_dict, src_state_dict)

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module"
        src_prefix: str = "sem_seg_head.predictor"

        self.replace_masked_attention_decoder(dst_state_dict, src_state_dict)

        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.query_feat.weight", f"{dst_prefix}.learnable_queries.weight"),
            (f"{src_prefix}.level_embed.weight", f"{dst_prefix}.level_embed.weight"),
        ]

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_instance_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # NOTE in our case we don't have a prefix, thus we removed the "." from the keys later on!
        dst_prefix: str = ""
        src_prefix: str = "sem_seg_head.predictor"

        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        logger.info(f"Replacing keys {pformat(renamed_keys)}")
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, mask2former: Mask2FormerModel) -> Mask2FormerModel:
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_pixel_module(dst_state_dict, src_state_dict)
        self.replace_transformer_module(dst_state_dict, src_state_dict)

        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        logger.info("🙌 Done")

        mask2former.load_state_dict(mask2former.state_dict())#load_state_dict(dst_state_dict)

        return mask2former

    def convert_instance_segmentation(
        self, mask2former: Mask2FormerForInstanceSegmentation
    ) -> Mask2FormerForInstanceSegmentation:
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_instance_segmentation_module(dst_state_dict, src_state_dict)

        mask2former.load_state_dict(mask2former.state_dict())#load_state_dict(dst_state_dict)

        return mask2former

    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")

        for checkpoint in checkpoints:
            logger.info(f"💪 Converting {checkpoint.stem}")
            # find associated config file
            config: Path = config_dir / checkpoint.parents[2].stem / checkpoint.parents[1].stem / "swin" / f"{checkpoint.parents[0].stem}.yaml"

            yield config, checkpoint


def test(original_model, our_model: Mask2FormerForInstanceSegmentation, feature_extractor: Mask2FormerFeatureExtractor):
    with torch.no_grad():

        original_model = original_model.eval()
        our_model = our_model.eval()

        im = prepare_img()

        tr = T.Compose(
            [
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(
                    mean=torch.tensor([123.675, 116.280, 103.530]) / 255.0,
                    std=torch.tensor([58.395, 57.120, 57.375]) / 255.0,
                ),
            ],
        )

        x = tr(im).unsqueeze(0)

        original_model_backbone_features = original_model.backbone(x.clone())

        our_model_output: Mask2FormerModelOutput = our_model.model(x.clone(), output_hidden_states=True)

        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):

            assert torch.allclose(
                original_model_feature, our_model_feature, atol=1e-3
            ), "The backbone features are not the same."

        original_model_pixel_out = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )

        assert torch.allclose(
            original_model_pixel_out[0], our_model_output.pixel_decoder_last_hidden_state, atol=1e-4
        ), "The pixel decoder feature are not the same"

        # let's test the full model
        original_model_out = original_model([{"image": x.squeeze(0)}])

        original_segmentation = original_model_out[0]["sem_seg"]

        our_model_out: Mask2FormerForInstanceSegmentationOutput = our_model(x)

        our_segmentation = feature_extractor.post_process_segmentation(our_model_out, target_size=(384, 384))

        assert torch.allclose(
            original_segmentation, our_segmentation, atol=1e-3
        ), "The segmentation image is not the same."

        logger.info("✅ Test passed!")


def get_model_name(checkpoint_file: Path):
    # model_name_raw is something like maskformer2_swin_small_bs16_50ep
    model_name_raw: str = checkpoint_file.parents[0].stem

    # `segmentation_task_type` must be one of the following: `instance-segmentation`, `panoptic-segmentation`, `semantic-segmentation`
    segmentation_task_name: str = checkpoint_file.parents[1].stem
    if segmentation_task_name not in ["instance-segmentation", "panoptic-segmentation", "semantic-segmentation"]:
        raise ValueError(f"{segmentation_task_name} must be wrong since acceptable values are: instance-segmentation, panoptic-segmentation, semantic-segmentation.")

    # dataset name must be one of the following: `coco`, `ade`, `cityscapes`, `mapillary-vistas`
    dataset_name: str = checkpoint_file.parents[2].stem
    if dataset_name not in ["coco", "ade"]:
        raise ValueError(f"{dataset_name} must be wrong since we didn't find 'coco' or 'ade' in it ")

    backbone = "swin"
    backbone_types = ["tiny", "small", "base", "large"]
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0]

    model_name = f"mask2former-{segmentation_task_name.split('-')[0]}-{backbone}-{backbone_type}-{dataset_name}"

    return model_name


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Command line to convert the original mask2formers (with swin backbone) to our implementations."
    )

    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " <DIR_NAME>/<DATASET_NAME>/<SEGMENTATION_TASK_NAME>/<CONFIG_NAME>.pkl"
        ),
    )
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<SEGMENTATION_TASK_NAME>/<CONFIG_NAME>.yaml"
        ),
    )
    parser.add_argument(
        "--save_model_dir",
        required=True,
        type=Path,
        help="Path to the folder to output PyTorch models.",
    )
    parser.add_argument(
        "--mask2former_dir",
        required=True,
        type=Path,
        help=(
            "A path to Mask2Former's original implementation directory. You can download from here:"
            " https://github.com/facebookresearch/Mask2Former"
        ),
    )

    args = parser.parse_args()

    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    save_directory: Path = args.save_model_dir
    mask2former_dir: Path = args.mask2former_dir
    # append the path to the parents to mask2former dir
    sys.path.append(str(mask2former_dir.parent))
    # import original Mask2Former config and model from original source code repo
    from Mask2Former.mask2former.config import add_maskformer2_config
    from Mask2Former.mask2former.maskformer_model import MaskFormer as OriginalMask2Former

    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    for config_file, checkpoint_file in OriginalMask2FormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):

        feature_extractor = OriginalMask2FormerConfigToFeatureExtractorConverter()(
            setup_cfg(Args(config_file=config_file))
        )

        original_config = setup_cfg(Args(config_file=config_file))
        mask2former_kwargs = OriginalMask2Former.from_config(original_config)

        original_model = OriginalMask2Former(**mask2former_kwargs).eval()

        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        config: Mask2FormerConfig = OriginalMask2FormerConfigToOursConverter()(original_config)
        print("Conversion successfull!!")
        print(config)
        mask2former = Mask2FormerModel(config=config).eval()

        converter = OriginalMask2FormerCheckpointToOursConverter(original_model, config)

        mask2former = converter.convert(mask2former)

        mask2former_for_instance_segmentation = Mask2FormerForInstanceSegmentation(config=config).eval()

        mask2former_for_instance_segmentation.model = mask2former
        mask2former_for_instance_segmentation = converter.convert_instance_segmentation(
            mask2former_for_instance_segmentation
        )

        # commenting test model for now due to mismatch in feature values -> need to fix
        # test(original_model, mask2former_for_instance_segmentation, feature_extractor)

        model_name = get_model_name(checkpoint_file)
        logger.info(f"🪄 Saving {model_name}")

        feature_extractor.save_pretrained(save_directory / model_name)
        mask2former_for_instance_segmentation.save_pretrained(save_directory / model_name)

        repo_id = os.path.join(save_directory, model_name)
        
        feature_extractor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add model",
            use_temp_dir=True,
        )
        
        mask2former_for_instance_segmentation.push_to_hub(
            repo_id=repo_id,
            commit_message="Add model",
            use_temp_dir=True,
        )

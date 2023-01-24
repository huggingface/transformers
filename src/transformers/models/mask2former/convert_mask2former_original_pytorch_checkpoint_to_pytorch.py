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
import json
import sys
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
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Mask2FormerModel,
    SwinConfig,
)
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
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

        repo_id = "huggingface/label-files"
        if model.SEM_SEG_HEAD.NUM_CLASSES == 847:
            filename = "mask2former-ade20k-full-id2label.json"
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 150:
            filename = "ade20k-id2label.json"
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 80:
            filename = "coco-detection-mmdet-id2label.json"
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 171:
            filename = "mask2former-coco-stuff-id2label.json"
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 133:
            filename = "coco-panoptic-id2label.json"
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 19:
            filename = "cityscapes-id2label.json"
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 8:
            filename = "cityscapes-instance-id2label.json"
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 65:
            filename = "mapillary-vistas-id2label.json"

        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {label: idx for idx, label in id2label.items()}

        if model.SWIN.EMBED_DIM == 96:
            backbone_config = SwinConfig.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224", out_features=["stage1", "stage2", "stage3", "stage4"]
            )
        elif model.SWIN.EMBED_DIM == 128:
            backbone_config = SwinConfig(
                embed_dim=128,
                window_size=12,
                depths=(2, 2, 18, 2),
                num_heads=(4, 8, 16, 32),
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )

        elif model.SWIN.EMBED_DIM == 192:
            backbone_config = SwinConfig.from_pretrained(
                "microsoft/swin-large-patch4-window12-384", out_features=["stage1", "stage2", "stage3", "stage4"]
            )
        else:
            raise ValueError(f"embed dim {model.SWIN.EMBED_DIM} not supported for Swin!")

        backbone_config.drop_path_rate = model.SWIN.DROP_PATH_RATE
        backbone_config.attention_probs_dropout_prob = model.SWIN.ATTN_DROP_RATE
        backbone_config.depths = model.SWIN.DEPTHS

        config: Mask2FormerConfig = Mask2FormerConfig(
            ignore_value=model.SEM_SEG_HEAD.IGNORE_VALUE,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            num_queries=model.MASK_FORMER.NUM_OBJECT_QUERIES,
            no_object_weight=model.MASK_FORMER.NO_OBJECT_WEIGHT,
            class_weight=model.MASK_FORMER.CLASS_WEIGHT,
            mask_weight=model.MASK_FORMER.MASK_WEIGHT,
            dice_weight=model.MASK_FORMER.DICE_WEIGHT,
            train_num_points=model.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=model.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=model.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            init_std=0.02,
            init_xavier_std=1.0,
            use_auxiliary_loss=model.MASK_FORMER.DEEP_SUPERVISION,
            feature_strides=[4, 8, 16, 32],
            backbone_config=backbone_config,
            id2label=id2label,
            label2id=label2id,
            feature_size=model.SEM_SEG_HEAD.CONVS_DIM,
            mask_feature_size=model.SEM_SEG_HEAD.MASK_DIM,
            hidden_dim=model.MASK_FORMER.HIDDEN_DIM,
            encoder_layers=model.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            encoder_feedforward_dim=1024,
            decoder_layers=model.MASK_FORMER.DEC_LAYERS,
            num_attention_heads=model.MASK_FORMER.NHEADS,
            dropout=model.MASK_FORMER.DROPOUT,
            dim_feedforward=model.MASK_FORMER.DIM_FEEDFORWARD,
            pre_norm=model.MASK_FORMER.PRE_NORM,
            enforce_input_proj=model.MASK_FORMER.ENFORCE_INPUT_PROJ,
            common_stride=model.SEM_SEG_HEAD.COMMON_STRIDE,
        )
        return config


class OriginalMask2FormerConfigToFeatureExtractorConverter:
    def __call__(self, original_config: object) -> Mask2FormerImageProcessor:
        model = original_config.MODEL
        model_input = original_config.INPUT

        return Mask2FormerImageProcessor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            size=model_input.MIN_SIZE_TEST,
            max_size=model_input.MAX_SIZE_TEST,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            ignore_index=model.SEM_SEG_HEAD.IGNORE_VALUE,
            size_divisibility=32,
        )


class OriginalMask2FormerCheckpointToOursConverter:
    def __init__(self, original_model: nn.Module, config: Mask2FormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    def replace_maskformer_swin_backbone(
        self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig
    ):
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

    def replace_swin_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig):
        dst_prefix: str = "pixel_level_module.encoder"
        src_prefix: str = "backbone"

        renamed_keys = [
            (
                f"{src_prefix}.patch_embed.proj.weight",
                f"{dst_prefix}.embeddings.patch_embeddings.projection.weight",
            ),
            (f"{src_prefix}.patch_embed.proj.bias", f"{dst_prefix}.embeddings.patch_embeddings.projection.bias"),
            (f"{src_prefix}.patch_embed.norm.weight", f"{dst_prefix}.embeddings.norm.weight"),
            (f"{src_prefix}.patch_embed.norm.bias", f"{dst_prefix}.embeddings.norm.bias"),
        ]

        for layer_idx in range(len(config.backbone_config.depths)):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend(
                    [  # src, dst
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table",
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
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight"
                ] = src_att_weight[:offset, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias"
                ] = src_att_bias[:offset]

                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight"
                ] = src_att_weight[offset : offset * 2, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias"
                ] = src_att_bias[offset : offset * 2]

                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight"
                ] = src_att_weight[-offset:, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias"
                ] = src_att_bias[-offset:]

                # let's pop them
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight")
                src_state_dict.pop(f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias")
                # proj
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias",
                        ),
                    ]
                )

                # second norm
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias",
                        ),
                    ]
                )

                # mlp
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias",
                        ),
                    ]
                )

                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index",
                        )
                    ]
                )

            if layer_idx < 3:
                # patch merging
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.reduction.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.downsample.reduction.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.weight",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.weight",
                        ),
                        (
                            f"{src_prefix}.layers.{layer_idx}.downsample.norm.bias",
                            f"{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.bias",
                        ),
                    ]
                )

            # hidden states norms
            renamed_keys.extend(
                [
                    (
                        f"{src_prefix}.norm{layer_idx}.weight",
                        f"{dst_prefix}.hidden_states_norms.stage{layer_idx+1}.weight",
                    ),
                    (
                        f"{src_prefix}.norm{layer_idx}.bias",
                        f"{dst_prefix}.hidden_states_norms.stage{layer_idx+1}.bias",
                    ),
                ]
            )
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # Backbone + Pixel Decoder
    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder"
        src_prefix: str = "sem_seg_head.pixel_decoder"

        self.replace_swin_backbone(dst_state_dict, src_state_dict, self.config)

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
            self_attn_keys = []
            self_attn_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.attention_weights", f"{dst_prefix}.attention_weights")
            )
            self_attn_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.output_proj", f"{dst_prefix}.output_proj")
            )
            self_attn_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.sampling_offsets", f"{dst_prefix}.sampling_offsets")
            )
            self_attn_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.value_proj", f"{dst_prefix}.value_proj"))

            return self_attn_keys

        def rename_keys_for_encoder_layer(src_prefix: str, dst_prefix: str):
            encoder_keys = []
            encoder_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.linear1", f"{dst_prefix}.fc1"))
            encoder_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.linear2", f"{dst_prefix}.fc2"))
            encoder_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.norm1", f"{dst_prefix}.self_attn_layer_norm")
            )
            encoder_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.norm2", f"{dst_prefix}.final_layer_norm"))
            encoder_keys.extend(rename_keys_for_self_attn(f"{src_prefix}.self_attn", f"{dst_prefix}.self_attn"))

            return encoder_keys

        # convolution layer for final features
        renamed_keys = [
            (f"{src_prefix}.adapter_1.weight", f"{dst_prefix}.adapter_1.0.weight"),
            (f"{src_prefix}.adapter_1.norm.weight", f"{dst_prefix}.adapter_1.1.weight"),
            (f"{src_prefix}.adapter_1.norm.bias", f"{dst_prefix}.adapter_1.1.bias"),
        ]

        renamed_keys.extend(
            [
                (f"{src_prefix}.layer_1.weight", f"{dst_prefix}.layer_1.0.weight"),
                (f"{src_prefix}.layer_1.norm.weight", f"{dst_prefix}.layer_1.1.weight"),
                (f"{src_prefix}.layer_1.norm.bias", f"{dst_prefix}.layer_1.1.bias"),
            ]
        )

        # proj layers
        for i in range(3):
            for j in range(2):
                renamed_keys.extend(
                    [
                        (f"{src_prefix}.input_proj.{i}.{j}.weight", f"{dst_prefix}.input_projections.{i}.{j}.weight"),
                        (f"{src_prefix}.input_proj.{i}.{j}.bias", f"{dst_prefix}.input_projections.{i}.{j}.bias"),
                    ]
                )

        renamed_keys.extend([(f"{src_prefix}.transformer.level_embed", f"{dst_prefix}.level_embed")])

        # layers
        for layer_idx in range(self.config.encoder_layers):
            renamed_keys.extend(
                rename_keys_for_encoder_layer(
                    f"{src_prefix}.transformer.encoder.layers.{layer_idx}", f"{dst_prefix}.encoder.layers.{layer_idx}"
                )
            )

        # proj
        renamed_keys.extend(
            [
                (f"{src_prefix}.mask_features.weight", f"{dst_prefix}.mask_projection.weight"),
                (f"{src_prefix}.mask_features.bias", f"{dst_prefix}.mask_projection.bias"),
            ]
        )
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # Transformer Decoder
    def rename_keys_in_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor"

        rename_keys = []
        for i in range(self.config.decoder_layers - 1):

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
                    f"{dst_prefix}.layers.{i}.cross_attn.in_proj_weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias",
                    f"{dst_prefix}.layers.{i}.cross_attn.in_proj_bias",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.cross_attn.out_proj.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.cross_attn.out_proj.bias",
                )
            )

            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.norm.weight",
                    f"{dst_prefix}.layers.{i}.cross_attn_layer_norm.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_cross_attention_layers.{i}.norm.bias",
                    f"{dst_prefix}.layers.{i}.cross_attn_layer_norm.bias",
                )
            )

            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear1.weight", f"{dst_prefix}.layers.{i}.fc1.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear1.bias", f"{dst_prefix}.layers.{i}.fc1.bias")
            )
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear2.weight", f"{dst_prefix}.layers.{i}.fc2.weight")
            )
            rename_keys.append(
                (f"{src_prefix}.transformer_ffn_layers.{i}.linear2.bias", f"{dst_prefix}.layers.{i}.fc2.bias")
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_ffn_layers.{i}.norm.weight",
                    f"{dst_prefix}.layers.{i}.final_layer_norm.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}.transformer_ffn_layers.{i}.norm.bias",
                    f"{dst_prefix}.layers.{i}.final_layer_norm.bias",
                )
            )

        return rename_keys

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
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.weight",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.weight",
                    ),
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.bias",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.bias",
                    ),
                ]
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder.layers"
        src_prefix: str = "sem_seg_head.predictor"
        for i in range(self.config.decoder_layers - 1):
            # read in weights + bias of input projection layer of self-attention
            in_proj_weight = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight"
            )
            in_proj_bias = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias"
            )
            # next, add query, keys and values (in that order) to the state dict
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module"
        src_prefix: str = "sem_seg_head.predictor"

        self.replace_masked_attention_decoder(dst_state_dict, src_state_dict)

        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.query_feat.weight", f"{dst_prefix}.queries_features.weight"),
            (f"{src_prefix}.level_embed.weight", f"{dst_prefix}.level_embed.weight"),
        ]

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)

    def replace_universal_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
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

        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        mask2former.load_state_dict(state_dict)
        return mask2former

    def convert_universal_segmentation(
        self, mask2former: Mask2FormerForUniversalSegmentation
    ) -> Mask2FormerForUniversalSegmentation:
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_universal_segmentation_module(dst_state_dict, src_state_dict)

        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        mask2former.load_state_dict(state_dict)

        return mask2former

    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")

        for checkpoint in checkpoints:
            logger.info(f"💪 Converting {checkpoint.stem}")
            # find associated config file

            # dataset_name e.g 'coco'
            dataset_name = checkpoint.parents[2].stem
            if dataset_name == "ade":
                dataset_name = dataset_name.replace("ade", "ade20k")

            # task type e.g 'instance-segmentation'
            segmentation_task = checkpoint.parents[1].stem

            # config file corresponding to checkpoint
            config_file_name = f"{checkpoint.parents[0].stem}.yaml"

            config: Path = config_dir / dataset_name / segmentation_task / "swin" / config_file_name
            yield config, checkpoint


def test(
    original_model,
    our_model: Mask2FormerForUniversalSegmentation,
    feature_extractor: Mask2FormerImageProcessor,
    tolerance: float,
):
    with torch.no_grad():
        original_model = original_model.eval()
        our_model = our_model.eval()

        im = prepare_img()
        x = feature_extractor(images=im, return_tensors="pt")["pixel_values"]

        original_model_backbone_features = original_model.backbone(x.clone())
        our_model_output: Mask2FormerModelOutput = our_model.model(x.clone(), output_hidden_states=True)

        # Test backbone
        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=tolerance
            ), "The backbone features are not the same."

        # Test pixel decoder
        mask_features, _, multi_scale_features = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )

        for original_model_feature, our_model_feature in zip(
            multi_scale_features, our_model_output.pixel_decoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=tolerance
            ), "The pixel decoder feature are not the same"

        # Let's test the full model
        tr_complete = T.Compose(
            [T.Resize((384, 384)), T.ToTensor()],
        )
        y = (tr_complete(im) * 255.0).to(torch.int).float()

        # modify original Mask2Former code to return mask and class logits
        original_class_logits, original_mask_logits = original_model([{"image": y.clone().squeeze(0)}])

        our_model_out: Mask2FormerForUniversalSegmentationOutput = our_model(x.clone())
        our_mask_logits = our_model_out.masks_queries_logits
        our_class_logits = our_model_out.class_queries_logits

        assert original_mask_logits.shape == our_mask_logits.shape, "Output masks shapes are not matching."
        assert original_class_logits.shape == our_class_logits.shape, "Output class logits shapes are not matching."
        assert torch.allclose(
            original_class_logits, our_class_logits, atol=tolerance
        ), "The class logits are not the same."
        assert torch.allclose(
            original_mask_logits, our_mask_logits, atol=tolerance
        ), "The predicted masks are not the same."

        logger.info("✅ Test passed!")


def get_model_name(checkpoint_file: Path):
    # model_name_raw is something like maskformer2_swin_small_bs16_50ep
    model_name_raw: str = checkpoint_file.parents[0].stem

    # `segmentation_task_type` must be one of the following: `instance-segmentation`, `panoptic-segmentation`, `semantic-segmentation`
    segmentation_task_name: str = checkpoint_file.parents[1].stem
    if segmentation_task_name not in ["instance-segmentation", "panoptic-segmentation", "semantic-segmentation"]:
        raise ValueError(
            f"{segmentation_task_name} must be wrong since acceptable values are: instance-segmentation,"
            " panoptic-segmentation, semantic-segmentation."
        )

    # dataset name must be one of the following: `coco`, `ade`, `cityscapes`, `mapillary-vistas`
    dataset_name: str = checkpoint_file.parents[2].stem
    if dataset_name not in ["coco", "ade", "cityscapes", "mapillary-vistas"]:
        raise ValueError(
            f"{dataset_name} must be wrong since we didn't find 'coco' or 'ade' or 'cityscapes' or 'mapillary-vistas'"
            " in it "
        )

    backbone = "swin"
    backbone_types = ["tiny", "small", "base_IN21k", "base", "large"]
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0].replace("_", "-")

    model_name = f"mask2former-{backbone}-{backbone_type}-{dataset_name}-{segmentation_task_name.split('-')[0]}"

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
    mask2former_dir: Path = args.mask2former_dir
    # append the path to the parents to mask2former dir
    sys.path.append(str(mask2former_dir.parent))
    # import original Mask2Former config and model from original source code repo
    from Mask2Former.mask2former.config import add_maskformer2_config
    from Mask2Former.mask2former.maskformer_model import MaskFormer as OriginalMask2Former

    for config_file, checkpoint_file in OriginalMask2FormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):
        model_name = get_model_name(checkpoint_file)
        feature_extractor = OriginalMask2FormerConfigToFeatureExtractorConverter()(
            setup_cfg(Args(config_file=config_file))
        )
        feature_extractor.size = {"height": 384, "width": 384}

        original_config = setup_cfg(Args(config_file=config_file))
        mask2former_kwargs = OriginalMask2Former.from_config(original_config)
        original_model = OriginalMask2Former(**mask2former_kwargs).eval()

        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        config: Mask2FormerConfig = OriginalMask2FormerConfigToOursConverter()(original_config)
        mask2former = Mask2FormerModel(config=config).eval()

        converter = OriginalMask2FormerCheckpointToOursConverter(original_model, config)
        mask2former = converter.convert(mask2former)

        mask2former_for_segmentation = Mask2FormerForUniversalSegmentation(config=config).eval()
        mask2former_for_segmentation.model = mask2former

        mask2former_for_segmentation = converter.convert_universal_segmentation(mask2former_for_segmentation)

        tolerance = 3e-1
        high_tolerance_models = [
            "mask2former-swin-base-IN21k-coco-instance",
            "mask2former-swin-base-coco-instance",
            "mask2former-swin-small-cityscapes-semantic",
        ]

        if model_name in high_tolerance_models:
            tolerance = 3e-1

        logger.info(f"🪄 Testing {model_name}...")
        test(original_model, mask2former_for_segmentation, feature_extractor, tolerance)
        logger.info(f"🪄 Pushing {model_name} to hub...")

        feature_extractor.push_to_hub(model_name)
        mask2former_for_segmentation.push_to_hub(model_name)

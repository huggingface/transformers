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

import requests
import torch
import torchvision
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn

from transformers import (
    Mask2FormerConfig,
    VideoMask2FormerForVideoSegmentation,
    VideoMask2FormerImageProcessor,
    VideoMask2FormerModel,
    SwinConfig,
)
from transformers.models.mask2former.modeling_video_mask2former import (
    VideoMask2FormerForVideoSegmentationOutput,
    VideoMask2FormerModelOutput,
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
        return set(self.to_track.keys()) - self._seen

    def copy(self) -> Dict:
        # proxy the call to the internal dictionary
        return self.to_track.copy()


@dataclass
class Args:
    """Fake command line arguments needed by video mask2former/detectron implementation"""

    config_file: str


def add_maskformer2_video_config(cfg):
    # video data
    # DataLoader
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = []  # "brightness", "contrast", "saturation", "rotation"

def setup_video_cfg(args: Args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


class OriginalVideoMask2FormerConfigToOursConverter:
    def __call__(self, original_config: object, dataset_name: str) -> Mask2FormerConfig:
        model = original_config.MODEL

        repo_id = "huggingface/label-files"

        if dataset_name == "youtubevis-2021" and model.SEM_SEG_HEAD.NUM_CLASSES == 40:
            filename = "youtubevis_2021-instance-id2label.json"
        elif dataset_name == "youtubevis-2019" and model.SEM_SEG_HEAD.NUM_CLASSES == 40:
            filename = "youtubevis_2019-instance-id2label.json"

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


class OriginalVideoMask2FormerConfigToImageProcessorConverter:
    def __call__(self, original_config: object) -> VideoMask2FormerImageProcessor:
        model = original_config.MODEL
        model_input = original_config.INPUT

        return VideoMask2FormerImageProcessor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            size=model_input.MIN_SIZE_TEST,
            max_size=model_input.MAX_SIZE_TEST,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            ignore_index=model.SEM_SEG_HEAD.IGNORE_VALUE,
            size_divisibility=32,
        )


class OriginalVideoMask2FormerCheckpointToOursConverter:
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

    def replace_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = ""
        src_prefix: str = "sem_seg_head.predictor"

        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        logger.info(f"Replacing keys {pformat(renamed_keys)}")
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, video_mask2former: VideoMask2FormerModel) -> VideoMask2FormerModel:
        dst_state_dict = TrackedStateDict(video_mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_pixel_module(dst_state_dict, src_state_dict)
        self.replace_transformer_module(dst_state_dict, src_state_dict)

        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        logger.info("🙌 Done")

        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        video_mask2former.load_state_dict(state_dict)
        return video_mask2former

    def convert_segmentation_head(
        self, video_mask2former: VideoMask2FormerForVideoSegmentation
    ) -> VideoMask2FormerForVideoSegmentation:
        dst_state_dict = TrackedStateDict(video_mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_segmentation_module(dst_state_dict, src_state_dict)

        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        video_mask2former.load_state_dict(state_dict)

        return video_mask2former

    @staticmethod
    def using_dirs(
        checkpoints_dir: Path, config_dir: Path
    ) -> Iterator[Tuple[object, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")

        for checkpoint in checkpoints:
            logger.info(f"💪 Converting {checkpoint.stem}")
            # find associated config file
            # dataset_name e.g 'youtubevis'
            dataset_name = checkpoint.parents[2].stem

            # task type e.g 'instance-segmentation'
            segmentation_task = checkpoint.parents[1].stem
            # config file corresponding to checkpoint
            config_file_name = f"{checkpoint.parents[0].stem}.yaml"
            
            config: Path = config_dir / dataset_name / "swin" / config_file_name

            yield config, checkpoint


def test(
    original_model,
    our_model: VideoMask2FormerForVideoSegmentation,
    image_processor: VideoMask2FormerImageProcessor,
    tolerance: float,
):
    with torch.no_grad():
        original_model = original_model.eval()
        our_model = our_model.eval()

        image_size = (480, 640)
        file_path = hf_hub_download(repo_id="shivi/video-demo", filename="cars.mp4", repo_type="dataset")
        video = torchvision.io.read_video(file_path)[0]
        
        img_processor_output = image_processor(images=list(video[:5]), return_tensors="pt", do_resize=True, size=image_size).pixel_values

        original_model_backbone_features = original_model.backbone(img_processor_output.clone())
        our_model_output: VideoMask2FormerModelOutput = our_model.model(
            img_processor_output.clone(), output_hidden_states=True
        )

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
        
        original_model_input = image_processor(images=list(video[:5]),return_tensors="pt",
                                    do_resize=True,do_rescale=False,
                                    do_normalize=False,size=image_size).pixel_values

        # modify original Mask2Former code to return mask and class logits
        original_class_logits, original_mask_logits = original_model([{"image": original_model_input}])

        our_model_out: VideoMask2FormerForVideoSegmentationOutput = our_model(img_processor_output.clone())
        our_mask_logits = our_model_out.masks_queries_logits
        our_class_logits = our_model_out.class_queries_logits

        assert original_mask_logits.shape == our_mask_logits.shape, "Output masks shapes are not matching."
        assert original_class_logits.shape == our_class_logits.shape, "Output class logits shapes are not matching."
        assert torch.allclose(
            original_mask_logits, our_mask_logits, atol=tolerance
        ), "The predicted masks are not the same."
        assert torch.allclose(
            original_class_logits, our_class_logits, atol=tolerance
        ), "The class logits are not the same."
        logger.info("✅ Test passed!")


def get_model_and_dataset_name(checkpoint_file: Path):
    # model_name_raw is something like video_maskformer2_swin_small_bs16_50ep
    model_name_raw: str = checkpoint_file.parents[0].stem

    # `segmentation_task_type` must be only `instance-segmentation`
    segmentation_task_name: str = checkpoint_file.parents[1].stem
    if segmentation_task_name != "instance-segmentation":
        raise ValueError(
            f"{segmentation_task_name} must be wrong since 'instance-segmentation' is the only acceptable value."
        )

    dataset_name: str = checkpoint_file.parents[2].stem
    if dataset_name not in ["youtubevis_2019", "youtubevis_2021"]:
        raise ValueError(
            f"{dataset_name} must be wrong since we didn't find 'youtubevis_2019' or 'youtubevis_2021' in it"
        )
    dataset_name = dataset_name.replace("_", "-")

    backbone = "swin"
    backbone_types = ["tiny", "small", "base_IN21k", "large"]
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0].replace("_", "-")

    model_name = (
        f"video-mask2former-{backbone}-{backbone_type}-{dataset_name}-{segmentation_task_name.split('-')[0]}"
    )

    return model_name, dataset_name


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
    sys.path.append(str(mask2former_dir))

    # import original VideoMask2Former model from original source code repo

    from Mask2Former.mask2former_video.video_maskformer_model import VideoMaskFormer as OriginalVideoMask2Former

    for config_file, checkpoint_file in OriginalVideoMask2FormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):
        model_name, dataset_name = get_model_and_dataset_name(checkpoint_file)

        
        height_width = (480, 640)
        # load video mask2former config
        original_config = setup_video_cfg(Args(config_file=config_file))
        # load original video mask2former model
        video_mask2former_kwargs = OriginalVideoMask2Former.from_config(original_config)
        original_model = OriginalVideoMask2Former(**video_mask2former_kwargs).eval()

        image_processor = OriginalVideoMask2FormerConfigToImageProcessorConverter()(original_config)
        image_processor.size = {"height": height_width[0], "width": height_width[1]}

        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        config: Mask2FormerConfig = OriginalVideoMask2FormerConfigToOursConverter()(
            original_config, dataset_name
        )
        video_mask2former = VideoMask2FormerModel(config=config).eval()

        converter = OriginalVideoMask2FormerCheckpointToOursConverter(original_model, config)
        mask2former = converter.convert(video_mask2former)

        video_mask2former_for_segmentation = VideoMask2FormerForVideoSegmentation(config=config).eval()
        video_mask2former_for_segmentation.model = video_mask2former

        video_mask2former_for_segmentation = converter.convert_segmentation_head(video_mask2former_for_segmentation)

        tolerance = 3e-3

        logger.info(f"🪄 Testing {model_name}...")
        test(original_model, video_mask2former_for_segmentation, image_processor, tolerance)
        logger.info(f"🪄 Pushing {model_name} to hub...")

        image_processor.push_to_hub(model_name)
        video_mask2former_for_segmentation.push_to_hub(model_name)

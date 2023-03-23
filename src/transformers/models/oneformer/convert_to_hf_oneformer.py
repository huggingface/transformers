# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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

"""Convert OneFormer checkpoints from the original repository. URL: https://github.com/SHI-Labs/OneFormer"""

import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple

import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn


try:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config
except ImportError:
    pass
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
    OneFormerConfig,
    OneFormerForUniversalSegmentation,
    OneFormerForUniversalSegmentationOutput,
    OneFormerModel,
    OneFormerModelOutput,
)
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
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


# Image to verify the result
def prepare_img():
    url = "https://praeclarumjj3.github.io/files/coco.jpeg"
    img_data = requests.get(url, stream=True).raw
    im = Image.open(img_data)
    return im


@dataclass
class Args:
    """Fake command line arguments needed by oneformer/detectron2 implementation"""

    config_file: str


def setup_cfg(args: Args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_oneformer_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


class OriginalOneFormerConfigToOursConverter:
    def __call__(self, original_config: object, is_swin: bool) -> OneFormerConfig:
        model = original_config.MODEL

        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST_PANOPTIC[0])
        id2label = dict(enumerate(dataset_catalog.stuff_classes))
        label2id = {label: idx for idx, label in id2label.items()}

        if is_swin:
            if model.SWIN.EMBED_DIM == 96:
                backbone_config = SwinConfig.from_pretrained(
                    "microsoft/swin-tiny-patch4-window7-224",
                    drop_path_rate=model.SWIN.DROP_PATH_RATE,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )
            elif model.SWIN.EMBED_DIM == 192:
                backbone_config = SwinConfig.from_pretrained(
                    "microsoft/swin-large-patch4-window12-384",
                    drop_path_rate=model.SWIN.DROP_PATH_RATE,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                )
            else:
                raise ValueError(f"embed dim {model.SWIN.EMBED_DIM} not supported for Swin!")
        else:
            backbone_config = DinatConfig.from_pretrained(
                "shi-labs/dinat-large-11x11-in22k-in1k-384",
                dilations=model.DiNAT.DILATIONS,
                kernel_size=model.DiNAT.KERNEL_SIZE,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )

        config: OneFormerConfig = OneFormerConfig(
            backbone_config=backbone_config,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            ignore_value=model.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=model.SEM_SEG_HEAD.NUM_CLASSES,
            num_queries=model.ONE_FORMER.NUM_OBJECT_QUERIES,
            no_object_weight=model.ONE_FORMER.NO_OBJECT_WEIGHT,
            class_weight=model.ONE_FORMER.CLASS_WEIGHT,
            mask_weight=model.ONE_FORMER.MASK_WEIGHT,
            dice_weight=model.ONE_FORMER.DICE_WEIGHT,
            contrastive_weight=model.ONE_FORMER.CONTRASTIVE_WEIGHT,
            contrastive_temperature=model.ONE_FORMER.CONTRASTIVE_TEMPERATURE,
            train_num_points=model.ONE_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=model.ONE_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=model.ONE_FORMER.IMPORTANCE_SAMPLE_RATIO,
            init_std=0.02,
            init_xavier_std=1.0,
            layer_norm_eps=1e-05,
            is_training=False,
            use_auxiliary_loss=model.ONE_FORMER.DEEP_SUPERVISION,
            output_auxiliary_logits=True,
            strides=[4, 8, 16, 32],
            task_seq_len=original_config.INPUT.TASK_SEQ_LEN,
            max_seq_len=original_config.INPUT.MAX_SEQ_LEN,
            text_encoder_width=model.TEXT_ENCODER.WIDTH,
            text_encoder_context_length=model.TEXT_ENCODER.CONTEXT_LENGTH,
            text_encoder_num_layers=model.TEXT_ENCODER.NUM_LAYERS,
            text_encoder_vocab_size=model.TEXT_ENCODER.VOCAB_SIZE,
            text_encoder_proj_layers=model.TEXT_ENCODER.PROJ_NUM_LAYERS,
            text_encoder_n_ctx=model.TEXT_ENCODER.N_CTX,
            conv_dim=model.SEM_SEG_HEAD.CONVS_DIM,
            mask_dim=model.SEM_SEG_HEAD.MASK_DIM,
            hidden_dim=model.ONE_FORMER.HIDDEN_DIM,
            norm=model.SEM_SEG_HEAD.NORM,
            encoder_layers=model.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            encoder_feedforward_dim=1024,
            decoder_layers=model.ONE_FORMER.DEC_LAYERS,
            use_task_norm=model.ONE_FORMER.USE_TASK_NORM,
            num_attention_heads=model.ONE_FORMER.NHEADS,
            dropout=model.ONE_FORMER.DROPOUT,
            dim_feedforward=model.ONE_FORMER.DIM_FEEDFORWARD,
            pre_norm=model.ONE_FORMER.PRE_NORM,
            enforce_input_proj=model.ONE_FORMER.ENFORCE_INPUT_PROJ,
            query_dec_layers=model.ONE_FORMER.CLASS_DEC_LAYERS,
            common_stride=model.SEM_SEG_HEAD.COMMON_STRIDE,
            id2label=id2label,
            label2id=label2id,
        )

        return config


class OriginalOneFormerConfigToProcessorConverter:
    def __call__(self, original_config: object, model_repo: str) -> OneFormerProcessor:
        model = original_config.MODEL
        model_input = original_config.INPUT
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST_PANOPTIC[0])

        if "ade20k" in model_repo:
            class_info_file = "ade20k_panoptic.json"
        elif "coco" in model_repo:
            class_info_file = "coco_panoptic.json"
        elif "cityscapes" in model_repo:
            class_info_file = "cityscapes_panoptic.json"
        else:
            raise ValueError("Invalid Dataset!")

        image_processor = OneFormerImageProcessor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            size=model_input.MIN_SIZE_TEST,
            max_size=model_input.MAX_SIZE_TEST,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            ignore_index=dataset_catalog.ignore_label,
            class_info_file=class_info_file,
        )

        tokenizer = CLIPTokenizer.from_pretrained(model_repo)

        return OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            task_seq_length=original_config.INPUT.TASK_SEQ_LEN,
            max_seq_length=original_config.INPUT.MAX_SEQ_LEN,
        )


class OriginalOneFormerCheckpointToOursConverter:
    def __init__(self, original_model: nn.Module, config: OneFormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    # Swin Backbone
    def replace_swin_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: OneFormerConfig):
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
        num_layers = len(config.backbone_config.depths)
        for layer_idx in range(num_layers):
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

            if layer_idx < num_layers - 1:
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

    # Dinat Backbone
    def replace_dinat_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: OneFormerConfig):
        dst_prefix: str = "pixel_level_module.encoder"
        src_prefix: str = "backbone"

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        renamed_keys = rename_keys_for_weight_bias(f"{src_prefix}.patch_embed.norm", f"{dst_prefix}.embeddings.norm")

        for i in range(2):
            renamed_keys.extend(
                rename_keys_for_weight_bias(
                    f"{src_prefix}.patch_embed.proj.{i}",
                    f"{dst_prefix}.embeddings.patch_embeddings.projection.{i}",
                )
            )

        num_layers = len(config.backbone_config.depths)
        for layer_idx in range(num_layers):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend(
                    rename_keys_for_weight_bias(
                        f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.norm1",
                        f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.layernorm_before",
                    )
                )

                renamed_keys.extend(
                    rename_keys_for_weight_bias(
                        f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.norm2",
                        f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.layernorm_after",
                    )
                )

                renamed_keys.extend(
                    [  # src, dst
                        (
                            f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.rpb",
                            f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.rpb",
                        ),
                    ]
                )
                # now we need to handle the attentions
                # read in weights + bias of input projection layer of cross-attention

                src_att_weight = src_state_dict[f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.weight"]
                src_att_bias = src_state_dict[f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.bias"]

                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[
                    f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.query.weight"
                ] = src_att_weight[:offset, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.query.bias"
                ] = src_att_bias[:offset]

                dst_state_dict[
                    f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.key.weight"
                ] = src_att_weight[offset : offset * 2, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.key.bias"
                ] = src_att_bias[offset : offset * 2]

                dst_state_dict[
                    f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.value.weight"
                ] = src_att_weight[-offset:, :]
                dst_state_dict[
                    f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.value.bias"
                ] = src_att_bias[-offset:]

                # let's pop them
                src_state_dict.pop(f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.weight")
                src_state_dict.pop(f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.bias")
                # proj

                renamed_keys.extend(
                    rename_keys_for_weight_bias(
                        f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.proj",
                        f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.output.dense",
                    )
                )

                # mlp
                renamed_keys.extend(
                    rename_keys_for_weight_bias(
                        f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.mlp.fc1",
                        f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.intermediate.dense",
                    )
                )

                renamed_keys.extend(
                    rename_keys_for_weight_bias(
                        f"{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.mlp.fc2",
                        f"{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.output.dense",
                    )
                )

            if layer_idx < num_layers - 1:
                # patch merging
                renamed_keys.extend(
                    [
                        (
                            f"{src_prefix}.levels.{layer_idx}.downsample.reduction.weight",
                            f"{dst_prefix}.encoder.levels.{layer_idx}.downsample.reduction.weight",
                        ),
                        (
                            f"{src_prefix}.levels.{layer_idx}.downsample.norm.weight",
                            f"{dst_prefix}.encoder.levels.{layer_idx}.downsample.norm.weight",
                        ),
                        (
                            f"{src_prefix}.levels.{layer_idx}.downsample.norm.bias",
                            f"{dst_prefix}.encoder.levels.{layer_idx}.downsample.norm.bias",
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
    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict, is_swin: bool):
        dst_prefix: str = "pixel_level_module.decoder"
        src_prefix: str = "sem_seg_head.pixel_decoder"

        if is_swin:
            self.replace_swin_backbone(dst_state_dict, src_state_dict, self.config)
        else:
            self.replace_dinat_backbone(dst_state_dict, src_state_dict, self.config)

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
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module"
        src_prefix: str = "sem_seg_head.predictor"

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
            attn_keys = [
                (f"{src_prefix}.in_proj_bias", f"{dst_prefix}.in_proj_bias"),
                (f"{src_prefix}.in_proj_weight", f"{dst_prefix}.in_proj_weight"),
            ]
            attn_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.out_proj", f"{dst_prefix}.out_proj"))

            return attn_keys

        def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
            attn_keys = []
            attn_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.out_proj", f"{dst_prefix}.out_proj"))

            return attn_keys

        def rename_keys_for_query_transformer_layer(src_prefix: str, dst_prefix: str):
            query_transformer_layer_keys = []

            query_transformer_layer_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.linear1", f"{dst_prefix}.linear1")
            )
            query_transformer_layer_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.linear2", f"{dst_prefix}.linear2")
            )
            query_transformer_layer_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.norm1", f"{dst_prefix}.norm1")
            )
            query_transformer_layer_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.norm2", f"{dst_prefix}.norm2")
            )
            query_transformer_layer_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.norm3", f"{dst_prefix}.norm3")
            )

            query_transformer_layer_keys.extend(
                rename_keys_for_attn(f"{src_prefix}.self_attn", f"{dst_prefix}.self_attn")
            )

            query_transformer_layer_keys.extend(
                rename_keys_for_attn(f"{src_prefix}.multihead_attn", f"{dst_prefix}.multihead_attn")
            )

            return query_transformer_layer_keys

        def rename_keys_for_cross_attn_layer(src_prefix: str, dst_prefix: str):
            cross_attn_layer_keys = []

            cross_attn_layer_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.norm", f"{dst_prefix}.norm"))
            cross_attn_layer_keys.extend(
                rename_keys_for_attn(f"{src_prefix}.multihead_attn", f"{dst_prefix}.multihead_attn")
            )

            return cross_attn_layer_keys

        def rename_keys_for_self_attn_layer(src_prefix: str, dst_prefix: str):
            self_attn_layer_keys = []

            self_attn_layer_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.norm", f"{dst_prefix}.norm"))
            self_attn_layer_keys.extend(
                rename_keys_for_self_attn(f"{src_prefix}.self_attn", f"{dst_prefix}.self_attn")
            )

            return self_attn_layer_keys

        def rename_keys_for_ffn_layer(src_prefix: str, dst_prefix: str):
            ffn_layer_keys = []

            ffn_layer_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.linear1", f"{dst_prefix}.linear1"))
            ffn_layer_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.linear2", f"{dst_prefix}.linear2"))
            ffn_layer_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.norm", f"{dst_prefix}.norm"))

            return ffn_layer_keys

        def rename_keys_for_transformer_decoder_layer(src_prefix: str, dst_prefix: str, idx: int):
            transformer_decoder_layer_keys = []

            transformer_decoder_layer_keys.extend(
                rename_keys_for_cross_attn_layer(
                    f"{src_prefix}.transformer_cross_attention_layers.{idx}", f"{dst_prefix}.{idx}.cross_attn"
                )
            )

            transformer_decoder_layer_keys.extend(
                rename_keys_for_self_attn_layer(
                    f"{src_prefix}.transformer_self_attention_layers.{idx}", f"{dst_prefix}.{idx}.self_attn"
                )
            )

            transformer_decoder_layer_keys.extend(
                rename_keys_for_ffn_layer(f"{src_prefix}.transformer_ffn_layers.{idx}", f"{dst_prefix}.{idx}.ffn")
            )

            return transformer_decoder_layer_keys

        # positional embedding for object queries
        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.level_embed.weight", f"{dst_prefix}.level_embed.weight"),
        ]

        # norm
        renamed_keys.extend(
            rename_keys_for_weight_bias(f"{src_prefix}.decoder_norm", f"{dst_prefix}.decoder.decoder_norm")
        )

        # proj
        renamed_keys.extend(
            rename_keys_for_weight_bias(
                f"{src_prefix}.class_input_proj", f"{dst_prefix}.decoder.query_input_projection"
            )
        )

        renamed_keys.extend(
            rename_keys_for_weight_bias(f"{src_prefix}.class_embed", f"{dst_prefix}.decoder.class_embed")
        )

        for i in range(3):
            renamed_keys.extend(
                rename_keys_for_weight_bias(
                    f"{src_prefix}.mask_embed.layers.{i}", f"{dst_prefix}.decoder.mask_embed.layers.{i}.0"
                )
            )

        # norm
        renamed_keys.extend(
            rename_keys_for_weight_bias(
                f"{src_prefix}.class_transformer.decoder.norm", f"{dst_prefix}.decoder.query_transformer.decoder.norm"
            )
        )

        # transformer to update queries with task tokens
        for i in range(self.config.query_dec_layers):
            renamed_keys.extend(
                rename_keys_for_query_transformer_layer(
                    f"{src_prefix}.class_transformer.decoder.layers.{i}",
                    f"{dst_prefix}.decoder.query_transformer.decoder.layers.{i}",
                )
            )

        # decoder layers
        for i in range(self.config.decoder_layers - 1):
            renamed_keys.extend(
                rename_keys_for_transformer_decoder_layer(
                    f"{src_prefix}",
                    f"{dst_prefix}.decoder.layers",
                    i,
                )
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)

    def replace_task_mlp(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "task_encoder"
        src_prefix: str = "task_mlp"

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        renamed_keys = []

        for i in range(2):
            renamed_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.layers.{i}", f"{dst_prefix}.task_mlp.layers.{i}.0")
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_text_projector(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "text_mapper.text_projector"
        src_prefix: str = "text_projector"

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        renamed_keys = []

        for i in range(self.config.text_encoder_config["text_encoder_proj_layers"]):
            renamed_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.layers.{i}", f"{dst_prefix}.{i}.0"))

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_text_mapper(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "text_mapper.text_encoder"
        src_prefix: str = "text_encoder"

        self.replace_text_projector(dst_state_dict, src_state_dict)

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
            attn_keys = [
                (f"{src_prefix}.in_proj_bias", f"{dst_prefix}.in_proj_bias"),
                (f"{src_prefix}.in_proj_weight", f"{dst_prefix}.in_proj_weight"),
            ]
            attn_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.out_proj", f"{dst_prefix}.out_proj"))

            return attn_keys

        def rename_keys_for_layer(src_prefix: str, dst_prefix: str):
            resblock_keys = []

            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.mlp.c_fc", f"{dst_prefix}.mlp.fc1"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.mlp.c_proj", f"{dst_prefix}.mlp.fc2"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_1", f"{dst_prefix}.layer_norm1"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_2", f"{dst_prefix}.layer_norm2"))
            resblock_keys.extend(rename_keys_for_attn(f"{src_prefix}.attn", f"{dst_prefix}.self_attn"))

            return resblock_keys

        renamed_keys = [
            ("prompt_ctx.weight", "text_mapper.prompt_ctx.weight"),
        ]

        renamed_keys.extend(
            [
                (f"{src_prefix}.positional_embedding", f"{dst_prefix}.positional_embedding"),
                (f"{src_prefix}.token_embedding.weight", f"{dst_prefix}.token_embedding.weight"),
            ]
        )

        renamed_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_final", f"{dst_prefix}.ln_final"))

        for i in range(self.config.text_encoder_config["text_encoder_num_layers"]):
            renamed_keys.extend(
                rename_keys_for_layer(
                    f"{src_prefix}.transformer.resblocks.{i}", f"{dst_prefix}.transformer.layers.{i}"
                )
            )

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, oneformer: OneFormerModel, is_swin: bool) -> OneFormerModel:
        dst_state_dict = TrackedStateDict(oneformer.state_dict())
        src_state_dict = self.original_model.state_dict()

        self.replace_pixel_module(dst_state_dict, src_state_dict, is_swin)
        self.replace_transformer_module(dst_state_dict, src_state_dict)
        self.replace_task_mlp(dst_state_dict, src_state_dict)
        if self.config.is_training:
            self.replace_text_mapper(dst_state_dict, src_state_dict)

        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        logger.info("🙌 Done")

        oneformer.load_state_dict(dst_state_dict)

        return oneformer

    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pth")

        for checkpoint in checkpoints:
            logger.info(f"💪 Converting {checkpoint.stem}")
            # find associated config file
            config: Path = config_dir / f"{checkpoint.stem}.yaml"

            yield config, checkpoint


def post_process_sem_seg_output(outputs: OneFormerForUniversalSegmentationOutput, target_size: Tuple[int, int]):
    # class_queries_logits has shape [BATCH, QUERIES, CLASSES + 1]
    class_queries_logits = outputs.class_queries_logits
    # masks_queries_logits has shape [BATCH, QUERIES, HEIGHT, WIDTH]
    masks_queries_logits = outputs.masks_queries_logits
    if target_size is not None:
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    # remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    # mask probs has shape [BATCH, QUERIES, HEIGHT, WIDTH]
    masks_probs = masks_queries_logits.sigmoid()
    # now we want to sum over the queries,
    # $ out_{c,h,w} =  \sum_q p_{q,c} * m_{q,h,w} $
    # where $ softmax(p) \in R^{q, c} $ is the mask classes
    # and $ sigmoid(m) \in R^{q, h, w}$ is the mask probabilities
    # b(atch)q(uery)c(lasses), b(atch)q(uery)h(eight)w(idth)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

    return segmentation


def test(
    original_model,
    our_model: OneFormerForUniversalSegmentation,
    processor: OneFormerProcessor,
    model_repo: str,
):
    def _preprocess_text(text_list=None, max_length=77):
        if text_list is None:
            raise ValueError("tokens cannot be None.")

        tokens = tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True)

        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        for attn_mask, input_id in zip(attention_masks, input_ids):
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs

    with torch.no_grad():
        tokenizer = CLIPTokenizer.from_pretrained(model_repo)
        original_model = original_model.eval()
        our_model = our_model.eval()

        im = prepare_img()

        tr = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
                T.Normalize(
                    mean=torch.tensor([123.675, 116.280, 103.530]) / 255.0,
                    std=torch.tensor([58.395, 57.120, 57.375]) / 255.0,
                ),
            ],
        )

        x = tr(im).unsqueeze(0)

        task_input = ["the task is semantic"]
        task_token = _preprocess_text(task_input, max_length=processor.task_seq_length)

        original_model_backbone_features = original_model.backbone(x.clone())

        our_model_output: OneFormerModelOutput = our_model.model(x.clone(), task_token, output_hidden_states=True)

        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=3e-3
            ), "The backbone features are not the same."
        mask_features, _, multi_scale_features, _, _ = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )

        original_pixel_decoder_features = []
        original_pixel_decoder_features.append(mask_features)
        for i in range(len(multi_scale_features)):
            original_pixel_decoder_features.append(multi_scale_features[i])

        for original_model_feature, our_model_feature in zip(
            original_pixel_decoder_features, our_model_output.pixel_decoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=3e-4
            ), "The pixel decoder feature are not the same"

        tr_complete = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ],
        )

        y = (tr_complete(im) * 255.0).to(torch.int).float()

        # let's test the full model
        original_model_out = original_model([{"image": y.clone(), "task": "The task is semantic"}])

        original_segmentation = original_model_out[0]["sem_seg"]

        our_model_out: OneFormerForUniversalSegmentationOutput = our_model(
            x.clone(), task_token, output_hidden_states=True
        )

        our_segmentation = post_process_sem_seg_output(our_model_out, target_size=(640, 640))[0]

        assert torch.allclose(
            original_segmentation, our_segmentation, atol=1e-3
        ), "The segmentation image is not the same."

        logger.info("✅ Test passed!")


def get_name(checkpoint_file: Path):
    model_name_raw: str = checkpoint_file.stem

    backbone = "swin" if "swin" in model_name_raw else "dinat"
    dataset = ""
    if "coco" in model_name_raw:
        dataset = "coco"
    elif "ade20k" in model_name_raw:
        dataset = "ade20k"
    elif "cityscapes" in model_name_raw:
        dataset = "cityscapes"
    else:
        raise ValueError(
            f"{model_name_raw} must be wrong since we didn't find 'coco' or 'ade20k' or 'cityscapes' in it "
        )

    backbone_types = ["tiny", "large"]

    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0]

    model_name = f"oneformer_{dataset}_{backbone}_{backbone_type}"

    return model_name


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Command line to convert the original oneformer models (with swin backbone) to transformers"
            " implementation."
        )
    )

    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.pth; where <CONFIG_NAME> name must follow the"
            " following nomenclature nomenclature: oneformer_<DATASET_NAME>_<BACKBONE>_<BACKBONE_TYPE>"
        ),
    )
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.yaml; where <CONFIG_NAME> name must follow the"
            " following nomenclature nomenclature: oneformer_<DATASET_NAME>_<BACKBONE>_<BACKBONE_TYPE>"
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=Path,
        help="Path to the folder to output PyTorch models.",
    )
    parser.add_argument(
        "--oneformer_dir",
        required=True,
        type=Path,
        help=(
            "A path to OneFormer's original implementation directory. You can download from here:"
            "https://github.com/SHI-Labs/OneFormer"
        ),
    )

    args = parser.parse_args()

    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    save_directory: Path = args.pytorch_dump_folder_path
    oneformer_dir: Path = args.oneformer_dir
    # append the path to the parents to oneformer dir
    sys.path.append(str(oneformer_dir.parent))
    # and import what's needed
    from OneFormer.oneformer import add_common_config, add_dinat_config, add_oneformer_config, add_swin_config
    from OneFormer.oneformer.oneformer_model import OneFormer as OriginalOneFormer

    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    for config_file, checkpoint_file in OriginalOneFormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):
        processor = OriginalOneFormerConfigToProcessorConverter()(
            setup_cfg(Args(config_file=config_file)), os.path.join("shi-labs", config_file.stem)
        )

        original_config = setup_cfg(Args(config_file=config_file))
        oneformer_kwargs = OriginalOneFormer.from_config(original_config)

        original_model = OriginalOneFormer(**oneformer_kwargs).eval()

        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        is_swin = "swin" in config_file.stem

        config: OneFormerConfig = OriginalOneFormerConfigToOursConverter()(original_config, is_swin)

        oneformer = OneFormerModel(config=config).eval()

        converter = OriginalOneFormerCheckpointToOursConverter(original_model, config)

        oneformer = converter.convert(oneformer, is_swin)

        oneformer_for_universal_segmentation = OneFormerForUniversalSegmentation(config=config).eval()

        oneformer_for_universal_segmentation.model = oneformer

        test(
            original_model,
            oneformer_for_universal_segmentation,
            processor,
            os.path.join("shi-labs", config_file.stem),
        )

        model_name = get_name(checkpoint_file)
        logger.info(f"🪄 Saving {model_name}")

        processor.save_pretrained(save_directory / model_name)
        oneformer_for_universal_segmentation.save_pretrained(save_directory / model_name)

        processor.push_to_hub(
            repo_id=os.path.join("shi-labs", config_file.stem),
            commit_message="Add configs",
            use_temp_dir=True,
        )
        oneformer_for_universal_segmentation.push_to_hub(
            repo_id=os.path.join("shi-labs", config_file.stem),
            commit_message="Add model",
            use_temp_dir=True,
        )

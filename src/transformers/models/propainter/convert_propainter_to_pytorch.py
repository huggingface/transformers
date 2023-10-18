# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert ViT and non-distilled DeiT checkpoints from the timm library."""


import argparse
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from PIL import Image

# from transformers import DeiTImageProcessor, ViTConfig, ViTForImageClassification, ProPainterConfig, ViTModel
from transformers.models.propainter.configuration_propainter import ProPainterConfig
from transformers.models.propainter.image_processing_propainter import ProPainterImageProcessor
from transformers.models.propainter.modeling_propainter import ProPainterForImageInPainting
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class depths:
    def __init__(self):
        self.optical_layers = 4
        self.optical_hidden_layers = 3
        self.encoder_depth = 8
        self.mid_dilation_depth = 6
        self.convoffset_depth = 7
        self.backbone_depth = 3
        self.fuse_depth = 3
        self.propainter_encoder_depth = 18
        self.transformer_depth = 8


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    ## OPTICAL FLOW
    # rename_keys.append((f"module.fnet.conv2.weight", f"OpticalFlow.FeatureNet.Norm1.weight"))
    # rename_keys.append((f"module.fnet.conv2.bias", f"OpticalFlow.FeatureNet.Norm1.bias"))
    rename_keys.append(("module.fnet.conv1.weight", "OpticalFlow.FeatureNet.Conv1.weight"))
    rename_keys.append(("module.fnet.conv1.bias", "OpticalFlow.FeatureNet.Conv1.bias"))
    rename_keys.append(("module.fnet.conv2.weight", "OpticalFlow.FeatureNet.Conv2.weight"))
    rename_keys.append(("module.fnet.conv2.bias", "OpticalFlow.FeatureNet.Conv2.bias"))
    for j in range(config.optical_layers):
        for i in range(config.optical_hidden_layers):
            for x in range(2):
                rename_keys.append(
                    (
                        f"module.fnet.layer{j}.{i}.downsample.{x}.weight",
                        f"OpticalFlow.FeatureNet.layer{j}.{i}.Downsample.{x}.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"module.fnet.layer{j}.{i}.downsample.{x}.bias",
                        f"OpticalFlow.FeatureNet.layer{j}.{i}.Downsample.{x}.bias",
                    )
                )

            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.conv1.weight", f"OpticalFlow.FeatureNet.layer{j}.{i}.Conv1.weight")
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.conv1.bias", f"OpticalFlow.FeatureNet.layer{j}.{i}.Conv1.bias")
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.conv2.weight", f"OpticalFlow.FeatureNet.layer{j}.{i}.Conv2.weight")
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.conv2.bias", f"OpticalFlow.FeatureNet.layer{j}.{i}.Conv2.bias")
            )

            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.norm1.weight", f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm1.weight")
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.norm1.bias", f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm1.bias")
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm1.running_mean",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm1.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm1.running_var",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm1.running_var",
                )
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm1.num_batched_tracked",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm1.num_batched_tracked",
                )
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.norm2.weight", f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm2.weight")
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.norm2.bias", f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm2.bias")
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm2.running_mean",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm2.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm2.running_var",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm2.running_var",
                )
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm2.num_batches_tracked",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm2.num_batches_tracked",
                )
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.norm3.weight", f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm3.weight")
            )
            rename_keys.append(
                (f"module.fnet.layer{j}.{i}.norm3.bias", f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm3.bias")
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm3.running_mean",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm3.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm3.running_var",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm3.running_var",
                )
            )
            rename_keys.append(
                (
                    f"module.fnet.layer{j}.{i}.norm3.num_batches_tracked",
                    f"OpticalFlow.FeatureNet.layer{j}.{i}.Norm3.num_batches_tracked",
                )
            )

    rename_keys.append(("module.cnet.norm1.weight", "OpticalFlow.ContextNet.Norm1.weight"))
    rename_keys.append(("module.cnet.norm1.bias", "OpticalFlow.ContextNet.Norm1.bias"))
    rename_keys.append(("module.cnet.norm1.running_mean", "OpticalFlow.ContextNet.Norm1.running_mean"))
    rename_keys.append(("module.cnet.norm1.running_var", "OpticalFlow.ContextNet.Norm1.running_var"))
    rename_keys.append(("module.cnet.norm1.num_batches_tracked", "OpticalFlow.ContextNet.Norm1.num_batches_tracked"))

    rename_keys.append(("module.cnet.conv1.weight", "OpticalFlow.ContextNet.Conv1.weight"))
    rename_keys.append(("module.cnet.conv1.bias", "OpticalFlow.ContextNet.Conv1.bias"))
    rename_keys.append(("module.cnet.conv2.weight", "OpticalFlow.ContextNet.Conv2.weight"))
    rename_keys.append(("module.cnet.conv2.bias", "OpticalFlow.ContextNet.Conv2.bias"))
    for j in range(config.optical_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        for i in range(config.optical_hidden_layers):
            for x in range(2):
                rename_keys.append(
                    (
                        f"module.cnet.layer{j}.{i}.downsample.{x}.weight",
                        f"OpticalFlow.ContextNet.layer{j}.{i}.Downsample.{x}.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"module.cnet.layer{j}.{i}.downsample.{x}.bias",
                        f"OpticalFlow.ContextNet.layer{j}.{i}.Downsample.{x}.bias",
                    )
                )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.downsample.1.running_mean",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Downsample.{x}.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.downsample.1.running_var",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Downsample.{x}.running_var",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.downsample.1.num_batches_tracked",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Downsample.{x}.num_batches_tracked",
                )
            )

            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.conv1.weight", f"OpticalFlow.ContextNet.layer{j}.{i}.Conv1.weight")
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.conv1.bias", f"OpticalFlow.ContextNet.layer{j}.{i}.Conv1.bias")
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.conv2.weight", f"OpticalFlow.ContextNet.layer{j}.{i}.Conv2.weight")
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.conv2.bias", f"OpticalFlow.ContextNet.layer{j}.{i}.Conv2.bias")
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.norm1.weight", f"OpticalFlow.ContextNet.layer{j}.{i}.Norm1.weight")
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.norm1.bias", f"OpticalFlow.ContextNet.layer{j}.{i}.Norm1.bias")
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm1.running_mean",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm1.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm1.running_var",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm1.running_var",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm1.num_batches_tracked",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm1.num_batches_tracked",
                )
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.norm2.weight", f"OpticalFlow.ContextNet.layer{j}.{i}.Norm2.weight")
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.norm2.bias", f"OpticalFlow.ContextNet.layer{j}.{i}.Norm2.bias")
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm2.running_mean",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm2.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm2.running_var",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm2.running_var",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm2.num_batches_tracked",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm2.num_batches_tracked",
                )
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.norm3.weight", f"OpticalFlow.ContextNet.layer{j}.{i}.Norm3.weight")
            )
            rename_keys.append(
                (f"module.cnet.layer{j}.{i}.norm3.bias", f"OpticalFlow.ContextNet.layer{j}.{i}.Norm3.bias")
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm3.running_mean",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm3.running_mean",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm3.running_var",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm3.running_var",
                )
            )
            rename_keys.append(
                (
                    f"module.cnet.layer{j}.{i}.norm3.num_batches_tracked",
                    f"OpticalFlow.ContextNet.layer{j}.{i}.Norm3.num_batches_tracked",
                )
            )

    for block_old, block in zip(
        ["convc1", "convc2", "convf1", "convf2", "conv"], ["Conv_c1", "Conv_c2", "Conv_f1", "Conv_f2", "Conv_"]
    ):
        rename_keys.append(
            (f"module.update_block.encoder.{block_old}.weight", f"OpticalFlow.UpdateBlock.Encoder.{block}.weight")
        )
        rename_keys.append(
            (f"module.update_block.encoder.{block_old}.bias", f"OpticalFlow.UpdateBlock.Encoder.{block}.bias")
        )

    for block_old, block in zip(
        ["convz1", "convr1", "convq1", "convz2", "convr2", "convq2"],
        ["Conv_z1", "Conv_r1", "Conv_q1", "Conv_z2", "Conv_r2", "Conv_q2"],
    ):
        rename_keys.append(
            (f"module.update_block.gru.{block_old}.weight", f"OpticalFlow.UpdateBlock.GRU.{block}.weight")
        )
        rename_keys.append((f"module.update_block.gru.{block_old}.bias", f"OpticalFlow.UpdateBlock.GRU.{block}.bias"))

    for block_old, block in zip(["conv1", "conv2"], ["Conv1", "Conv2"]):
        rename_keys.append(
            (f"module.update_block.flow_head.{block_old}.weight", f"OpticalFlow.UpdateBlock.FlowHead.{block}.weight")
        )
        rename_keys.append(
            (f"module.update_block.flow_head.{block_old}.bias", f"OpticalFlow.UpdateBlock.FlowHead.{block}.bias")
        )

    rename_keys.append(("module.update_block.mask.0.weight", "OpticalFlow.UpdateBlock.Mask.0.weight"))
    rename_keys.append(("module.update_block.mask.0.bias", "OpticalFlow.UpdateBlock.Mask.0.bias"))
    rename_keys.append(("module.update_block.mask.2.weight", "OpticalFlow.UpdateBlock.Mask.2.weight"))
    rename_keys.append(("module.update_block.mask.2.bias", "OpticalFlow.UpdateBlock.Mask.2.bias"))

    ## RecurentFlowCompleteNet
    rename_keys.append(("downsample.0.weight", "FlowComplete.Downsample.0.weight"))
    rename_keys.append(("downsample.0.bias", "FlowComplete.Downsample.0.bias"))

    for enc in range(1, 3):
        for i in range(0, config.encoder_depth, 2):  # 7
            rename_keys.append((f"encoder{enc}.{i}.conv1.0.weight", f"FlowComplete.Encoder{enc}.{i}.Conv1.0.weight"))
            rename_keys.append((f"encoder{enc}.{i}.conv1.0.bias", f"FlowComplete.Encoder{enc}.{i}.Conv1.0.bias"))
            rename_keys.append((f"encoder{enc}.{i}.conv2.0.weight", f"FlowComplete.Encoder{enc}.{i}.Conv2.0.weight"))
            rename_keys.append((f"encoder{enc}.{i}.conv2.0.bias", f"FlowComplete.Encoder{enc}.{i}.Conv2.0.bias"))

    for i in range(0, config.mid_dilation_depth, 2):  # 5
        rename_keys.append((f"mid_dilation.{i}.weight", f"FlowComplete.MidDilation.{i}.weight"))
        rename_keys.append((f"mid_dilation.{i}.bias", f"FlowComplete.MidDilation.{i}.bias"))

    for block_old, block in zip(["backward_", "forward_"], ["backward_1", "forward_1"]):
        rename_keys.append(
            (
                f"feat_prop_module.deform_align.{block_old}.weight",
                f"FlowComplete.feat_prop_module.deform_align.{block_old}.weight",
            )
        )
        rename_keys.append(
            (
                f"feat_prop_module.deform_align.{block_old}.bias",
                f"FlowComplete.feat_prop_module.deform_align.{block_old}.bias",
            )
        )

        rename_keys.append(
            (
                f"feat_prop_module.deform_align.{block}.weight",
                f"InPainting.feat_prop_module.deform_align.{block}.weight",
            )
        )
        rename_keys.append(
            (f"feat_prop_module.deform_align.{block}.bias", f"InPainting.feat_prop_module.deform_align.{block}.bias")
        )
        for i in range(0, config.convoffset_depth, 2):  # 6
            rename_keys.append(
                (
                    f"feat_prop_module.deform_align.{block_old}.conv_offset.{i}.weight",
                    f"FlowComplete.feat_prop_module.deform_align.{block_old}.ConvOffset.{i}.weight",
                )
            )
            rename_keys.append(
                (
                    f"feat_prop_module.deform_align.{block_old}.conv_offset.{i}.bias",
                    f"FlowComplete.feat_prop_module.deform_align.{block_old}.ConvOffset.{i}.bias",
                )
            )

            rename_keys.append(
                (
                    f"feat_prop_module.deform_align.{block}.conv_offset.{i}.weight",
                    f"InPainting.feat_prop_module.deform_align.{block}.ConvOffset.{i}.weight",
                )
            )
            rename_keys.append(
                (
                    f"feat_prop_module.deform_align.{block}.conv_offset.{i}.bias",
                    f"InPainting.feat_prop_module.deform_align.{block}.ConvOffset.{i}.bias",
                )
            )

        for i in range(0, config.backbone_depth, 2):  # 2
            rename_keys.append(
                (
                    f"feat_prop_module.backbone.{block_old}.{i}.weight",
                    f"FlowComplete.feat_prop_module.backbone.{block_old}.{i}.weight",
                )
            )
            rename_keys.append(
                (
                    f"feat_prop_module.backbone.{block_old}.{i}.bias",
                    f"FlowComplete.feat_prop_module.backbone.{block_old}.{i}.bias",
                )
            )

            rename_keys.append(
                (
                    f"feat_prop_module.backbone.{block}.{i}.weight",
                    f"InPainting.feat_prop_module.backbone.{block}.{i}.weight",
                )
            )
            rename_keys.append(
                (
                    f"feat_prop_module.backbone.{block}.{i}.bias",
                    f"InPainting.feat_prop_module.backbone.{block}.{i}.bias",
                )
            )

    rename_keys.append(("feat_prop_module.fusion.weight", "FlowComplete.feat_prop_module.fusion.weight"))
    rename_keys.append(("feat_prop_module.fusion.bias", "FlowComplete.feat_prop_module.fusion.bias"))

    rename_keys.append(("feat_prop_module.fuse.0.weight", "InPainting.feat_prop_module.fuse.0.weight"))
    rename_keys.append(("feat_prop_module.fuse.0.bias", "InPainting.feat_prop_module.fuse.0.bias"))
    rename_keys.append(("feat_prop_module.fuse.2.weight", "InPainting.feat_prop_module.fuse.2.weight"))
    rename_keys.append(("feat_prop_module.fuse.2.bias", "InPainting.feat_prop_module.fuse.2.bias"))

    for block_old, block in zip(["decoder1", "decoder2", "upsample"], ["Decoder1", "Decoder2", "Upsample"]):
        rename_keys.append((f"{block_old}.0.weight", f"FlowComplete.{block}.0.weight"))
        rename_keys.append((f"{block_old}.0.bias", f"FlowComplete.{block}.0.bias"))
        rename_keys.append((f"{block_old}.2.conv.weight", f"FlowComplete.{block}.2.Conv.weight"))
        rename_keys.append((f"{block_old}.2.conv.bias", f"FlowComplete.{block}.2.Conv.bias"))

    for block_old, block in zip(
        ["projection", "mid_layer_1", "mid_layer_2"], ["edge_projection", "edge_layer_1", "edge_layer_2"]
    ):
        rename_keys.append((f"edgeDetector.{block_old}.0.weight", f"FlowComplete.EdgeDetector.{block}.0.weight"))
        rename_keys.append((f"edgeDetector.{block_old}.0.bias", f"FlowComplete.EdgeDetector.{block}.0.bias"))

    rename_keys.append(("edgeDetector.out_layer.weight", "FlowComplete.EdgeDetector.edge_out.weight"))
    rename_keys.append(("edgeDetector.out_layer.bias", "FlowComplete.EdgeDetector.edge_out.bias"))

    ##ProPainter
    for i in range(0, config.propainter_encoder_depth, 2):  # 17
        rename_keys.append((f"encoder.layers.{i}.weight", f"InPainting.encoder.Layers.{i}.weight"))
        rename_keys.append((f"encoder.layers.{i}.bias", f"InPainting.encoder.Layers.{i}.bias"))
        rename_keys.append((f"encoder.layers.{i}.conv1.weight", f"InPainting.encoder.Layers.{i}.Conv1.weight"))
        rename_keys.append((f"encoder.layers.{i}.conv1.bias", f"InPainting.encoder.Layers.{i}.Conv1.bias"))
        rename_keys.append((f"encoder.layers.{i}.conv2.weight", f"InPainting.encoder.Layers.{i}.Conv2.weight"))
        rename_keys.append((f"encoder.layers.{i}.conv2.bias", f"InPainting.encoder.Layers.{i}.Conv2.bias"))

    rename_keys.append(("decoder.0.conv.weight", "InPainting.decoder.0.Conv.weight"))
    rename_keys.append(("decoder.0.conv.bias", "InPainting.decoder.0.Conv.bias"))
    rename_keys.append(("decoder.4.conv.weight", "InPainting.decoder.4.Conv.weight"))
    rename_keys.append(("decoder.4.conv.bias", "InPainting.decoder.4.Conv.bias"))
    for i in [2, 6]:  # 6
        rename_keys.append((f"decoder.{i}.weight", f"InPainting.decoder.{i}.weight"))
        rename_keys.append((f"decoder.{i}.bias", f"InPainting.decoder.{i}.bias"))

    for block in ["sc", "ss"]:
        rename_keys.append((f"{block}.embedding.weight", f"InPainting.{block}.embedding.weight"))
        rename_keys.append((f"{block}.embedding.bias", f"InPainting.{block}.embedding.bias"))

    rename_keys.append(("sc.bias_conv.weight", "InPainting.sc.bias_conv.weight"))
    rename_keys.append(("sc.bias_conv.bias", "InPainting.sc.bias_conv.bias"))

    for i in range(config.transformer_depth):  # 7
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.key.weight",
                f"InPainting.transformers.transformer.{i}.attention.key.weight",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.key.bias",
                f"InPainting.transformers.transformer.{i}.attention.key.bias",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.query.weight",
                f"InPainting.transformers.transformer.{i}.attention.query.weight",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.query.bias",
                f"InPainting.transformers.transformer.{i}.attention.query.bias",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.value.weight",
                f"InPainting.transformers.transformer.{i}.attention.value.weight",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.value.bias",
                f"InPainting.transformers.transformer.{i}.attention.value.bias",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.proj.weight",
                f"InPainting.transformers.transformer.{i}.attention.proj.weight",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.proj.bias",
                f"InPainting.transformers.transformer.{i}.attention.proj.bias",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.pool_layer.weight",
                f"InPainting.transformers.transformer.{i}.attention.pool_layer.weight",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.pool_layer.bias",
                f"InPainting.transformers.transformer.{i}.attention.pool_layer.bias",
            )
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.attention.valid_ind_rolled",
                f"InPainting.transformers.transformer.{i}.attention.valid_ind_rolled",
            )
        )

        rename_keys.append(
            (f"transformers.transformer.{i}.norm1.weight", f"InPainting.transformers.transformer.{i}.norm1.weight")
        )
        rename_keys.append(
            (f"transformers.transformer.{i}.norm1.bias", f"InPainting.transformers.transformer.{i}.norm1.bias")
        )
        rename_keys.append(
            (f"transformers.transformer.{i}.norm2.weight", f"InPainting.transformers.transformer.{i}.norm2.weight")
        )
        rename_keys.append(
            (f"transformers.transformer.{i}.norm2.bias", f"InPainting.transformers.transformer.{i}.norm2.bias")
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.mlp.fc1.0.weight",
                f"InPainting.transformers.transformer.{i}.mlp.fc1.0.weight",
            )
        )
        rename_keys.append(
            (f"transformers.transformer.{i}.mlp.fc1.0.bias", f"InPainting.transformers.transformer.{i}.mlp.fc1.0.bias")
        )
        rename_keys.append(
            (
                f"transformers.transformer.{i}.mlp.fc2.1.weight",
                f"InPainting.transformers.transformer.{i}.mlp.fc2.1.weight",
            )
        )
        rename_keys.append(
            (f"transformers.transformer.{i}.mlp.fc2.1.bias", f"InPainting.transformers.transformer.{i}.mlp.fc2.1.bias")
        )

    rename_keys.append(
        (f"transformers.transformer.{i}.mlp.fc2.1.weight", f"InPainting.transformers.transformer.{i}.mlp.fc2.1.weight")
    )
    rename_keys.append(
        (f"transformers.transformer.{i}.mlp.fc2.1.bias", f"InPainting.transformers.transformer.{i}.mlp.fc2.1.bias")
    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(state_dict, old, new):
    if old in state_dict:
        val = state_dict.pop(old)
        if "InPainting" in new:
            new = new.replace("InPainting.","")
        state_dict["model."+new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_propainter_checkpoint(
    checkpoint_path_optical_flow, checkpoint_path_recurrent_flow, checkpoint_path_pro_painter, pytorch_dump_folder_path
):
    """
    Copy/paste/tweak model's weights to our ProPainter structure.
    """

    # define default ViT configuration
    config = ProPainterConfig()

    # load original models

    state_dict = OrderedDict()
    state_dict.update(torch.load(checkpoint_path_optical_flow, map_location="cpu"))
    state_dict.update(torch.load(checkpoint_path_recurrent_flow, map_location="cpu"))
    state_dict.update(torch.load(checkpoint_path_pro_painter, map_location="cpu"))

    model = ProPainterForImageInPainting(config)
    model.eval()

    for i in state_dict:
        print(i)

    # load state_dict of original model, remove and rename some keys
    rename_keys = create_rename_keys(depths())
    print(rename_keys)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    model.load_state_dict(state_dict)
    print(model)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    image_processor = ProPainterImageProcessor(size=config.image_size)
    # encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # pixel_values = encoding["pixel_values"]
    # outputs = model(pixel_values)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path_optical_flow",
        default="./raft-things.pth",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_path_recurrent_flow",
        default="./recurrent_flow_completion.pth",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_path_pro_painter",
        default="./ProPainter.pth",
        type=str,
        help="Path to the original state dict (.pth file).",
    )

    parser.add_argument(
        "--pytorch_dump_folder_path", default="./", type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_propainter_checkpoint(
        args.checkpoint_path_optical_flow,
        args.checkpoint_path_recurrent_flow,
        args.checkpoint_path_pro_painter,
        args.pytorch_dump_folder_path,
    )

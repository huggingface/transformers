# coding=utf-8
# Copyright 2024 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ProPainter model."""

from collections import namedtuple
import itertools
import math
import numpy as np
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torchvision

from functools import reduce
from torch import nn
from torch.nn import L1Loss
from torch.nn.modules.utils import _pair, _single
from torch.cuda.amp import autocast
from torch.nn.functional import normalize
from torchvision import models as tv



from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithNoAttention,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel,TORCH_INIT_FUNCTIONS
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_propainter import ProPainterConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ProPainterConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "ruffy369/propainter"
_EXPECTED_OUTPUT_SHAPE = [80, 240, 432, 3]

class ProPainterResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, norm_fn='group', stride=1):
        super(ProPainterResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = channels // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(channels)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(channels)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride), self.norm3)


    def forward(self, hidden_states, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        residual = hidden_states
        residual = self.relu(self.norm1(self.conv1(residual)))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (residual.half(),)
        residual = self.relu(self.norm2(self.conv2(residual)))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (residual.half(),)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.relu(hidden_states+residual)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterBasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(ProPainterBasicEncoder, self).__init__()

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        in_channels = (64,64,96)
        channels = (64,96,128)
        strides = (1,2,2)

        self.resblocks = [[ProPainterResidualBlock(in_channel, num_channels, norm_fn, stride),ProPainterResidualBlock(num_channels, num_channels, norm_fn, stride=1)] for in_channel,num_channels,stride in zip(in_channels,channels,strides)]
        #using itertools makes flattening a little faster :)
        self.resblocks = nn.ModuleList(list(itertools.chain.from_iterable(self.resblocks))) 

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        #TODO: CHECKKK FOR THIS WEIGHT INITTTTTTTTTTTTTTTTT
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image,output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # if input is list, combine batch dimension
        is_list = isinstance(image, tuple) or isinstance(image, list)
        if is_list:
            batch_dim = image[0].shape[0]
            image = torch.cat(image, dim=0)

        hidden_states = self.conv1(image)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = self.norm1(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = self.relu1(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        for resblock in self.resblocks:
            resblock_output = resblock(hidden_states, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
            hidden_states, _all_hidden_states = resblock_output[0], resblock_output[1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (_all_hidden_states,)

        hidden_states = self.conv2(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if self.training and self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        if is_list:
            hidden_states = torch.split(hidden_states, [batch_dim, batch_dim], dim=0)

        if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterBasicMotionEncoder(nn.Module):
    def __init__(self, config):
        super(ProPainterBasicMotionEncoder, self).__init__()
        corr_planes = config.corr_levels * (2*config.corr_radius + 1)**2
        self.conv_corr1 = nn.Conv2d(corr_planes, 256, 1, padding=0)
        self.conv_corr2 = nn.Conv2d(256, 192, 3, padding=1)
        self.conv_flow1 = nn.Conv2d(2, 128, 7, padding=3)
        self.conv_flow2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        hidden_states_corr = F.relu(self.conv_corr1(corr))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_corr.half(),)
        hidden_states_corr = F.relu(self.conv_corr2(hidden_states_corr))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_corr.half(),)
        hidden_states_flow = F.relu(self.conv_flow1(flow))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_flow.half(),)
        hidden_states_flow = F.relu(self.conv_flow2(hidden_states_flow))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_flow.half(),)

        hidden_states = torch.cat([hidden_states_corr, hidden_states_flow], dim=1)
        hidden_states = F.relu(self.conv(hidden_states))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = torch.cat([hidden_states, flow], dim=1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ProPainterSepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, hidden_states, motion_features, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # horizontal
        hidden_states_motion_features = torch.cat([hidden_states, motion_features], dim=1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_motion_features.half(),)
        z = torch.sigmoid(self.convz1(hidden_states_motion_features))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (z.half(),)
        r = torch.sigmoid(self.convr1(hidden_states_motion_features))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (r.half(),)
        q = torch.tanh(self.convq1(torch.cat([r*hidden_states, motion_features], dim=1)))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (q.half(),)        
        hidden_states = (1-z) * hidden_states + z * q
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        # vertical
        hidden_states_motion_features = torch.cat([hidden_states, motion_features], dim=1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_motion_features.half(),)
        z = torch.sigmoid(self.convz2(hidden_states_motion_features))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (z.half(),)
        r = torch.sigmoid(self.convr2(hidden_states_motion_features))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (r.half(),)
        q = torch.tanh(self.convq2(torch.cat([r*hidden_states, motion_features], dim=1)))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (q.half(),)    
        hidden_states = (1-z) * hidden_states + z * q
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterFlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(ProPainterFlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hidden_states, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        hidden_states = self.relu(self.conv1(hidden_states))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = self.conv2(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterBasicUpdateBlock(nn.Module):
    def __init__(self, config, hidden_dim=128, input_dim=128):
        super(ProPainterBasicUpdateBlock, self).__init__()
        self.config = config
        self.encoder = ProPainterBasicMotionEncoder(config)
        self.gru = ProPainterSepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = ProPainterFlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        encoder_outputs = self.encoder(flow, corr, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        motion_features, _all_hidden_states = encoder_outputs[0], encoder_outputs[1]
        inp = torch.cat([inp, motion_features], dim=1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (inp.half(),)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)

        gru_output = self.gru(net, inp, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        net, _all_hidden_states  = gru_output[0], gru_output[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        flow_head_outputs = self.flow_head(net, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        delta_flow, _all_hidden_states = flow_head_outputs[0], flow_head_outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (mask.half(),)
        return net, mask, delta_flow, all_hidden_states


def coords_grid(batch_size, height, width):
    coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)


def sample_point(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates """
    height, width = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(width-1) - 1
    ygrid = 2*ygrid/(height-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img


class ProPainterCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = ProPainterCorrBlock.corr(fmap1, fmap2)

        batch_size, height1, width1, dimension, height2, width2 = corr.shape
        corr = corr.reshape(batch_size*height1*width1, dimension, height2, width2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        radius = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch_size, height1, width1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-radius, radius, 2*radius+1)
            dy = torch.linspace(-radius, radius, 2*radius+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch_size*height1*width1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*radius+1, 2*radius+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = sample_point(corr, coords_lvl)
            corr = corr.view(batch_size, height1, width1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch_size, dimension, height, width = fmap1.shape
        fmap1 = fmap1.view(batch_size, dimension, height*width)
        fmap2 = fmap2.view(batch_size, dimension, height*width)

        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch_size, height, width, 1, height, width)
        return corr  / torch.sqrt(torch.tensor(dimension).float())


class ProPainterRaftOpticalFlow(nn.Module):
    def __init__(self, config):
        super(ProPainterRaftOpticalFlow, self).__init__()
        self.config = config
        self.hidden_dim = 128
        self.context_dim = 128
        
        self.feature_network = ProPainterBasicEncoder(output_dim=256, norm_fn='instance', dropout=self.config.dropout)
        self.context_network = ProPainterBasicEncoder(output_dim=self.hidden_dim+self.context_dim, norm_fn='batch', dropout=self.config.dropout)
        self.update_block = ProPainterBasicUpdateBlock(self.config, hidden_dim=self.hidden_dim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, image):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, height, width = image.shape
        coords0 = coords_grid(N, height//8, width//8).to(image.device)
        coords1 = coords_grid(N, height//8, width//8).to(image.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [height/8, width/8, 2] -> [height, width, 2] using convex combination """
        N, _, height, width = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, height, width)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, height, width)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*height, 8*width)

    def _forward(self, image1, image2, iters=12, flow_init=None,output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        """ Estimate optical flow between pair of frames """
        all_hidden_states = () if output_hidden_states else None

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with autocast(enabled=False):
            feature_network_output = self.feature_network([image1, image2], output_attentions,output_hidden_states,return_dict)
            fmaps, _all_hidden_states = feature_network_output[0], feature_network_output[1]
            fmap1, fmap2 = fmaps
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (_all_hidden_states,)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        
        corr_fn = ProPainterCorrBlock(fmap1, fmap2, radius=self.config.corr_radius)

        # run the context network
        with autocast(enabled=False):
            context_network_output = self.context_network(image1, output_attentions,output_hidden_states,return_dict)
            context_network_out, _all_hidden_states = context_network_output[0], context_network_output[1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (_all_hidden_states,)
            net, inp = torch.split(context_network_out, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=False):
                net, up_mask, delta_flow, _all_hidden_states = self.update_block(net, inp, corr, flow, output_attentions,output_hidden_states,return_dict)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (_all_hidden_states,)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                new_size = (8 * (coords1 - coords0).shape[2], 8 * (coords1 - coords0).shape[3])
                flow_up =   8 * F.interpolate((coords1 - coords0), size=new_size, mode='bilinear', align_corners=True)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)


        return coords1 - coords0, flow_up, all_hidden_states

    def forward(self, gt_local_frames, iters=20,output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size, temporal_length, num_channels, height, width = gt_local_frames.size()

        gt_local_frames_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, num_channels, height, width)
        gt_local_frames_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, num_channels, height, width)

        _, gt_flows_forward, _all_hidden_states = self._forward(gt_local_frames_1, gt_local_frames_2, iters,output_attentions,output_hidden_states,return_dict)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        _, gt_flows_backward, _all_hidden_states = self._forward(gt_local_frames_2, gt_local_frames_1, iters,output_attentions,output_hidden_states,return_dict)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)

        
        gt_flows_forward = gt_flows_forward.view(batch_size, temporal_length-1, 2, height, width)
        gt_flows_backward = gt_flows_backward.view(batch_size, temporal_length-1, 2, height, width)

        if not return_dict:
            return tuple(v for v in [(gt_flows_forward, gt_flows_backward), all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=(gt_flows_forward, gt_flows_backward),
            hidden_states=all_hidden_states,
        )

class ProPainterP3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_residual=0, bias=True):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size),
                                    stride=(1, stride, stride), padding=(0, padding, padding), bias=bias),
                        nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                    padding=(2, 0, 0), dilation=(2, 1, 1), bias=bias)
        )
        self.use_residual = use_residual

    def forward(self, hidden_states):
        features1 = self.conv1(hidden_states)
        features2 = self.conv2(features1)
        if self.use_residual:
            hidden_states = hidden_states + features2
        else:
            hidden_states = features2
        return hidden_states


class ProPainterEdgeDetection(nn.Module):
    def __init__(self, in_channel=2, out_channel=1, intermediate_channel=16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channel, intermediate_channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.intermediate_layer_1 = nn.Sequential(
            nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.intermediate_layer_2 = nn.Sequential(
            nn.Conv2d(intermediate_channel, intermediate_channel, 3, 1, 1)
        )        

        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self.out_layer = nn.Conv2d(intermediate_channel, out_channel, 1, 1, 0)

    def forward(self, flow, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        flow = self.projection(flow)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (flow.half(),)
        edge = self.intermediate_layer_1(flow)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (edge.half(),)
        edge = self.intermediate_layer_2(edge)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (edge.half(),)
        edge = self.relu(flow + edge)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (edge.half(),)
        edge = self.out_layer(edge)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (edge.half(),)
        edge = torch.sigmoid(edge)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (edge.half(),)

        if not return_dict:
            return tuple(v for v in [edge, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=edge,
            hidden_states=all_hidden_states,
        )

class ProPainterBidirectionalPropagationFlowComplete(nn.Module):
    def __init__(self, num_channels):
        super(ProPainterBidirectionalPropagationFlowComplete, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.num_channels = num_channels

        for i, module in enumerate(modules):
            self.deform_align[module] = ProPainterSecondOrderDeformableAlignment(
                2 * num_channels, num_channels, 3, padding=1, deform_groups=16)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * num_channels, num_channels, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * num_channels, num_channels, 1, 1, 0)

    def forward(self, hidden_states, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        """
        hidden_states shape : [batch_size, timesteps, num_channels, height, width]
        return [batch_size, timesteps, num_channels, height, width]
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size, timesteps, _, height, width = hidden_states.shape
        features = {}
        features['spatial'] = [hidden_states[:, i, :, :, :] for i in range(0, timesteps)]

        for module_name in ['backward_', 'forward_']:

            features[module_name] = []

            frame_idx = range(0, timesteps)
            mapping_idx = list(range(0, len(features['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]

            feature_propagation = hidden_states.new_zeros(batch_size, self.num_channels, height, width)
            for i, idx in enumerate(frame_idx):
                feat_current = features['spatial'][mapping_idx[idx]]
                if i > 0:
                    cond_n1 = feature_propagation

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feature_propagation)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:  # second-order features
                        feat_n2 = features[module_name][-2]
                        cond_n2 = feat_n2

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1) # condition information, cond(flow warped 1st/2nd feature)
                    feature_propagation = torch.cat([feature_propagation, feat_n2], dim=1) # two order feature_propagation -1 & -2
                    deform_align_outputs = self.deform_align[module_name](feature_propagation, cond, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
                    feature_propagation, _all_hidden_states = deform_align_outputs[0], deform_align_outputs[1]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (_all_hidden_states,)
                # fuse current features
                feat = [feat_current] + \
                    [features[k][idx] for k in features if k not in ['spatial', module_name]] \
                    + [feature_propagation]

                feat = torch.cat(feat, dim=1)
                # embed current features
                feature_propagation = feature_propagation + self.backbone[module_name](feat)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (feature_propagation.half(),)
                features[module_name].append(feature_propagation)

            # end for
            if 'backward' in module_name:
                features[module_name] = features[module_name][::-1]

        outputs = []
        for i in range(0, timesteps):
            align_feats = [features[k].pop(0) for k in features if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        hidden_states = torch.stack(outputs, dim=1) + hidden_states
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

def flow_warp(features,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        features (Tensor): Tensor with size (n, num_channels, height, width).
        flow (Tensor): Tensor with size (n, height, width, 2). The last dimension is
            a two-num_channels, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if features.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({features.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, height, width = features.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height, device=device), torch.arange(0, width, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(features)  # (width, height, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(width - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(height - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(features,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output

def fbConsistencyCheck(flow_forward, flow_backward, alpha1=0.01, alpha2=0.5):
    flow_backward_warped = flow_warp(flow_backward, flow_forward.permute(0, 2, 3, 1))
    flow_diff_forward = flow_forward + flow_backward_warped

    normalized_forward = torch.norm(flow_forward, p=2, dim=1, keepdim=True)**2 + torch.norm(flow_backward_warped, p=2, dim=1, keepdim=True)**2  # |wf| + |wb(wf(x))|
    occ_thresh_forward = alpha1 * normalized_forward + alpha2

    fb_valid_forward = (torch.norm(flow_diff_forward, p=2, dim=1, keepdim=True)**2 < occ_thresh_forward).to(flow_forward)
    return fb_valid_forward


class ProPainterBidirectionalPropagationInPaint(nn.Module):
    def __init__(self, num_channels, learnable=True):
        super(ProPainterBidirectionalPropagationInPaint, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.num_channels = num_channels
        self.propagation_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for _, module in enumerate(self.propagation_list):
                self.deform_align[module] = ProPainterDeformableAlignment(
                    num_channels, num_channels, 3, padding=1, deform_groups=16)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2*num_channels+2, num_channels, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(num_channels, num_channels, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                    nn.Conv2d(2*num_channels+2, num_channels, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(num_channels, num_channels, 3, 1, 1),
                ) 
            
    def binary_mask(self, mask, th=0.1):
        return torch.where(mask > th, 1, 0).to(mask)

    def forward(self, masked_frames, flows_forward, flows_backward, mask, interpolation='bilinear', output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        """
        masked_frames shape : [batch_size, timesteps, num_channels, height, width]
        return [batch_size, timesteps, num_channels, height, width]
        """
        all_hidden_states = () if output_hidden_states else None

        # For backward warping, pred_flows_forward for backward feature propagation, pred_flows_backward for forward feature propagation
        batch_size, timesteps, num_channels, height, width = masked_frames.shape
        features, masks = {}, {}
        features['input'] = [masked_frames[:, i, :, :, :] for i in range(0, timesteps)]
        masks['input'] = [mask[:, i, :, :, :] for i in range(0, timesteps)]

        propagation_list = ['backward_1', 'forward_1']
        cache_list = ['input'] +  propagation_list

        for p_i, module_name in enumerate(propagation_list):
            features[module_name] = []
            masks[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, timesteps)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, timesteps)
                flow_idx = range(-1, timesteps - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = features[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask, mask_current], dim=1)
                        deform_align_outputs = self.deform_align[module_name](feat_prop, cond, flow_prop, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
                        feat_prop, _all_hidden_states = deform_align_outputs[0], deform_align_outputs[1]
                        if output_hidden_states:
                            all_hidden_states = all_hidden_states + (_all_hidden_states,)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)

                        union_vaild_mask = self.binary_mask(mask_current*flow_vaild_mask*(1-mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + (1-union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = self.binary_mask(mask_current*(1-(flow_vaild_mask*(1-mask_prop_valid))))
                
                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (feat_prop.half(),)

                features[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if 'backward' in module_name:
                features[module_name] = features[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(features['backward_1'], dim=1).view(-1, num_channels, height, width)
        outputs_f = torch.stack(features['forward_1'], dim=1).view(-1, num_channels, height, width)

        if self.learnable:
            mask_in = mask.view(-1, 2, height, width)
            masks_f = None
            outputs = self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) + masked_frames.view(-1, num_channels, height, width)
        else:
            masks_f = torch.stack(masks['forward_1'], dim=1)
            outputs = outputs_f

        return outputs_b.view(batch_size, -1, num_channels, height, width), outputs_f.view(batch_size, -1, num_channels, height, width), \
               outputs.view(batch_size, -1, num_channels, height, width), masks_f, all_hidden_states

class ProPainterDeconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(hidden_states)

class ProPainterModulatedDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ProPainterModulatedDeformConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, hidden_states, offset, mask):
        pass

class ProPainterDeformableAlignment(ProPainterModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)

        super(ProPainterDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

    def forward(self, features_propagation, cond_features, flow, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        output = self.conv_offset(cond_features)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output.half(),)
        output1, output2, mask = torch.chunk(output, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((output1, output2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)
        hidden_states = torchvision.ops.deform_conv2d(features_propagation, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation, mask)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterSecondOrderDeformableAlignment(ProPainterModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 5)

        super(ProPainterSecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

    def forward(self, features, extra_features, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        output = self.conv_offset(extra_features)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output.half(),)
        output1, output2, mask = torch.chunk(output, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((output1, output2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        hidden_states = torchvision.ops.deform_conv2d(features, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation, mask)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class ProPainterRecurrentFlowCompleteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(
                        nn.Conv3d(3, 32, kernel_size=(1, 5, 5), stride=(1, 2, 2), 
                                        padding=(0, 2, 2), padding_mode='replicate'),
                        nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder1 = nn.Sequential(
            ProPainterP3DBlock(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterP3DBlock(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 4x

        self.encoder2 = nn.Sequential(
            ProPainterP3DBlock(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterP3DBlock(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 8x

        self.intermediate_dilation = nn.Sequential(
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)), # p = d*(k-1)/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # feature propagation module
        self.feature_propagation_module = ProPainterBidirectionalPropagationFlowComplete(128)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(128, 64, 3, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 4x

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(64, 32, 3, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 2x

        self.upsample = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(32, 2, 3, 1)
        )

        # edge loss
        self.edgeDetector = ProPainterEdgeDetection(in_channel=2, out_channel=1, intermediate_channel=16)

    def forward(self, masked_flows, masks, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        batch_size, timesteps, _, height, width = masked_flows.size()
        masked_flows = masked_flows.permute(0,2,1,3,4)
        masks = masks.permute(0,2,1,3,4)

        inputs = torch.cat((masked_flows, masks), dim=1)
        
        downsample_inputs = self.downsample(inputs)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (downsample_inputs.half(),)

        features_enc1 = self.encoder1(downsample_inputs)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (features_enc1.half(),)
        features_enc2 = self.encoder2(features_enc1) # batch_size num_channels timesteps height width
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (features_enc2.half(),)
        features_intermediate = self.intermediate_dilation(features_enc2) # batch_size num_channels timesteps height width
        features_intermediate = features_intermediate.permute(0,2,1,3,4) # batch_size timesteps num_channels height width
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (features_intermediate.half(),)

        feature_propagation_module_outputs = self.feature_propagation_module(features_intermediate, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        features_prop, _all_hidden_states = feature_propagation_module_outputs[0], feature_propagation_module_outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        features_prop = features_prop.view(-1, 128, height//8, width//8) # batch_size*timesteps num_channels height width

        _, num_channels, _, h_f, w_f = features_enc1.shape
        features_enc1 = features_enc1.permute(0,2,1,3,4).contiguous().view(-1, num_channels, h_f, w_f) # batch_size*timesteps num_channels height width
        features_dec2 = self.decoder2(features_prop) + features_enc1
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (features_dec2.half(),)

        _, num_channels, _, h_f, w_f = downsample_inputs.shape
        downsample_inputs = downsample_inputs.permute(0,2,1,3,4).contiguous().view(-1, num_channels, h_f, w_f) # batch_size*timesteps num_channels height width

        features_dec1 = self.decoder1(features_dec2)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (features_dec1.half(),)

        flow = self.upsample(features_dec1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (flow.half(),)
        # if self.training:
        edge_detector_outputs = self.edgeDetector(flow)
        edge, _all_hidden_states = edge_detector_outputs[0], edge_detector_outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        edge = edge.view(batch_size, timesteps, 1, height, width)
        # else:
        #     edge = None

        flow = flow.view(batch_size, timesteps, 2, height, width)

        if not return_dict:
            return tuple(v for v in [(flow, edge), all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=(flow, edge),
            hidden_states=all_hidden_states,
        )        

    def forward_bidirect_flow(self, masked_flows_bi, masks, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        """
        Args:
            masked_flows_bi: [masked_flows_f, masked_flows_b] | (batch_size, timesteps-1, 2, height, width), (batch_size, timesteps-1, 2, height, width)
            masks: batch_size, timesteps, 1, height, width
        """
        all_hidden_states = () if output_hidden_states else None

        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        # mask flow
        masked_flows_forward = masked_flows_bi[0] * (1-masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1-masks_backward)
        
        # -- completion --
        outputs = self.forward(masked_flows_forward, masks_forward, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        pred_flows_forward, pred_edges_forward = outputs[0]
        _all_hidden_states = outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)

        # backward
        masked_flows_backward = torch.flip(masked_flows_backward, dims=[1])
        masks_backward = torch.flip(masks_backward, dims=[1])
        outputs = self.forward(masked_flows_backward, masks_backward, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        pred_flows_backward, pred_edges_backward = outputs[0]
        _all_hidden_states = outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        pred_flows_backward = torch.flip(pred_flows_backward, dims=[1])
        if self.training:
            pred_edges_backward = torch.flip(pred_edges_backward, dims=[1])

        if not return_dict:
            return tuple(v for v in [([pred_flows_forward, pred_flows_backward], [pred_edges_forward, pred_edges_backward]), all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=([pred_flows_forward, pred_flows_backward], [pred_edges_forward, pred_edges_backward]),
            hidden_states=all_hidden_states,
        )


    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1-masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1-masks_backward)

        return pred_flows_forward, pred_flows_backward


class ProPainterEncoder(nn.Module):
    def __init__(self):
        super(ProPainterEncoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        negative_slope = 0.2
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(negative_slope, inplace=True)
        ])

    def forward(self, masked_inputs, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None

        batch_size, _, _, _ = masked_inputs.size()
        features = masked_inputs
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = features
                _, _, height, width = x0.size()
            if i > 8 and i % 2 == 0:
                group = self.group[(i - 8) // 2]
                masked_inputs = x0.view(batch_size, group, -1, height, width)
                feature = features.view(batch_size, group, -1, height, width)
                features = torch.cat([masked_inputs, feature], 2).view(batch_size, -1, height, width)
            features = layer(features)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (features.half(),)

        if not return_dict:
            return tuple(v for v in [features, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=features,
            hidden_states=all_hidden_states,
        )

class ProPainterSoftSplit(nn.Module):
    def __init__(self, num_channels, hidden_size, kernel_size, stride, padding):
        super(ProPainterSoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        input_features = reduce((lambda x, y: x * y), kernel_size) * num_channels
        self.embedding = nn.Linear(input_features, hidden_size)

    def forward(self, hidden_states, batch_size, output_size, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        features_height = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        features_width = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = self.embedding(hidden_states)
        hidden_states = hidden_states.view(batch_size, -1, features_height, features_width, hidden_states.size(2))
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterSoftComp(nn.Module):
    def __init__(self, num_channels, hidden_size, kernel_size, stride, padding):
        super(ProPainterSoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        output_features = reduce((lambda x, y: x * y), kernel_size) * num_channels
        self.embedding = nn.Linear(hidden_size, output_features)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(num_channels,
                                   num_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, hidden_states, timestep, output_size, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None

        num_batch_, _, _, _, channel_ = hidden_states.shape
        hidden_states = hidden_states.view(num_batch_, -1, channel_)
        hidden_states = self.embedding(hidden_states)   
        batch_size, _, num_channels = hidden_states.size()
        hidden_states = hidden_states.view(batch_size * timestep, -1, num_channels).permute(0, 2, 1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = F.fold(hidden_states,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = self.bias_conv(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size, num_attention_heads):
    """
    Args:
        input_feature: shape is (batch_size, timesteps, height, width, num_channels)
        window_size (tuple[int]): window size
    Returns:
        windows: (batch_size, num_windows_h, num_windows_w, num_attention_heads, timesteps, window_size, window_size, num_channels//num_attention_heads)
    """
    batch_size, timesteps, height, width, num_channels= input_feature.shape
    input_feature = input_feature.view(
        batch_size, timesteps, height // window_size[0], window_size[0], width // window_size[1], window_size[1], num_attention_heads, num_channels//num_attention_heads)
    windows = input_feature.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows


class ProPainterSparseWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0., 
                pooling_token=True):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(hidden_size, hidden_size, qkv_bias)
        self.query = nn.Linear(hidden_size, hidden_size, qkv_bias)
        self.value = nn.Linear(hidden_size, hidden_size, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            kernel_size, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=(0, 0), groups=hidden_size)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))


    def forward(self, hidden_states, mask=None, token_indices=None, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_self_attentions = () if output_attentions else None

        batch_size, timesteps, height, width, num_channels = hidden_states.shape # 20 36
        window_height, window_width = self.window_size[0], self.window_size[1]
        channel_head = num_channels // self.num_attention_heads
        n_window_height = math.ceil(height / self.window_size[0])
        n_window_width = math.ceil(width / self.window_size[1])
        new_height = n_window_height * self.window_size[0] # 20
        new_width = n_window_width * self.window_size[1] # 36
        pad_r = new_width - width
        pad_b = new_height - height
        # reverse order
        if pad_r > 0 or pad_b > 0:
            hidden_states = F.pad(hidden_states,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 
            mask = F.pad(mask,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        window_query = window_partition(query.contiguous(), self.window_size, self.num_attention_heads).view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, window_height*window_width, channel_head)
        window_key = window_partition(key.contiguous(), self.window_size, self.num_attention_heads).view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, window_height*window_width, channel_head)
        window_value = window_partition(value.contiguous(), self.window_size, self.num_attention_heads).view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, window_height*window_width, channel_head)
        # roll_k and roll_v
        if any(i > 0 for i in self.expand_size):
            (key_top_left, value_top_left) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (key, value))
            (key_top_right, value_top_right) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (key, value))
            (key_bottom_left, value_bottom_left) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (key, value))
            (key_bottom_right, value_bottom_right) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (key, value))

            (key_top_left_windows, key_top_right_windows, key_bottom_left_windows, key_bottom_right_windows) = map(
                lambda a: window_partition(a, self.window_size, self.num_attention_heads).view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, window_height*window_width, channel_head), 
                (key_top_left, key_top_right, key_bottom_left, key_bottom_right))
            (value_top_left_windows, value_top_right_windows, value_bottom_left_windows, value_bottom_right_windows) = map(
                lambda a: window_partition(a, self.window_size, self.num_attention_heads).view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, window_height*window_width, channel_head), 
                (value_top_left, value_top_right, value_bottom_left, value_bottom_right))
            rool_key = torch.cat((key_top_left_windows, key_top_right_windows, key_bottom_left_windows, key_bottom_right_windows), 4).contiguous()
            rool_value = torch.cat((value_top_left_windows, value_top_right_windows, value_bottom_left_windows, value_bottom_right_windows), 4).contiguous() # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
            # mask output tokens in current window
            rool_key = rool_key[:, :, :, :, self.valid_ind_rolled]
            rool_value = rool_value[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_key.shape[4]
            rool_key = rool_key.view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, roll_N, num_channels // self.num_attention_heads)
            rool_value = rool_value.view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, roll_N, num_channels // self.num_attention_heads)
            window_key = torch.cat((window_key, rool_key), dim=4)
            window_value = torch.cat((window_value, rool_value), dim=4)
        else:
            window_key = window_key
            window_value = window_value
        
        # pool_k and pool_v
        if self.pooling_token:
            pool_x = self.pool_layer(hidden_states.view(batch_size*timesteps, new_height, new_width, num_channels).permute(0,3,1,2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0,2,3,1).view(batch_size, timesteps, p_h, p_w, num_channels)
            # pool_k
            pool_k = self.key(pool_x).unsqueeze(1).repeat(1, n_window_height*n_window_width, 1, 1, 1, 1) # [batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, num_channels]
            pool_k = pool_k.view(batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, self.num_attention_heads, channel_head).permute(0,1,5,2,3,4,6)
            pool_k = pool_k.contiguous().view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, p_h*p_w, channel_head)
            window_key = torch.cat((window_key, pool_k), dim=4)
            # pool_v
            pool_v = self.value(pool_x).unsqueeze(1).repeat(1, n_window_height*n_window_width, 1, 1, 1, 1) # [batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, num_channels]
            pool_v = pool_v.view(batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, self.num_attention_heads, channel_head).permute(0,1,5,2,3,4,6)
            pool_v = pool_v.contiguous().view(batch_size, n_window_height*n_window_width, self.num_attention_heads, timesteps, p_h*p_w, channel_head)
            window_value = torch.cat((window_value, pool_v), dim=4)

        # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
        output = torch.zeros_like(window_query)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(batch_size * l_t, new_height, new_width))
        mask = mask.view(batch_size, l_t, n_window_height*n_window_width)
        mask = torch.sum(mask, dim=1) # [batch_size, n_window_height*n_window_width]
        for i in range(window_query.shape[0]):
            ### For masked windows
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # mask output quary in current window
            # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                window_query_t = window_query[i, mask_ind_i].view(mask_n, self.num_attention_heads, timesteps*window_height*window_width, channel_head)
                window_key_t = window_key[i, mask_ind_i] 
                window_value_t = window_value[i, mask_ind_i] 
                # mask output key and value
                if token_indices is not None:
                    # key [n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
                    window_key_t = window_key_t[:, :, token_indices.view(-1)].view(mask_n, self.num_attention_heads, -1, channel_head)
                    # value
                    window_value_t = window_value_t[:, :, token_indices.view(-1)].view(mask_n, self.num_attention_heads, -1, channel_head)
                else:
                    window_key_t = window_key_t.view(n_window_height*n_window_width, self.num_attention_heads, timesteps*window_height*window_width, channel_head)
                    window_value_t = window_value_t.view(n_window_height*n_window_width, self.num_attention_heads, timesteps*window_height*window_width, channel_head)

                att_t = (window_query_t @ window_key_t.transpose(-2, -1)) * (1.0 / math.sqrt(window_query_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (att_t.half())
                y_t = att_t @ window_value_t 
                
                output[i, mask_ind_i] = y_t.view(-1, self.num_attention_heads, timesteps, window_height*window_width, channel_head)

            ### For unmasked windows
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask output quary in current window
            # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
            window_query_s = window_query[i, unmask_ind_i]
            window_key_s = window_key[i, unmask_ind_i, :, :, :window_height*window_width]
            window_value_s = window_value[i, unmask_ind_i, :, :, :window_height*window_width]

            att_s = (window_query_s @ window_key_s.transpose(-2, -1)) * (1.0 / math.sqrt(window_query_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            if output_attentions:
                all_self_attentions = all_self_attentions + (att_s.half())
            y_s = att_s @ window_value_s
            output[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        output = output.view(batch_size, n_window_height, n_window_width, self.num_attention_heads, timesteps, window_height, window_width, channel_head)
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(batch_size, timesteps, new_height, new_width, num_channels)


        if pad_r > 0 or pad_b > 0:
            output = output[:, :, :height, :width, :]

        # output projection
        output = self.proj_drop(self.proj(output))
        return output, all_self_attentions

class ProPainterFusionFeedForward(nn.Module):
    def __init__(self, hidden_size, hidden_dim=1960, token_to_token_params=None):
        super(ProPainterFusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_size))
        assert token_to_token_params is not None
        self.token_to_token_params = token_to_token_params
        self.kernel_shape = reduce((lambda x, y: x * y), token_to_token_params['kernel_size']) # 49

    def forward(self, hidden_states, output_size, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None

        num_vecs = 1
        for i, d in enumerate(self.token_to_token_params['kernel_size']):
            num_vecs *= int((output_size[i] + 2 * self.token_to_token_params['padding'][i] -
                           (d - 1) - 1) / self.token_to_token_params['stride'][i] + 1)

        hidden_states = self.fc1(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        batch_size, timestep, num_channel = hidden_states.size()
        normalizer = hidden_states.new_ones(batch_size, timestep, self.kernel_shape).view(-1, num_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.token_to_token_params['kernel_size'],
                            padding=self.token_to_token_params['padding'],
                            stride=self.token_to_token_params['stride'])

        hidden_states = F.fold(hidden_states.view(-1, num_vecs, num_channel).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.token_to_token_params['kernel_size'],
                   padding=self.token_to_token_params['padding'],
                   stride=self.token_to_token_params['stride'])
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = F.unfold(hidden_states / normalizer,
                     kernel_size=self.token_to_token_params['kernel_size'],
                     padding=self.token_to_token_params['padding'],
                     stride=self.token_to_token_params['stride']).permute(
                         0, 2, 1).contiguous().view(batch_size, timestep, num_channel)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)
        hidden_states = self.fc2(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.half(),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

class ProPainterTemporalSparseTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, pool_size,
                layer_norm=nn.LayerNorm, token_to_token_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = ProPainterSparseWindowAttention(hidden_size, num_attention_heads, window_size, pool_size)
        self.layer_norm1 = layer_norm(hidden_size)
        self.layer_norm2 = layer_norm(hidden_size)
        self.mlp = ProPainterFusionFeedForward(hidden_size, token_to_token_params=token_to_token_params)

    def forward(self, image_tokens, fold_x_size, mask=None, token_indices=None, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        """
        Args:
            image_tokens: shape [batch_size, timesteps, height, width, num_channels]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [batch_size, timesteps, height, width, 1]
        Returns:
            out_tokens: shape [batch_size, timesteps, height, width, 1]
        """
        all_hidden_states = () if output_hidden_states else None

        batch_size, timesteps, height, width, num_channels = image_tokens.shape # 20 36

        shortcut = image_tokens
        image_tokens = self.layer_norm1(image_tokens)
        att_x, all_self_attentions = self.attention(image_tokens, mask, token_indices, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)

        # FFN
        image_tokens = shortcut + att_x
        y = self.layer_norm2(image_tokens)
        mlp_outputs = self.mlp(y.view(batch_size, timesteps * height * width, num_channels), fold_x_size, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        hidden_states, _all_hidden_states = mlp_outputs[0], mlp_outputs[1]
        hidden_states = hidden_states.view(batch_size, timesteps, height, width, num_channels)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)

        image_tokens = image_tokens + hidden_states
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (image_tokens.half(),)

        if not return_dict:
            return tuple(v for v in [image_tokens, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=image_tokens,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

class ProPainterTemporalSparseTransformer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, window_size, pool_size, num_hidden_layers, token_to_token_params=None):
        super().__init__()
        blocks = []
        for _ in range(num_hidden_layers):
             blocks.append(
                ProPainterTemporalSparseTransformerBlock(hidden_size, num_attention_heads, window_size, pool_size, token_to_token_params=token_to_token_params)
             )
        self.transformer = nn.Sequential(*blocks)
        self.num_hidden_layers = num_hidden_layers

    def forward(self, image_tokens, fold_x_size, local_mask=None, t_dilation=2, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        """
        Args:
            image_tokens: shape [batch_size, timesteps, height, width, num_channels]
            fold_x_size: fold feature size, shape [60 108]
            local_mask: local mask tokens, shape [batch_size, timesteps, height, width, 1]
        Returns:
            out_tokens: shape [batch_size, timesteps, height, width, num_channels]
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        assert self.num_hidden_layers % t_dilation == 0, 'wrong t_dilation input.'
        timesteps = image_tokens.size(1)
        token_indices = [torch.arange(i, timesteps, t_dilation) for i in range(t_dilation)] * (self.num_hidden_layers // t_dilation)

        for i in range(0, self.num_hidden_layers):
            transformer_outputs = self.transformer[i](image_tokens, fold_x_size, local_mask, token_indices[i], output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
            image_tokens, _all_hidden_states, _all_self_attentions = transformer_outputs[0], transformer_outputs[1], transformer_outputs[2]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (_all_hidden_states,)
            if output_attentions:
                all_self_attentions = all_self_attentions + (_all_self_attentions,)

        if not return_dict:
            return tuple(v for v in [image_tokens, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=image_tokens,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

class ProPainterInpaintGenerator(nn.Module):
    def __init__(self,config):
        super(ProPainterInpaintGenerator, self).__init__()

        self.encoder = ProPainterEncoder()

        # decoder
        self.decoder = nn.Sequential(
            ProPainterDeconv(config.num_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ProPainterDeconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # soft split and soft composition
        token_to_token_params = {
            'kernel_size': config.kernel_size,
            'stride': config.stride,
            'padding': config.padding
        }
        self.soft_split = ProPainterSoftSplit(config.num_channels, config.hidden_size, config.kernel_size, config.stride, config.padding)
        self.soft_comp = ProPainterSoftComp(config.num_channels, config.hidden_size, config.kernel_size, config.stride, config.padding)
        self.max_pool = nn.MaxPool2d(config.kernel_size, config.stride, config.padding)

        # feature propagation module
        self.img_prop_module = ProPainterBidirectionalPropagationInPaint(3, learnable=False)
        self.feature_propagation_module = ProPainterBidirectionalPropagationInPaint(128, learnable=True)
        
        
        self.transformers = ProPainterTemporalSparseTransformer(hidden_size=config.hidden_size,
                                                num_attention_heads=config.num_attention_heads,
                                                window_size=config.window_size,
                                                pool_size=config.pool_size,
                                                num_hidden_layers=config.num_hidden_layers,
                                                token_to_token_params=token_to_token_params)

    def img_propagation(self, masked_frames, completed_flows, masks, interpolation='nearest', output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        _, _, prop_frames, updated_masks, all_hidden_states = self.img_prop_module(masked_frames, completed_flows[0], completed_flows[1], masks, interpolation, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)

        if not return_dict:
            return tuple(v for v in [(prop_frames, updated_masks), all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=(prop_frames, updated_masks),
            hidden_states=all_hidden_states,
        )

    def forward(self, masked_frames, completed_flows, masks_in, masks_updated, num_local_frames, interpolation='bilinear', t_dilation=2, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        local_timestep = num_local_frames
        batch_size, timestep, _, original_height, original_width = masked_frames.size()

        # extracting features
        encoder_outputs = self.encoder(torch.cat([masked_frames.view(batch_size * timestep, 3, original_height, original_width),
                                        masks_in.view(batch_size * timestep, 1, original_height, original_width),
                                        masks_updated.view(batch_size * timestep, 1, original_height, original_width)], dim=1), output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        encoder_hidden_states, _all_hidden_states = encoder_outputs[0], encoder_outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        _, num_channels, height, width = encoder_hidden_states.size()
        local_features = encoder_hidden_states.view(batch_size, timestep, num_channels, height, width)[:, :local_timestep, ...]
        ref_features = encoder_hidden_states.view(batch_size, timestep, num_channels, height, width)[:, local_timestep:, ...]
        fold_feat_size = (height, width)

        ds_flows_f = F.interpolate(completed_flows[0].view(-1, 2, original_height, original_width), scale_factor=1/4, mode='bilinear', align_corners=False).view(batch_size, local_timestep-1, 2, height, width)/4.0
        ds_flows_b = F.interpolate(completed_flows[1].view(-1, 2, original_height, original_width), scale_factor=1/4, mode='bilinear', align_corners=False).view(batch_size, local_timestep-1, 2, height, width)/4.0
        ds_mask_in = F.interpolate(masks_in.reshape(-1, 1, original_height, original_width), scale_factor=1/4, mode='nearest').view(batch_size, timestep, 1, height, width)
        ds_mask_in_local = ds_mask_in[:, :local_timestep]
        ds_mask_updated_local =  F.interpolate(masks_updated[:,:local_timestep].reshape(-1, 1, original_height, original_width), scale_factor=1/4, mode='nearest').view(batch_size, local_timestep, 1, height, width)


        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, height, width))
            mask_pool_l = mask_pool_l.view(batch_size, timestep, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, height, width))
            mask_pool_l = mask_pool_l.view(batch_size, local_timestep, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (mask_pool_l.half(),)

        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        _, _, local_features, _, _all_hidden_states = self.feature_propagation_module(local_features, ds_flows_f, ds_flows_b, prop_mask_in, interpolation, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        encoder_hidden_states = torch.cat((local_features, ref_features), dim=1)

        soft_split_outputs = self.soft_split(encoder_hidden_states.view(-1, num_channels, height, width), batch_size, fold_feat_size, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        trans_feat, _all_hidden_states = soft_split_outputs[0], soft_split_outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        mask_pool_l = mask_pool_l.permute(0,1,3,4,2).contiguous()
        transformers_outputs = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        trans_feat, _all_hidden_states, _all_self_attentions = transformers_outputs[0], transformers_outputs[1], transformers_outputs[2]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        if output_attentions:
                all_self_attentions = all_self_attentions + (_all_self_attentions,)
        soft_comp_outputs = self.soft_comp(trans_feat, timestep, fold_feat_size, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        trans_feat, _all_hidden_states = soft_comp_outputs[0], soft_comp_outputs[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hidden_states,)
        trans_feat = trans_feat.view(batch_size, timestep, -1, height, width)

        encoder_hidden_states = encoder_hidden_states + trans_feat

        if self.training:
            output = self.decoder(encoder_hidden_states.view(-1, num_channels, height, width))
            output = torch.tanh(output).view(batch_size, timestep, 3, original_height, original_width)
        else:
            output = self.decoder(encoder_hidden_states[:, :local_timestep].view(-1, num_channels, height, width))
            output = torch.tanh(output).view(batch_size, local_timestep, 3, original_height, original_width)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output.half(),)

        if not return_dict:
            return tuple(v for v in [output, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=output,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )


# ######################################################################
#  ProPainterDiscriminator for Temporal Patch GAN
# ######################################################################
class ProPainterSpectralNorm(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1

    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, name='weight', num_power_iterations=1, dimension=0, eps=1e-12):
        self.name = name
        self.dimension = dimension
        if num_power_iterations <= 0:
            raise ValueError(
                'Expected num_power_iterations to be positive, but '
                'got num_power_iterations={}'.format(num_power_iterations))
        self.num_power_iterations = num_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dimension != 0:
            # permute dimension to front
            weight_mat = weight_mat.permute(
                self.dimension,
                *[d for d in range(weight_mat.dim()) if d != self.dimension])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.num_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u),
                                  dim=0,
                                  eps=self.eps,
                                  out=v)
                    u = normalize(torch.mv(weight_mat, v),
                                  dim=0,
                                  eps=self.eps,
                                  out=u)
                if self.num_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma#🐺
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name,
                                  torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(
            module, self.name,
            self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(),
                               weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, num_power_iterations, dimension, eps):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, ProPainterSpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name))

        func = ProPainterSpectralNorm(name, num_power_iterations, dimension, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = func.reshape_weight_to_matrix(weight)

            height, width = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=func.eps)
            v = normalize(weight.new_empty(width).normal_(0, 1), dim=0, eps=func.eps)

        delattr(module, func.name)
        module.register_parameter(func.name + "_orig", weight)
        # We still need to assign weight back as func.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, func.name, weight.data)
        module.register_buffer(func.name + "_u", u)
        module.register_buffer(func.name + "_v", v)

        module.register_forward_pre_hook(func)

        module._register_state_dict_hook(ProPainterSpectralNormStateDictHook(func))
        module._register_load_state_dict_pre_hook(
            ProPainterSpectralNormLoadStateDictPreHook(func))
        return func

# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class ProPainterSpectralNormLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, func):
        self.func = func

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        func = self.func
        version = local_metadata.get('spectral_norm',
                                     {}).get(func.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + func.name + '_orig']
                weight_mat = func.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + func.name + '_u']


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class ProPainterSpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, func):
        self.func = func

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.func.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError(
                "Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.func._version


def spectral_norm(module,
                  name='weight',
                  num_power_iterations=1,
                  eps=1e-12,
                  dimension=None):
    r"""Applies spectral normalization to a parameter in the given module.

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        num_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dimension (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dimension is None:
        if isinstance(module,
                      (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
                       torch.nn.ConvTranspose3d)):
            dimension = 1
        else:
            dimension = 0
    ProPainterSpectralNorm.apply(module, name, num_power_iterations, dimension, eps)
    return module

class ProPainterDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 ):
        super(ProPainterDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        num_features = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=num_features * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 1,
                          num_features * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 2,
                          num_features * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 4,
                          num_features * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(num_features * 4,
                          num_features * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(num_features * 4,
                      num_features * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))


    def forward(self, completed_frames):
        completed_frames_t = torch.transpose(completed_frames, 1, 2)
        hidden_states = self.conv(completed_frames_t)
        if self.use_sigmoid:
            hidden_states = torch.sigmoid(hidden_states)
        hidden_states = torch.transpose(hidden_states, 1, 2)  # batch_size, timesteps, num_channels, height, width
        return hidden_states

#Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/pretrained_networks.py
class ProPainterVgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(ProPainterVgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, frames):
        hidden_states = self.slice1(frames)
        hidden_states_relu1_2 = hidden_states
        hidden_states = self.slice2(hidden_states)
        hidden_states_relu2_2 = hidden_states
        hidden_states = self.slice3(hidden_states)
        hidden_states_relu3_3 = hidden_states
        hidden_states = self.slice4(hidden_states)
        hidden_states_relu4_3 = hidden_states
        hidden_states = self.slice5(hidden_states)
        hidden_states_relu5_3 = hidden_states
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        hidden_states = vgg_outputs(hidden_states_relu1_2, hidden_states_relu2_2, hidden_states_relu3_3, hidden_states_relu4_3, hidden_states_relu5_3)

        return hidden_states

#Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
class ProPainterScalingLayer(nn.Module):
    def __init__(self):
        super(ProPainterScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, frames):
        return (frames - self.shift) / self.scale

#Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
class ProPainterIntermediateLossLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, num_channels, use_dropout=False):
        super(ProPainterIntermediateLossLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(num_channels, num_channels, 1, stride=1, padding=0, bias=False),]
        self.loss_layers = nn.Sequential(*layers)

    def forward(self, hidden_states):
        return self.loss_layers(hidden_states)

def spatial_average(input_tensor, keepdim=True):
    return input_tensor.mean([2,3],keepdim=keepdim)

def upsample(input_tensor, out_HW=(64,64)): # assumes scale factor is same for height and W
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(input_tensor)

def normalize_tensor(hidden_states,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(hidden_states**2,dim=1,keepdim=True))
    return hidden_states/(norm_factor+eps)

#Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# Learned perceptual metric
class ProPainterLpips(nn.Module):
    def __init__(self, use_dropout=True,):
        """ Initializes a perceptual loss torch.nn.Module
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(ProPainterLpips, self).__init__()
        
        self.scaling_layer = ProPainterScalingLayer()

        self.num_channels = [64,128,256,512,512]
        self.length = len(self.num_channels)

        self.net = ProPainterVgg16()

        
        self.layer0 = ProPainterIntermediateLossLayer(self.num_channels[0], use_dropout=use_dropout)
        self.layer1 = ProPainterIntermediateLossLayer(self.num_channels[1], use_dropout=use_dropout)
        self.layer2 = ProPainterIntermediateLossLayer(self.num_channels[2], use_dropout=use_dropout)
        self.layer3 = ProPainterIntermediateLossLayer(self.num_channels[3], use_dropout=use_dropout)
        self.layer4 = ProPainterIntermediateLossLayer(self.num_channels[4], use_dropout=use_dropout)
        self.layers = [self.layer0,self.layer1,self.layer2,self.layer3,self.layer4]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, frames, pred_images):
        frames = 2 * frames  - 1
        pred_images = 2 * pred_images  - 1

        frames, pred_images = (self.scaling_layer(frames), self.scaling_layer(pred_images))
        hidden_states0, hidden_states1 = self.net.forward(frames), self.net.forward(pred_images)
        feats0, feats1, diffs = {}, {}, {}

        for i in range(self.length):
            feats0[i], feats1[i] = normalize_tensor(hidden_states0[i]), normalize_tensor(hidden_states1[i])
            diffs[i] = (feats0[i]-feats1[i])**2

        layer_perceptual_losses = [spatial_average(self.layers[i](diffs[i]), keepdim=True) for i in range(self.length)]

        return sum(layer_perceptual_losses)

class ProPainterLpipsLoss(nn.Module):
    def __init__(self, 
            loss_weight=1.0, 
            use_input_norm=True,
            range_norm=False,):
        super(ProPainterLpipsLoss, self).__init__()
        self.perceptual = ProPainterLpips().eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_images, frames):
        if self.range_norm:
            pred_images   = (pred_images + 1) / 2
            frames = (frames + 1) / 2
        if self.use_input_norm:
            pred_images   = (pred_images - self.mean) / self.std
            frames = (frames - self.mean) / self.std
        lpips_loss = self.perceptual(frames.contiguous(), pred_images.contiguous())
        return self.loss_weight * lpips_loss.mean(), None

class ProPainterAdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self,
                 type='nsgan',
                 target_real_label=1.0,
                 target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(ProPainterAdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, generated_frames, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    generated_frames = -generated_frames
                return self.criterion(1 + generated_frames).mean()
            else:
                return (-generated_frames).mean()
        else:
            labels = (self.real_label
                      if is_real else self.fake_label).expand_as(generated_frames)
            loss = self.criterion(generated_frames, labels)
            return loss

def create_mask(flow, paddings):
    """
    flow shape: [batch_size, num_channels, height, width]
    paddings: [2 x 2] shape list, the first row indicates up and down paddings
    the second row indicates left and right paddings
    |            |
    |       x    |
    |     x * x  |
    |       x    |
    |            |
    """
    shape = flow.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])
    torch_paddings = [paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]]  # left, right, up and down
    mask2d = F.pad(inner, pad=torch_paddings)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()

def smoothness_deltas(flow):
    """
    flow: [batch_size, num_channels, height, width]
    """
    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])
    mask = torch.cat((mask_x, mask_y), dim=1)
    mask = mask.to(flow.device)
    filter_x = torch.tensor([[0, 0, 0.], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 0, 0.], [0, 1, 0], [0, -1, 0]])
    weights = torch.ones([2, 1, 3, 3])
    weights[0, 0] = filter_x
    weights[1, 0] = filter_y
    weights = weights.to(flow.device)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask

def charbonnier_loss(delta, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """
    Compute the generalized charbonnier loss of the difference tensor x
    All positions where mask == 0 are not taken into account
    delta: a tensor of shape [batch_size, num_channels, height, width]
    mask: a mask of shape [batch_size, mc, height, width], where mask channels must be either 1 or the same as
    the number of channels of delta. Entries should be 0 or 1
    return: loss
    """
    batch_size, num_channels, height, width = delta.shape
    norm = batch_size * num_channels * height * width
    error = torch.pow(torch.square(delta * beta) + torch.square(torch.tensor(epsilon)), alpha)
    if mask is not None:
        error = mask * error
    if truncate is not None:
        error = torch.min(error, truncate)
    return torch.sum(error) / norm

def second_order_deltas(flow):
    """
    consider the single flow first
    flow shape: [batch_size, num_channels, height, width]
    """
    # create mask
    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])
    mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), dim=1)
    mask = mask.to(flow.device)

    filter_x = torch.tensor([[0, 0, 0.], [1, -2, 1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 1, 0.], [0, -2, 0], [0, 1, 0]])
    filter_diag1 = torch.tensor([[1, 0, 0.], [0, -2, 0], [0, 0, 1]])
    filter_diag2 = torch.tensor([[0, 0, 1.], [0, -2, 0], [1, 0, 0]])
    weights = torch.ones([4, 1, 3, 3])
    weights[0] = filter_x
    weights[1] = filter_y
    weights[2] = filter_diag1
    weights[3] = filter_diag2
    weights = weights.to(flow.device)

    # split the flow into flow_u and flow_v, conv them with the weights
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask

def smoothness_loss(flow, cmask):
    delta_u, delta_v, _ = smoothness_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v

def second_order_loss(flow, cmask):
    delta_u, delta_v, _ = second_order_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v

def convert_rgb_to_grayscale(image, rgb_weights = None):
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, height, width). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b

def ternary_transform(image, max_distance=1):
    device = image.device
    patch_size = 2 * max_distance + 1
    intensities = convert_rgb_to_grayscale(image) * 255
    out_channels = patch_size * patch_size
    weights = np.eye(out_channels).reshape(out_channels, 1, patch_size, patch_size)
    weights = torch.from_numpy(weights).float().to(device)
    patches = F.conv2d(intensities, weights, stride=1, padding=1)
    transf = patches - intensities
    transf_norm = transf / torch.sqrt(0.81 + torch.square(transf))
    return transf_norm

def hamming_distance(ternary_transform_frame1, ternary_transform_frame2):
    distance = torch.square(ternary_transform_frame1 - ternary_transform_frame2)
    distance_norm = distance / (0.1 + distance)
    distance_sum = torch.sum(distance_norm, dim=1, keepdim=True)
    return distance_sum

def ternary_loss(flow_comp, flow_gt, mask, current_frame, shift_frame, scale_factor=1):
    if scale_factor != 1:
        current_frame = F.interpolate(current_frame, scale_factor=1 / scale_factor, mode='bilinear')
        shift_frame = F.interpolate(shift_frame, scale_factor=1 / scale_factor, mode='bilinear')
    warped_sc = flow_warp(shift_frame, flow_gt.permute(0, 2, 3, 1))
    confidence_mask = torch.exp(-50. * torch.sum(torch.abs(current_frame - warped_sc), dim=1).pow(2)).unsqueeze(1)
    warped_comp_sc = flow_warp(shift_frame, flow_comp.permute(0, 2, 3, 1))
    
    ternary_transform1 = ternary_transform(current_frame) #current_frame: [batch_size * timesteps, num_channels, height, width]
    ternary_transform21 = ternary_transform(warped_comp_sc) #warped_comp_sc: [batch_size * timesteps, num_channels, height, width]
    dist = hamming_distance(ternary_transform1, ternary_transform21) 
    loss = torch.mean(dist * confidence_mask * mask) / torch.mean(mask) #confidence_mask, mask: [batch_size * timesteps, num_channels, height, width]
    
    return loss

class ProPainterFlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_flows, masks, frames):
        # pred_flows: bacth_size timestep-1 2 height width
        loss = 0
        warp_loss = 0
        height, width = pred_flows[0].shape[-2:]
        masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        frames0 = frames[:,:-1,...]
        frames1 = frames[:,1:,...]
        current_frames = [frames0, frames1]
        next_frames = [frames1, frames0]
        for i in range(len(pred_flows)):
            combined_flow = pred_flows[i] * masks[i] + gt_flows[i] * (1-masks[i])
            l1_loss = self.l1_criterion(pred_flows[i] * masks[i], gt_flows[i] * masks[i]) / torch.mean(masks[i])
            l1_loss += self.l1_criterion(pred_flows[i] * (1-masks[i]), gt_flows[i] * (1-masks[i])) / torch.mean((1-masks[i]))

            smooth_loss = smoothness_loss(combined_flow.reshape(-1,2,height,width), masks[i].reshape(-1,1,height,width))
            smooth_loss2 = second_order_loss(combined_flow.reshape(-1,2,height,width), masks[i].reshape(-1,1,height,width))
            
            warp_loss_i = ternary_loss(combined_flow.reshape(-1,2,height,width), gt_flows[i].reshape(-1,2,height,width), 
                            masks[i].reshape(-1,1,height,width), current_frames[i].reshape(-1,3,height,width), next_frames[i].reshape(-1,3,height,width)) 

            loss += l1_loss + smooth_loss + smooth_loss2

            warp_loss += warp_loss_i
            
        return loss, warp_loss

class ProPainterEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def edgeLoss(self,pred_edges, edges):
        """

        Args:
            pred_edges: with shape [batch_size, num_channels, height, width]
            edges: with shape [batch_size, num_channels, height, width]

        Returns: Edge losses

        """
        mask = (edges > 0.5).float()
        _, num_channels, height, width = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float() # Shape: [batch_size,].
        num_neg = num_channels * height * width - num_pos # Shape: [batch_size,].
        neg_weights = (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        pos_weights = (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        weight = neg_weights * mask + pos_weights * (1 - mask)  # weight for debug
        losses = F.binary_cross_entropy_with_logits(pred_edges.float(), edges.float(), weight=weight, reduction='none')
        loss = torch.mean(losses)
        return loss

    def forward(self, pred_edges, gt_edges, masks):
        # pred_flows: batch_size timestep-1 1 height width
        loss = 0
        height, width = pred_edges[0].shape[-2:]
        masks = [masks[:,:-1,...].contiguous(), masks[:, 1:, ...].contiguous()]
        for i in range(len(pred_edges)):
            combined_edge = pred_edges[i] * masks[i] + gt_edges[i] * (1-masks[i])
            edge_loss = (self.edgeLoss(pred_edges[i].reshape(-1,1,height,width), gt_edges[i].reshape(-1,1,height,width)) \
                        + 5 * self.edgeLoss(combined_edge.reshape(-1,1,height,width), gt_edges[i].reshape(-1,1,height,width)))
            loss += edge_loss 

        return loss

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    offsets = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        offsets = offsets + 0.5

    gauss = torch.exp((-offsets.pow(2.0) / (2 * sigma**2)).float())
    return gauss / gauss.sum()

def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d

def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth num_channels of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(batch_size, num_channels, height, width)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(batch_size, num_channels, height, width)`.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
        raise ValueError(
            f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}"
        )

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ['valid', 'same']:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    batch_size, num_channels, height, width = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, num_channels, -1, -1)

    height_, width_ = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: List[int] = _compute_padding([height_, width_])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height_, width_)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == 'same':
        out = output.view(batch_size, num_channels, height, width)
    else:
        out = output.view(batch_size, num_channels, height - height_ + 1, width - width_ + 1)

    return out

def gaussian_blur2d(
    input: torch.Tensor,
    kernel_size: Tuple[int, int],
    sigma: Tuple[float, float],
    border_type: str = 'reflect',
    separable: bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each num_channels. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(B,C,height,width)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(batch_size, num_channels, height, width)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       gaussian_blur.html>`__.
    """
    if separable:
        kernel_x: torch.Tensor = get_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel_y: torch.Tensor = get_gaussian_kernel1d(kernel_size[0], sigma[0])
        # Convolve a tensor with two 1d kernels, in x and y directions.The kernel is applied
        # independently at each depth num_channels of the tensor. Before applying the
        # kernel, the function applies padding according to the specified mode so
        # that the output remains in the same shape.
        output_x = filter2d(input, kernel_x[None].unsqueeze(0), border_type, normalized = False, padding = 'same')
        output = filter2d(output_x, kernel_y[None].unsqueeze(-1), border_type, normalized = False, padding = 'same')
    else:
        #returns Gaussian filter matrix coefficients.
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
        ksize_x, ksize_y = kernel_size
        sigma_x, sigma_y = sigma
        kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even = False)
        kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even = False)
        kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())        
        output = filter2d(input, kernel_2d[None], border_type)

    return output

def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def get_sobel_kernel_5x5_2nd_order() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ]
    )


def _get_sobel_kernel_5x5_2nd_order_xy() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ]
    )

def get_diff_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3."""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]])


def get_sobel_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_sobel_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_diff_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    return torch.stack([gxx, gxy, gyy])

def get_spatial_gradient_kernel2d(mode: str, order: int) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients, using one of the following operators:

    sobel, diff.
    """
    if mode not in ['sobel', 'diff']:
        raise TypeError(
            "mode should be either sobel\
                         or diff. Got {}".format(
                mode
            )
        )
    if order not in [1, 2]:
        raise TypeError(
            "order should be either 1 or 2\
                         Got {}".format(
                order
            )
        )
    if mode == 'sobel' and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel

def normalize_kernel2d(kernel: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(kernel.size()) < 2:
        raise TypeError(f"kernel should be at least 2D tensor. Got {kernel.size()}")
    norm: torch.Tensor = kernel.abs().sum(dim=-1).sum(dim=-1)
    return kernel / (norm.unsqueeze(-1).unsqueeze(-1))


def spatial_gradient(input: torch.Tensor, mode: str = 'sobel', order: int = 1, normalized: bool = True) -> torch.Tensor:
    r"""Compute the first order image derivative in both x and y using a Sobel operator.

    Args:
        input: input image tensor with shape :math:`(batch_size, num_channels, height, width)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, height, width)`.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    # allocate kernel
    kernel: torch.Tensor = get_spatial_gradient_kernel2d(mode, order)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    batch_size, num_channels, height, width = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip: torch.Tensor = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for num_channels
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: torch.Tensor = F.pad(input.reshape(batch_size * num_channels, 1, height, width), spatial_pad, 'replicate')[:, :, None]

    return F.conv3d(padded_inp, kernel_flip, padding=0).view(batch_size, num_channels, out_channels, height, width)

def get_canny_nms_kernel(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    kernel: torch.Tensor = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)

def get_hysteresis_kernel(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    kernel: torch.Tensor = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)

class ProPainterCanny(nn.Module):
    r"""Module that finds edges of the input image and filters them using the Canny algorithm.

    Args:
        input: input image tensor with shape :math:`(B,C,height,width)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,height,width)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,height,width)`.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Tuple[float, float] = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if low_threshold > high_threshold:
            raise ValueError(
                "Invalid input thresholds. low_threshold should be\
                             smaller than the high_threshold. Got: {}>{}".format(
                    low_threshold, high_threshold
                )
            )

        if low_threshold < 0 or low_threshold > 1:
            raise ValueError(f"Invalid input threshold. low_threshold should be in range (0,1). Got: {low_threshold}")

        if high_threshold < 0 or high_threshold > 1:
            raise ValueError(f"Invalid input threshold. high_threshold should be in range (0,1). Got: {high_threshold}")

        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Double threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Hysteresis
        self.hysteresis = hysteresis

        self.eps: float = eps

    def canny(
        self,
        input: torch.Tensor,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Tuple[float, float] = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Find edges of the input image and filters them using the Canny algorithm.
        Args:
            input: input image tensor with shape :math:`(B,C,height,width)`.
            low_threshold: lower threshold for the hysteresis procedure.
            high_threshold: upper threshold for the hysteresis procedure.
            kernel_size: the size of the kernel for the gaussian blur.
            sigma: the standard deviation of the kernel for the gaussian blur.
            hysteresis: if True, applies the hysteresis edge tracking.
                Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
            eps: regularization number to avoid NaN during backprop.

        Returns:
            - the canny edge magnitudes map, shape of :math:`(B,1,height,width)`.
            - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,height,width)`.

        .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
        canny.html>`__.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

        if low_threshold > high_threshold:
            raise ValueError(
                "Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: {}>{}".format(
                    low_threshold, high_threshold
                )
            )

        if low_threshold < 0 and low_threshold > 1:
            raise ValueError(f"Invalid input threshold. low_threshold should be in range (0,1). Got: {low_threshold}")

        if high_threshold < 0 and high_threshold > 1:
            raise ValueError(f"Invalid input threshold. high_threshold should be in range (0,1). Got: {high_threshold}")

        device: torch.device = input.device
        dtype: torch.dtype = input.dtype

        # To Grayscale
        if input.shape[1] == 3:
            input = convert_rgb_to_grayscale(input)

        # Gaussian filter
        blurred: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)

        # Compute the gradients
        gradients: torch.Tensor = spatial_gradient(blurred, normalized=False)

        # Unpack the edges
        gx: torch.Tensor = gradients[:, :, 0]
        gy: torch.Tensor = gradients[:, :, 1]

        # Compute gradient magnitude and angle
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)
        angle: torch.Tensor = torch.atan2(gy, gx)

        # Radians to Degrees
        angle = 180.0 * angle / math.pi

        # Round angle to the nearest 45 degree
        angle = torch.round(angle / 45) * 45

        # Non-maximal suppression
        nms_kernels: torch.Tensor = get_canny_nms_kernel(device, dtype)
        nms_magnitude: torch.Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

        # Get the indices for both directions
        positive_idx: torch.Tensor = (angle / 45) % 8
        positive_idx = positive_idx.long()

        negative_idx: torch.Tensor = ((angle / 45) + 4) % 8
        negative_idx = negative_idx.long()

        # Apply the non-maximum suppression to the different directions
        channel_select_filtered_positive: torch.Tensor = torch.gather(nms_magnitude, 1, positive_idx)
        channel_select_filtered_negative: torch.Tensor = torch.gather(nms_magnitude, 1, negative_idx)

        channel_select_filtered: torch.Tensor = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative], 1
        )

        is_max: torch.Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

        magnitude = magnitude * is_max

        # Threshold
        edges: torch.Tensor = F.threshold(magnitude, low_threshold, 0.0)

        low: torch.Tensor = magnitude > low_threshold
        high: torch.Tensor = magnitude > high_threshold

        edges = low * 0.5 + high * 0.5
        edges = edges.to(dtype)

        # Hysteresis
        if hysteresis:
            edges_old: torch.Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
            hysteresis_kernels: torch.Tensor = get_hysteresis_kernel(device, dtype)

            while ((edges_old - edges).abs() != 0).any():
                weak: torch.Tensor = (edges == 0.5).float()
                strong: torch.Tensor = (edges == 1).float()

                hysteresis_magnitude: torch.Tensor = F.conv2d(
                    edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2
                )
                hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
                hysteresis_magnitude = hysteresis_magnitude * weak + strong

                edges_old = edges.clone()
                edges = hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5

            edges = hysteresis_magnitude

        return magnitude, edges

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.canny(
            input, self.low_threshold, self.high_threshold, self.kernel_size, self.sigma, self.hysteresis, self.eps
        )

class ProPainterLosses():
    def __init__(self, config) -> None:
        self.config = config
        self.l1_loss = L1Loss()
        self.perc_loss = ProPainterLpipsLoss(use_input_norm=True, range_norm=True)
        self.adversarial_loss = ProPainterAdversarialLoss(type=config.GAN_LOSS)
        self.flow_loss = ProPainterFlowLoss()
        self.edge_loss = ProPainterEdgeLoss()
        self.canny = ProPainterCanny(sigma=(2,2), low_threshold=0.1, high_threshold=0.2)

    def get_edges(self, flows): 
        # (batch_size, timesteps, 2, height, width)
        batch_size, timesteps, _, height, width = flows.shape
        flows = flows.view(-1, 2, height, width)
        flows_gray = (flows[:, 0, None] ** 2 + flows[:, 1, None] ** 2) ** 0.5
        if flows_gray.max() < 1:
            flows_gray = flows_gray*0
        else:
            flows_gray = flows_gray / flows_gray.max()
            
        _, edges = self.canny(flows_gray.float())
        edges = edges.view(batch_size, timesteps, 1, height, width)
        return edges

    def calculate_losses(self, pred_imgs,masks_dilated, frames, comp_frames ,discriminator, pred_flows_bi, gt_flows_bi,flow_masks,pred_edges_bi):
        _,_,_, height, width = frames.size()
        
        gt_edges_forward = self.get_edges(gt_flows_bi[0])
        gt_edges_backward = self.get_edges(gt_flows_bi[1])
        gt_edges_bi = [gt_edges_forward, gt_edges_backward]

        gen_loss = 0
        dis_loss = 0
        # generator l1 loss
        hole_loss = self.l1_loss(pred_imgs * masks_dilated, frames * masks_dilated)
        hole_loss = hole_loss / torch.mean(masks_dilated) * self.config.hole_weight
        gen_loss += hole_loss

        valid_loss = self.l1_loss(pred_imgs * (1 - masks_dilated), frames * (1 - masks_dilated))
        valid_loss = valid_loss / torch.mean(1-masks_dilated) * self.config.valid_weight
        gen_loss += valid_loss

        # perceptual loss
        if self.config.perceptual_weight > 0:
            perc_loss = self.perc_loss(pred_imgs.view(-1,3,height,width), frames.view(-1,3,height,width))[0] * self.config['losses']['perceptual_weight']
            gen_loss += perc_loss

        # gan loss
        if not self.config.no_dis:
            # generator adversarial loss
            gen_clip = discriminator(comp_frames)
            gan_loss = self.adversarial_loss(gen_clip, True, False)
            gan_loss = gan_loss * self.config.adversarial_weight
            gen_loss += gan_loss

        if not self.config.no_dis:
            # discriminator adversarial loss
            real_clip = discriminator(frames)
            fake_clip = discriminator(comp_frames.detach())
            dis_real_loss = self.adversarial_loss(real_clip, True, True)
            dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

        #these losses are for training flow completion network
        # compulte flow_loss
        flow_loss, warp_loss = self.flow_loss(pred_flows_bi, gt_flows_bi, flow_masks, frames)
        flow_loss = flow_loss * self.config.flow_weight

        # compute edge loss
        edge_loss = self.edge_loss(pred_edges_bi, gt_edges_bi, flow_masks)
        edge_loss = edge_loss*1.0

        flow_complete_loss = flow_loss + warp_loss * 0.01 + edge_loss
        return gen_loss, dis_loss, flow_complete_loss

class ProPainterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProPainterConfig
    base_model_prefix = "propainter"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ProPainterSecondOrderDeformableAlignment) or isinstance(module, ProPainterDeformableAlignment):
            num_channels = module.in_channels
            for k in module.kernel_size:
                num_channels *= k
            stdv = 1. / math.sqrt(num_channels)
            module.weight.data.uniform_(-stdv, stdv)
            if module.bias is not None:
                module.bias.data.zero_()     
            if hasattr(module.conv_offset[-1], 'weight') and module.conv_offset[-1].weight is not None:
                TORCH_INIT_FUNCTIONS["constant_"](module.conv_offset[-1].weight, 0)
            if hasattr(module.conv_offset[-1], 'bias') and module.conv_offset[-1].bias is not None:
                TORCH_INIT_FUNCTIONS["constant_"](module.conv_offset[-1].bias, 0)   
        elif isinstance(module, ProPainterInpaintGenerator) or isinstance(module, ProPainterDiscriminator):
            for child in module.children():
                classname = child.__class__.__name__
                if classname.find('InstanceNorm2d') != -1:
                    if hasattr(child, 'weight') and child.weight is not None:
                        nn.init.constant_(child.weight.data, 1.0)
                    if hasattr(child, 'bias') and child.bias is not None:
                        nn.init.constant_(child.bias.data, 0.0)
                elif hasattr(child, 'weight') and (classname.find('Conv') != -1
                                            or classname.find('Linear') != -1):
                    nn.init.normal_(child.weight.data, 0.0, 0.02)
                    if hasattr(child, 'bias') and child.bias is not None:
                        nn.init.constant_(child.bias.data, 0.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


PROPAINTER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ProPainterConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PROPAINTER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ProPainterImageProcessor.__call__`]
            for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ProPainter Model outputting composed frames without any specific head on top.",
    PROPAINTER_START_DOCSTRING,
)
class ProPainterModel(ProPainterPreTrainedModel):
    def __init__(self, config: ProPainterConfig):
        super().__init__(config)
        self.config = config
        self.optical_flow_model = ProPainterRaftOpticalFlow(config)
        self.flow_completion_net = ProPainterRecurrentFlowCompleteNet()
        self.inpaint_generator = ProPainterInpaintGenerator(config)
        self.discriminator = ProPainterDiscriminator(use_sigmoid=self.config.GAN_LOSS != 'hinge')
        # Initialize weights and apply final processing
        self.post_init()

    def _get_ref_index(self,mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def _get_short_clip_len(self, width):
        if width <= 640:
            return 12
        elif width <= 720:
            return 8
        elif width <= 1280:
            return 4
        else:
            return 2

    @add_start_docstrings_to_model_forward(PROPAINTER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedImageModelingOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )

    def compute_flow(self,pixel_values,output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        short_clip_len = self._get_short_clip_len(pixel_values.size(-1))
        if pixel_values.size(1) > short_clip_len:
            all_hidden_states = () if output_hidden_states else None
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, self.video_length, short_clip_len):
                end_f = min(self.video_length, f + short_clip_len)
                if f == 0:
                    optical_flow_model_outputs = self.optical_flow_model(pixel_values[:,f:end_f], iters=self.config.raft_iter)
                    flows_f, flows_b = optical_flow_model_outputs[0]
                    _all_hidden_states = optical_flow_model_outputs[1]
                else:
                    optical_flow_model_outputs = self.optical_flow_model(pixel_values[:,f-1:end_f], iters=self.config.raft_iter)
                    flows_f, flows_b = optical_flow_model_outputs[0]
                    _all_hidden_states = optical_flow_model_outputs[1]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (_all_hidden_states,)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            optical_flow_model_outputs = self.optical_flow_model(pixel_values, iters=self.config.raft_iter)
            gt_flows_bi, all_hidden_states = optical_flow_model_outputs[0], optical_flow_model_outputs[1]
            torch.cuda.empty_cache()
        return gt_flows_bi, all_hidden_states
    
    def complete_flow(self,gt_flows_bi, flow_masks, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):        
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.config.subvideo_length:
            all_hidden_states = () if output_hidden_states else None
            pred_flows_f, pred_flows_b, pred_flows_bi_loss, pred_edges_bi_loss= [], [], []
            pad_len = 5
            for f in range(0, flow_length, self.config.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.config.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.config.subvideo_length)
                flow_completion_net_outputs = self.flow_completion_net.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1], output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
                pred_flows_bi_sub, pred_edges_bi = flow_completion_net_outputs[0]
                _all_hidden_states = flow_completion_net_outputs[1]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (_all_hidden_states,)

                pred_flows_bi_loss.append(pred_flows_bi_sub)
                pred_edges_bi_loss.append(pred_edges_bi)
                pred_flows_bi_sub = self.flow_completion_net.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)

            pred_flows_bi_loss = torch.cat(pred_flows_bi_loss)
            pred_edges_bi_loss = torch.cat(pred_edges_bi_loss)
        else:
            flow_completion_net_outputs = self.flow_completion_net.forward_bidirect_flow(gt_flows_bi, flow_masks, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
            pred_flows_bi, pred_edges_bi = flow_completion_net_outputs[0]
            all_hidden_states = flow_completion_net_outputs[1]
            pred_flows_bi_loss = pred_flows_bi

            pred_flows_bi = self.flow_completion_net.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)

            torch.cuda.empty_cache()

        return pred_flows_bi, pred_flows_bi_loss, pred_edges_bi, all_hidden_states

    def image_propagation(self,pixel_values,masks_dilated, pred_flows_bi, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        width, height = self.size
        masked_frames = pixel_values * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, self.config.subvideo_length) # ensure a minimum of 100 frames for image propagation
        if self.video_length > subvideo_length_img_prop:
            all_hidden_states = () if output_hidden_states else None
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, self.video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(self.video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(self.video_length, f + subvideo_length_img_prop)

                batch_size, timesteps, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                inpaint_generator_outputs = self.inpaint_generator.img_propagation(masked_frames[:, s_f:e_f], 
                                                                    pred_flows_bi_sub, 
                                                                    masks_dilated[:, s_f:e_f], 
                                                                    'nearest', output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
                prop_imgs_sub, updated_local_masks_sub = inpaint_generator_outputs[0]
                _all_hidden_states = inpaint_generator_outputs[1]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (_all_hidden_states,)
                updated_frames_sub = pixel_values[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(batch_size, timesteps, 3, height, width) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(batch_size, timesteps, 1, height, width)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            batch_size, timesteps, _, _, _ = masks_dilated.size()
            inpaint_generator_outputs = self.inpaint_generator.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest', output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
            prop_imgs, updated_local_masks = inpaint_generator_outputs[0]
            all_hidden_states = inpaint_generator_outputs[1]
            updated_frames = pixel_values * (1 - masks_dilated) + prop_imgs.view(batch_size, timesteps, 3, height, width) * masks_dilated
            updated_masks = updated_local_masks.view(batch_size, timesteps, 1, height, width)
            torch.cuda.empty_cache()

        return updated_frames,updated_masks, all_hidden_states

    def feature_propagation(self,updated_frames,updated_masks,masks_dilated,pred_flows_bi,original_frames, output_attentions: bool = False,output_hidden_states: bool = False,return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        width, height = self.size
        comp_frames = [None] * self.video_length

        neighbor_stride = self.config.neighbor_length // 2
        if self.video_length > self.config.subvideo_length:
            ref_num = self.config.subvideo_length // self.config.ref_stride
        else:
            ref_num = -1

        pred_imgs_loss = [None] * self.video_length
        # ---- feature propagation + transformer ----
        for f in range(0, self.video_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(self.video_length, f + neighbor_stride + 1))
            ]
            ref_ids = self._get_ref_index(f, neighbor_ids, self.video_length, self.config.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                inpaint_generator_outputs = self.inpaint_generator(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
                pred_img, _all_hidden_states, _all_self_attentions = inpaint_generator_outputs[0], inpaint_generator_outputs[1], inpaint_generator_outputs[2]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (_all_hidden_states,)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (_all_self_attentions,)
                
                pred_img = pred_img.view(-1, 3, height, width)

                
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + original_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else: 
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
                    pred_imgs_loss[idx]=pred_img[i]
        return comp_frames, pred_imgs_loss, all_hidden_states, all_self_attentions
    
 
    def forward(
        self,
        pixel_values_inp: Optional[List[np.ndarray]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        flow_masks: Optional[torch.BoolTensor] = None,
        masks_dilated: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int,int]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedImageModelingOutput]:
        r"""

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        losses = ProPainterLosses(self.config)

        self.size = size
        
        self.video_length = pixel_values.size(1)
        with torch.no_grad():
            gt_flows_bi, _all_hiddens_states = self.compute_flow(pixel_values, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (_all_hiddens_states,)
            
            pred_flows_bi, pred_flows_bi_loss, pred_edges_bi, _all_hiddens_states = self.complete_flow(gt_flows_bi,flow_masks, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (_all_hiddens_states,)

            updated_frames,updated_masks, _all_hiddens_states = self.image_propagation(pixel_values,masks_dilated,pred_flows_bi)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (_all_hiddens_states,)
                
        comp_frames, pred_imgs_loss, _all_hiddens_states, _all_self_attentions = self.feature_propagation(updated_frames, updated_masks, masks_dilated, pred_flows_bi, pixel_values_inp)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (_all_hiddens_states,)
        if output_attentions:
                all_self_attentions = all_self_attentions + (_all_self_attentions,)

        pred_imgs_loss = torch.tensor(np.array(pred_imgs_loss)).permute(0, 3, 1, 2).unsqueeze(0).to(masks_dilated.device)
        comp_frames_loss = torch.tensor(np.array(comp_frames)).permute(3,0,1,2).to(masks_dilated.device).to(torch.float32)
        #ADD LOCAL FRAMES and training mode
        gen_loss, dis_loss, flow_complete_loss = losses.calculate_losses(pred_imgs_loss,masks_dilated, pixel_values, comp_frames_loss,self.discriminator,pred_flows_bi_loss,gt_flows_bi,flow_masks,pred_edges_bi)

        if not return_dict:
            return tuple(v for v in [(gen_loss, dis_loss, flow_complete_loss), comp_frames, all_hidden_states, all_self_attentions] if v is not None)

        return MaskedImageModelingOutput(
            loss=(gen_loss, dis_loss, flow_complete_loss),
            reconstruction=comp_frames,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

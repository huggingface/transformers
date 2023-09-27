# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViT model."""


import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vit import ViTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ProPainterConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


VIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/vit-base-patch16-224",
    # See all ViT models at https://huggingface.co/models?filter=vit
]


class P3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_residual=0, bias=True, config:ProPainterConfig):
        super().__init__()
        self.Conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size),
                                    stride=(1, stride, stride), padding=(0, padding, padding), bias=bias),
                        nn.LeakyReLU(config.threshold, inplace=True)
        )
        self.Conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                    padding=(2, 0, 0), dilation=(2, 1, 1), bias=bias)
        )
        self.use_residual = use_residual

    def forward(self, pixel_values):
        feat1 = self.Conv1(pixel_values)
        feat2 = self.Conv2(feat1)
        if self.use_residual:
            output = feats + feat2
        else:
            output = feat2
        return output

class EdgeDetection(nn.Module):
    def __init__(self, config:ProPainterConfig):
        super().__init__()

        self.config = config
        self.edge_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True)
        )

        self.edge_layer_1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True)
        )

        self.edge_layer_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
        )

        self.edge_act = nn.LeakyReLU(0.01, inplace=True)

        self.edge_out = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)

    def forward(self, flow):
        flow = self.edge_projection(flow)
        edge = self.edge_layer_1(flow)
        edge = self.edge_layer_2(edge)
        edge = self.edge_act(flow + edge)
        edge = self.edge_out(edge)
        return torch.sigmoid(edge)


class ReccurrentFlowCompleteNet(nn.Module):
    def __init__(self, config:ProPainterConfig):
        super().__init__()

        self.Downsample = nn.Sequence(
            nn.Conv3d(3,32,kernel_size=(1,5,3), stride=(1,2,2),
                padding=(0,2,2), padding_mode="replicate"),
            nn.LeakyReLU(config.threshold,  inpalce=True)
        )

        self.Encoder = []
        for depth in range(config.encoder_depth):
            self.Encoder += [P3DBlock(32*(depth+1),32*(depth+1),3,1,1)]
            self.Encoder += [nn.LeakyReLU(config.threshold, inplace=True)]
            self.Encoder += [P3DBlock(32*(depth+1),64*(depth+1),3,2,1)]
            self.Encoder += [nn.LeakyReLU(config.threshold, inplace=True)]

        self.Encoder = nn.Sequential(nn.ModuleList(self.Encoder))

        self.MidDilation = nn.Sequential(
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)),
            nn.LeakyReLU(config.threshold, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)),
            nn.LeakyReLU(config.threshold, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)),
            nn.LeakyReLU(config.threshold, inplace=True)
        )

        self.feat_prop_module = BidirectionalPropagation(128)

        self.Decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
            deconv(64, 32, 3, 1),
            nn.LeakyReLU(config.threshold, inplace=True)
        ) # 2x

        self.Decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
            deconv(128, 64, 3, 1),
            nn.LeakyReLU(config.threshold, inplace=True)
        ) # 4x

        self.Upsample = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(config.threshold, inplace=True),
            deconv(32, 2, 3, 1)
        )

        self.EdgeDetector = EdgeDetection(in_channels=2, out_channels=1, hidden_channels=16)

        #for m in self.modules():
        #    if isinstance(m, SecondOrderDeformableAlignment):
        #        m.init_offset()
  
    def forward(self, masked_flows, masks):
        b, t, _, h, w = masked_flows.size()
        masked_flows = masked_flows.permute(0,2,1,3,4)
        masks = masks.permute(0,2,1,3,4)

        inputs = torch.cat((masked_flows, masks), dim=1)
        
        x = self.Downsample(inputs)

        feat_e1 = self.Encoder1(x)
        feat_e2 = self.Encoder2(feat_e1) # b c t h w
        feat_mid = self.MidDilation(feat_e2) # b c t h w
        feat_mid = feat_mid.permute(0,2,1,3,4) # b t c h w

        feat_prop = self.feat_prop_module(feat_mid)
        feat_prop = feat_prop.view(-1, 128, h//8, w//8) # b*t c h w

        _, c, _, h_f, w_f = feat_e1.shape
        feat_e1 = feat_e1.permute(0,2,1,3,4).contiguous().view(-1, c, h_f, w_f) # b*t c h w
        feat_d2 = self.Decoder2(feat_prop) + feat_e1

        _, c, _, h_f, w_f = x.shape
        x = x.permute(0,2,1,3,4).contiguous().view(-1, c, h_f, w_f) # b*t c h w

        feat_d1 = self.Decoder1(feat_d2)

        flow = self.Upsample(feat_d1)
        if self.training:
            edge = self.EdgeDetector(flow)
            edge = edge.view(b, t, 1, h, w)
        else:
            edge = None

        flow = flow.view(b, t, 2, h, w)

        return flow, edge
    

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in args._get_kwargs():
            args.dropout = 0

        if 'alternate_corr' not in args._get_kwargs():
            args.alternate_corr = False
        
        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=True):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions

class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        # self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation, mask)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
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

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)

class BidirectionalPropagation(nn.Module):
    def __init__(self, channel, learnable=True):
        super(BidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2*channel+2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                    nn.Conv2d(2*channel+2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                ) 
            
    def binary_mask(self, mask, th=0.1):
        mask[mask>th] = 1
        mask[mask<=th] = 0
        # return mask.float()
        return mask.to(mask)

    def forward(self, x, flows_forward, flows_backward, mask, interpolation='bilinear'):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats, masks = {}, {}
        feats['input'] = [x[:, i, :, :, :] for i in range(0, t)]
        masks['input'] = [mask[:, i, :, :, :] for i in range(0, t)]

        prop_list = ['backward_1', 'forward_1']
        cache_list = ['input'] +  prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
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
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
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
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats['backward_1'], dim=1).view(-1, c, h, w)
        outputs_f = torch.stack(feats['forward_1'], dim=1).view(-1, c, h, w)

        if self.learnable:
            mask_in = mask.view(-1, 2, h, w)
            masks_b, masks_f = None, None
            outputs = self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) + x.view(-1, c, h, w)
        else:
            masks_b = torch.stack(masks['backward_1'], dim=1)
            masks_f = torch.stack(masks['forward_1'], dim=1)
            outputs = outputs_f

        return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), \
               outputs.view(b, -1, c, h, w), masks_f

class InpaintGenerator(PretrainedModel):
    def __init__(self):
        super(InpaintGenerator, self).__init__()
        channel = 128
        hidden = 512

        self.encoder = Encoder(o)

        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        kernel_size = (7,7)
        padding = (3,3)
        stride = (3,3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }

        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True)

        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(dim=hidden,
                                                n_head=num_heads,
                                                window_size=window_size,
                                                pool_size=pool_size,
                                                depths=depths,
                                                t2t_params=t2t_params)
        if init_weights:
            self.init_weights()


        if model_path is not None:
            print('Pretrained ProPainter has loaded...')
            ckpt = torch.load(model_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=True)

        # print network parameter number
        self.print_network()

    def img_propagation(self, masked_frames, completed_flows, masks, interpolation='nearest'):
        _, _, prop_frames, updated_masks = self.img_prop_module(masked_frames, completed_flows[0], completed_flows[1], masks, interpolation)
        return prop_frames, updated_masks

    def forward(self, masked_frames, compeleted_flows, masks_in, masks_updated, num_local_frames, interpolation='bilinear', t_dilation=2):

        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.size()

        # extracting features
        enc_feat = self.encoder(torch.cat([masked_frames.view(b * t, 3, ori_h, ori_w),
                                        masks_in.view(b * t, 1, ori_h, ori_w),
                                        masks_updated.view(b * t, 1, ori_h, ori_w)], dim=1))
        _, c, h, w = enc_feat.size()
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        fold_feat_size = (h, w)

        ds_flows_f = F.interpolate(completed_flows[0].view(-1, 2, ori_h, ori_w), scale_factor=1/4, mode='bilinear', align_corners=False).view(b, l_t-1, 2, h, w)/4.0
        ds_flows_b = F.interpolate(completed_flows[1].view(-1, 2, ori_h, ori_w), scale_factor=1/4, mode='bilinear', align_corners=False).view(b, l_t-1, 2, h, w)/4.0
        ds_mask_in = F.interpolate(masks_in.reshape(-1, 1, ori_h, ori_w), scale_factor=1/4, mode='nearest').view(b, t, 1, h, w)
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local =  F.interpolate(masks_updated[:,:l_t].reshape(-1, 1, ori_h, ori_w), scale_factor=1/4, mode='nearest').view(b, l_t, 1, h, w)


        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, l_t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))


        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        _, _, local_feat, _ = self.feat_prop_module(local_feat, ds_flows_f, ds_flows_b, prop_mask_in, interpolation)
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        mask_pool_l = rearrange(mask_pool_l, 'b t c h w -> b t h w c').contiguous()
        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)

        enc_feat = enc_feat + trans_feat

        if self.training:
            output = self.decoder(enc_feat.view(-1, c, h, w))
            output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)
        else:
            output = self.decoder(enc_feat[:, :l_t].view(-1, c, h, w))
            output = torch.tanh(output).view(b, l_t, 3, ori_h, ori_w)

        return output

class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


class Discriminator_2D(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator_2D, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(1, 5, 5),
                      stride=(1, 2, 2),
                      padding=(0, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module

class ProPainterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProPainterConfig
    base_model_prefix = "propainter"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTEmbeddings", "ViTLayer"]

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

    def _set_gradient_checkpointing(self, module: ViTEncoder, value: bool = False) -> None:
        if isinstance(module, ViTEncoder):
            module.gradient_checkpointing = value


VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class ProPainterModel(ProPainterPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ProPainterForImageInPainting(ProPainterPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedImageModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input."
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ProPainterForImageOutPainting(ProPainterPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


#class ViTEmbeddings(nn.Module):
#    """
#    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
#    """
#
#    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
#        super().__init__()
#
#        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
#        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
#        self.patch_embeddings = ViTPatchEmbeddings(config)
#        num_patches = self.patch_embeddings.num_patches
#        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#        self.config = config
#
#    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
#        """
#        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
#        resolution images.
#
#        Source:
#        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
#        """
#
#        num_patches = embeddings.shape[1] - 1
#        num_positions = self.position_embeddings.shape[1] - 1
#        if num_patches == num_positions and height == width:
#            return self.position_embeddings
#        class_pos_embed = self.position_embeddings[:, 0]
#        patch_pos_embed = self.position_embeddings[:, 1:]
#        dim = embeddings.shape[-1]
#        h0 = height // self.config.patch_size
#        w0 = width // self.config.patch_size
#        # we add a small number to avoid floating point error in the interpolation
#        # see discussion at https://github.com/facebookresearch/dino/issues/8
#        h0, w0 = h0 + 0.1, w0 + 0.1
#        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
#        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
#        patch_pos_embed = nn.functional.interpolate(
#            patch_pos_embed,
#            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
#            mode="bicubic",
#            align_corners=False,
#        )
#        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
#        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
#
#    def forward(
#        self,
#        pixel_values: torch.Tensor,
#        bool_masked_pos: Optional[torch.BoolTensor] = None,
#        interpolate_pos_encoding: bool = False,
#    ) -> torch.Tensor:
#        batch_size, num_channels, height, width = pixel_values.shape
#        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
#
#        if bool_masked_pos is not None:
#            seq_length = embeddings.shape[1]
#            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
#            # replace the masked visual tokens by mask_tokens
#            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
#            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
#
#        # add the [CLS] token to the embedded patch tokens
#        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
#
#        # add positional encoding to each token
#        if interpolate_pos_encoding:
#            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
#        else:
#            embeddings = embeddings + self.position_embeddings
#
#        embeddings = self.dropout(embeddings)
#
#        return embeddings
#
#
#class ViTPatchEmbeddings(nn.Module):
#    """
#    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
#    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
#    Transformer.
#    """
#
#    def __init__(self, config):
#        super().__init__()
#        image_size, patch_size = config.image_size, config.patch_size
#        num_channels, hidden_size = config.num_channels, config.hidden_size
#
#        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
#        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
#        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
#        self.image_size = image_size
#        self.patch_size = patch_size
#        self.num_channels = num_channels
#        self.num_patches = num_patches
#
#        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
#
#    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
#        batch_size, num_channels, height, width = pixel_values.shape
#        if num_channels != self.num_channels:
#            raise ValueError(
#                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
#                f" Expected {self.num_channels} but got {num_channels}."
#            )
#        if not interpolate_pos_encoding:
#            if height != self.image_size[0] or width != self.image_size[1]:
#                raise ValueError(
#                    f"Input image size ({height}*{width}) doesn't match model"
#                    f" ({self.image_size[0]}*{self.image_size[1]})."
#                )
#        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
#        return embeddings
#
#
#class ViTSelfAttention(nn.Module):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
#            raise ValueError(
#                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
#                f"heads {config.num_attention_heads}."
#            )
#
#        self.num_attention_heads = config.num_attention_heads
#        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#        self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
#        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
#        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
#
#        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#
#    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
#        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#        x = x.view(new_x_shape)
#        return x.permute(0, 2, 1, 3)
#
#    def forward(
#        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
#    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#        mixed_query_layer = self.query(hidden_states)
#
#        key_layer = self.transpose_for_scores(self.key(hidden_states))
#        value_layer = self.transpose_for_scores(self.value(hidden_states))
#        query_layer = self.transpose_for_scores(mixed_query_layer)
#
#        # Take the dot product between "query" and "key" to get the raw attention scores.
#        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#
#        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#
#        # Normalize the attention scores to probabilities.
#        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
#
#        # This is actually dropping out entire tokens to attend to, which might
#        # seem a bit unusual, but is taken from the original Transformer paper.
#        attention_probs = self.dropout(attention_probs)
#
#        # Mask heads if we want to
#        if head_mask is not None:
#            attention_probs = attention_probs * head_mask
#
#        context_layer = torch.matmul(attention_probs, value_layer)
#
#        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#        context_layer = context_layer.view(new_context_layer_shape)
#
#        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#
#        return outputs
#
#
#class ViTSelfOutput(nn.Module):
#    """
#    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
#    layernorm applied before each block.
#    """
#
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#        hidden_states = self.dense(hidden_states)
#        hidden_states = self.dropout(hidden_states)
#
#        return hidden_states
#
#
#class ViTAttention(nn.Module):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.attention = ViTSelfAttention(config)
#        self.output = ViTSelfOutput(config)
#        self.pruned_heads = set()
#
#    def prune_heads(self, heads: Set[int]) -> None:
#        if len(heads) == 0:
#            return
#        heads, index = find_pruneable_heads_and_indices(
#            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
#        )
#
#        # Prune linear layers
#        self.attention.query = prune_linear_layer(self.attention.query, index)
#        self.attention.key = prune_linear_layer(self.attention.key, index)
#        self.attention.value = prune_linear_layer(self.attention.value, index)
#        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
#
#        # Update hyper params and store pruned heads
#        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
#        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
#        self.pruned_heads = self.pruned_heads.union(heads)
#
#    def forward(
#        self,
#        hidden_states: torch.Tensor,
#        head_mask: Optional[torch.Tensor] = None,
#        output_attentions: bool = False,
#    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
#
#        attention_output = self.output(self_outputs[0], hidden_states)
#
#        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
#        return outputs
#
#
#class ViTIntermediate(nn.Module):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#        if isinstance(config.hidden_act, str):
#            self.intermediate_act_fn = ACT2FN[config.hidden_act]
#        else:
#            self.intermediate_act_fn = config.hidden_act
#
#    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#        hidden_states = self.dense(hidden_states)
#        hidden_states = self.intermediate_act_fn(hidden_states)
#
#        return hidden_states
#
#
#class ViTOutput(nn.Module):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#        hidden_states = self.dense(hidden_states)
#        hidden_states = self.dropout(hidden_states)
#
#        hidden_states = hidden_states + input_tensor
#
#        return hidden_states
#
#
#class ViTLayer(nn.Module):
#    """This corresponds to the Block class in the timm implementation."""
#
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.chunk_size_feed_forward = config.chunk_size_feed_forward
#        self.seq_len_dim = 1
#        self.attention = ViTAttention(config)
#        self.intermediate = ViTIntermediate(config)
#        self.output = ViTOutput(config)
#        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#
#    def forward(
#        self,
#        hidden_states: torch.Tensor,
#        head_mask: Optional[torch.Tensor] = None,
#        output_attentions: bool = False,
#    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#        self_attention_outputs = self.attention(
#            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
#            head_mask,
#            output_attentions=output_attentions,
#        )
#        attention_output = self_attention_outputs[0]
#        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
#
#        # first residual connection
#        hidden_states = attention_output + hidden_states
#
#        # in ViT, layernorm is also applied after self-attention
#        layer_output = self.layernorm_after(hidden_states)
#        layer_output = self.intermediate(layer_output)
#
#        # second residual connection is done here
#        layer_output = self.output(layer_output, hidden_states)
#
#        outputs = (layer_output,) + outputs
#
#        return outputs
#
#
#class ViTEncoder(nn.Module):
#    def __init__(self, config: ViTConfig) -> None:
#        super().__init__()
#        self.config = config
#        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
#        self.gradient_checkpointing = False
#
#    def forward(
#        self,
#        hidden_states: torch.Tensor,
#        head_mask: Optional[torch.Tensor] = None,
#        output_attentions: bool = False,
#        output_hidden_states: bool = False,
#        return_dict: bool = True,
#    ) -> Union[tuple, BaseModelOutput]:
#        all_hidden_states = () if output_hidden_states else None
#        all_self_attentions = () if output_attentions else None
#
#        for i, layer_module in enumerate(self.layer):
#            if output_hidden_states:
#                all_hidden_states = all_hidden_states + (hidden_states,)
#
#            layer_head_mask = head_mask[i] if head_mask is not None else None
#
#            if self.gradient_checkpointing and self.training:
#
#                def create_custom_forward(module):
#                    def custom_forward(*inputs):
#                        return module(*inputs, output_attentions)
#
#                    return custom_forward
#
#                layer_outputs = torch.utils.checkpoint.checkpoint(
#                    create_custom_forward(layer_module),
#                    hidden_states,
#                    layer_head_mask,
#                )
#            else:
#                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
#
#            hidden_states = layer_outputs[0]
#
#            if output_attentions:
#                all_self_attentions = all_self_attentions + (layer_outputs[1],)
#
#        if output_hidden_states:
#            all_hidden_states = all_hidden_states + (hidden_states,)
#
#        if not return_dict:
#            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
#        return BaseModelOutput(
#            last_hidden_state=hidden_states,
#            hidden_states=all_hidden_states,
#            attentions=all_self_attentions,
#        )
#
#
#class ViTPooler(nn.Module):
#    def __init__(self, config: ViTConfig):
#        super().__init__()
#        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#        self.activation = nn.Tanh()
#
#    def forward(self, hidden_states):
#        # We "pool" the model by simply taking the hidden state corresponding
#        # to the first token.
#        first_token_tensor = hidden_states[:, 0]
#        pooled_output = self.dense(first_token_tensor)
#        pooled_output = self.activation(pooled_output)
#        return pooled_output
#
#

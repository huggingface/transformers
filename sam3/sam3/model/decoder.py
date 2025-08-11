# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved
"""
Transformer decoder.
Inspired from Pytorch's version, adds the pre-norm variant
"""

from typing import Any, Dict, List, Optional, Optional

import numpy as np

import torch

from torch import nn, Tensor
from torchvision.ops.roi_align import RoIAlign

from .act_ckpt_utils import activation_ckpt_wrapper

from .box_ops import box_cxcywh_to_xyxy

from .model_misc import (
    gen_sineembed_for_position,
    get_activation_fn,
    get_clones,
    inverse_sigmoid,
    MLP,
)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        cross_attention: nn.Module,
        n_heads: int,
        use_text_cross_attention: bool = False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = cross_attention
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # num_token, bs, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
        # dac
        dac=False,
        dac_use_selfatt_ln=True,
        # skip inside deformable attn
        identity=0.0,
        **kwargs,  # additional kwargs for compatibility
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # self attention
        if self.self_attn is not None:
            if dac:
                # we only apply self attention to the first half of the queries
                assert tgt.shape[0] % 2 == 0
                num_o2o_queries = tgt.shape[0] // 2
                tgt_o2o = tgt[:num_o2o_queries]
                tgt_query_pos_o2o = tgt_query_pos[:num_o2o_queries]
                tgt_o2m = tgt[num_o2o_queries:]
            else:
                tgt_o2o = tgt
                tgt_query_pos_o2o = tgt_query_pos

            q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o)
            tgt2 = self.self_attn(q, k, tgt_o2o, attn_mask=self_attn_mask)[0]
            tgt_o2o = tgt_o2o + self.dropout2(tgt2)
            if dac:
                if not dac_use_selfatt_ln:
                    tgt_o2o = self.norm2(tgt_o2o)
                tgt = torch.cat((tgt_o2o, tgt_o2m), dim=0)  # Recombine
                if dac_use_selfatt_ln:
                    tgt = self.norm2(tgt)
            else:
                tgt = tgt_o2o
                tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=cross_attn_mask,
            key_padding_mask=(
                memory_key_padding_mask.transpose(0, 1)
                if memory_key_padding_mask is not None
                else None
            ),
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        interaction_layer,
        layer,
        num_layers: int,
        num_queries: int,
        return_intermediate: bool,
        box_refine: bool = False,
        num_o2m_queries: int = 0,
        dac: bool = False,
        boxRPB: str = "none",
        # Experimental: An object query for SAM 2 tasks
        instance_query: bool = False,
        # Defines the number of additional instance queries,
        # 1 or 4 are the most likely for single vs multi mask support
        num_instances: int = 1,  # Irrelevant if instance_query is False
        dac_use_selfatt_ln: bool = True,
        use_act_checkpoint: bool = False,
        compile_mode=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.fine_layers = (
            get_clones(interaction_layer, num_layers)
            if interaction_layer is not None
            else [None] * num_layers
        )
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dac = dac
        if dac:
            self.num_o2m_queries = num_queries
            tot_num_queries = num_queries
        else:
            self.num_o2m_queries = num_o2m_queries
            tot_num_queries = num_queries + num_o2m_queries
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(tot_num_queries, d_model)
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query = instance_query
        self.num_instances = num_instances
        if instance_query:
            self.instance_query_embed = nn.Embedding(num_instances, d_model)
        self.box_refine = box_refine
        if box_refine:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

            self.reference_points = nn.Embedding(num_queries, 4)
            if instance_query:
                self.instance_reference_points = nn.Embedding(num_instances, 4)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            try:
                nheads = self.layers[0].cross_attn_image.num_heads
            except AttributeError:
                nheads = self.layers[0].cross_attn.num_heads

            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = MLP(n_input, d_model, nheads, 2)
            self.boxRPB_embed_y = MLP(n_input, d_model, nheads, 2)
            self.compilable_cord_cache = None
            self.compilable_stored_size = None
            self.coord_cache = {}

        self.roi_pooler = (
            RoIAlign(output_size=7, spatial_scale=1, sampling_ratio=-1, aligned=True)
            if interaction_layer is not None
            else None
        )
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)
        self.dac_use_selfatt_ln = dac_use_selfatt_ln
        self.use_act_checkpoint = use_act_checkpoint

        nn.init.normal_(self.query_embed.weight.data)
        if self.instance_query_embed is not None:
            nn.init.normal_(self.instance_query_embed.weight.data)

        assert self.roi_pooler is None
        # TODO(chay): add support for turning off box_refine and return_intermediate
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"

        self.compile_mode = compile_mode
        self.compiled = False
        # We defer compilation till after the first forward, to first warm-up the boxRPB cache

        # assign layer index to each layer so that some layers can decide what to do
        # based on which layer index they are (e.g. cross attention to memory bank only
        # in selected layers)
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def _get_coords(H, W, device):
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(self, reference_boxes, feat_size):
        H, W = feat_size
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
        bs, num_queries, _ = boxes_xyxy.shape
        if self.compilable_cord_cache is None:
            self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
            self.compilable_stored_size = (H, W)

        if torch.compiler.is_dynamo_compiling() or self.compilable_stored_size == (
            H,
            W,
        ):
            # good, hitting the cache, will be compilable
            coords_h, coords_w = self.compilable_cord_cache
        else:
            # cache miss, will create compilation issue
            # In case we're not compiling, we'll still rely on the dict-based cache
            if feat_size not in self.coord_cache:
                self.coord_cache[feat_size] = self._get_coords(
                    H, W, reference_boxes.device
                )
            coords_h, coords_w = self.coord_cache[feat_size]

            assert coords_h.shape == (H,)
            assert coords_w.shape == (W,)

        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)

        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8  # normalize to -8, 8
            deltas_x_log = (
                torch.sign(deltas_x_log)
                * torch.log2(torch.abs(deltas_x_log) + 1.0)
                / np.log2(8)
            )

            deltas_y_log = deltas_y * 8  # normalize to -8, 8
            deltas_y_log = (
                torch.sign(deltas_y_log)
                * torch.log2(torch.abs(deltas_y_log) + 1.0)
                / np.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)

        deltas_x = activation_ckpt_wrapper(self.boxRPB_embed_x)(
            x=deltas_x,
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # bs, num_queries, W, n_heads
        deltas_y = activation_ckpt_wrapper(self.boxRPB_embed_y)(
            x=deltas_y,
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # bs, num_queries, H, n_heads

        if not torch.compiler.is_dynamo_compiling():
            assert deltas_x.shape[:3] == (bs, num_queries, W)
            assert deltas_y.shape[:3] == (bs, num_queries, H)

        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(
            2
        )  # bs, num_queries, H, W, n_heads
        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[:4] == (bs, num_queries, H, W)
        B = B.flatten(2, 3)  # bs, num_queries, H*W, n_heads
        B = B.permute(0, 3, 1, 2)  # bs, n_heads, num_queries, H*W
        B = B.contiguous()  # memeff attn likes ordered strides
        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[2:] == (num_queries, H * W)
        return B

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        reference_boxes: Optional[Tensor] = None,  # num_queries, bs, 4
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        # if `apply_dac` is None, it will default to `self.dac`
        apply_dac: Optional[bool] = None,
        decoder_extra_kwargs: Optional[Dict] = None,
        # ROI memory bank
        obj_roi_memory_feat=None,
        obj_roi_memory_mask=None,
        box_head_trk=None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: \\sum{hw}, bs, d_model
            - pos: \\sum{hw}, bs, d_model
            - reference_boxes: nq, bs, 4 (after sigmoid)
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        if memory_mask is not None:
            assert (
                self.boxRPB == "none"
            ), "inputting a memory_mask in the presence of boxRPB is unexpected/not implemented"

        apply_dac = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            assert (tgt.shape[0] == self.num_queries) or (
                self.use_instance_query
                and (tgt.shape[0] == self.instance_query_embed.num_embeddings)
            )

            tgt = tgt.repeat(2, 1, 1)
            # note that we don't tile tgt_mask, since DAC doesn't
            # use self-attention in o2m queries
            if reference_boxes is not None:
                assert (reference_boxes.shape[0] == self.num_queries) or (
                    self.use_instance_query
                    and (
                        reference_boxes.shape[0]
                        == self.instance_query_embed.num_embeddings
                    )
                )
                reference_boxes = reference_boxes.repeat(2, 1, 1)

        bs = tgt.shape[1]
        intermediate = []

        if self.box_refine:
            if reference_boxes is None:
                # In this case, we're in a one-stage model, so we generate the reference boxes
                reference_boxes = self.reference_points.weight.unsqueeze(1)
                reference_boxes = (
                    reference_boxes.repeat(2, bs, 1)
                    if apply_dac
                    else reference_boxes.repeat(1, bs, 1)
                )
                reference_boxes = reference_boxes.sigmoid()
            intermediate_ref_boxes = [reference_boxes]
        else:
            reference_boxes = None
            intermediate_ref_boxes = None

        output = tgt

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = (
                reference_boxes[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4

            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )  # nq, bs, d_model*2

            # conditional query
            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, d_model

            if self.boxRPB != "none" and reference_boxes is not None:
                assert (
                    spatial_shapes.shape[0] == 1
                ), "only single scale support implemented"
                memory_mask = self._get_rpb_matrix(
                    reference_boxes,
                    (spatial_shapes[0, 0], spatial_shapes[0, 1]),
                )
                memory_mask = memory_mask.flatten(0, 1)  # (bs*n_heads, nq, H*W)

            output = activation_ckpt_wrapper(layer)(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                dac=apply_dac,
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,
                **(decoder_extra_kwargs or {}),
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                # ROI memory bank
                obj_roi_memory_feat=obj_roi_memory_feat,
                obj_roi_memory_mask=obj_roi_memory_mask,
            )

            # iter update
            if self.box_refine:
                reference_before_sigmoid = inverse_sigmoid(reference_boxes)
                if box_head_trk is None:
                    delta_unsig = self.bbox_embed(output)
                else:
                    # box_head_trk use a separate box head for tracking queries
                    Q_det = decoder_extra_kwargs["Q_det"]
                    assert output.size(0) >= Q_det
                    delta_unsig_det = self.bbox_embed(output[:Q_det])
                    delta_unsig_trk = box_head_trk(output[Q_det:])
                    delta_unsig = torch.cat([delta_unsig_det, delta_unsig_trk], dim=0)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_boxes = new_reference_points.detach()
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                raise NotImplementedError("not implemented yet")

            intermediate.append(self.norm(output))

        if not self.compiled and self.compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=self.compile_mode, fullgraph=True
            )
            self.compiled = True

        return torch.stack(intermediate), torch.stack(intermediate_ref_boxes)

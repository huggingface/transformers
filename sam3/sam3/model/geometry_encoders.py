# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from typing_extensions import override

from .act_ckpt_utils import activation_ckpt_wrapper
from .box_ops import box_cxcywh_to_xyxy

from .encoder import TransformerEncoderLayer

from .model_misc import get_clones, NestedTensor


def is_right_padded(mask):
    """Given a padding mask (following pytorch convention, 1s for padded values),
    returns whether the padding is on the right or not."""
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """
    Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Following pytorch's convention, tensors are sequence first, and the mask are
    batch first, with 1s for padded values.

    :param seq1: A tensor of shape (seq1_length, batch_size, hidden_size).
    :param mask1: A tensor of shape (batch_size, seq1_length).
    :param seq2: A tensor of shape (seq2_length, batch_size,  hidden_size).
    :param mask2: A tensor of shape (batch_size, seq2_length).
    :param return_index: If True, also returns the index of the ids of the element of seq2
        in the concatenated sequence. This can be used to retrieve the elements of seq2
    :return: A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape

    assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
    assert hidden_size == seq1.size(2) == seq2.size(2)
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    torch._assert_async(is_right_padded(mask1))
    torch._assert_async(is_right_padded(mask2))

    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1)
        >= final_lengths[:, None]
    )

    # (max_len, batch_size, hidden_size)
    concatenated_sequence = torch.zeros(
        (max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype
    )
    concatenated_sequence[:seq1_length, :, :] = seq1

    # At this point, the element of seq1 are in the right place
    # We just need to shift the elements of seq2

    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]

    concatenated_sequence = concatenated_sequence.scatter(
        0, index[:, :, None].expand(-1, -1, hidden_size), seq2
    )

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


class Prompt:
    """Utility class to manipulate geometric prompts.

    We expect the sequences in pytorch convention, that is sequence first, batch second
    The dimensions are expected as follows:
    box_embeddings shape: N_boxes x B x C_box
    box_mask shape: B x N_boxes. Can be None if nothing is masked out
    point_embeddings shape: N_points x B x C_point
    point_mask shape: B x N_points. Can be None if nothing is masked out
    mask_embeddings shape: N_masks x B x 1 x H_mask x W_mask
    mask_mask shape: B x N_masks. Can be None if nothing is masked out

    We also store positive/negative labels. These tensors are also stored batch-first
    If they are None, we'll assume positive labels everywhere
    box_labels: long tensor of shape N_boxes x B
    point_labels: long tensor of shape N_points x B
    mask_labels: long tensor of shape N_masks x B
    """

    def __init__(
        self,
        box_embeddings=None,
        box_mask=None,
        point_embeddings=None,
        point_mask=None,
        box_labels=None,
        point_labels=None,
        mask_embeddings=None,
        mask_mask=None,  # Attention mask for mask prompt
        mask_labels=None,
    ):
        # Check for null prompt
        if (
            box_embeddings is None
            and point_embeddings is None
            and mask_embeddings is None
        ):
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            self.point_embeddings = None
            self.point_labels = None
            self.point_mask = None
            self.mask_embeddings = None
            self.mask_mask = None
            # Masks are assumed positive only for now.
            self.mask_labels = None
            return
        # Get sequence lengths and device
        box_seq_len, point_seq_len, mask_seq_len, bs, device = (
            self._init_seq_len_and_device(
                box_embeddings, point_embeddings, mask_embeddings
            )
        )

        # Initialize embeds, labels, attention masks.
        box_embeddings, box_labels, box_mask = self._init_box(
            box_embeddings, box_labels, box_mask, box_seq_len, bs, device
        )
        point_embeddings, point_labels, point_mask = self._init_point(
            point_embeddings, point_labels, point_mask, point_seq_len, bs, device
        )
        mask_embeddings, mask_labels, mask_mask = self._init_mask(
            mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device
        )

        # Dimension checks
        assert box_embeddings is not None and list(box_embeddings.shape[:2]) == [
            box_seq_len,
            bs,
        ], f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        assert box_mask is not None and list(box_mask.shape) == [
            bs,
            box_seq_len,
        ], f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        assert point_embeddings is not None and list(point_embeddings.shape[:2]) == [
            point_seq_len,
            bs,
        ], f"Wrong dimension for point embeddings. Expected [{point_seq_len}, {bs}, *] got {point_embeddings.shape}"
        assert point_mask is not None and list(point_mask.shape) == [
            bs,
            point_seq_len,
        ], f"Wrong dimension for point mask. Expected [{bs}, {point_seq_len}] got {point_mask.shape}"
        assert box_labels is not None and list(box_labels.shape) == [
            box_seq_len,
            bs,
        ], f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        assert point_labels is not None and list(point_labels.shape) == [
            point_seq_len,
            bs,
        ], f"Wrong dimension for point labels. Expected [{point_seq_len}, {bs}] got {point_labels.shape}"
        assert (
            # Allowed to be None, we leave it to the encoder to check for validity before encoding.
            mask_embeddings is None
            or list(mask_embeddings.shape[:2])
            == [
                mask_seq_len,
                bs,
            ]
        ), f"Wrong dimension for mask embeddings. Expected [{mask_seq_len}, {bs}, *] got {mask_embeddings.shape}"
        assert mask_mask is None or list(mask_mask.shape) == [
            bs,
            mask_seq_len,
        ], f"Wrong dimension for mask attn. mask. Expected [{bs}, {mask_seq_len}] got {mask_mask.shape}"

        # Device checks
        assert (
            box_embeddings is not None and box_embeddings.device == device
        ), f"Expected box embeddings to be on device {device}, got {box_embeddings.device}"
        assert (
            box_mask is not None and box_mask.device == device
        ), f"Expected box mask to be on device {device}, got {box_mask.device}"
        assert (
            box_labels is not None and box_labels.device == device
        ), f"Expected box labels to be on device {device}, got {box_labels.device}"
        assert (
            point_embeddings is not None and point_embeddings.device == device
        ), f"Expected point embeddings to be on device {device}, got {point_embeddings.device}"
        assert (
            point_mask is not None and point_mask.device == device
        ), f"Expected point mask to be on device {device}, got {point_mask.device}"
        assert (
            point_labels is not None and point_labels.device == device
        ), f"Expected point labels to be on device {device}, got {point_labels.device}"
        assert (
            mask_embeddings is None or mask_embeddings.device == device
        ), f"Expected mask embeddings to be on device {device}, got {mask_embeddings.device}"
        assert (
            mask_mask is None or mask_mask.device == device
        ), f"Expected mask attn. mask to be on device {device}, got {mask_mask.device}"

        self.box_embeddings = box_embeddings
        self.point_embeddings = point_embeddings
        self.box_mask = box_mask
        self.point_mask = point_mask
        self.box_labels = box_labels
        self.point_labels = point_labels
        self.mask_embeddings = mask_embeddings
        self.mask_labels = mask_labels
        self.mask_mask = mask_mask

    def _init_seq_len_and_device(
        self, box_embeddings, point_embeddings, mask_embeddings
    ):
        box_seq_len = point_seq_len = mask_seq_len = 0
        bs = None
        device = None
        if box_embeddings is not None:
            bs = box_embeddings.shape[1]
            box_seq_len = box_embeddings.shape[0]
            device = box_embeddings.device

        if point_embeddings is not None:
            point_seq_len = point_embeddings.shape[0]
            if bs is not None:
                assert (
                    bs == point_embeddings.shape[1]
                ), f"Batch size mismatch between box and point embeddings. Got {bs} and {point_embeddings.shape[1]}."
            else:
                bs = point_embeddings.shape[1]
            if device is not None:
                assert (
                    device == point_embeddings.device
                ), "Device mismatch between box and point embeddings"
            else:
                device = point_embeddings.device

        if mask_embeddings is not None:
            mask_seq_len = mask_embeddings.shape[0]
            if bs is not None:
                assert (
                    bs == mask_embeddings.shape[1]
                ), f"Batch size mismatch between box/point and mask embedding. Got {bs} and {mask_embeddings.shape[1]}"
            else:
                bs = mask_embeddings.shape[1]
            if device is not None:
                assert (
                    device == mask_embeddings.device
                ), "Device mismatch between box/point and mask embeddings."
            else:
                device = mask_embeddings.device

        return box_seq_len, point_seq_len, mask_seq_len, bs, device

    def _init_box(self, box_embeddings, box_labels, box_mask, box_seq_len, bs, device):
        if box_embeddings is None:
            box_embeddings = torch.zeros(box_seq_len, bs, 4, device=device)
        if box_labels is None:
            box_labels = torch.ones(box_seq_len, bs, device=device, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(bs, box_seq_len, device=device, dtype=torch.bool)
        return box_embeddings, box_labels, box_mask

    def _init_point(
        self, point_embeddings, point_labels, point_mask, point_seq_len, bs, device
    ):
        """
        Identical to _init_box. Except that C=2 for points (vs. 4 for boxes).
        """
        if point_embeddings is None:
            point_embeddings = torch.zeros(point_seq_len, bs, 2, device=device)
        if point_labels is None:
            point_labels = torch.ones(
                point_seq_len, bs, device=device, dtype=torch.long
            )
        if point_mask is None:
            point_mask = torch.zeros(bs, point_seq_len, device=device, dtype=torch.bool)
        return point_embeddings, point_labels, point_mask

    def _init_mask(
        self, mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device
    ):
        # NOTE: Mask embeddings can be of arbitrary resolution, so we don't initialize it here.
        # In case we append new mask, we check that its resolution matches exisiting ones (if any).
        # In case mask_embeddings is None, we should never encode it.
        if mask_labels is None:
            mask_labels = torch.ones(mask_seq_len, bs, device=device, dtype=torch.long)
        if mask_mask is None:
            mask_mask = torch.zeros(bs, mask_seq_len, device=device, dtype=torch.bool)
        return mask_embeddings, mask_labels, mask_mask

    def append_boxes(self, boxes, labels, mask=None):
        if self.box_embeddings is None:
            self.box_embeddings = boxes
            self.box_labels = labels
            self.box_mask = mask
            return

        bs = self.box_embeddings.shape[1]
        assert boxes.shape[1] == labels.shape[1] == bs
        assert list(boxes.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(
                bs, boxes.shape[0], dtype=torch.bool, device=boxes.device
            )

        self.box_labels, _ = concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask, labels.unsqueeze(-1), mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(
            self.box_embeddings, self.box_mask, boxes, mask
        )

    def append_points(self, points, labels, mask=None):
        if self.point_embeddings is None:
            self.point_embeddings = points
            self.point_labels = labels
            self.point_mask = mask
            return

        bs = self.point_embeddings.shape[1]
        assert points.shape[1] == labels.shape[1] == bs
        assert list(points.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(
                bs, points.shape[0], dtype=torch.bool, device=points.device
            )

        self.point_labels, _ = concat_padded_sequences(
            self.point_labels.unsqueeze(-1), self.point_mask, labels.unsqueeze(-1), mask
        )
        self.point_labels = self.point_labels.squeeze(-1)
        self.point_embeddings, self.point_mask = concat_padded_sequences(
            self.point_embeddings, self.point_mask, points, mask
        )

    def append_masks(self, masks, labels=None, attn_mask=None):
        if labels is not None:
            assert list(masks.shape[:2]) == list(labels.shape[:2])
        if self.mask_embeddings is None:
            self.mask_embeddings = masks
            mask_seq_len, bs = masks.shape[:2]
            if labels is None:
                self.mask_labels = torch.ones(
                    mask_seq_len, bs, device=masks.device, dtype=torch.long
                )
            else:
                self.mask_labels = labels
            if attn_mask is None:
                self.mask_mask = torch.zeros(
                    bs, mask_seq_len, device=masks.device, dtype=torch.bool
                )
            else:
                self.mask_mask = attn_mask
        else:
            raise NotImplementedError("Only one mask per prompt is supported.")

    def clone(self):
        return Prompt(
            box_embeddings=(
                None if self.box_embeddings is None else self.box_embeddings.clone()
            ),
            box_mask=None if self.box_mask is None else self.box_mask.clone(),
            point_embeddings=(
                None if self.point_embeddings is None else self.point_embeddings.clone()
            ),
            point_mask=None if self.point_mask is None else self.point_mask.clone(),
            box_labels=None if self.box_labels is None else self.box_labels.clone(),
            point_labels=(
                None if self.point_labels is None else self.point_labels.clone()
            ),
        )


class MaskEncoder(nn.Module):
    """
    Base class for mask encoders.
    """

    def __init__(
        self,
        mask_downsampler: nn.Module,
        position_encoding: nn.Module,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.position_encoding = position_encoding

    def forward(self, masks, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = self.mask_downsampler(masks)
        masks_nt = NestedTensor(
            tensors=masks,
            mask=None,
        )
        masks_pos = self.position_encoding(masks_nt).to(masks_nt.tensors.dtype)

        return masks, masks_pos


class FusedMaskEncoder(MaskEncoder):
    """
    Identical to memory.SimpleMaskEncoder but follows the interface of geometry_encoders.MaskEncoder.
    We also remove the `skip_mask_sigmoid` option (to be handled outside the MaskEncoder).
    Fuses backbone image features with mask features.
    """

    def __init__(
        self,
        mask_downsampler: nn.Module,
        position_encoding: nn.Module,
        fuser: nn.Module,
        in_dim: int = 256,
        out_dim: int = 256,
    ):
        super().__init__(mask_downsampler, position_encoding)
        self.fuser = fuser
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    @override
    def forward(
        self,
        masks: torch.Tensor,
        pix_feat: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        x_pos = NestedTensor(
            tensors=x,
            mask=None,
        )
        pos = self.position_encoding(x_pos).to(x_pos.tensors.dtype)

        return x, pos


class SequenceGeometryEncoder(nn.Module):
    """
    This a fully fledged encoder for geometric prompts.
    It assumes boxes are passed in the "normalized CxCyWH" format, and points in normalized xy
    This allows flexibility in how to encode the features (eg do pooling)

    Points and boxes can be encoded with any of the three possibilities:
     - direct projection: we just compute a linear from coordinate space to d_model
     - pooling: pool features from the backbone in the requested location.
                For boxes, it's a roi align
                For points it's a grid sample
     - pos encoder: Take the position encoding of the point or box center

    These three options are mutually compatible. If several are selected, we'll take a simple addition

    As an alternative, we offer the possibility to encode points only.
    In that case, the boxes are converted to two points for the top left and bottom right corners (with appropriate labels)

    On top of these encodings, we offer the possibility to further encode the prompt sequence with a transformer.
    """

    def __init__(
        self,
        encode_boxes_as_points: bool,
        points_direct_project: bool,
        points_pool: bool,
        points_pos_enc: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc,
        num_layers: int,
        layer: nn.Module,
        roi_size: int = 7,  # for boxes pool
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        mask_encoder: MaskEncoder = None,
        add_mask_label: bool = False,
        use_act_ckpt: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size
        # There usually are two labels: positive and negatives.
        # If we encode boxes as points, we have 3 types of points: regular, top left, bottom right
        # These 3 types can be positives or negatives, hence 2*3 = 6 labels
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = torch.nn.Embedding(num_labels, self.d_model)

        # This is a cls token, can be used for pooling if need be.
        # It also ensures that the encoded sequences are always non-empty
        self.cls_embed = None
        if add_cls:
            self.cls_embed = torch.nn.Embedding(1, self.d_model)

        assert (
            points_direct_project or points_pos_enc or points_pool
        ), "Error: need at least one way to encode points"
        assert (
            encode_boxes_as_points
            or boxes_direct_project
            or boxes_pos_enc
            or boxes_pool
        ), "Error: need at least one way to encode boxes"

        self.points_direct_project = None
        if points_direct_project:
            self.points_direct_project = nn.Linear(2, self.d_model)
        self.points_pool_project = None
        if points_pool:
            self.points_pool_project = nn.Linear(self.d_model, self.d_model)
        self.points_pos_enc_project = None
        if points_pos_enc:
            self.points_pos_enc_project = nn.Linear(self.d_model, self.d_model)

        self.boxes_direct_project = None
        self.boxes_pool_project = None
        self.boxes_pos_enc_project = None
        if not encode_boxes_as_points:
            if boxes_direct_project:
                self.boxes_direct_project = nn.Linear(4, self.d_model)
            if boxes_pool:
                self.boxes_pool_project = nn.Conv2d(
                    self.d_model, self.d_model, self.roi_size
                )
            if boxes_pos_enc:
                self.boxes_pos_enc_project = nn.Linear(self.d_model + 2, self.d_model)

        self.final_proj = None
        if add_post_encode_proj:
            self.final_proj = nn.Linear(self.d_model, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)

        self.img_pre_norm = nn.Identity()
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = nn.LayerNorm(self.d_model)

        self.encode = None
        if num_layers > 0:
            assert (
                add_cls
            ), "It's currently highly recommended to add a CLS when using a transformer"
            self.encode = get_clones(layer, num_layers)
            self.encode_norm = nn.LayerNorm(self.d_model)

        if mask_encoder is not None:
            assert isinstance(
                mask_encoder, MaskEncoder
            ), f"Expected mask_encoder of type MaskEncoder. Got {type(mask_encoder)}."
            if add_mask_label:
                self.mask_label_embed = torch.nn.Embedding(2, self.d_model)
        self.add_mask_label = add_mask_label
        self.mask_encoder = mask_encoder
        self.use_act_ckpt = use_act_ckpt

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        n_points, bs = points.shape[:2]

        if self.points_direct_project is not None:
            proj = self.points_direct_project(points)
            assert points_embed is None
            points_embed = proj

        if self.points_pool_project is not None:
            # points are [Num_points, bs, 2], normalized in [0, 1]
            # the grid needs to be [Bs, H_out, W_out, 2] normalized in [-1,1]
            # Will take H_out = num_points, w_out = 1
            grid = points.transpose(0, 1).unsqueeze(2)
            # re normalize to [-1, 1]
            grid = (grid * 2) - 1
            sampled = torch.nn.functional.grid_sample(img_feats, grid)
            assert list(sampled.shape) == [bs, self.d_model, n_points, 1]
            sampled = sampled.squeeze(-1).permute(2, 0, 1)
            proj = self.points_pool_project(sampled)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        if self.points_pos_enc_project is not None:
            x, y = points.unbind(-1)
            enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
            enc_x = enc_x.view(n_points, bs, enc_x.shape[-1])
            enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
            enc = torch.cat([enc_x, enc_y], -1)

            proj = self.points_pos_enc_project(enc)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]

        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes)
            assert boxes_embed is None
            boxes_embed = proj

        if self.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]

            # boxes are [Num_boxes, bs, 4], normalized in [0, 1]
            # We need to denormalize, and convert to [x, y, x, y]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor(
                [W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device
            ).view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale
            sampled = torchvision.ops.roi_align(
                img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size
            )
            assert list(sampled.shape) == [
                bs * n_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(
                cx.flatten(), cy.flatten(), w.flatten(), h.flatten()
            )
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

            proj = self.boxes_pos_enc_project(enc)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    def _encode_masks(
        self,
        masks: torch.Tensor,
        attn_mask: torch.Tensor,
        mask_labels: torch.Tensor,
        img_feats: torch.Tensor = None,
    ):
        n_masks, bs = masks.shape[:2]
        assert (
            n_masks == 1
        ), "We assume one mask per prompt for now. Code should still be functional if this assertion is removed."
        assert list(attn_mask.shape) == [
            bs,
            n_masks,
        ], f"Expected attn_mask to be of shape {bs}x{n_masks}. Got {list(attn_mask.shape)}."
        masks, pos = self.mask_encoder(
            masks=masks.flatten(0, 1).float(),
            pix_feat=img_feats,
        )
        H, W = masks.shape[-2:]
        n_tokens_per_mask = H * W
        # NOTE: We directly add pos enc here as we usually don't keep track of pos encoding
        # for the concatenated prompt (text, other geometric prompts).
        # Might need to do some refactoring for more flexibility.
        masks = masks + pos
        masks = masks.view(n_masks, bs, *masks.shape[1:]).flatten(
            -2
        )  # n_masks x bs x C x H*W
        masks = masks.permute(0, 3, 1, 2).flatten(0, 1)  # n_masks * H*W x bs x C
        attn_mask = attn_mask.repeat_interleave(n_tokens_per_mask, dim=1)
        if self.add_mask_label:
            masks = masks + self.mask_label_embed(mask_labels.long())
        return masks, attn_mask

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        points = geo_prompt.point_embeddings
        points_mask = geo_prompt.point_mask
        points_labels = geo_prompt.point_labels
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels
        masks = geo_prompt.mask_embeddings
        masks_mask = geo_prompt.mask_mask
        masks_labels = geo_prompt.mask_labels
        seq_first_img_feats = img_feats[-1]  # [H*W, B, C]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1]
            if img_pos_embeds is not None
            else torch.zeros_like(seq_first_img_feats)
        )

        if self.points_pool_project or self.boxes_pool_project:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = self.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            # Put back in NxCxHxW
            cur_img_feat = cur_img_feat.permute(1, 2, 0)
            cur_img_feat = cur_img_feat.view(N, C, H, W)
            img_feats = cur_img_feat

        if self.encode_boxes_as_points:
            assert boxes is not None
            assert geo_prompt.box_mask is not None
            assert geo_prompt.box_labels is not None
            assert boxes.shape[-1] == 4

            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            top_left, bottom_right = boxes_xyxy.split(split_size=2, dim=-1)

            labels_tl = geo_prompt.box_labels + 2
            labels_br = geo_prompt.box_labels + 4

            # Append to the existing points
            points, _ = concat_padded_sequences(
                points, points_mask, top_left, boxes_mask
            )
            points_labels, points_mask = concat_padded_sequences(
                points_labels.unsqueeze(-1),
                points_mask,
                labels_tl.unsqueeze(-1),
                boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)

            points, _ = concat_padded_sequences(
                points, points_mask, bottom_right, boxes_mask
            )
            points_labels, points_mask = concat_padded_sequences(
                points_labels.unsqueeze(-1),
                points_mask,
                labels_br.unsqueeze(-1),
                boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)

        final_embeds, final_mask = self._encode_points(
            points=points,
            points_mask=points_mask,
            points_labels=points_labels,
            img_feats=img_feats,
        )

        if not self.encode_boxes_as_points:
            boxes_embeds, boxes_mask = self._encode_boxes(
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=img_feats,
            )

            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, boxes_embeds, boxes_mask
            )

        if masks is not None and self.mask_encoder is not None:
            masks_embed, masks_mask = self._encode_masks(
                masks=masks,
                attn_mask=masks_mask,
                mask_labels=masks_labels,
                img_feats=img_feats,
            )
            if points.size(0) == boxes.size(0) == 0:
                return masks_embed, masks_mask
        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs
        if self.cls_embed is not None:
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(
                bs, 1, dtype=final_mask.dtype, device=final_mask.device
            )
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, cls, cls_mask
            )

        if self.final_proj is not None:
            final_embeds = self.norm(self.final_proj(final_embeds))

        if self.encode is not None:
            for lay in self.encode:
                if isinstance(
                    lay, TransformerEncoderLayer
                ):  # Note: TransformerDecoderLayer renamed for code release
                    final_embeds = activation_ckpt_wrapper(lay)(
                        tgt=final_embeds,
                        memory=seq_first_img_feats,
                        tgt_key_padding_mask=final_mask,
                        pos=seq_first_img_pos_embeds,
                        act_ckpt_enable=self.training and self.use_act_ckpt,
                    )
                else:
                    raise NotImplementedError(
                        f"Only `TransformerEncoderLayer` or `TransformerDecoderLayer` are supported. Got {type(lay)}"
                    )

            final_embeds = self.encode_norm(final_embeds)
        # Finally, concat mask embeddings if any
        if masks is not None and self.mask_encoder is not None:
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, masks_embed, masks_mask
            )
        return final_embeds, final_mask

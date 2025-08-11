# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from torch.nn import MultiheadAttention

from .model.decoder import TransformerDecoder, TransformerDecoderLayer
from .model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .model.geometry_encoders import FusedMaskEncoder, SequenceGeometryEncoder
from .model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .model.memory import CXBlock, SimpleFuser, SimpleMaskDownSampler
from .model.model_misc import DotProductScoring, MLP, TransformerWrapper
from .model.necks import OriginalViTDetNeck
from .model.position_encoding import PositionEmbeddingSine
from .model.sam3_demo import Sam3ImageInteractiveDemo
from .model.text_encoder_ve import VETextEncoder
from .model.tokenizer_ve import SimpleTokenizer
from .model.vitdet import ViT
from .model.vl_combiner import NonFusionVLBackbone


def build_sam3_image_model(
    bpe_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
):
    """
    This function replaces the Hydra-based configuration in sam3_image_v1.4.yaml
    for image - only setting.

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary

    Returns:
        A SAM3 image model for interactive segmentation
    """
    # Create position encoding for visual backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000
    )

    # Create ViT backbone
    vit_backbone = ViT(
        img_size=1008,
        # weights_path=None,
        # freezing="NoFreeze",
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        # use_tiled_rope=False,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        # use_act_checkpoint=True,
        compile_mode="default",
    )

    # Create ViT neck
    vit_neck = OriginalViTDetNeck(
        position_encoding=position_encoding,
        # neck_norm=None,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
    )

    # Create text tokenizer
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)

    # Create text encoder
    text_encoder = VETextEncoder(
        # frozen=False,
        tokenizer=tokenizer,
        d_model=256,
        # weights_path=None,
        width=1024,
        heads=16,
        layers=24,
        # use_act_checkpoint=True
    )

    # Create visual-language backbone
    backbone = NonFusionVLBackbone(visual=vit_neck, text=text_encoder, scalp=1)

    # Create transformer encoder layer
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    # Create transformer encoder
    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )

    # Create transformer decoder layer
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,  #  attn_type=AttentionType.Vanilla,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    # Create transformer decoder
    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        use_act_checkpoint=True,
        instance_query=True,
        num_instances=4,
    )

    # Create transformer
    transformer = TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)

    # Create MLP for dot product scorer
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )

    # Create dot product scorer
    dot_prod_scoring = DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)

    # Create pixel decoder for segmentation head
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode="default",
    )

    # Create cross attention for segmentation head
    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,  # attn_type=AttentionType.Vanilla,
    )

    # Create segmentation head
    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=True,
        dot_product_scorer=DotProductScoring(
            d_model=256,
            d_proj=256,
            prompt_mlp=MLP(
                input_dim=256,
                hidden_dim=2048,
                output_dim=256,
                num_layers=2,
                dropout=0.1,
                residual=True,
                out_norm=nn.LayerNorm(256),
            ),
        ),
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )

    # Create position encoding for geometry encoder
    geo_pos_enc = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000
    )

    # Create mask downsampler
    mask_downsampler = SimpleMaskDownSampler(
        interpol_size=[288, 288], kernel_size=3, stride=2, padding=1, total_stride=4
    )

    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    # Create fuser
    fuser = SimpleFuser(layer=cx_block, num_layers=2)

    # Create mask encoder
    mask_encoder = FusedMaskEncoder(
        out_dim=256,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=1008,
        ),
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
        mask_encoder=mask_encoder,
    )

    # Create the SAM3 model
    model = Sam3ImageInteractiveDemo(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=True,
        multimask_output=True,
    )

    # move to eval mode
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=True)

    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()

    return model

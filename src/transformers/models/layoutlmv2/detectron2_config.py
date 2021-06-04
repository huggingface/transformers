# -*- coding: utf-8 -*-
def add_layoutlmv2_config(cfg):
    _C = cfg
    # -----------------------------------------------------------------------------
    # Config definition
    # -----------------------------------------------------------------------------
    _C.MODEL.MASK_ON = True

    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    _C.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]

    # ---------------------------------------------------------------------------- #
    # Backbone options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"

    # ---------------------------------------------------------------------------- #
    # FPN options
    # ---------------------------------------------------------------------------- #
    # Names of the input feature maps to be used by FPN
    # They must have contiguous power of 2 strides
    # e.g., ["res2", "res3", "res4", "res5"]
    _C.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]

    # ---------------------------------------------------------------------------- #
    # Anchor generator options
    # ---------------------------------------------------------------------------- #
    # Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
    # Format: list[list[float]]. SIZES[i] specifies the list of sizes
    # to use for IN_FEATURES[i]; len(SIZES) == len(IN_FEATURES) must be true,
    # or len(SIZES) == 1 is true and size list SIZES[0] is used for all
    # IN_FEATURES.
    _C.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]

    # ---------------------------------------------------------------------------- #
    # RPN options
    # ---------------------------------------------------------------------------- #
    # Names of the input feature maps to be used by RPN
    # e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
    _C.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    # Number of top scoring RPN proposals to keep before applying NMS
    # When FPN is used, this is *per FPN level* (not total)
    _C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    _C.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    # Number of top scoring RPN proposals to keep after applying NMS
    # When FPN is used, this limit is applied per level and then again to the union
    # of proposals from all levels
    # NOTE: When FPN is used, the meaning of this config is different from Detectron1.
    # It means per-batch topk in Detectron1, but per-image topk here.
    # See the "find_top_rpn_proposals" function for details.
    _C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    _C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    # ---------------------------------------------------------------------------- #
    # ROI HEADS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    # Number of foreground classes
    _C.MODEL.ROI_HEADS.NUM_CLASSES = 5
    # Names of the input feature maps to be used by ROI heads
    # Currently all heads (box, mask, ...) use the same input feature map list
    # e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
    _C.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    # ---------------------------------------------------------------------------- #
    # Box Head
    # ---------------------------------------------------------------------------- #
    # C4 don't use head name option
    # Options for non-C4 models: FastRCNNConvFCHead,
    _C.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    _C.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    _C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14

    # ---------------------------------------------------------------------------- #
    # Mask Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    _C.MODEL.ROI_MASK_HEAD.NUM_CONV = 4  # The number of convs in the mask head
    _C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 7

    # ---------------------------------------------------------------------------- #
    # ResNe[X]t options (ResNets = {ResNet, ResNeXt}
    # Note that parts of a resnet may be used for both the backbone and the head
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RESNETS.DEPTH = 101
    _C.MODEL.RESNETS.SIZES = [[32], [64], [128], [256], [512]]
    _C.MODEL.RESNETS.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    _C.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]  # res4 for C4 backbone, res2..5 for FPN backbone

    # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
    _C.MODEL.RESNETS.NUM_GROUPS = 32

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    _C.MODEL.RESNETS.WIDTH_PER_GROUP = 8

    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    _C.MODEL.RESNETS.STRIDE_IN_1X1 = False

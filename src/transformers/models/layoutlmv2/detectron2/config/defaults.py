# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)
# Sample size of smallest side by choice or random selection from range give by
# INPUT.MIN_SIZE_TRAIN
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Mode for flipping images used in data augmentation during training
# choose one of ["horizontal, "vertical", "none"]
_C.INPUT.RANDOM_FLIP = "horizontal"

# `True` if cropping is used for data augmentation during training
_C.INPUT.CROP = CN({"ENABLED": False})
# Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
_C.INPUT.CROP.TYPE = "relative_range"
# Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
# pixels if CROP.TYPE is "absolute"
_C.INPUT.CROP.SIZE = [0.9, 0.9]


# Whether the model needs RGB, YUV, HSV etc.
# Should be one of the modes defined here, as we use PIL to read the image:
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# with BGR being the one exception. One can set image format to BGR, we will
# internally use RGB for conversion and flip the channels over
_C.INPUT.FORMAT = "BGR"
# The ground truth mask format that the model will use.
# Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
_C.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
# Samples from these datasets will be merged and used as one dataset.
_C.DATASETS.TRAIN = ()
# List of the pre-computed proposal files for training, which must be consistent
# with datasets listed in DATASETS.TRAIN.
_C.DATASETS.PROPOSAL_FILES_TRAIN = ()
# Number of top scoring precomputed proposals to keep for training
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()
# List of the pre-computed proposal files for test, which must be consistent
# with datasets listed in DATASETS.TEST.
_C.DATASETS.PROPOSAL_FILES_TEST = ()
# Number of top scoring precomputed proposals to keep for test
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
# Tf True, when working on datasets that have instance annotations, the
# training dataloader will filter out images without associated annotations
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
_C.MODEL.BACKBONE.FREEZE_AT = 2


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.IN_FEATURES = []
_C.MODEL.FPN.OUT_CHANNELS = 256

# Options: "" (no norm), "GN"
_C.MODEL.FPN.NORM = ""

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.FPN.FUSE_TYPE = "sum"


# ---------------------------------------------------------------------------- #
# Proposal generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.PROPOSAL_GENERATOR = CN()
# Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# Proposal height and width both need to be greater than MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0


# ---------------------------------------------------------------------------- #
# Anchor generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.ANCHOR_GENERATOR = CN()
# The generator can be any name in the ANCHOR_GENERATOR registry
_C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
# Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
# Format: list[list[float]]. SIZES[i] specifies the list of sizes to use for
# IN_FEATURES[i]; len(SIZES) must be equal to len(IN_FEATURES) or 1.
# When len(SIZES) == 1, SIZES[0] is used for all IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# Anchor aspect ratios. For each area given in `SIZES`, anchors with different aspect
# ratios are generated by an anchor generator.
# Format: list[list[float]]. ASPECT_RATIOS[i] specifies the list of aspect ratios (H/W)
# to use for IN_FEATURES[i]; len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
# or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0] is used
# for all IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
# Anchor angles.
# list[list[float]], the angle in degrees, for each input feature map.
# ANGLES[i] specifies the list of angles for IN_FEATURES[i].
_C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
# Relative offset between the center of the first anchor and the top-left corner of the image
# Value has to be in [0, 1). Recommend to use 0.5, which means half stride.
# The value is not expected to affect model accuracy.
_C.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY

# Names of the input feature maps to be used by RPN
# e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
_C.MODEL.RPN.IN_FEATURES = ["res4"]
# Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.BOUNDARY_THRESH = -1
# IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example: 1)
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example: 0)
# Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
# are ignored (-1)
_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
# Number of regions per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Options are: "smooth_l1", "giou"
_C.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
_C.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
# Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
_C.MODEL.RPN.LOSS_WEIGHT = 1.0
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See the "find_top_rpn_proposals" function for details.
_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Set this to -1 to use the same number of output channels as input channels.
_C.MODEL.RPN.CONV_DIMS = [-1]

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
# Number of foreground classes
_C.MODEL.ROI_HEADS.NUM_CLASSES = 80
# Names of the input feature maps to be used by ROI heads
# Currently all heads (box, mask, ...) use the same input feature map list
# e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
_C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 16 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# inference.
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# If True, augment proposals with ground-truth boxes before sampling proposals to
# train ROI heads.
_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True

# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()
# C4 don't use head name option
# Options for non-C4 models: FastRCNNConvFCHead,
_C.MODEL.ROI_BOX_HEAD.NAME = ""
# Options are: "smooth_l1", "giou"
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
# The final scaling coefficient on the box regression loss, used to balance the magnitude of its
# gradients with other losses in the model. See also `MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT`.
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0
# Hidden layer dimension for FC layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
# Channel dimension for Conv layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_BOX_HEAD.NORM = ""
# Whether to use class agnostic for bbox regression
_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
# If true, RoI heads use bounding boxes predicted by the box head rather than proposal boxes.
_C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False

# ---------------------------------------------------------------------------- #
# Cascaded Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_CASCADE_HEAD = CN()
# The number of cascade stages is implicitly defined by the length of the following two configs.
_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)


# ---------------------------------------------------------------------------- #
# Mask Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  # The number of convs in the mask head
_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_MASK_HEAD.NORM = ""
# Whether to use class agnostic for mask prediction
_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"


# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  # 17 is the number of keypoints in COCO.

# Images with too few (or no) keypoints are excluded from training.
_C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
# Normalize by the total number of visible keypoints in the minibatch if True.
# Otherwise, normalize by the total number of keypoints that could ever exist
# in the minibatch.
# The keypoint softmax loss is only calculated on visible keypoints.
# Since the number of visible keypoints can vary significantly between
# minibatches, this has the effect of up-weighting the importance of
# minibatches with few visible keypoints. (Imagine the extreme case of
# only one visible keypoint versus N: in the case of N, each one
# contributes 1/N to the gradient compared to the single keypoint
# determining the gradient direction). Instead, we can normalize the
# loss by the total number of keypoints, if it were the case that all
# keypoints were visible in a full minibatch. (Returning to the example,
# this means that the one visible keypoint contributes as much as each
# of the N keypoints.)
_C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
# Multi-task loss weight to use for keypoints
# Recommended values:
#   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
#   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
_C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"

# ---------------------------------------------------------------------------- #
# Semantic Segmentation Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SEM_SEG_HEAD = CN()
_C.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
_C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# the correposnding pixel.
_C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
# Number of classes in the semantic segmentation head
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54
# Number of channels in the 3x3 convs inside semantic-FPN heads.
_C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
# Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
_C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
_C.MODEL.SEM_SEG_HEAD.NORM = "GN"
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0

_C.MODEL.PANOPTIC_FPN = CN()
# Scaling of all losses from instance detection / segmentation head.
_C.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0

# options when combining instance & semantic segmentation outputs
_C.MODEL.PANOPTIC_FPN.COMBINE = CN({"ENABLED": True})  # "COMBINE.ENABLED" is deprecated & not used
_C.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
_C.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
_C.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5


# ---------------------------------------------------------------------------- #
# RetinaNet Head
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes.
_C.MODEL.RETINANET.NUM_CLASSES = 80

_C.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
_C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
# Select topk candidates before NMS
_C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

# Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
_C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Loss parameters
_C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1
# Options are: "smooth_l1", "giou"
_C.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"

# One of BN, SyncBN, FrozenBN, GN
# Only supports GN until unshared norm is implemented
_C.MODEL.RETINANET.NORM = ""


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.RESNETS.NORM = "FrozenBN"

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# For R18 and R34, this needs to be set to 64
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# Apply Deformable Convolution in stages
# Specify if apply deform_conv on Res2, Res3, Res4, Res5
_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
# Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
# Use False for DeformableV1.
_C.MODEL.RESNETS.DEFORM_MODULATED = False
# Number of groups in deformable conv.
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# See detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration). If we use 16 GPUs
# and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.SOLVER.IMS_PER_BATCH = 16

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.AMP = CN({"ENABLED": False})

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0
# The sigmas used to calculate keypoint OKS. See http://cocodataset.org/#keypoints-eval
# When empty, it will use the defaults in COCO.
# Otherwise it should be a list[float] with the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
_C.TEST.KEYPOINT_OKS_SIGMAS = []
# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
_C.TEST.DETECTIONS_PER_IMAGE = 100

_C.TEST.AUG = CN({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
_C.TEST.AUG.MAX_SIZE = 4000
_C.TEST.AUG.FLIP = True

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0

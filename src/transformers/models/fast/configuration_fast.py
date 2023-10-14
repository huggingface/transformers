from transformers import PretrainedConfig


class FastConfig(PretrainedConfig):
    def __init__(
            self,
            backbone_kernel_size=3,
            backbone_stride=2,
            backbone_dilation=1,
            backbone_groups=1,
            backbone_bias=False,
            backbone_has_shuffle=False,
            backbone_in_channels=3,
            backbone_out_channels=64,
            backbone_use_bn=True,
            backbone_act_func="relu",
            backbone_dropout_rate=0,
            backbone_ops_order="weight_bn_act",
            backbone_stage1_in_channels=[64, 64, 64],
            backbone_stage1_out_channels=[64, 64, 64],
            backbone_stage1_kernel_size=[[3, 3], [3, 3], [3, 3]],
            backbone_stage1_stride=[1, 2, 1],
            backbone_stage1_dilation=[1, 1, 1],
            backbone_stage1_groups=[1, 1, 1],
            backbone_stage2_in_channels=[64, 128, 128, 128],
            backbone_stage2_out_channels=[128, 128, 128, 128],
            backbone_stage2_kernel_size=[[3, 3], [1, 3], [3, 3], [3, 1]],
            backbone_stage2_stride=[2, 1, 1, 1],
            backbone_stage2_dilation=[1, 1, 1, 1],
            backbone_stage2_groups=[1, 1, 1, 1],
            backbone_stage3_in_channels=[128, 256, 256, 256],
            backbone_stage3_out_channels=[256, 256, 256, 256],
            backbone_stage3_kernel_size=[[3, 3], [3, 3], [3, 1], [1, 3]],
            backbone_stage3_stride=[2, 1, 1, 1],
            backbone_stage3_dilation=[1, 1, 1, 1],
            backbone_stage3_groups=[1, 1, 1, 1],
            backbone_stage4_in_channels=[256, 512, 512, 512],
            backbone_stage4_out_channels=[512, 512, 512, 512],
            backbone_stage4_kernel_size=[[3, 3], [3, 1], [1, 3], [3, 3]],
            backbone_stage4_stride=[2, 1, 1, 1],
            backbone_stage4_dilation=[1, 1, 1, 1],
            backbone_stage4_groups=[1, 1, 1, 1],
            neck_in_channels=[64, 128, 256, 512],
            neck_out_channels=[128, 128, 128, 128],
            neck_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3]],
            neck_stride=[1, 1, 1, 1],
            neck_dilation=[1, 1, 1, 1],
            neck_groups=[1, 1, 1, 1],
            head_pooling_size=9,
            head_dropout_ratio=0.1,
            head_conv_in_channels=512,
            head_conv_out_channels=128,
            head_conv_kernel_size=[3, 3],
            head_conv_stride=1,
            head_conv_dilation=1,
            head_conv_groups=1,
            head_final_kernel_size=1,
            head_final_stride=1,
            head_final_dilation=1,
            head_final_groups=1,
            head_final_bias=False,
            head_final_has_shuffle=False,
            head_final_in_channels=128,
            head_final_out_channels=5,
            head_final_use_bn=False,
            head_final_act_func=None,
            head_final_dropout_rate=0,
            head_final_ops_order="weight",
            min_area=250,
            min_score=0.88,
            bbox_type='rect',
            loss_bg=False,
            initializer_range=0.02,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_kernel_size = backbone_kernel_size
        self.backbone_stride = backbone_stride
        self.backbone_dilation = backbone_dilation
        self.backbone_groups = backbone_groups
        self.backbone_bias = backbone_bias
        self.backbone_has_shuffle = backbone_has_shuffle
        self.backbone_in_channels = backbone_in_channels
        self.backbone_out_channels = backbone_out_channels
        self.backbone_use_bn = backbone_use_bn
        self.backbone_act_func = backbone_act_func
        self.backbone_dropout_rate = backbone_dropout_rate
        self.backbone_ops_order = backbone_ops_order

        self.backbone_stage1_in_channels = backbone_stage1_in_channels
        self.backbone_stage1_out_channels = backbone_stage1_out_channels
        self.backbone_stage1_kernel_size = backbone_stage1_kernel_size
        self.backbone_stage1_stride = backbone_stage1_stride
        self.backbone_stage1_dilation = backbone_stage1_dilation
        self.backbone_stage1_groups = backbone_stage1_groups

        self.backbone_stage2_in_channels = backbone_stage2_in_channels
        self.backbone_stage2_out_channels = backbone_stage2_out_channels
        self.backbone_stage2_kernel_size = backbone_stage2_kernel_size
        self.backbone_stage2_stride = backbone_stage2_stride
        self.backbone_stage2_dilation = backbone_stage2_dilation
        self.backbone_stage2_groups = backbone_stage2_groups

        self.backbone_stage3_in_channels = backbone_stage3_in_channels
        self.backbone_stage3_out_channels = backbone_stage3_out_channels
        self.backbone_stage3_kernel_size = backbone_stage3_kernel_size
        self.backbone_stage3_stride = backbone_stage3_stride
        self.backbone_stage3_dilation = backbone_stage3_dilation
        self.backbone_stage3_groups = backbone_stage3_groups

        self.backbone_stage4_in_channels = backbone_stage4_in_channels
        self.backbone_stage4_out_channels = backbone_stage4_out_channels
        self.backbone_stage4_kernel_size = backbone_stage4_kernel_size
        self.backbone_stage4_stride = backbone_stage4_stride
        self.backbone_stage4_dilation = backbone_stage4_dilation
        self.backbone_stage4_groups = backbone_stage4_groups

        self.neck_in_channels = neck_in_channels
        self.neck_out_channels = neck_out_channels
        self.neck_kernel_size = neck_kernel_size
        self.neck_stride = neck_stride
        self.neck_dilation = neck_dilation
        self.neck_groups = neck_groups

        self.head_pooling_size = head_pooling_size
        self.head_dropout_ratio = head_dropout_ratio

        self.head_conv_in_channels = head_conv_in_channels
        self.head_conv_out_channels = head_conv_out_channels
        self.head_conv_kernel_size = head_conv_kernel_size
        self.head_conv_stride = head_conv_stride
        self.head_conv_dilation = head_conv_dilation
        self.head_conv_groups = head_conv_groups

        self.head_final_kernel_size = head_final_kernel_size
        self.head_final_stride = head_final_stride
        self.head_final_dilation = head_final_dilation
        self.head_final_groups = head_final_groups
        self.head_final_bias = head_final_bias
        self.head_final_has_shuffle = head_final_has_shuffle
        self.head_final_in_channels = head_final_in_channels
        self.head_final_out_channels = head_final_out_channels
        self.head_final_use_bn = head_final_use_bn
        self.head_final_act_func = head_final_act_func
        self.head_final_dropout_rate = head_final_dropout_rate
        self.head_final_ops_order = head_final_ops_order

        self.min_area = min_area
        self.min_score = min_score
        self.bbox_type = bbox_type
        self.loss_bg = loss_bg
        self.initializer_range = initializer_range
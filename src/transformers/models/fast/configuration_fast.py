from transformers import PretrainedConfig


class FastConfig(PretrainedConfig):

    def __init__(self,
                 backbone_config=None,
                 backbone_stage1_in_channels=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
                 backbone_stage1_out_channels=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
                 backbone_stage1_kernel_size=[(3, 3), (3, 3), (3, 1), (3, 3), (3, 1), (3, 3), (3, 3), (1, 3), (3, 3),
                                              (3, 3)],
                 backbone_stage1_stride=[1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage1_dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage1_groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

                 backbone_stage2_in_channels=[64, 128, 128, 128, 128, 128, 128, 128, 128, 128],
                 backbone_stage2_out_channels=[128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
                 backbone_stage2_kernel_size=[(3, 3), (3, 3), (3, 1), (3, 3), (3, 1), (3, 3), (3, 3), (1, 3), (3, 3),
                                              (3, 3)],
                 backbone_stage2_stride=[1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage2_dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage2_groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

                 backbone_stage3_in_channels=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
                 backbone_stage3_out_channels=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
                 backbone_stage3_kernel_size=[(3, 3), (3, 3), (3, 1), (3, 3), (3, 1), (3, 3), (3, 3), (1, 3), (3, 3),
                                              (3, 3)],
                 backbone_stage3_stride=[1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage3_dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage3_groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

                 backbone_stage4_in_channels=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
                 backbone_stage4_out_channels=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
                 backbone_stage4_kernel_size=[(3, 3), (3, 3), (3, 1), (3, 3), (3, 1), (3, 3), (3, 3), (1, 3), (3, 3),
                                              (3, 3)],
                 backbone_stage4_stride=[1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage4_dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 backbone_stage4_groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

                 neck_in_channels=[64, 128, 256, 512],
                 neck_out_channels=[128, 128, 128, 128],
                 neck_kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3)],
                 neck_stride=[1, 1, 1, 1],
                 neck_dilation=[1, 1, 1, 1],
                 neck_groups=[1, 1, 1, 1],
                 **kwargs
                 ):
        self.backbone_config = {
            "kernel_size": 3,
            "stride": 2,
            "dilation": 1,
            "groups": 1,
            "bias": False,
            "has_shuffle": False,
            "in_channels": 3,
            "out_channels": 64,
            "use_bn": True,
            "act_func": "relu",
            "dropout_rate": 0,
            "ops_order": "weight_bn_act"
        }
        super.__init__(**kwargs)
        if backbone_config is not None:
            self.backbone_config.update(backbone_config)

        self.backbone_stage1_in_channels = backbone_stage1_in_channels
        self.backbone_stage1_out_channels = backbone_stage1_out_channels
        self.backbone_stage1_kernel_size = backbone_stage1_kernel_size,
        self.backbone_stage1_stride = backbone_stage1_stride,
        self.backbone_stage1_dilation = backbone_stage1_dilation,
        self.backbone_stage1_groups = backbone_stage1_groups,

        self.neck_in_channels = neck_in_channels,
        self.neck_out_channels = neck_out_channels,
        self.neck_kernel_size_channels = neck_kernel_size,
        self.neck_stride_channels = neck_stride,
        self.neck_dilation_channels = neck_dilation,
        self.neck_groups_channels = neck_groups,

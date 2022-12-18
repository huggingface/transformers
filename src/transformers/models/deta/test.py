from transformers import DetaConfig, DetaForObjectDetection, SwinConfig

backbone_config = SwinConfig(embed_dim = 192,
                                 depths = (2, 2, 18, 2),
                                 num_heads = (6, 12, 24, 48),
                                 out_features=["stage2", "stage3", "stage4"]
)

config = DetaConfig(
    backbone_config=backbone_config,
    num_queries=900,
    encoder_ffn_dim=2048,
    decoder_ffn_dim=2048,
    num_feature_levels=5,
    assign_first_stage=True,
    with_box_refine=True,
    two_stage=True,
)

model = DetaForObjectDetection(config)

for name, param in model.named_parameters():
    print(name, param.shape)

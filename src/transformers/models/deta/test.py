from transformers import DetaConfig, DetaForObjectDetection


config = DetaConfig(
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

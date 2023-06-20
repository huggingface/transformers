from transformers import BeitConfig, DPTConfig, DPTForDepthEstimation


backbone_config = BeitConfig(
    image_size=512,
    num_hidden_layers=24,
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=16,
    use_relative_position_bias=True,
    out_features=["stage5", "stage11", "stage17", "stage23"],
)

config = DPTConfig(backbone_config=backbone_config, hidden_size=1024, neck_hidden_sizes=[256, 512, 1024, 1024])

model = DPTForDepthEstimation(config)

for name, param in model.named_parameters():
    print(name, param.shape)

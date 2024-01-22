from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation


backbone_config = Dinov2Config.from_pretrained(
    "facebook/dinov2-small", out_indices=[3, 6, 9, 12], apply_layernorm=False, reshape_hidden_states=False
)
neck_hidden_sizes = [48, 96, 192, 384]

config = DPTConfig(
    backbone_config=backbone_config,
    neck_hidden_sizes=neck_hidden_sizes,
    use_bias_in_fusion_residual=False,
    add_projection=True,
)

model = DPTForDepthEstimation(config)

for name, param in model.named_parameters():
    print(name, param.shape)

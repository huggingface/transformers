from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation

# initialize with a Transformer-based backbone such as DINOv2
# in that case, we also specify `reshape_hidden_states=False` to get feature maps of shape (batch_size, num_channels, height, width)
backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False)

config = DPTConfig(backbone_config=backbone_config)
model = DPTForDepthEstimation(config=config)
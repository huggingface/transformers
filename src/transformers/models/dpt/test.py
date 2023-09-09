from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation

backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-small", out_indices=[2, 5, 8, 11])

neck_hidden_sizes = [48, 96, 192, 384]
config = DPTConfig(backbone_config=backbone_config, neck_hidden_sizes=neck_hidden_sizes)

model = DPTForDepthEstimation(config)

for name, param in model.named_parameters():
    print(name, param.shape)


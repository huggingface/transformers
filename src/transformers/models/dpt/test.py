from transformers import BeitConfig, DPTConfig, DPTForDepthEstimation


backbone_config = BeitConfig(num_hidden_layers=24, hidden_size=1024, num_attention_heads=16,
                             out_features=["stage5", "stage11", "stage17", "stage23"])

config = DPTConfig(backbone_config=backbone_config)

model = DPTForDepthEstimation(config)

for name, param in model.named_parameters():
    print(name, param.shape)
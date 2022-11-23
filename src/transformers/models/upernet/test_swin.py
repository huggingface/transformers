from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation


backbone_config = SwinConfig(
    output_hidden_states_before_downsampling=True,
    out_features=["stage1", "stage2", "stage3", "stage4"],
)

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)

for name, param in model.named_parameters():
    print(name, param.shape)

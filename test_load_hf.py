import torch
test_sample = torch.randn(1, 3, 224, 224)
model = torch.hub.load("./dinov3", 'dinov3_vit7b16_lc', source="local", weights="/tmp/lc.pth", backbone_weights="/tmp/backbone.pth")
outputs_torch = model(test_sample)
del(model)

from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTForImageClassification, DINOv3ViTBackbone, DINOv3ViTModel
import torch
from transformers import DINOv3ViTForImageClassification
model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m" #"facebook/dinov3-vits16-pretrain-lvd1689m"
dino_backbone = DINOv3ViTModel.from_pretrained(model_name)
config = DINOv3ViTConfig.from_pretrained(model_name, num_labels=1000)
with torch.device("meta"):
    model = DINOv3ViTForImageClassification(config)
    model.dinov3 = dino_backbone


lc_weights = torch.load("/tmp/lc.pth")
model.classifier.to_empty(device="cpu")
model.classifier.load_state_dict(lc_weights)
output_dir = "/tmp/dinov3_temp"
model.save_pretrained(output_dir)
model = DINOv3ViTForImageClassification.from_pretrained(output_dir)


outputs = model(test_sample).logits

del(model)

breakpoint()
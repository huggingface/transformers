from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTForImageClassification, DINOv3ViTBackbone, DINOv3ViTModel
import torch
import os
from transformers import DINOv3ViTForImageClassification
output_dir = "/tmp/dinov3_temp"
if not os.path.exists(output_dir):
    model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m" #"facebook/dinov3-vits16-pretrain-lvd1689m"
    dino_backbone = DINOv3ViTModel.from_pretrained(model_name)
    config = DINOv3ViTConfig.from_pretrained(model_name, num_labels=1000)
    with torch.device("meta"):
        model = DINOv3ViTForImageClassification(config)
        model.dinov3 = dino_backbone


    lc_weights = torch.load("/tmp/lc.pth")
    model.classifier.to_empty(device="cpu")
    model.classifier.load_state_dict(lc_weights)

    model.save_pretrained(output_dir)
model = DINOv3ViTForImageClassification.from_pretrained(output_dir)
model.eval()
tensor = torch.ones(1,3,224,224)
with torch.no_grad():
    bbone_output = model.dinov3(tensor).last_hidden_state
    outputs = model(tensor).logits
print(outputs.std())
torch.save(outputs, "/tmp/dinov3_vit7b16_lc_output_hf.pth")
torch.save(bbone_output, "/tmp/dinov3_vit7b16_lc_output_hf_bbone.pth")
breakpoint()
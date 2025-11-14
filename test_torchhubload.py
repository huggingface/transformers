import torch
# init a random tensor with fixed seed
tensor = torch.ones(1,3,224,224)
# DINOv3
dinov3_vit7b16_lc = torch.hub.load("/workspaces/transformers/dinov3", 'dinov3_vit7b16_lc', source="local", weights="/tmp/lc.pth", backbone_weights="/tmp/backbone.pth")
dinov3_vit7b16_lc.eval()
with torch.inference_mode():
    output = dinov3_vit7b16_lc(tensor)
    bbone_output = dinov3_vit7b16_lc.backbone(tensor)
breakpoint()
print(output.std())
torch.save(output, "/tmp/dinov3_vit7b16_lc_output.pth")
torch.save(bbone_output, "/tmp/dinov3_vit7b16_lc_bbone_output.pth")
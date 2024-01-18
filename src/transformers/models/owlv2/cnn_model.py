import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from .resnet import ResNet


class ConvModel(nn.Module):
    def __init__(self, name, out_dim=768, hidden_dim=256):
        super(ConvModel, self).__init__()
        if name =='resnet18':
            subset_model = ResNet(order='18', pretrained=True)
            z_dim = 256
        elif name =='resnet34':
            subset_model = ResNet(order='34', pretrained=True)
            z_dim = 256
        elif name =='resnet50':
            subset_model = ResNet(order='50', pretrained=True)
            z_dim = 1024
        elif name =='resnet101':
            subset_model = ResNet(order='101', pretrained=True)
            z_dim = 1024
        elif name =='convnext_tiny':
            backbone = convnext_tiny()
            z_dim = 384
            subset_model = torch.nn.Sequential(*list(backbone.features[:-2]))
        elif name =='convnext_small':
            backbone = convnext_small()
            z_dim = 384
            subset_model = torch.nn.Sequential(*list(backbone.features[:-2]))
        elif name =='convnext_base':
            backbone = convnext_base()
            z_dim = 512
            subset_model = torch.nn.Sequential(*list(backbone.features[:-2]))
        elif name =='convnext_large':
            backbone = convnext_large()
            z_dim = 768
            subset_model = torch.nn.Sequential(*list(backbone.features[:-2]))
        
        projection_layer = nn.Conv2d(z_dim, out_dim, kernel_size=1)
        torch.nn.init.xavier_uniform_(projection_layer.weight)
        torch.nn.init.zeros_(projection_layer.bias)
        self.backbone = nn.Sequential(
            subset_model,
            projection_layer
        )
        self.fc1 = nn.Linear(out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        patch_embeds = self.backbone(x)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        embeds = nn.functional.gelu(self.fc1(patch_embeds.mean(dim=1)))
        class_embeds = self.fc2(embeds).unsqueeze(dim=1)
        last_hidden_state = torch.cat([class_embeds, patch_embeds], dim=1)
        return last_hidden_state
    

if __name__ == "__main__":
    import time
    input_tensor = torch.randn((1, 3, 960, 960)).cuda().half()
    custom_resnet = ConvModel(name='convnext_large').cuda().eval().half()

    for i in range(10000):
        start = time.time()
        output = custom_resnet(input_tensor)
        # nn.functional.gelu(input_tensor)
        print(1000*(time.time() - start))
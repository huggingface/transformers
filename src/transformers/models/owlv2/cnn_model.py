import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models import convnext_tiny
from torchvision.models import convnext_small
from torchvision.models import convnext_base
from torchvision.models import convnext_large


class MenteeVisionModel(nn.Module):
    def __init__(self, name, out_dim=768, hidden_dim=256):
        super(MenteeVisionModel, self).__init__()
        if name =='resnet18':
            backbone = resnet18(pretrained=True)
            z_dim = 256
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-3])
        elif name =='resnet34':
            backbone = resnet34(pretrained=True)
            z_dim = 256
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-3])
        elif name =='resnet50':
            backbone = resnet50(pretrained=True)
            z_dim = 1024
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-3])
        elif name =='resnet101':
            backbone = resnet101(pretrained=True)
            z_dim = 1024
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-3])
        elif name =='convnext_tiny':
            backbone = list(convnext_tiny().children())[0]
            z_dim = 384
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-2])
        elif name =='convnext_small':
            backbone = list(convnext_small().children())[0]
            z_dim = 384
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-2])
        elif name =='convnext_base':
            backbone = list(convnext_base().children())[0]
            z_dim = 512
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-2])
        elif name =='convnext_large':
            backbone = list(convnext_large().children())[0]
            z_dim = 768
            subset_model = torch.nn.Sequential(*list(backbone.children())[:-2])
        
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
    

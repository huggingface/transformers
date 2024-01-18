from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, order='18', pretrained=True):
        super(ResNet, self).__init__()
        if order == '18':
            resnet = resnet18(pretrained=pretrained)
        elif order == '34':
            resnet = resnet34(pretrained=pretrained)
        elif order == '50':
            resnet = resnet50(pretrained=pretrained)
        elif order == '101':
            resnet = resnet101(pretrained=pretrained)
        elif order == '152':
            resnet = resnet152(pretrained=pretrained)
        else:
            raise ValueError("Unsupported ResNet type. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool 
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

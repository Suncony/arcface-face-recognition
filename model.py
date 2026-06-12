import torch
import torch.nn as nn

from torchvision import models, transforms, datasets

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes):
        super(ArcFaceModel, self).__init__()
        self.base = models.resnet50(pretrained=True)
        
        # Selective layer unfreezing
        # layers_to_unfreeze = ['layer4', 'layer3']
        layers_to_unfreeze = ['layer4']

        for name, param in self.base.named_parameters():
            param.requires_grad = any(layer in name for layer in layers_to_unfreeze)
        
        # Channel attention module
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid()
        )
        
        # Improved classifier
        in_features = self.base.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# model.py
import torch
import torch.nn as nn
import timm

class InceptionResNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        full = timm.create_model('inception_resnet_v2', pretrained=True)
        children = list(full.children())

        stem = children[:8]
        repeat_block = children[8]
        repeat_truncated = nn.Sequential(*list(repeat_block.children())[:10])

        self.backbone = nn.Sequential(
            *stem,
            repeat_truncated
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

class FCDDHead(nn.Module):
    def __init__(self, in_ch=320, num_anomaly=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_anomaly, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return torch.sqrt(x * x + 1) - 1

class FCDDNet(nn.Module):
    def __init__(self, num_anomaly=1):
        super().__init__()
        self.backbone = InceptionResNetV2Backbone()
        self.head = FCDDHead(in_ch=320, num_anomaly=num_anomaly)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
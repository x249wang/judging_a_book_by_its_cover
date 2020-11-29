import torch
import torch.nn as nn
from torchvision import models


class CNNModel(nn.Module):
    def __init__(self, fc_hidden_dim=256, dropout=0.1, freeze_layers=False):

        super(CNNModel, self).__init__()

        self.feature_extractor = models.resnet18(pretrained=True)

        if freeze_layers:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(num_features, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, 1),
        )

    def forward(self, x):
        x_out = self.feature_extractor(x)
        return x_out

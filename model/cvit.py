import torch
import torch.nn as nn


class CViT(nn.Module):
    def __init__(self, dim=512):
        super(CViT, self).__init__()

        # Simple CNN backbone (safe fallback)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 14 * 14, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
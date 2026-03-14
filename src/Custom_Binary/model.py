import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# The custom Mushroom-Roulette Model
class MR(nn.Module):
    def __init__(self):
        super(MR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling to fight overfitting
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        # Conv Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        # Conv Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        # Conv Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        # Conv Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # GAP Layer
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Dropout Layer
        x = self.dropout(x)

        # Linear Layer
        x = self.fc(x)

        return x

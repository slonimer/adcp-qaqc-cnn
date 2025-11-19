import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=4):
        """
        Args:
            input_channels (int): Number of input channels (3 by default: vel, backscatter, corr)
            num_classes (int): Number of output classes
        """
        super(TemporalCNN, self).__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # Collapse range dimension with adaptive pooling
        self.pool_range = nn.AdaptiveAvgPool2d((None, 1))  # Output shape: (B, C, T, 1)

        # Final classification layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x: Tensor of shape (B, 3, T, R)
        Returns: logits of shape (B, T, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))     # (B, 16, T, R)
        x = F.relu(self.bn2(self.conv2(x)))     # (B, 32, T, R)
        x = F.relu(self.bn3(self.conv3(x)))     # (B, 64, T, R)
        x = self.pool_range(x)                  # (B, 64, T, 1)
        x = x.squeeze(-1).permute(0, 2, 1)      # (B, T, 64)
        logits = self.fc(x)                     # (B, T, num_classes)
        return logits

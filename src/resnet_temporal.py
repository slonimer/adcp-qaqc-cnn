import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetTemporalClassifier(nn.Module):
    """
    A time-aware wrapper around a ResNet backbone.
    Input:
    x: Tensor of shape (B, C, T, R)

    Output:
    logits: Tensor of shape (B, T, num_classes)
    """
    def __init__(self, num_classes, pretrained=True, variant='resnet50', resize=(224, 224)):
        super().__init__()
        # Choose ResNet variant
        if variant == 'resnet18':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif variant == 'resnet34':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif variant == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif variant == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif variant == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        # Replace final classifier to match your num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        self.num_classes = num_classes
        # self.resize = resize  # (H, W) for ResNet input, e.g., (224, 224) # I DONT NEED THIS

    def forward(self, x):
        """
        x: (B, C, T, R)
        Returns: logits of shape (B, T, num_classes)
        """
        B, C, T, R = x.shape
        # Prepare frames: move time to batch dimension
        # x_perm: (B, T, C, R)
        x_perm = x.permute(0, 2, 1, 3).contiguous()
        # Flatten to (B*T, C, 1, R) to feed into 2D CNN
        frames = x_perm.view(B * T, C, 1, R)
        # Upsample to standard ResNet input size
        
        #frames_resized = F.interpolate(frames, size=self.resize, mode='bilinear', align_corners=False) # I DONT NEED THIS
        # frames_resized: (B*T, C, H, W)

        frames_resized = frames

        logits_flat = self.backbone(frames_resized)  # (B*T, num_classes)
        logits = logits_flat.view(B, T, self.num_classes)  # (B, T, num_classes)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationEncoder(nn.Module):
    """
    Encoder: Downsamples image and extracts hierarchical features
    
    Architecture: ResNet-like convolutional blocks
    Outputs feature maps at different scales (1/2, 1/4, 1/8)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Initial convolution layer
        # [B, 3, H, W] -> [B, 64, H, W]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # First downsampling stage
        # [B, 64, H/2, W/2] -> [B, 128, H/4, W/4]
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Second downsampling stage
        # [B, 128, H/4, W/4] -> [B, 256, H/8, W/8]
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            features: list of [c1, c2, c3]
                c1: [B, 64, H/2, W/2]
                c2: [B, 128, H/4, W/4]
                c3: [B, 256, H/8, W/8]
        """
        c1 = self.conv1(x)      # 1/2 resolution
        c2 = self.conv2(c1)     # 1/4 resolution
        c3 = self.conv3(c2)     # 1/8 resolution
        
        return [c1, c2, c3]

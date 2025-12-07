import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDecoder(nn.Module):
    """
    Decoder: Upsamples feature maps and produces semantic labels
    
    Architecture: Transposed convolutions with skip connections
    Progressively upsamples from 1/8 to full resolution
    """
    def __init__(self, num_classes=10, base_channels=64):
        super().__init__()
        
        # Upsample from 1/8 to 1/4 resolution
        # [B, 256, H/8, W/8] -> [B, 128, H/4, W/4]
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # After skip connection: [B, 256, H/4, W/4] (doubled channels)
        # Reduce channels: [B, 256, H/4, W/4] -> [B, 128, H/4, W/4]
        self.reduce3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Upsample from 1/4 to 1/2 resolution
        # [B, 128, H/4, W/4] -> [B, 64, H/2, W/2]
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # After skip connection: [B, 128, H/2, W/2] (doubled channels)
        # Reduce channels: [B, 128, H/2, W/2] -> [B, 64, H/2, W/2]
        self.reduce2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Upsample from 1/2 to full resolution
        # [B, 64, H/2, W/2] -> [B, num_classes, H, W]
        self.deconv1 = nn.ConvTranspose2d(base_channels, num_classes, 
                                          kernel_size=4, stride=2, padding=1)
        
    def forward(self, features):
        """
        Args:
            features: list of [c1, c2, c3] from encoder
                c1: [B, 64, H/2, W/2]
                c2: [B, 128, H/4, W/4]
                c3: [B, 256, H/8, W/8]
                
        Returns:
            logits: [B, num_classes, H, W] semantic class logits
        """
        c1, c2, c3 = features
        
        # Upsample from 1/8 to 1/4
        x = self.deconv3(c3)  # [B, 128, H/4, W/4]
        
        # Skip connection: concatenate with encoder features
        x = torch.cat([x, c2], dim=1)  # [B, 256, H/4, W/4]
        x = self.reduce3(x)  # [B, 128, H/4, W/4]
        
        # Upsample from 1/4 to 1/2
        x = self.deconv2(x)  # [B, 64, H/2, W/2]
        
        # Skip connection: concatenate with encoder features
        x = torch.cat([x, c1], dim=1)  # [B, 128, H/2, W/2]
        x = self.reduce2(x)  # [B, 64, H/2, W/2]
        
        # Upsample from 1/2 to full resolution
        x = self.deconv1(x)  # [B, num_classes, H, W]
        
        return x

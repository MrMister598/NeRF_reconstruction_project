import torch
import torch.nn as nn
from encoder import SegmentationEncoder
from decoder import SegmentationDecoder

class SemanticSegmentationNet(nn.Module):
    """
    Full semantic segmentation network
    
    Architecture: U-Net style with encoder-decoder
    Encoder: downsamples and extracts features
    Decoder: upsamples and produces semantic labels
    """
    def __init__(self, num_classes=10, in_channels=3, base_channels=64):
        super().__init__()
        
        self.encoder = SegmentationEncoder(in_channels, base_channels)
        self.decoder = SegmentationDecoder(num_classes, base_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            logits: [B, num_classes, H, W] semantic class logits
        """
        features = self.encoder(x)
        logits = self.decoder(features)
        
        return logits

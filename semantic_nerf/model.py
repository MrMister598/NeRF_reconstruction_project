import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticNeRF(nn.Module):
    """
    Semantic-aware Neural Radiance Field
    
    Predicts RGB, density, AND semantic class logits
    Architecture: MLP with dual-head output
    """
    def __init__(self, 
                 D=8,                    # Number of layers
                 W=256,                  # Hidden layer width
                 input_ch=63,            # Positional encoding dimension
                 num_classes=10,         # Number of semantic classes
                 skip_layer=4):          # Skip connection at layer
        super().__init__()
        
        self.D = D
        self.W = W
        self.skip_layer = skip_layer
        
        # Main network layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in [skip_layer] else nn.Linear(W + input_ch, W) 
             for i in range(D-1)]
        )
        
        # Density prediction (sigma)
        self.density_linear = nn.Linear(W, 1)
        
        # RGB prediction (color)
        self.rgb_linear = nn.Linear(W, 3)
        
        # Semantic prediction (class logits) - OUR ORIGINAL HEAD
        self.semantic_linear = nn.Linear(W, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [N, input_ch] - positionally encoded 3D points
            
        Returns:
            rgb: [N, 3] - predicted RGB color
            sigma: [N, 1] - predicted density
            semantic_logits: [N, num_classes] - semantic class predictions
        """
        input_pts = x
        h = x
        
        # Forward through MLP layers with skip connection
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            
            # Skip connection at middle layer
            if i == self.skip_layer:
                h = torch.cat([h, input_pts], dim=-1)
        
        # Predict density (volume density sigma)
        # Softplus ensures positive values with bias toward empty space
        sigma = F.softplus(self.density_linear(h) - 1.0)
        
        # Predict RGB (0-1 range via sigmoid)
        rgb = torch.sigmoid(self.rgb_linear(h))
        
        # Predict semantic logits (unnormalized class scores)
        semantic_logits = self.semantic_linear(h)
        
        return rgb, sigma, semantic_logits

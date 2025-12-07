import torch
import numpy as np

class PositionalEncoding:
    """
    Positional encoding for 3D coordinates
    
    Maps (x,y,z) to high-dimensional space using Fourier features
    Enables MLP to learn high-frequency details
    
    Formula: enc(x) = [sin(2^0*x), cos(2^0*x), sin(2^1*x), cos(2^1*x), ..., sin(2^(L-1)*x), cos(2^(L-1)*x)]
    """
    def __init__(self, num_freqs=10, include_input=True):
        """
        Args:
            num_freqs: number of frequency bands
            include_input: whether to include original coordinates
        """
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Frequency bands: 2^0, 2^1, ..., 2^(num_freqs-1)
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        
    def __call__(self, coords):
        """
        Encode coordinates
        
        Args:
            coords: [N, 3] - 3D coordinates (x, y, z)
            
        Returns:
            encoded: [N, output_dim] - encoded coordinates
        """
        encoded = []
        
        # Include original coordinates
        if self.include_input:
            encoded.append(coords)
        
        # For each frequency band
        for freq in self.freq_bands:
            # sin and cos at this frequency for each coordinate
            encoded.append(torch.sin(freq * coords))
            encoded.append(torch.cos(freq * coords))
        
        return torch.cat(encoded, dim=-1)
    
    @property
    def output_dim(self):
        """Calculate output dimension"""
        dim = 0
        if self.include_input:
            dim += 3  # Original (x, y, z)
        dim += 3 * 2 * self.num_freqs  # sin/cos for each freq, each coordinate
        return dim

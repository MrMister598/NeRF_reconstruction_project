import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .feature_extractor import FeatureExtractor

def train_sfm_features(config):
    """
    Train feature extractor for Structure-from-Motion
    
    Args:
        config: dict with training configuration
    """
    
    device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Training on device: {device}")
    
    # Create model
    model = FeatureExtractor(out_channels=128).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    
    # Create dummy dataset
    num_samples = config.get('num_samples', 100)
    H, W = 512, 512
    
    dummy_images = torch.randn(num_samples, 3, H, W)
    dummy_features = torch.randn(num_samples, 128, H//4, W//4)
    
    dataset = TensorDataset(dummy_images, dummy_features)
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 4), shuffle=True)
    
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    
    num_epochs = config.get('num_epochs', 50)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            features = model(images)
            
            # Compute loss
            loss = criterion(features, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            checkpoint_path = checkpoint_dir / f'feature_extractor_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / 'feature_extractor_best.pth'
    torch.save(model.state_dict(), final_path)
    print(f"✓ Final model saved: {final_path}")
    
    return model

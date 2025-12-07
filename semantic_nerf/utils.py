import torch
import numpy as np
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='checkpoints'):
    """Save model checkpoint"""
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    path = Path(checkpoint_dir) / f'model_epoch_{epoch}.pth'
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")
    
    return path

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    
    return epoch, loss

def get_device():
    """Get available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU (training will be slow)")
    
    return device

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

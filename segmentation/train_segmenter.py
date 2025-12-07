# segmentation/train_segmenter.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset import StatueSegmentationDataset
from model import SemanticSegmentationNet  # your U-Net


def train_segmenter(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Segmenter] Training on device: {device}")

    dataset = StatueSegmentationDataset(
        images_dir=config['images_dir'],
        heuristic_masks_dir=config['heuristic_masks_dir'],
        manual_masks_dir=config.get('manual_masks_dir', None),
    )
    print("len(dataset) =", len(dataset))
    dataloader = DataLoader(dataset,
                            batch_size=config.get('batch_size', 2),
                            shuffle=True)

    print(f"[Segmenter] Found {len(dataset)} image+mask pairs.")

    model = SemanticSegmentationNet(
        num_classes=config.get('num_classes', 3),  # 0 bg, 1 statue, 2 base/ground
        in_channels=3,
        base_channels=64
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config.get('checkpoint_dir', '../checkpoints/segmenter'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = config.get('num_epochs', 15)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)  # [B,H,W], values 0..2

            logits = model(imgs)  # [B,3,H,W]
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"[Segmenter] Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if (epoch + 1) % config.get('save_every', 5) == 0:
            ckpt_path = checkpoint_dir / f"segmenter_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)

    final_path = checkpoint_dir / "segmenter_best.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[Segmenter] Saved final model to {final_path}")
    return model


if __name__ == "__main__":
    config = {
        'images_dir': "../data/Family/images",
        'heuristic_masks_dir': "../data/Family/processed/semantic_masks",
        'manual_masks_dir': "../data/Family/manual_masks",
        'num_classes': 3,   # 0 bg, 1 statue, 2 base/ground
        'num_epochs': 15,
        'batch_size': 2,
        'lr': 1e-3,
        'checkpoint_dir': "../checkpoints/segmenter",
        'save_every': 5,
    }
    train_segmenter(config)

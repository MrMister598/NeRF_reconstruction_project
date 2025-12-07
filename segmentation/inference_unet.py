import torch
from pathlib import Path
from PIL import Image
import numpy as np

from model import SemanticSegmentationNet

def generate_unet_masks(images_dir, output_dir, checkpoint_path, num_classes=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[U-Net Inference] Using device: {device}")

    model = SemanticSegmentationNet(
        num_classes=num_classes,
        in_channels=3,
        base_channels=64
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"[U-Net Inference] Found {len(image_paths)} images in {images_dir}")

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            logits = model(img_tensor)           # [1,C,H,W]
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            out_path = output_dir / (img_path.stem + ".png")
            Image.fromarray(pred).save(out_path)

            print(f"✓ U-Net mask for {img_path.name} -> {out_path.name}")

if __name__ == "__main__":
    images_dir = "../data/Family/images"  # or ../data/Family/images
    output_dir = "../data/Family/processed/semantic_masks_unet"
    checkpoint_path = "../checkpoints/segmenter/segmenter_best.pth"

    generate_unet_masks(images_dir, output_dir, checkpoint_path, num_classes=3)

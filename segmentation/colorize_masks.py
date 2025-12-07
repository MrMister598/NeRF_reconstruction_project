# segmentation/colorize_masks.py
from pathlib import Path
from PIL import Image
import numpy as np

PALETTE = {
    0: (0, 0, 0),        # bg - black
    1: (255, 0, 0),      # statue - red
    2: (0, 0, 255),      # base/ground - blue
}

def colorize_mask(mask_np):
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in PALETTE.items():
        color[mask_np == cid] = rgb
    return color

if __name__ == "__main__":
    masks_dir = Path("../data/Family/processed/final_masks")
    out_dir   = Path("../data/Family/processed/semantic_masks_unet_color")
    out_dir.mkdir(parents=True, exist_ok=True)

    for mask_path in sorted(masks_dir.glob("*.png")):
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        color = colorize_mask(mask)
        Image.fromarray(color).save(out_dir / mask_path.name)
        print("✓", mask_path.name)

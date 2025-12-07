# segmentation/dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class StatueSegmentationDataset(Dataset):
    def __init__(self, images_dir, heuristic_masks_dir, manual_masks_dir=None):
        self.images_dir = Path(images_dir)
        self.heuristic_masks_dir = Path(heuristic_masks_dir)
        self.manual_masks_dir = Path(manual_masks_dir) if manual_masks_dir else None

        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        self.pairs = []
        for img_path in self.image_paths:
            base = img_path.stem
            heuristic_path = self.heuristic_masks_dir / f"{base}.png"
            if not heuristic_path.exists():
                continue

            manual_path = None
            if self.manual_masks_dir is not None:
                cand = self.manual_masks_dir / f"{base}.png"
                if cand.exists():
                    manual_path = cand

            self.pairs.append((img_path, heuristic_path, manual_path))

    def __len__(self):
        return len(self.pairs)

    def _load_manual_mask(self, manual_path):
        # grayscale -> class ids {0,1,2}
        mask_img = Image.open(manual_path).convert("L")
        mask_np = np.array(mask_img, dtype=np.uint8)

        manual_ids = np.zeros_like(mask_np, dtype=np.uint8)
        # background: near 0
        manual_ids[mask_np < 42] = 0
        # statue: mid gray ~85
        manual_ids[(mask_np >= 42) & (mask_np < 128)] = 1
        # base: light gray ~170
        manual_ids[mask_np >= 128] = 2

        return manual_ids

    def __getitem__(self, idx):
        img_path, heuristic_path, manual_path = self.pairs[idx]

        # load RGB
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C,H,W]

        # choose mask: manual or heuristic
        if manual_path is not None:
            mask_np = self._load_manual_mask(manual_path)  # values 0,1,2
        else:
            mask_img = Image.open(heuristic_path)
            raw = np.array(mask_img, dtype=np.uint8)  # may be 0,1,2,3

            # map to 0,1,2 only: merge any label >=2 into "2" (non-statue foreground)
            mask_np = np.zeros_like(raw, dtype=np.uint8)
            mask_np[raw == 1] = 1           # statue
            mask_np[raw >= 2] = 2           # base/ground merged

        # ensure values are in [0,2]
        mask_np = np.clip(mask_np, 0, 2).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np)  # [H,W], int64

        return img_tensor, mask_tensor

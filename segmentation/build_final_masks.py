# segmentation/build_final_masks.py
from pathlib import Path
from PIL import Image
import numpy as np

def load_manual_mask(path):
    # grayscale to {0,1,2}
    mask_img = Image.open(path).convert("L")
    m = np.array(mask_img, dtype=np.uint8)
    out = np.zeros_like(m, dtype=np.uint8)
    out[m < 42] = 0
    out[(m >= 42) & (m < 128)] = 1  # statue
    out[m >= 128] = 2               # base
    return out

if __name__ == "__main__":
    images_dir    = Path("../data/Family/images")
    heuristic_dir = Path("../data/Family/processed/semantic_masks")
    manual_dir    = Path("../data/Family/manual_masks")
    out_dir       = Path("../data/Family/processed/final_masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    print("Found", len(image_paths), "images")

    for img_path in image_paths:
        stem = img_path.stem
        heur_path  = heuristic_dir / f"{stem}.png"
        manual_path = manual_dir / f"{stem}.png"

        if manual_path.exists():
            # use detailed manual mask
            mask = load_manual_mask(manual_path)  # 0,1,2
        else:
            # fallback: heuristic mask
            if not heur_path.exists():
                print("No mask for", stem, "– skipping")
                continue
            raw = np.array(Image.open(heur_path), dtype=np.uint8)
            # map: 0 stays 0, 1 statue stays 1, 2/3 → 2 (base/ground)
            mask = np.zeros_like(raw, dtype=np.uint8)
            mask[raw == 1] = 1
            mask[raw >= 2] = 2

        Image.fromarray(mask).save(out_dir / f"{stem}.png")
        print("✓", stem)

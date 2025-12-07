import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm


def compute_cluster_stats(labels, h, w):
    """
    Compute stats per cluster:
      - center_frac: fraction of pixels in central box
      - mean_y: average normalized vertical position (0 top, 1 bottom)
      - mean_L: average brightness in LAB space
    """
    stats = {}
    labels_2d = labels.reshape(h, w)

    # central region: 40% x 40% around center
    cx, cy = w // 2, h // 2
    box_w, box_h = int(0.4 * w), int(0.4 * h)
    x0, x1 = cx - box_w // 2, cx + box_w // 2
    y0, y1 = cy - box_h // 2, cy + box_h // 2

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    center_mask = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    y_norm = yy.astype(np.float32) / max(h - 1, 1)  # 0 top, 1 bottom

    for cid in np.unique(labels):
        mask = labels_2d == cid
        total = mask.sum()
        if total == 0:
            center_frac = 0.0
            mean_y = 0.0
        else:
            center_frac = (mask & center_mask).sum() / float(total)
            mean_y = y_norm[mask].mean()

        stats[int(cid)] = {
            "center_frac": float(center_frac),
            "mean_y": float(mean_y),
            # mean_L will be filled outside
            "mean_L": 0.0,
        }

    return stats


def generate_heuristic_masks(
    images_dir,
    output_dir,
    K=4,
    resize_long_side=640
):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"[Heuristic] Found {len(image_paths)} images in {images_dir}")

    for img_path in tqdm(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h0, w0 = img_bgr.shape[:2]

        # optional resize for speed
        scale = 1.0
        if max(h0, w0) > resize_long_side:
            scale = resize_long_side / max(h0, w0)
            img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

        h, w = img_bgr.shape[:2]

        # LAB color
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        L = img_lab[:, :, 0]

        # edges (Canny)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # normalized vertical position
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        y_norm = yy.astype(np.float32) / max(h - 1, 1)

        # feature vector per pixel: [L, A, B, edges, y_norm]
        feats = np.concatenate(
            [
                img_lab,                     # [h,w,3]
                edges[..., None],           # [h,w,1]
                y_norm[..., None],          # [h,w,1]
            ],
            axis=-1,
        ).reshape(-1, 5)

        # K-means
        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        labels = kmeans.fit_predict(feats)

        # compute stats
        stats = compute_cluster_stats(labels, h, w)

        # mean_L per cluster
        labels_2d = labels.reshape(h, w)
        for cid in stats.keys():
            mask = labels_2d == cid
            if mask.any():
                stats[cid]["mean_L"] = float(L[mask].mean())
            else:
                stats[cid]["mean_L"] = 0.0

        # === improved cluster -> class mapping ===
        cluster_ids = sorted(stats.keys())
        center_arr = np.array([stats[cid]["center_frac"] for cid in cluster_ids])
        mean_y_arr = np.array([stats[cid]["mean_y"] for cid in cluster_ids])
        mean_L_arr = np.array([stats[cid]["mean_L"] for cid in cluster_ids])

        # 1) Choose statue cluster: central and mid-brightness
        brightness_penalty = np.abs(mean_L_arr - 0.5)  # prefer mid brightness
        statue_score = center_arr - 0.7 * brightness_penalty
        statue_idx = int(statue_score.argmax())
        statue_cid = cluster_ids[statue_idx]

        # 2) Choose base cluster: near bottom and somewhat darker
        remaining_ids = [cid for cid in cluster_ids if cid != statue_cid]
        base_cid = None
        if remaining_ids:
            mean_y_rem = np.array([stats[cid]["mean_y"] for cid in remaining_ids])
            mean_L_rem = np.array([stats[cid]["mean_L"] for cid in remaining_ids])
            base_score = mean_y_rem - 0.5 * mean_L_rem  # bottom & dark
            base_idx = int(base_score.argmax())
            base_cid = remaining_ids[base_idx]

        # 3) build final mask: 0=background, 1=statue, 2=base, 3=ground/other
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[labels_2d == statue_cid] = 1
        if base_cid is not None:
            mask[labels_2d == base_cid] = 2

        for cid in cluster_ids:
            if cid not in (statue_cid, base_cid):
                mask[labels_2d == cid] = 3

        # upsample back to original resolution if we resized
        if scale != 1.0:
            mask = cv2.resize(
                mask,
                (w0, h0),
                interpolation=cv2.INTER_NEAREST,
            )

        out_path = output_dir / (img_path.stem + ".png")
        Image.fromarray(mask).save(out_path)

    print(f"[Heuristic] Saved masks to {output_dir}")


if __name__ == "__main__":
    images_dir = "../data/Family/images"
    output_dir = "../data/Family/processed/semantic_masks"
    generate_heuristic_masks(images_dir, output_dir, K=4, resize_long_side=640)

# Semantic NeRF for Heritage Reconstruction – Family Statue

This repository contains the code and experiments for our project on **Semantic NeRF** applied to a heritage scene (family statue).  
We extend a NeRF model with a **semantic head** that predicts per‑point class labels (background / statue / base) and train it using a **small number of manually labeled keyframes**.

---

## 1. Project overview

### Goal

Recover not only **geometry and color**, but also **semantic meaning** in 3D:

- Class 0 – background  
- Class 1 – statue  
- Class 2 – base  

We train a Semantic‑NeRF that:

- Uses **RGB loss on all images** to learn geometry and appearance.
- Uses **semantic loss only on 7 manually annotated keyframes** to learn a 3D semantic field.

The final model can:

- Render **RGB and semantic views** from novel camera poses.
- Produce a **semantic point cloud** with statue vs base separated.

---

## 2. Repository structure

```
.
├─ semantic_nerf/
│  ├─ model.py            # SemanticNeRF architecture (NeRF + semantic head)
│  ├─ dataset.py          # HeritageNeRFDataset with sparse semantic supervision
│  ├─ rendering.py        # Ray sampling + volume rendering
│  ├─ loss.py             # RGB + semantic loss, PSNR, SSIM
│  ├─ encoding.py         # Positional encoding
│  ├─ metrics.py          # Evaluation metrics
│  └─ utils.py            # Checkpointing, parameter counting, etc.
│
├─ data/
│  └─ Family/
│     ├─ images/          # All input images (RGB)
│     ├─ manual_masks/    # 7 manually labeled keyframes (PNG)
│     ├─ processed/
│     │  └─ transforms.json  # Camera poses & intrinsics from SfM
│     └─ key_list.txt     # List of keyframe stems (e.g. 00012, 00027, ...)
│
├─ train_nerf.py          # Train Semantic-NeRF
├─ render_nerf.py         # Render views + semantic point cloud
└─ README.md
```

---

## 3. Setup

### Requirements

- Python 3.9+
- PyTorch with CUDA (for GPU training / rendering)
- Common Python packages: `numpy`, `Pillow`, `tqdm`, etc.

Install:

```bash
pip install -r requirements.txt
```

### Data

Place your dataset under `data/Family/`:

- `images/` – all RGB images. Filenames must match `transforms.json`

- `processed/transforms.json` – camera poses and intrinsics from COLMAP / SfM.

- `manual_masks/` – 7 PNG masks for selected keyframes.
  - Grayscale values thresholded into: background (0), statue (1), base (2).

- `key_list.txt` – text file listing the stems of the 7 keyframes, one per line

---

## 4. Training Semantic‑NeRF

We train on **all images** but only apply semantic loss on rays from the 7 keyframes.

Run:

```bash
python train_nerf.py
```

Key config (inside `train_nerf.py`):

```python
config = {
    'transforms_path': 'data/Family/processed/transforms.json',
    'images_dir':      'data/Family',
    'masks_dir':       'data/Family/manual_masks',
    'key_list_path':   'data/Family/key_list.txt',

    'checkpoint_dir':  'checkpoints',
    'output_dir':      'outputs',

    'N_rays':      1024,
    'N_samples':   64,
    'near':        2.0,
    'far':         6.0,

    'num_classes': 3,
    'num_epochs':  200,
    'num_freqs':   10,
    'D':           8,
    'W':           256,
    'skip_layer':  4,
    'lr':          5e-4,
    'lambda_semantic': 0.5,

    'downscale':   1,
    'white_bkgd':  True,
}
```

During training:

- `rgb_loss` is computed for all rays.
- `semantic_loss` is computed only where `sem_valid=True` (rays from keyframes).

---

## 5. Rendering and point cloud export

After training, use `render_nerf.py` to:

- Render RGB, depth, and semantic images from selected views.
- Export a **semantic point cloud** as `.ply`.

Run:

```bash
python render_nerf.py
```

Outputs (per rendered view):

- `view_XXX_rgb.png` – NeRF RGB render.
- `view_XXX_depth.png` – depth map (uint16).
- `view_XXX_sem.png` – raw semantic IDs.
- `view_XXX_sem_color.png` – colorized semantics (0=black, 1=red, 2=blue).

Point cloud:

- `scene_pointcloud.ply` – merged colored point cloud with semantic labels.

Inspect it in Meshlab, CloudCompare, etc.

---

## 6. Method summary

- Start from a NeRF-style MLP with positional encoding.
- Add a **semantic head** that outputs class logits at each sampled 3D point.
- Train with:
  - **Photometric loss** on all rays.
  - **Cross-entropy semantic loss** only on rays from 7 manually annotated keyframes.
- Use volume rendering to obtain RGB and semantic predictions for each ray.
- Aggregate depth and semantic predictions across views to build a semantic 3D point cloud.

---

## 7. Limitations & future work

- Heuristic K-means and U-Net based segmentation were attempted but not robust.
  - Often separated shadows vs non-shadows instead of statue vs base.
  - Therefore, they were not used for final Semantic-NeRF supervision.
- Final semantics are strongest near the manually labeled keyframes (sparse labels).
- Future directions:
  - Better 2D segmentation (or more labels) to supervise semantics across more views.
  - Scaling to larger, more complex heritage scenes.

---

## 8. Acknowledgements

Built for a course project on 3D reconstruction and neural rendering.

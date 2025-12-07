import torch
import numpy as np
from pathlib import Path
from PIL import Image

from semantic_nerf.model import SemanticNeRF
from semantic_nerf.encoding import PositionalEncoding
from semantic_nerf.rendering import sample_points_along_rays, volume_rendering
from semantic_nerf.dataset import HeritageNeRFDataset

PALETTE = {
    0: (0, 0, 0),       # background
    1: (255, 0, 0),     # statue
    2: (0, 0, 255),     # base
    3: (0, 255, 0),     # ground (unused here)
}

def colorize_semantic(sem_ids):
    h, w = sem_ids.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in PALETTE.items():
        color[sem_ids == cid] = rgb
    return color

def generate_rays_full_image(H, W, focal, c2w):
    i_coords, j_coords = np.meshgrid(
        np.arange(W), np.arange(H), indexing="xy"
    )
    dirs = np.stack([
        (i_coords - W / 2) / focal,
        -(j_coords - H / 2) / focal,
        -np.ones_like(i_coords),
    ], axis=-1)  # [H,W,3]

    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    return rays_o, rays_d

def render_novel_views(config, checkpoint_path, output_dir, num_views=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dataset (for poses & intrinsics)
    dataset = HeritageNeRFDataset(
        transforms_path=config['transforms_path'],
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        downscale=config.get('downscale', 1),
        N_rays=config.get('N_rays', 1024),
        key_list_path=config.get('key_list_path', None),  # optional, safe
    )
    H, W, focal = dataset.H, dataset.W, dataset.focal
    poses = dataset.poses
    num_imgs = len(poses)
    indices = np.linspace(0, num_imgs - 1, num_views).astype(int)

    # 2) Model
    pos_enc = PositionalEncoding(num_freqs=config.get('num_freqs', 10))
    model = SemanticNeRF(
        D=config.get('D', 8),
        W=config.get('W', 256),
        input_ch=pos_enc.output_dim,
        num_classes=config.get('num_classes', 3),  # must be 3
        skip_layer=config.get('skip_layer', 4),
    ).to(device)

    # Load checkpoint (unwrap if it's a full dict)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    model.load_state_dict(state_dict)
    model.eval()

    all_points, all_colors, all_semantic = [], [], []
    chunk_size = 4096

    with torch.no_grad():
        for i, idx in enumerate(indices):
            c2w = poses[idx]

            rays_o_np, rays_d_np = generate_rays_full_image(H, W, focal, c2w)
            rays_o_all = torch.from_numpy(rays_o_np).float().to(device)
            rays_d_all = torch.from_numpy(rays_d_np).float().to(device)
            N_rays_total = rays_o_all.shape[0]

            rgb_chunks = []
            depth_chunks = []
            sem_chunks = []

            for start in range(0, N_rays_total, chunk_size):
                end = min(start + chunk_size, N_rays_total)
                rays_o = rays_o_all[start:end]
                rays_d = rays_d_all[start:end]

                pts, z_vals = sample_points_along_rays(
                    rays_o, rays_d,
                    near=config.get('near', 2.0),
                    far=config.get('far', 6.0),
                    N_samples=config.get('N_samples', 32),
                    perturb=False,
                )
                pts_flat = pts.reshape(-1, 3)
                pts_encoded = pos_enc(pts_flat)

                rgb, sigma, semantic_logits = model(pts_encoded)
                N_rays = pts.shape[0]
                N_samples = pts.shape[1]

                rgb = rgb.reshape(N_rays, N_samples, 3)
                sigma = sigma.reshape(N_rays, N_samples, 1)
                semantic_logits = semantic_logits.reshape(N_rays, N_samples, -1)

                rgb_pred, depth_map, _, semantic_pred = volume_rendering(
                    rgb, sigma, z_vals, rays_d,
                    semantic_logits=semantic_logits,
                    white_bkgd=config.get('white_bkgd', True),
                )

                rgb_chunks.append(rgb_pred.cpu())
                depth_chunks.append(depth_map.cpu())
                sem_chunks.append(semantic_pred.cpu())

            rgb_pred = torch.cat(rgb_chunks, dim=0)
            depth_map = torch.cat(depth_chunks, dim=0)
            semantic_pred = torch.cat(sem_chunks, dim=0)

            rgb_img = rgb_pred.reshape(H, W, 3).numpy()
            depth_img = depth_map.reshape(H, W).numpy()
            sem_logits_img = semantic_pred.reshape(H, W, -1)
            sem_ids = torch.argmax(sem_logits_img, dim=-1).numpy().astype(np.uint8)

            rgb_uint8 = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(rgb_uint8).save(output_dir / f"view_{i:03d}_rgb.png")

            depth_norm = depth_img / (depth_img.max() + 1e-8)
            depth_uint16 = (depth_norm * 65535).astype(np.uint16)
            Image.fromarray(depth_uint16).save(output_dir / f"view_{i:03d}_depth.png")

            Image.fromarray(sem_ids).save(output_dir / f"view_{i:03d}_sem.png")
            sem_color = colorize_semantic(sem_ids)
            Image.fromarray(sem_color).save(output_dir / f"view_{i:03d}_sem_color.png")

            print(f"✓ Rendered view {i+1}/{len(indices)}")

            depth_flat = depth_img.reshape(-1)
            valid = depth_flat > 0
            pts_cam = rays_o_np + rays_d_np * depth_flat[:, None]
            pts_cam = pts_cam[valid]

            cols = rgb_uint8.reshape(-1, 3)[valid]
            sem_flat = sem_ids.reshape(-1)[valid]

            all_points.append(pts_cam)
            all_colors.append(cols)
            all_semantic.append(sem_flat)

    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_semantic = np.concatenate(all_semantic, axis=0)
        save_pointcloud_ply(
            all_points, all_colors, all_semantic,
            output_dir / "scene_pointcloud.ply"
        )
        print(f"✓ Saved point cloud with {all_points.shape[0]} points")

def save_pointcloud_ply(points, colors, semantics, path):
    path = Path(path)
    N = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("end_header\n")
        for p, c, s in zip(points, colors, semantics):
            f.write(
                f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])} {int(s)}\n"
            )

if __name__ == "__main__":
    config = {
        'transforms_path': 'data/Family/processed/transforms.json',
        'images_dir': 'data/Family',
        'masks_dir': 'data/Family/manual_masks',   # not used at render time but ok
        'N_rays': 1024,
        'N_samples': 128,
        'near': 2.0,
        'far': 6.0,
        'num_classes': 3,
        'num_freqs': 10,
        'D': 8,
        'W': 256,
        'skip_layer': 4,
        'downscale': 1,
        'white_bkgd': True,
        'key_list_path': 'data/Family/key_list.txt',  # optional
    }
    checkpoint_path = "checkpoints/model_epoch_200.pth"
    output_dir = "outputs/rendered_views"

    render_novel_views(config, checkpoint_path, output_dir, num_views=12)

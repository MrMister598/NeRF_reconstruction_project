import torch
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
from pathlib import Path


class HeritageNeRFDataset(Dataset):
    """
    PyTorch Dataset for Heritage NeRF with semantic labels

    Loads images, semantic masks, and camera poses
    Samples rays and returns ray data for training
    """

    def __init__(self, transforms_path, images_dir, masks_dir,
                 downscale=1, N_rays=1024, key_list_path=None):
        """
        Args:
            transforms_path: path to transforms.json (from SfM)
            images_dir: directory with RGB images
            masks_dir: directory with semantic mask PNGs (manual for key images)
            downscale: image downscaling factor (1 = no downscale)
            N_rays: number of rays to sample per batch
            key_list_path: txt file with stems of key images (one per line)
        """
        # Load transforms.json
        with open(transforms_path, 'r') as f:
            self.meta = json.load(f)

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.downscale = downscale
        self.N_rays = N_rays

        # load key stems (e.g. 00012, 00027, ...)
        self.key_stems = set()
        if key_list_path is not None:
            txt = Path(key_list_path).read_text().strip().splitlines()
            self.key_stems = {s.strip() for s in txt if s.strip()}

        self.frames = self.meta['frames']
        self.num_frames = len(self.frames)

        # Camera intrinsics
        self.H = self.meta['h'] // downscale
        self.W = self.meta['w'] // downscale
        self.focal = self.meta['fl_x'] / downscale

        # Load all images, masks, poses
        self.images = []
        self.masks = []          # 0,1,2 for key images, -1 elsewhere
        self.has_semantics = []  # per-frame flag
        self.poses = []

        for frame in self.frames:
            rel_path = frame['file_path']                 # e.g. "images/00001.jpg"
            img_path = self.images_dir / rel_path         # data/Family/images/00001.jpg
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.W, self.H), Image.LANCZOS)
            img = np.array(img, dtype=np.float32) / 255.0
            self.images.append(img)

            stem = Path(rel_path).stem                    # "00001"
            mask_name = f"{stem}.png"
            mask_path = self.masks_dir / mask_name

            if (stem in self.key_stems) and mask_path.exists():
                # manual mask: grayscale -> {0,1,2}
                mask_img = Image.open(mask_path).convert("L")
                mask_img = mask_img.resize((self.W, self.H), Image.NEAREST)
                m = np.array(mask_img, dtype=np.uint8)

                mask_ids = np.zeros_like(m, dtype=np.int64)
                mask_ids[m < 42] = 0                      # background
                mask_ids[(m >= 42) & (m < 128)] = 1       # statue
                mask_ids[m >= 128] = 2                    # base
                self.masks.append(mask_ids)
                self.has_semantics.append(True)
            else:
                # no semantics for this frame
                self.masks.append(np.full((self.H, self.W), -1, dtype=np.int64))
                self.has_semantics.append(False)

            # camera pose
            pose = np.array(frame['transform_matrix'], dtype=np.float32)
            self.poses.append(pose)

        # Stack into arrays
        self.images = np.stack(self.images, axis=0)          # [N, H, W, 3]
        self.masks = np.stack(self.masks, axis=0)            # [N, H, W]
        self.has_semantics = np.array(self.has_semantics)    # [N]
        self.poses = np.stack(self.poses, axis=0)            # [N, 4, 4]

    def __len__(self):
        """Number of images (= number of batches)"""
        return self.num_frames

    def __getitem__(self, idx):
        """
        Get one image's worth of rays

        Returns:
            dict with:
                - rays_o: [N_rays, 3] ray origins
                - rays_d: [N_rays, 3] ray directions
                - rgb_gt: [N_rays, 3] ground truth RGB
                - semantic_gt: [N_rays] ground truth semantic class (0,1,2) or -1
                - sem_valid: [N_rays] bool, True where semantic_gt is valid
        """
        img = self.images[idx]       # [H, W, 3]
        mask = self.masks[idx]       # [H, W]
        pose = self.poses[idx]       # [4, 4]

        # Generate camera rays for this image
        rays_o, rays_d = self._get_rays(self.H, self.W, self.focal, pose)

        # Flatten
        rays_o = rays_o.reshape(-1, 3)      # [H*W, 3]
        rays_d = rays_d.reshape(-1, 3)
        rgb_gt = img.reshape(-1, 3)         # [H*W, 3]
        semantic_gt = mask.reshape(-1)      # [H*W]

        # Sample random rays for training batch
        num_rays_available = rays_o.shape[0]
        if self.N_rays <= num_rays_available:
            indices = np.random.choice(num_rays_available, self.N_rays, replace=False)
        else:
            indices = np.random.choice(num_rays_available, self.N_rays, replace=True)

        rgb_gt = rgb_gt[indices]
        semantic_gt = semantic_gt[indices]
        sem_valid = semantic_gt >= 0        # True only for key images

        return {
            'rays_o': torch.from_numpy(rays_o[indices]).float(),
            'rays_d': torch.from_numpy(rays_d[indices]).float(),
            'rgb_gt': torch.from_numpy(rgb_gt).float(),
            'semantic_gt': torch.from_numpy(semantic_gt).long(),
            'sem_valid': torch.from_numpy(sem_valid).bool(),
        }

    @staticmethod
    def _get_rays(H, W, focal, c2w):
        """
        Generate camera rays for an image

        Args:
            H, W: image height, width
            focal: focal length (pixels)
            c2w: [4, 4] camera-to-world matrix

        Returns:
            rays_o: [H, W, 3] ray origins
            rays_d: [H, W, 3] ray directions
        """
        i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

        dirs = np.stack([
            (i - W/2) / focal,
            -(j - H/2) / focal,
            -np.ones_like(i)
        ], axis=-1)  # [H, W, 3]

        rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)  # [H, W, 3]
        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)          # [H, W, 3]

        return rays_o, rays_d

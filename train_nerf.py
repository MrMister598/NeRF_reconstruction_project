import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import time

from semantic_nerf.model import SemanticNeRF
from semantic_nerf.encoding import PositionalEncoding
from semantic_nerf.rendering import sample_points_along_rays, volume_rendering
from semantic_nerf.loss import compute_loss, compute_psnr
from semantic_nerf.dataset import HeritageNeRFDataset
from semantic_nerf.utils import save_checkpoint, count_parameters
from semantic_nerf.metrics import MetricsComputer


def train_nerf(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    num_epochs = config.get('num_epochs', 200)
    near = config.get('near', 2.0)
    far  = config.get('far', 6.0)
    N_samples = config.get('N_samples', 64)
    lambda_semantic = config.get('lambda_semantic', 0.1)

    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)

    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(exist_ok=True)

    # 1/4: Dataset
    print("\n[1/4] Loading dataset...")
    dataset = HeritageNeRFDataset(
        transforms_path=config['transforms_path'],
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        downscale=config.get('downscale', 1),
        N_rays=config.get('N_rays', 1024),
        key_list_path=config.get('key_list_path', None),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"✓ Loaded {len(dataset)} images")

    # 2/4: Model
    print("\n[2/4] Initializing model...")
    pos_enc = PositionalEncoding(num_freqs=config.get('num_freqs', 10))

    model = SemanticNeRF(
        D=config.get('D', 8),
        W=config.get('W', 256),
        input_ch=pos_enc.output_dim,
        num_classes=config.get('num_classes', 3),  # 0 bg, 1 statue, 2 base
        skip_layer=config.get('skip_layer', 4)
    ).to(device)

    num_params = count_parameters(model)
    print(f"✓ Model created: {num_params:,} trainable parameters")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 5e-4),
        betas=(0.9, 0.999),
        eps=1e-8
    )

    metrics_computer = MetricsComputer()

    training_history = {
        'epoch': [],
        'total_loss': [],
        'rgb_loss': [],
        'semantic_loss': [],
        'psnr': [],
        'ssim': [],
        'semantic_accuracy': [],
        'f1_weighted': [],
        'miou': [],
        'gpu_memory_gb': [],
        'inference_fps': [],
        'training_time': []
    }

    epoch_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_rgb_loss = 0
        epoch_sem_loss = 0
        epoch_sem_acc = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            rays_o = batch['rays_o'][0].to(device)
            rays_d = batch['rays_d'][0].to(device)
            rgb_gt = batch['rgb_gt'][0].to(device)
            semantic_gt = batch['semantic_gt'][0].to(device)
            sem_valid = batch['sem_valid'][0].to(device)   # [N_rays] bool

            # Sample points along rays
            pts, z_vals = sample_points_along_rays(
                rays_o, rays_d,
                near=near,
                far=far,
                N_samples=N_samples,
                perturb=True
            )

            # Flatten and encode points
            pts_flat = pts.reshape(-1, 3)
            pts_encoded = pos_enc(pts_flat)

            # Forward pass
            rgb, sigma, semantic_logits = model(pts_encoded)

            N_rays_batch = pts.shape[0]
            rgb = rgb.reshape(N_rays_batch, N_samples, 3)
            sigma = sigma.reshape(N_rays_batch, N_samples, 1)
            semantic_logits = semantic_logits.reshape(N_rays_batch, N_samples, -1)

            rgb_pred, depth_map, acc_map, semantic_pred = volume_rendering(
                rgb, sigma, z_vals, rays_d,
                semantic_logits=semantic_logits,
                white_bkgd=config.get('white_bkgd', True)
            )

            # Use semantics only where valid
            if sem_valid.any():
                sem_pred_valid = semantic_pred[sem_valid]
                sem_gt_valid = semantic_gt[sem_valid]
            else:
                sem_pred_valid = None
                sem_gt_valid = None

            loss, rgb_loss, sem_loss, sem_acc = compute_loss(
                rgb_pred, rgb_gt,
                sem_pred_valid, sem_gt_valid,
                lambda_semantic=lambda_semantic
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = compute_psnr(rgb_pred, rgb_gt)

            epoch_loss += loss.item()
            epoch_rgb_loss += rgb_loss.item()
            epoch_sem_loss += sem_loss.item() if isinstance(sem_loss, torch.Tensor) else sem_loss
            epoch_sem_acc += sem_acc.item() if isinstance(sem_acc, torch.Tensor) else sem_acc
            num_batches += 1

            pbar.set_postfix({
                'loss': loss.item(),
                'psnr': psnr.item(),
                'sem_acc': sem_acc.item() if isinstance(sem_acc, torch.Tensor) else 0.0
            })

        avg_loss = epoch_loss / num_batches
        avg_rgb_loss = epoch_rgb_loss / num_batches
        avg_sem_loss = epoch_sem_loss / num_batches
        avg_sem_acc = epoch_sem_acc / num_batches

        # Validation (you should also ignore semantics where invalid here, similar idea)
        with torch.no_grad():
            all_rgb_pred = []
            all_rgb_gt = []
            all_sem_pred = []
            all_sem_gt = []

            for batch in dataloader:
                rays_o = batch['rays_o'][0].to(device)
                rays_d = batch['rays_d'][0].to(device)
                rgb_gt = batch['rgb_gt'][0].to(device)
                semantic_gt = batch['semantic_gt'][0].to(device)
                sem_valid = batch['sem_valid'][0].to(device)

                pts, z_vals = sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
                pts_flat = pts.reshape(-1, 3)
                pts_encoded = pos_enc(pts_flat)

                rgb, sigma, semantic_logits = model(pts_encoded)
                N_rays_batch = pts.shape[0]
                rgb = rgb.reshape(N_rays_batch, N_samples, 3)
                sigma = sigma.reshape(N_rays_batch, N_samples, 1)
                semantic_logits = semantic_logits.reshape(N_rays_batch, N_samples, -1)

                rgb_pred, _, _, semantic_pred = volume_rendering(
                    rgb, sigma, z_vals, rays_d, semantic_logits=semantic_logits
                )

                all_rgb_pred.append(rgb_pred)
                all_rgb_gt.append(rgb_gt)
                all_sem_pred.append(semantic_pred)
                all_sem_gt.append(semantic_gt)

            all_rgb_pred = torch.cat(all_rgb_pred, dim=0)
            all_rgb_gt = torch.cat(all_rgb_gt, dim=0)
            all_sem_pred = torch.cat(all_sem_pred, dim=0)
            all_sem_gt = torch.cat(all_sem_gt, dim=0)

            validation_metrics = metrics_computer.compute_all_metrics(
                all_rgb_pred, all_rgb_gt,
                all_sem_pred, all_sem_gt,
                model=model
            )

        training_history['epoch'].append(epoch + 1)
        training_history['total_loss'].append(avg_loss)
        training_history['rgb_loss'].append(avg_rgb_loss)
        training_history['semantic_loss'].append(avg_sem_loss)
        training_history['psnr'].append(validation_metrics['psnr'])
        training_history['ssim'].append(validation_metrics['ssim'])
        training_history['semantic_accuracy'].append(validation_metrics['semantic_accuracy'])
        training_history['f1_weighted'].append(validation_metrics['f1_weighted'])
        training_history['miou'].append(validation_metrics['miou'])
        training_history['gpu_memory_gb'].append(validation_metrics['gpu_memory_used_gb'])
        training_history['inference_fps'].append(validation_metrics.get('inference_fps', 0))

        elapsed_time = time.time() - epoch_start_time
        training_history['training_time'].append(elapsed_time)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {elapsed_time/60:.1f} min")
        print(f"{'='*70}")
        print(f"Loss: {avg_loss:.4f} (RGB: {avg_rgb_loss:.4f}, Sem: {avg_sem_loss:.4f})")
        print(f"PSNR: {validation_metrics['psnr']:.2f} dB | SSIM: {validation_metrics['ssim']:.4f}")
        print(f"Semantic Acc: {validation_metrics['semantic_accuracy']:.4f} | F1: {validation_metrics['f1_weighted']:.4f} | mIoU: {validation_metrics['miou']:.4f}")
        print(f"GPU Memory: {validation_metrics['gpu_memory_used_gb']:.2f} GB / {validation_metrics['gpu_memory_total_gb']:.2f} GB")
        if 'inference_fps' in validation_metrics:
            print(f"Inference FPS: {validation_metrics['inference_fps']:.2f}")

        if (epoch + 1) % config.get('save_every', 20) == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_dir)

    # Final report & save history/model same as before...
    # (keep your existing code here)

if __name__ == '__main__':
    config = {
        'transforms_path': 'data/Family/processed/transforms.json',
        'images_dir': 'data/Family',
        'masks_dir': 'data/Family/manual_masks',
        'key_list_path': 'data/Family/key_list.txt',
        'checkpoint_dir': 'checkpoints',
        'output_dir': 'outputs',
        'N_rays': 1024,
        'N_samples': 64,
        'near': 2.0,
        'far': 6.0,
        'num_classes': 3,          # 0 bg, 1 statue, 2 base
        'num_epochs': 200,
        'num_freqs': 10,
        'D': 8,
        'W': 256,
        'skip_layer': 4,
        'lr': 5e-4,
        'lambda_semantic': 0.5,
        'save_every': 20,
        'downscale': 1,
        'white_bkgd': True
    }
    train_nerf(config)

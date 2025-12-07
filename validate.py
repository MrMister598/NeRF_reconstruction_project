#!/usr/bin/env python3

"""
Comprehensive validation and ablation studies
Tests model with/without semantic head
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from semantic_nerf.model import SemanticNeRF
from semantic_nerf.encoding import PositionalEncoding
from semantic_nerf.rendering import sample_points_along_rays, volume_rendering
from semantic_nerf.loss import compute_psnr, compute_ssim
from semantic_nerf.metrics import MetricsComputer
from semantic_nerf.dataset import HeritageNeRFDataset


def validate_full_model(checkpoint_path, config):
    """Validate full Semantic-NeRF model"""
    print("\n[VALIDATION] Full Semantic-NeRF (RGB + Semantic)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    pos_enc = PositionalEncoding(num_freqs=config.get('num_freqs', 10))
    model = SemanticNeRF(
        D=config.get('D', 8),
        W=config.get('W', 256),
        input_ch=pos_enc.output_dim,
        num_classes=config.get('num_classes', 10),
        skip_layer=config.get('skip_layer', 4)
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load dataset
    dataset = HeritageNeRFDataset(
        transforms_path=config['transforms_path'],
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        downscale=config.get('downscale', 1),
        N_rays=config.get('N_rays', 1024)
    )
    
    metrics_computer = MetricsComputer()
    
    all_rgb_pred = []
    all_rgb_gt = []
    all_sem_pred = []
    all_sem_gt = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Validating"):
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            rgb_gt = batch['rgb_gt'].to(device)
            semantic_gt = batch['semantic_gt'].to(device)
            
            # Forward pass
            pts, z_vals = sample_points_along_rays(
                rays_o, rays_d,
                near=config.get('near', 2.0),
                far=config.get('far', 6.0),
                N_samples=config.get('N_samples', 64),
                perturb=False
            )
            
            pts_flat = pts.reshape(-1, 3)
            pts_encoded = pos_enc(pts_flat)
            
            rgb, sigma, semantic_logits = model(pts_encoded)
            
            N_rays = pts.shape[0]
            rgb = rgb.reshape(N_rays, -1, 3)
            sigma = sigma.reshape(N_rays, -1, 1)
            semantic_logits = semantic_logits.reshape(N_rays, -1, -1)
            
            rgb_pred, _, _, semantic_pred = volume_rendering(
                rgb, sigma, z_vals, rays_d,
                semantic_logits=semantic_logits
            )
            
            all_rgb_pred.append(rgb_pred)
            all_rgb_gt.append(rgb_gt)
            all_sem_pred.append(semantic_pred)
            all_sem_gt.append(semantic_gt)
    
    all_rgb_pred = torch.cat(all_rgb_pred, dim=0)
    all_rgb_gt = torch.cat(all_rgb_gt, dim=0)
    all_sem_pred = torch.cat(all_sem_pred, dim=0)
    all_sem_gt = torch.cat(all_sem_gt, dim=0)
    
    metrics = metrics_computer.compute_all_metrics(
        all_rgb_pred, all_rgb_gt,
        all_sem_pred, all_sem_gt,
        model=model
    )
    
    return metrics


def validate_rgb_only_model(checkpoint_path, config):
    """Ablation: Test model with semantic head disabled"""
    print("\n[ABLATION] RGB-Only Model (No Semantic)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with semantic head disabled
    pos_enc = PositionalEncoding(num_freqs=config.get('num_freqs', 10))
    model = SemanticNeRF(
        D=config.get('D', 8),
        W=config.get('W', 256),
        input_ch=pos_enc.output_dim,
        num_classes=config.get('num_classes', 10),
        skip_layer=config.get('skip_layer', 4)
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Disable semantic head (set to zero output)
    original_semantic_linear = model.semantic_linear
    model.semantic_linear = nn.Identity()
    model.eval()
    
    # Load dataset
    dataset = HeritageNeRFDataset(
        transforms_path=config['transforms_path'],
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        downscale=config.get('downscale', 1),
        N_rays=config.get('N_rays', 1024)
    )
    
    metrics_computer = MetricsComputer()
    
    all_rgb_pred = []
    all_rgb_gt = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Validating (RGB-only)"):
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            rgb_gt = batch['rgb_gt'].to(device)
            
            # Forward pass (without semantic)
            pts, z_vals = sample_points_along_rays(
                rays_o, rays_d,
                near=config.get('near', 2.0),
                far=config.get('far', 6.0),
                N_samples=config.get('N_samples', 64),
                perturb=False
            )
            
            pts_flat = pts.reshape(-1, 3)
            pts_encoded = pos_enc(pts_flat)
            
            rgb, sigma, _ = model(pts_encoded)
            
            N_rays = pts.shape[0]
            rgb = rgb.reshape(N_rays, -1, 3)
            sigma = sigma.reshape(N_rays, -1, 1)
            
            rgb_pred, _, _ = volume_rendering(
                rgb, sigma, z_vals, rays_d,
                semantic_logits=None  # No semantic
            )
            
            all_rgb_pred.append(rgb_pred)
            all_rgb_gt.append(rgb_gt)
    
    all_rgb_pred = torch.cat(all_rgb_pred, dim=0)
    all_rgb_gt = torch.cat(all_rgb_gt, dim=0)
    
    # Compute only RGB metrics
    psnr = metrics_computer.psnr(all_rgb_pred, all_rgb_gt)
    ssim = metrics_computer.ssim(all_rgb_pred, all_rgb_gt)
    
    metrics = {
        'psnr': psnr,
        'ssim': ssim,
        'gpu_memory_used_gb': metrics_computer.gpu_memory_usage()[0],
        'gpu_memory_total_gb': metrics_computer.gpu_memory_usage()[1],
    }
    
    return metrics


def run_ablation_study(checkpoint_path, config):
    """Run full ablation study"""
    print("\n" + "="*70)
    print("ABLATION STUDY: Semantic Head Impact")
    print("="*70)
    
    # Full model
    full_metrics = validate_full_model(checkpoint_path, config)
    
    # RGB-only ablation
    rgb_metrics = validate_rgb_only_model(checkpoint_path, config)
    
    # Compare
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    print("\nFull Model (RGB + Semantic):")
    print(f"  PSNR: {full_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {full_metrics['ssim']:.4f}")
    print(f"  Semantic Accuracy: {full_metrics['semantic_accuracy']:.4f}")
    print(f"  F1 Score: {full_metrics['f1_weighted']:.4f}")
    print(f"  mIoU: {full_metrics['miou']:.4f}")
    
    print("\nRGB-Only Model (No Semantic Head):")
    print(f"  PSNR: {rgb_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {rgb_metrics['ssim']:.4f}")
    
    print("\nDifference (Impact of Semantic Head):")
    psnr_diff = full_metrics['psnr'] - rgb_metrics['psnr']
    ssim_diff = full_metrics['ssim'] - rgb_metrics['ssim']
    print(f"  PSNR change: {psnr_diff:+.2f} dB ({(psnr_diff/rgb_metrics['psnr'])*100:+.1f}%)")
    print(f"  SSIM change: {ssim_diff:+.4f} ({(ssim_diff/rgb_metrics['ssim'])*100:+.1f}%)")
    
    print("\n✓ Ablation study complete!")
    print("="*70)
    
    # Save results
    ablation_results = {
        'full_model': full_metrics,
        'rgb_only': rgb_metrics,
        'differences': {
            'psnr_diff': psnr_diff,
            'ssim_diff': ssim_diff,
        }
    }
    
    return ablation_results


if __name__ == '__main__':
    config = {
        'transforms_path': 'data/Family/processed/transforms.json',
        'images_dir': 'data/Family',
        'masks_dir': 'data/Family/processed/semantic_masks',
        'N_rays': 1024,
        'N_samples': 64,
        'near': 2.0,
        'far': 6.0,
        'num_classes': 10,
        'num_freqs': 10,
        'D': 8,
        'W': 256,
        'skip_layer': 4,
        'downscale': 1,
    }
    
    checkpoint_path = 'checkpoints/model_final.pth'
    ablation_results = run_ablation_study(checkpoint_path, config)
    
    # Save
    with open('outputs/ablation_study.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

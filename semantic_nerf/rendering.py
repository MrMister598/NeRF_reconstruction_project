import torch
import numpy as np

def sample_points_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
    """
    Sample 3D points along camera rays
    
    Args:
        rays_o: [N_rays, 3] - ray origins (camera positions)
        rays_d: [N_rays, 3] - ray directions (normalized)
        near: float - near depth bound
        far: float - far depth bound
        N_samples: int - number of samples per ray
        perturb: bool - whether to perturb samples (stratified sampling)
        
    Returns:
        pts: [N_rays, N_samples, 3] - 3D points along rays
        z_vals: [N_rays, N_samples] - depth values
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device
    
    # Create evenly-spaced depth samples
    t_vals = torch.linspace(0.0, 1.0, N_samples, device=device)
    
    # Map from [0,1] to [near, far]
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [N_samples]
    z_vals = z_vals.expand(N_rays, N_samples)  # [N_rays, N_samples]
    
    # Stratified sampling: perturb within bins
    if perturb:
        # Midpoints between samples
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # [N_rays, N_samples-1]
        
        # Upper and lower bounds for perturbation
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)  # [N_rays, N_samples]
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)    # [N_rays, N_samples]
        
        # Random perturbation
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
    
    # Calculate 3D points: pts = o + t*d
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    # Result: [N_rays, N_samples, 3]
    
    return pts, z_vals


def volume_rendering(rgb, sigma, z_vals, rays_d, semantic_logits=None, white_bkgd=True):
    """
    Volume rendering using alpha compositing
    
    Args:
        rgb: [N_rays, N_samples, 3] - predicted RGB at each sample
        sigma: [N_rays, N_samples, 1] - density at each sample
        z_vals: [N_rays, N_samples] - depth values
        rays_d: [N_rays, 3] - ray directions
        semantic_logits: [N_rays, N_samples, num_classes] - semantic logits (optional)
        white_bkgd: bool - use white background
        
    Returns:
        rgb_map: [N_rays, 3] - final RGB for each ray
        depth_map: [N_rays] - estimated depth
        acc_map: [N_rays] - accumulated alpha (opacity)
        semantic_map: [N_rays, num_classes] - rendered semantic probabilities (if semantic_logits provided)
    """
    # Calculate distances between adjacent samples along ray
    dists = z_vals[:, 1:] - z_vals[:, :-1]  # [N_rays, N_samples-1]
    
    # Add large distance at end (to infinity)
    dists = torch.cat([dists, torch.ones_like(dists[:, :1]) * 1e10], dim=-1)  # [N_rays, N_samples]
    
    # Adjust for ray direction (account for non-unit ray directions)
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
    
    # ===== Alpha compositing equation =====
    # alpha_i = 1 - exp(-sigma_i * delta_i)
    # where delta_i is distance to next sample
    
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # [N_rays, N_samples]
    
    # Transmittance: T_i = prod(1 - alpha_j) for j < i
    # This is the probability that light reaches point i without being occluded
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1]], dim=-1),
        dim=-1
    )  # [N_rays, N_samples]
    
    # Weights: w_i = T_i * alpha_i
    # This is the contribution of sample i to the final color
    weights = alpha * transmittance  # [N_rays, N_samples]
    
    # Composite RGB: final_rgb = sum(w_i * rgb_i)
    rgb_map = torch.sum(weights[:, :, None] * rgb, dim=1)  # [N_rays, 3]
    
    # Composite depth: estimated_depth = sum(w_i * z_i)
    depth_map = torch.sum(weights * z_vals, dim=1)  # [N_rays]
    
    # Accumulated opacity: acc = sum(w_i)
    acc_map = torch.sum(weights, dim=1)  # [N_rays]
    
    # Add white background (1 - acc_map is the background)
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[:, None])
    
    # ===== Semantic rendering (OUR CONTRIBUTION) =====
    semantic_map = None
    if semantic_logits is not None:
        # Convert logits to probabilities via softmax
        semantic_probs = torch.softmax(semantic_logits, dim=-1)  # [N_rays, N_samples, num_classes]
        
        # Weight by rendering weights
        # weights[:, :, None] is [N_rays, N_samples, 1]
        # semantic_probs is [N_rays, N_samples, num_classes]
        semantic_map = torch.sum(
            weights[:, :, None] * semantic_probs,
            dim=1
        )  # [N_rays, num_classes]
    
    return rgb_map, depth_map, acc_map, semantic_map


def get_rays(H, W, focal, c2w):
    """
    Generate camera rays for an image
    
    Args:
        H, W: image height and width
        focal: focal length (in pixels)
        c2w: [4, 4] camera-to-world transformation matrix
        
    Returns:
        rays_o: [H*W, 3] ray origins
        rays_d: [H*W, 3] ray directions
    """
    # Create pixel coordinate grids
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    
    # Convert pixel coordinates to normalized device coordinates
    # [-1, 1] in x direction, [-1, 1] in y direction
    dirs = torch.stack([
        (i - W/2) / focal,
        -(j - H/2) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    # dirs shape: [H, W, 3]
    
    # Rotate ray directions to world space
    # c2w[:3, :3] is the rotation matrix
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)  # [H, W, 3]
    
    # Ray origin is camera position (same for all rays)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # [H, W, 3]
    
    # Reshape to [H*W, 3]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    return rays_o, rays_d

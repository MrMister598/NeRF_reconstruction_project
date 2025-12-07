import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
import time
import psutil
import GPUtil

class MetricsComputer:
    """
    Comprehensive metrics computation for Semantic-NeRF
    Includes: PSNR, SSIM, Semantic Accuracy, F1, IoU, GPU memory, FPS
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== RGB Metrics ==========
    
    @staticmethod
    def psnr(pred, target):
        """
        Peak Signal-to-Noise Ratio (dB)
        Higher is better (>25 dB = good, >30 dB = excellent)
        
        Args:
            pred: [N, 3] predicted RGB
            target: [N, 3] ground truth RGB
            
        Returns:
            psnr: scalar PSNR value
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'), device=pred.device)
        
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def ssim(pred, target, window_size=11, reduction='mean'):
        """
        Structural Similarity Index Measure
        Range: -1 to 1 (1 = identical, higher is better)
        
        Args:
            pred: [N, 3] or [H, W, 3]
            target: [N, 3] or [H, W, 3]
            window_size: Gaussian window size
            
        Returns:
            ssim: scalar SSIM value
        """
        # Constants to avoid division by zero
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        # Reshape if needed
        if pred.dim() == 2:
            B = 1
            pred = pred.unsqueeze(0).permute(0, 2, 1).reshape(B, 3, -1, 1)
            target = target.unsqueeze(0).permute(0, 2, 1).reshape(B, 3, -1, 1)
        
        # Compute means
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        
        # Compute variances
        sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu1 ** 2
        sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu2 ** 2
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1 * mu2
        
        # SSIM formula
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if reduction == 'mean':
            return ssim.mean().item()
        else:
            return ssim
    
    # ========== Semantic Metrics ==========
    
    @staticmethod
    def semantic_accuracy(pred, target, ignore_index=-1):
        """
        Semantic segmentation accuracy
        
        Args:
            pred: [N] or [H, W] predicted class indices
            target: [N] or [H, W] ground truth class indices
            ignore_index: class index to ignore (typically -1 for unlabeled)
            
        Returns:
            accuracy: scalar accuracy value (0-1)
        """
        # Flatten
        pred_flat = pred.reshape(-1).cpu().numpy()
        target_flat = target.reshape(-1).cpu().numpy()
        
        # Create mask for valid pixels
        valid_mask = target_flat != ignore_index
        
        if valid_mask.sum() == 0:
            return 0.0
        
        # Compute accuracy on valid pixels
        accuracy = accuracy_score(
            target_flat[valid_mask],
            pred_flat[valid_mask]
        )
        
        return float(accuracy)
    
    @staticmethod
    def f1_score(pred, target, num_classes=10, ignore_index=-1, average='weighted'):
        """
        F1 Score (micro, macro, or weighted)
        Range: 0-1 (1 = perfect, higher is better)
        
        Args:
            pred: [N] or [H, W] predicted class indices
            target: [N] or [H, W] ground truth class indices
            num_classes: number of semantic classes
            ignore_index: class index to ignore
            average: 'micro', 'macro', or 'weighted'
            
        Returns:
            f1: scalar F1 score
        """
        # Flatten
        pred_flat = pred.reshape(-1).cpu().numpy()
        target_flat = target.reshape(-1).cpu().numpy()
        
        # Create mask for valid pixels
        valid_mask = target_flat != ignore_index
        
        if valid_mask.sum() == 0:
            return 0.0
        
        # Compute F1 score on valid pixels
        f1 = f1_score(
            target_flat[valid_mask],
            pred_flat[valid_mask],
            labels=list(range(num_classes)),
            average=average,
            zero_division=0
        )
        
        return float(f1)
    
    @staticmethod
    def iou(pred, target, num_classes=10, ignore_index=-1):
        """
        Intersection over Union (Jaccard Index)
        Per-class and mean IoU
        Range: 0-1 (1 = perfect, higher is better)
        
        Args:
            pred: [N] or [H, W] predicted class indices
            target: [N] or [H, W] ground truth class indices
            num_classes: number of semantic classes
            ignore_index: class index to ignore
            
        Returns:
            miou: mean IoU across all classes
            iou_per_class: dict of per-class IoU values
        """
        # Flatten
        pred_flat = pred.reshape(-1).cpu().numpy()
        target_flat = target.reshape(-1).cpu().numpy()
        
        # Create mask for valid pixels
        valid_mask = target_flat != ignore_index
        
        if valid_mask.sum() == 0:
            return 0.0, {}
        
        iou_per_class = {}
        valid_classes = []
        
        for class_idx in range(num_classes):
            # Binary mask for this class
            pred_class = (pred_flat[valid_mask] == class_idx)
            target_class = (target_flat[valid_mask] == class_idx)
            
            # Intersection and Union
            intersection = (pred_class & target_class).sum()
            union = (pred_class | target_class).sum()
            
            if union == 0:
                # No pixels of this class in target
                iou_per_class[f"class_{class_idx}"] = 0.0
            else:
                iou = intersection / union
                iou_per_class[f"class_{class_idx}"] = float(iou)
                valid_classes.append(iou)
        
        # Mean IoU over classes that exist in target
        miou = np.mean(valid_classes) if valid_classes else 0.0
        
        return float(miou), iou_per_class
    
    # ========== Efficiency Metrics ==========
    
    @staticmethod
    def gpu_memory_usage():
        """
        Get current GPU memory usage in GB
        
        Returns:
            used_gb: used memory (GB)
            total_gb: total memory (GB)
            percent: percentage used
        """
        if torch.cuda.is_available():
            gpu_info = GPUtil.getGPUs()[0]
            used_gb = gpu_info.memoryUsed / 1024.0
            total_gb = gpu_info.memoryTotal / 1024.0
            percent = gpu_info.memoryUtil
            
            return used_gb, total_gb, percent
        else:
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def cpu_memory_usage():
        """
        Get CPU memory usage in GB
        
        Returns:
            used_gb: used memory (GB)
            total_gb: total memory (GB)
            percent: percentage used
        """
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        
        return used_gb, total_gb, mem.percent
    
    @staticmethod
    def inference_fps(model, num_samples=100, num_rays=1024, num_points=64):
        """
        Measure inference FPS for NeRF
        
        Args:
            model: trained NeRF model
            num_samples: number of inference runs
            num_rays: rays per batch
            num_points: points sampled per ray
            
        Returns:
            fps: frames per second
            avg_time_ms: average inference time (milliseconds)
        """
        device = next(model.parameters()).device
        
        # Create dummy input
        x = torch.randn(num_rays * num_points, 63, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_samples):
                _ = model(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / num_samples) * 1000
        fps = num_samples / elapsed
        
        return fps, avg_time_ms
    
    # ========== Composite Metrics ==========
    
    def compute_all_metrics(self, rgb_pred, rgb_gt, semantic_pred, semantic_gt, model=None):
        """
        Compute all metrics at once
        
        Args:
            rgb_pred: [N, 3] predicted RGB
            rgb_gt: [N, 3] ground truth RGB
            semantic_pred: [N, num_classes] predicted semantic logits
            semantic_gt: [N] ground truth semantic labels
            model: NeRF model (optional, for FPS measurement)
            
        Returns:
            metrics_dict: dictionary of all computed metrics
        """
        # Get predicted semantic classes
        semantic_pred_classes = torch.argmax(semantic_pred, dim=-1)
        
        metrics = {
            # RGB metrics
            'psnr': self.psnr(rgb_pred, rgb_gt),
            'ssim': self.ssim(rgb_pred, rgb_gt),
            
            # Semantic metrics
            'semantic_accuracy': self.semantic_accuracy(semantic_pred_classes, semantic_gt),
            'f1_weighted': self.f1_score(semantic_pred_classes, semantic_gt, average='weighted'),
            'f1_macro': self.f1_score(semantic_pred_classes, semantic_gt, average='macro'),
            'miou': self.iou(semantic_pred_classes, semantic_gt)[0],
        }
        
        # GPU metrics
        gpu_used, gpu_total, gpu_pct = self.gpu_memory_usage()
        metrics['gpu_memory_used_gb'] = gpu_used
        metrics['gpu_memory_total_gb'] = gpu_total
        metrics['gpu_memory_percent'] = gpu_pct
        
        # CPU metrics
        cpu_used, cpu_total, cpu_pct = self.cpu_memory_usage()
        metrics['cpu_memory_used_gb'] = cpu_used
        metrics['cpu_memory_total_gb'] = cpu_total
        metrics['cpu_memory_percent'] = cpu_pct
        
        # FPS (if model provided)
        if model is not None:
            fps, avg_time_ms = self.inference_fps(model)
            metrics['inference_fps'] = fps
            metrics['inference_time_ms'] = avg_time_ms
        
        return metrics
    
    def print_metrics(self, metrics):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("COMPREHENSIVE METRICS REPORT")
        print("="*60)
        
        print("\n📊 RGB RECONSTRUCTION METRICS:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB (target: >25 dB)")
        print(f"  SSIM: {metrics['ssim']:.4f} (range: -1 to 1)")
        
        print("\n🎯 SEMANTIC SEGMENTATION METRICS:")
        print(f"  Accuracy: {metrics['semantic_accuracy']:.4f} ({metrics['semantic_accuracy']*100:.2f}%)")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  mIoU: {metrics['miou']:.4f}")
        
        print("\n💾 MEMORY USAGE:")
        print(f"  GPU: {metrics['gpu_memory_used_gb']:.2f} / {metrics['gpu_memory_total_gb']:.2f} GB ({metrics['gpu_memory_percent']:.1f}%)")
        print(f"  CPU: {metrics['cpu_memory_used_gb']:.2f} / {metrics['cpu_memory_total_gb']:.2f} GB ({metrics['cpu_memory_percent']:.1f}%)")
        
        print("\n⚡ INFERENCE SPEED:")
        if 'inference_fps' in metrics:
            print(f"  FPS: {metrics['inference_fps']:.2f} frames/second")
            print(f"  Time per frame: {metrics['inference_time_ms']:.2f} ms")
        
        print("="*60 + "\n")

import torch
import torch.nn.functional as F


def compute_loss(rgb_pred, rgb_gt, semantic_pred, semantic_gt, lambda_semantic=0.1):
    """
    Combined RGB + Semantic loss with accuracy tracking

    Args:
        rgb_pred: [N, 3] - predicted RGB
        rgb_gt: [N, 3] - ground truth RGB
        semantic_pred: [N, num_classes] - predicted semantic logits,
                       or None if no valid semantics in this batch
        semantic_gt: [N] - ground truth semantic class indices,
                     or None if no valid semantics in this batch
        lambda_semantic: weight for semantic loss (balance term)

    Returns:
        total_loss: combined loss
        rgb_loss: RGB reconstruction loss
        semantic_loss: semantic classification loss
        semantic_accuracy: accuracy of semantic predictions
    """
    # RGB reconstruction loss (MSE)
    rgb_loss = F.mse_loss(rgb_pred, rgb_gt)

    # If we have no semantic supervision in this batch, skip semantic loss
    if semantic_pred is None or semantic_gt is None:
        semantic_loss = torch.tensor(0.0, device=rgb_pred.device)
        semantic_accuracy = torch.tensor(0.0, device=rgb_pred.device)
    else:
        # Only compute where we have valid ground truth labels (>= 0)
        valid_mask = semantic_gt >= 0  # -1 means unlabeled

        if valid_mask.sum() > 0:
            semantic_loss = F.cross_entropy(
                semantic_pred[valid_mask],
                semantic_gt[valid_mask]
            )

            with torch.no_grad():
                pred_classes = torch.argmax(semantic_pred[valid_mask], dim=-1)
                semantic_accuracy = (pred_classes == semantic_gt[valid_mask]).float().mean()
        else:
            semantic_loss = torch.tensor(0.0, device=rgb_pred.device)
            semantic_accuracy = torch.tensor(0.0, device=rgb_pred.device)

    total_loss = rgb_loss + lambda_semantic * semantic_loss
    return total_loss, rgb_loss, semantic_loss, semantic_accuracy


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


def compute_ssim(pred, target, window_size=11):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu2 ** 2
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1 * mu2

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.mean()

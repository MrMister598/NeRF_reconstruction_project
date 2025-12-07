import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class FeatureExtractor(nn.Module):
    """
    Learn to detect keypoints (like SIFT but learnable)
    Uses CNN to learn discriminative image patches
    """
    def __init__(self, out_channels=128):
        super().__init__()
        
        # Learned feature pyramid
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, out_channels, 1)
        
    def forward(self, img):
        """
        img: [B, 3, H, W]
        returns: features [B, C, H/4, W/4]
        """
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def extract_keypoints_simple(image):
    """
    Extract keypoints using Harris corner detection
    (simpler than SIFT, but fast and learnable)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Harris corners
    corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    
    # Threshold and get coordinates
    _, corners_binary = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
    contours, _ = cv2.findContours(corners_binary, cv2.RETR_TREE, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            keypoints.append([cx, cy])
    
    return np.array(keypoints) if keypoints else np.array([])

def extract_descriptors(image, keypoints, patch_size=16):
    """
    Extract local patches around keypoints as descriptors
    """
    descriptors = []
    h, w = image.shape[:2]
    
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        
        # Get patch (with bounds checking)
        x1 = max(0, x - patch_size//2)
        x2 = min(w, x + patch_size//2)
        y1 = max(0, y - patch_size//2)
        y2 = min(h, y + patch_size//2)
        
        patch = image[y1:y2, x1:x2]
        
        # Resize to fixed size
        patch = cv2.resize(patch, (patch_size, patch_size))
        
        # Normalize
        patch = (patch.astype(np.float32) - patch.mean()) / (patch.std() + 1e-6)
        
        descriptors.append(patch.flatten())
    
    return np.array(descriptors) if descriptors else np.array([])

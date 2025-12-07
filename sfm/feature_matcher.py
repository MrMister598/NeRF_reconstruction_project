import numpy as np
from scipy.spatial.distance import cdist

def match_features(desc1, desc2, ratio_test=0.75):
    """
    Match descriptors between two images using Lowe's ratio test
    
    desc1, desc2: [N, descriptor_dim]
    returns: matches (pairs of keypoint indices)
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return np.array([])
    
    # Compute pairwise distances
    distances = cdist(desc1, desc2, metric='euclidean')
    
    matches = []
    for i in range(len(desc1)):
        # Find two nearest neighbors in desc2
        sorted_indices = np.argsort(distances[i])
        
        if len(sorted_indices) >= 2:
            nearest = sorted_indices[0]
            second_nearest = sorted_indices[1]
            
            # Lowe's ratio test: accept if best is significantly better than second-best
            if distances[i, nearest] < ratio_test * distances[i, second_nearest]:
                matches.append([i, nearest])
    
    return np.array(matches)

def compute_fundamental_matrix(pts1, pts2):
    """
    Compute fundamental matrix using 8-point algorithm
    F relates corresponding points in two images: p2^T F p1 = 0
    """
    # Normalize points
    pts1_norm = (pts1 - pts1.mean(axis=0)) / (pts1.std() + 1e-6)
    pts2_norm = (pts2 - pts2.mean(axis=0)) / (pts2.std() + 1e-6)
    
    # Build A matrix
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    
    # SVD: F is last column of V
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    
    return F

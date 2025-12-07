import numpy as np

def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate 3D points from two camera views
    
    P1, P2: [3, 4] projection matrices
    pts1, pts2: [N, 2] image coordinates
    
    returns: points_3d [N, 3]
    """
    points_3d = []
    
    for pt1, pt2 in zip(pts1, pts2):
        # Build system of equations
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1]
        ])
        
        # SVD: solution is last column of V
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        # Normalize (convert from homogeneous to 3D)
        X = X / (X[3] + 1e-6)
        points_3d.append(X[:3])
    
    return np.array(points_3d)

def recover_pose(F, pts1, pts2, K):
    """
    Recover camera pose [R | t] from fundamental matrix
    
    F: fundamental matrix
    pts1, pts2: corresponding points
    K: camera intrinsic matrix
    
    returns: R, t (rotation and translation)
    """
    # E = K^T F K (essential matrix)
    E = K.T @ F @ K
    
    # SVD: E = U D V^T, where D = diag(1, 1, 0)
    U, _, Vt = np.linalg.svd(E)
    
    # W matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Four possible solutions
    solutions = [
        (U @ W @ Vt, U[:, 2]),
        (U @ W @ Vt, -U[:, 2]),
        (U @ W.T @ Vt, U[:, 2]),
        (U @ W.T @ Vt, -U[:, 2]),
    ]
    
    # Choose solution with most points in front of both cameras
    best_R, best_t = solutions[0]
    max_points_in_front = 0
    
    for R, t in solutions:
        points_3d = triangulate_points(
            K @ np.hstack([np.eye(3), np.zeros((3,1))]),
            K @ np.hstack([R, t[:, None]]),
            pts1, pts2
        )
        
        # Check how many points are in front
        points_in_front = np.sum(points_3d[:, 2] > 0)
        
        if points_in_front > max_points_in_front:
            max_points_in_front = points_in_front
            best_R, best_t = R, t
    
    return best_R, best_t

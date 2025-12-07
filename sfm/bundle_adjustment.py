import numpy as np
from scipy.optimize import minimize

def bundle_adjustment(cameras, points_3d, observations):
    """
    Optimize camera poses and 3D points to minimize reprojection error
    
    cameras: list of (R, t, K) for each view
    points_3d: [M, 3] 3D point coordinates
    observations: [M, N] - which points are visible in which cameras
    
    returns: optimized cameras and points_3d
    """
    
    def reprojection_error(params):
        """Compute sum of squared reprojection errors"""
        error = 0
        
        # Unpack params
        num_cameras = len(cameras)
        camera_params = params[:num_cameras * 6]  # 6 params per camera (3 rot, 3 trans)
        point_params = params[num_cameras * 6:]    # 3 params per point
        
        for cam_idx, (cam) in enumerate(cameras):
            R, t, K = cam
            
            for point_idx in range(len(points_3d)):
                if observations[point_idx, cam_idx] == 0:
                    continue  # Point not visible in this camera
                
                # Project point
                pt_3d = point_params[point_idx*3:(point_idx+1)*3]
                pt_2d_proj = K @ (R @ pt_3d + t)
                pt_2d_proj = pt_2d_proj[:2] / pt_2d_proj[2]
                
                # Observed point (would come from detected features)
                # pt_2d_obs = ... (from initial feature detection)
                
                # Squared error
                # error += np.sum((pt_2d_proj - pt_2d_obs) ** 2)
        
        return error
    
    # Initial guess
    initial_params = np.concatenate([
        np.concatenate([[r.flatten(), t] for r, t, k in cameras]),
        points_3d.flatten()
    ])
    
    # Minimize
    result = minimize(reprojection_error, initial_params, method='Levenberg-Marquardt')
    
    return result.x

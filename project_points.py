import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """

    # [TODO] get image coordinates
    homogeneous_2d  = K @ points_3d.T

    x_i = homogeneous_2d[0, :] / homogeneous_2d[2, :]
    y_i = homogeneous_2d[1, :] / homogeneous_2d[2, :]

    image_coords = np.column_stack((x_i, y_i))
    
    # [TODO] apply distortion
    
    projected_points = distort_points(image_coords, D, K) 

    return projected_points

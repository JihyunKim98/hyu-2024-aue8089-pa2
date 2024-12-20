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
    num_points = points_3d.shape[1]

    homogeneous_3d = np.vstack((points_3d, np.ones((1, num_points))))
    homogeneous_2d = np.dot(K, homogeneous_3d[:3, :])
    
    x_c = homogeneous_2d[0, :]
    y_c = homogeneous_2d[1, :]
    z_c = homogeneous_2d[2, :]

    x_i = x_c/z_c
    y_i = y_c/z_c

    r_squared = x_i**2 + y_i**2
    
    # [TODO] apply distortion
    
    radial_distortion = 1 + D[0] * r_squared + D[1] * r_squared**2

    x_distorted = x_i * radial_distortion
    y_distorted = y_i * radial_distortion     

    projected_points = np.vstack((x_distorted, y_distorted)).T

    return projected_points

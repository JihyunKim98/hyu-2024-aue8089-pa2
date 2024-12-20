import math
import numpy as np

from distort_points import distort_points


def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """

    height, width, *_ = img.shape
    undistorted_img = np.zeros_like(img)

    for x in range(width):
        for y in range(height):

            # apply distortion
            x_d = distort_points(np.array([[x, y]]), D, K)
            u, v = x_d[0, :]

            # bilinear interpolation
            u1 = math.floor(u)
            v1 = math.floor(v)

            in_image = (u1 >= 0) & (u1+1 < width) & (v1 >= 0) & (v1+1 < height)
            if not in_image:
                continue

            if bilinear_interpolation:
                a = u - u1
                b = v - v1

                # [TODO] weighted sum of pixel values in img
                w_0 = (1-a) * (1-b)
                w_1 = a * (1-b)
                w_2 = (1-a) * b
                w_3 = a * b

                value = (img[v1, u1] * w_0 +
                         img[v1, u1+1] * w_1 +
                         img[v1+1, u1] * w_2 +
                         img[v1+1, u1] * w_3)
                
                undistorted_img[y, x] = value
            else:
                # [TODO] nearest neighbor
                u_round = int(round(u))
                v_round = int(round(v))

                if 0 <= u_round < width and 0 <= v_round < height:
                    undistorted_img[y, x] = img[v_round, u_round]

    return undistorted_img
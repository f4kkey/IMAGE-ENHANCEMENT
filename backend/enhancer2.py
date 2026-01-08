import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def bilateral_filter_paper_gray(image, sigma_d=3.0, sigma_r=25.0, window_radius=None):
    f = image.astype(np.float64)
    H, W = f.shape

    if window_radius is None:
        window_radius = int(3 * sigma_d)
    win_size = 2 * window_radius + 1
    yy, xx = np.mgrid[-window_radius:window_radius+1,
                      -window_radius:window_radius+1]
    spatial_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_d**2))
    h = np.zeros_like(f)

    for i in range(H):
        i_min = max(i - window_radius, 0)
        i_max = min(i + window_radius + 1, H)

        for j in range(W):
            j_min = max(j - window_radius, 0)
            j_max = min(j + window_radius + 1, W)

            patch = f[i_min:i_max, j_min:j_max]


            k_i_min = i_min - (i - window_radius)
            k_i_max = win_size - ((i + window_radius + 1) - i_max)
            k_j_min = j_min - (j - window_radius)
            k_j_max = win_size - ((j + window_radius + 1) - j_max)
            c_xi_x = spatial_kernel[k_i_min:k_i_max, k_j_min:k_j_max]
            center_val = f[i, j]
            s_fx_fxi = np.exp(-(patch - center_val)**2 / (2 * sigma_r**2))
            weights = c_xi_x * s_fx_fxi
            k_x = np.sum(weights)

            if k_x > 1e-12:
                h[i, j] = np.sum(patch * weights) / k_x
            else:
                h[i, j] = center_val

    return h

def run_bilateral_enhancement(input_path, img_id, sigma_color=25.0, sigma_space=3.0):
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    filtered = bilateral_filter_paper_gray(gray, sigma_d=sigma_space, sigma_r=sigma_color)

    result_name = f"{img_id}_bilateral.png"
    result_path = os.path.join(UPLOAD_DIR, result_name)
    cv2.imwrite(result_path, filtered)

    return {"bilateral": result_name}

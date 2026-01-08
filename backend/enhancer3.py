import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def box_filter(img, r):
    return cv2.boxFilter(img, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter_gray(I, p, r, eps):
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    ones = np.ones_like(I, dtype=np.float32)
    N = box_filter(ones, r) 
    mean_I = box_filter(I, r) / N
    mean_p = box_filter(p, r) / N

    mean_Ip = box_filter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box_filter(a, r) / N
    mean_b = box_filter(b, r) / N

    q = mean_a * I + mean_b

    return q

def guided_filter_color(I, p, r, eps):
    I = I.astype(np.float32)
    if p.ndim == 2:
        p = p.astype(np.float32)
        return _guided_filter_color_single_channel(I, p, r, eps)
    elif p.ndim == 3 and p.shape[2] == 3:
        out = np.zeros_like(p, dtype=np.float32)
        for c in range(3):
            out[:, :, c] = _guided_filter_color_single_channel(I, p[:, :, c], r, eps)
        return out
    else:
        raise ValueError("must be shape (H,W) or (H,W,3)")


def _guided_filter_color_single_channel(I, p, r, eps):
    h, w, _ = I.shape
    I_r = I[:, :, 0]
    I_g = I[:, :, 1]
    I_b = I[:, :, 2]

    p = p.astype(np.float32)

    ones = np.ones((h, w), dtype=np.float32)

    N = box_filter(ones, r)

    mean_I_r = box_filter(I_r, r) / N
    mean_I_g = box_filter(I_g, r) / N
    mean_I_b = box_filter(I_b, r) / N

    mean_p = box_filter(p, r) / N

    mean_Ip_r = box_filter(I_r * p, r) / N
    mean_Ip_g = box_filter(I_g * p, r) / N
    mean_Ip_b = box_filter(I_b * p, r) / N

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p

    mean_II_rr = box_filter(I_r * I_r, r) / N
    mean_II_rg = box_filter(I_r * I_g, r) / N
    mean_II_rb = box_filter(I_r * I_b, r) / N
    mean_II_gg = box_filter(I_g * I_g, r) / N
    mean_II_gb = box_filter(I_g * I_b, r) / N
    mean_II_bb = box_filter(I_b * I_b, r) / N

    var_rr = mean_II_rr - mean_I_r * mean_I_r
    var_rg = mean_II_rg - mean_I_r * mean_I_g
    var_rb = mean_II_rb - mean_I_r * mean_I_b
    var_gg = mean_II_gg - mean_I_g * mean_I_g
    var_gb = mean_II_gb - mean_I_g * mean_I_b
    var_bb = mean_II_bb - mean_I_b * mean_I_b

    a_r = np.zeros((h, w), dtype=np.float32)
    a_g = np.zeros((h, w), dtype=np.float32)
    a_b = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)

    eps_I = eps

    for y in range(h):
        for x in range(w):
            Sigma = np.array([
                [var_rr[y, x] + eps_I, var_rg[y, x],           var_rb[y, x]],
                [var_rg[y, x],          var_gg[y, x] + eps_I,  var_gb[y, x]],
                [var_rb[y, x],          var_gb[y, x],          var_bb[y, x] + eps_I]
            ], dtype=np.float32)

            cov_Ip = np.array([
                cov_Ip_r[y, x],
                cov_Ip_g[y, x],
                cov_Ip_b[y, x]
            ], dtype=np.float32)

            a = np.linalg.solve(Sigma, cov_Ip)

            a_r[y, x], a_g[y, x], a_b[y, x] = a[0], a[1], a[2]

    b = mean_p - (a_r * mean_I_r + a_g * mean_I_g + a_b * mean_I_b)
    mean_a_r = box_filter(a_r, r) / N
    mean_a_g = box_filter(a_g, r) / N
    mean_a_b = box_filter(a_b, r) / N
    mean_b = box_filter(b, r) / N

    q = mean_a_r * I_r + mean_a_g * I_g + mean_a_b * I_b + mean_b

    return q

def run_guided_filter(input_path, img_id, radius=8, eps=0.001):
    img = cv2.imread(input_path)  
    guide = img.copy()
    src = img.copy()

    guide_f = guide.astype(np.float32) / 255.0
    src_f = src.astype(np.float32) / 255.0
    guided = guided_filter_gray(guide_f[:, :, 0], src_f[:, :, 0], radius, eps)

    out = np.clip(guided * 255.0, 0, 255).astype(np.uint8)
    result_name = f"{img_id}_guided.png"
    result_path = os.path.join(UPLOAD_DIR, result_name)
    cv2.imwrite(result_path, out)

    return {"guided": result_name}

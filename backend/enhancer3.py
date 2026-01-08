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

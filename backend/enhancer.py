import os
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def clustering_filter(img, alpha=0.5, radius=7, num_inner_iter=3):
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    ys, xs = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    spatial_w = np.exp(-alpha * (xs**2 + ys**2)).astype(np.float32)

    for y in range(H):
        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)
        ky0 = y0 - (y - radius)
        ky1 = ky0 + (y1 - y0)

        for x in range(W):
            x0 = max(0, x - radius)
            x1 = min(W, x + radius + 1)
            kx0 = x0 - (x - radius)
            kx1 = kx0 + (x1 - x0)

            patch = img[y0:y1, x0:x1]
            w_spatial = spatial_w[ky0:ky1, kx0:kx1]

            sum_w = np.sum(w_spatial)
            if sum_w <= 0:
                out[y, x] = img[y, x]
                continue

            y_bar = np.sum(patch * w_spatial) / sum_w
            var_y = np.sum((patch - y_bar)**2 * w_spatial) / sum_w
            var_y = max(var_y, 1e-8) 

            p = 1.0 / (2.0 * var_y)

            y_curr = y_bar
            for _ in range(num_inner_iter):
                w_range = np.exp(-p * (patch - y_curr)**2).astype(np.float32)
                w_total = w_spatial * w_range

                sum_wt = np.sum(w_total)
                if sum_wt <= 1e-12:
                    break

                y_next = np.sum(patch * w_total) / sum_wt

                if abs(y_next - y_curr) < 1e-4:
                    y_curr = y_next
                    break
                y_curr = y_next

            out[y, x] = y_curr

    return out


def edge_preserving_smoothing(img, iterations=5, alpha=0.5, radius=7):
    out = img.copy()
    for _ in range(iterations):
        out = clustering_filter(out, alpha=alpha, radius=radius, num_inner_iter=3)
    return out


def run_enhancement(
        input_path, img_id,
        alpha=0.5,
        kernel_size=15,
        k=5,
        local_var_size=40,
        threshold=2.5,
        final_weight=0.5
    ):

    img = Image.open(input_path).convert("L")
    I_uint8 = np.array(img, dtype=np.uint8)
    I = I_uint8.astype(np.float32) / 255.0 

    step_paths = {}

    # Step 1:
    radius = kernel_size // 2
    print(f"Edge-preserving smoothing: k={k}, alpha={alpha}, radius={radius}")
    Ik = edge_preserving_smoothing(I, iterations=k, alpha=alpha, radius=radius)

    Ik_show = np.clip(Ik * 255.0, 0, 255).astype(np.uint8)
    mask_name = f"{img_id}_mask.png"
    Image.fromarray(Ik_show).save(os.path.join(UPLOAD_DIR, mask_name))
    step_paths["mask"] = mask_name

    # Step 2:
    Id = I - Ik

    Id_min, Id_max = Id.min(), Id.max()
    Id_norm = (Id - Id_min) / (Id_max - Id_min + 1e-8)
    Id_show = np.clip(Id_norm * 255.0, 0, 255).astype(np.uint8)
    diff_name = f"{img_id}_diff.png"
    Image.fromarray(Id_show).save(os.path.join(UPLOAD_DIR, diff_name))
    step_paths["diff"] = diff_name

    # Step 3:
    win_size = local_var_size
    kernel = (win_size, win_size)

    Id_f32 = Id.astype(np.float32)
    M = cv2.blur(Id_f32, kernel)       
    Id2 = Id_f32 * Id_f32
    M2 = cv2.blur(Id2, kernel)            
    V = M2 - M * M                      
    V = np.maximum(V, 1e-8)

    # Step 4:
    thresh_img = threshold * np.sqrt(V)
    mask_condition = np.abs(Id_f32 - M) < thresh_img

    Im = np.where(mask_condition, Ik, I)

    Im_show = np.clip(Im * 255.0, 0, 255).astype(np.uint8)
    Im_name = f"{img_id}_Im.png"
    Image.fromarray(Im_show).save(os.path.join(UPLOAD_DIR, Im_name))
    step_paths["Im"] = Im_name

    # Step 5:
    s = final_weight
    Ie = I - s * Im
    
    Ie_raw = np.clip(Ie, 0.0, 1.0)
    Ie_raw_u8 = (Ie_raw * 255.0).astype(np.uint8)
    final_name = f"{img_id}_final.png"
    Image.fromarray(Ie_raw_u8).save(os.path.join(UPLOAD_DIR, final_name))
    step_paths["final"] = final_name

    # Step 6: 
    m = Ie.mean()
    std = Ie.std()

    low = m - threshold * std
    high = m + threshold * std

    if high <= low:
        low, high = Ie.min(), Ie.max() + 1e-6

    Ie_clipped = np.clip(Ie, low, high)
    Ie_norm = (Ie_clipped - low) / (high - low + 1e-8)

    result_u8 = np.clip(Ie_norm * 255.0, 0, 255).astype(np.uint8)
    result_name = f"{img_id}_result.png"
    Image.fromarray(result_u8).save(os.path.join(UPLOAD_DIR, result_name))
    step_paths["result"] = result_name

    return step_paths

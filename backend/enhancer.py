import numpy as np
from PIL import Image
import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def exponetial_matrix(image, i, j, beta, kernel_size):
    padding = kernel_size // 2
    perception_view = image[i-padding:i+padding+1, j-padding:j+padding+1].copy()
    cell_value = image[i, j].copy()
    dif = perception_view - cell_value
    return np.exp(-beta * dif ** 2)

def enhance_a_pixel(image, i, j, w, beta, kernel_size):
    padding = kernel_size // 2
    perception_view = image[i-padding:i+padding+1, j-padding:j+padding+1].copy()
    exponetial = exponetial_matrix(image, i, j, beta, kernel_size)
    return np.sum(w * exponetial * perception_view) / np.sum(w * exponetial)

def cluster_filter(image, alpha=0.5, kernel_size=15):

    I = image.astype(np.float32)
    dist = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for x in range(kernel_size):
        for y in range(kernel_size):
            dist[x, y] = np.sqrt((x - kernel_size//2)**2 + (y - kernel_size//2)**2)
    w = np.exp(-alpha * dist)

    padded = np.pad(I, (kernel_size // 2,kernel_size // 2), mode='edge')
    res = np.zeros(I.shape, dtype=np.float32)
    
    # for i in range(res.shape[0]):
    #     for j in range(res.shape[1]):
    #         beta = calculate_beta(padded, i + kernel_size // 2, j + kernel_size // 2, w, kernel_size)
    #         res[i, j] = enhance_a_pixel(padded, i + kernel_size // 2, j + kernel_size // 2, w, beta, kernel_size)
    
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):

            patch = padded[i:i+kernel_size, j:j+kernel_size].copy()

            y_bar = np.sum(w * patch) / np.sum(w)

            sigma2 = np.sum(w * (patch - y_bar)**2) / np.sum(w)

            p = 1.0 / (2 * sigma2 + 1e-8)

            similarity = np.exp(-p * (patch - padded[i + kernel_size // 2, j + kernel_size // 2])**2)

            numerator = np.sum((w * similarity) * patch)
            denominator = np.sum(w * similarity)

            res[i, j] = numerator / (denominator + 1e-8)
            
    return res


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
    I = np.array(img, dtype=np.float32)

    step_paths = {}

    # Step 1:
    I_i = I.copy()
    for _ in range(k):
        I_i = cluster_filter(I_i, alpha, kernel_size)
    I_i_img = Image.fromarray(np.clip(I_i, 0, 255).astype(np.uint8))
    I_i_name = f"{img_id}_mask.png"
    I_i_img.save(os.path.join(UPLOAD_DIR, I_i_name))
    step_paths["mask"] = I_i_name

    # Step 2:
    I_d = (I - I_i).copy()

    diff_img = Image.fromarray(np.clip(I_d, 0, 255).astype(np.uint8))
    diff_name = f"{img_id}_diff.png"
    diff_img.save(os.path.join(UPLOAD_DIR, diff_name))
    step_paths["diff"] = diff_name

    # Step 3:
    from scipy.ndimage import uniform_filter

    M = uniform_filter(I_d, size=local_var_size)
    V = uniform_filter((I_d - M)**2, size=local_var_size)

    # Step 4: 

    cond = np.abs(I_d - M) > threshold * np.sqrt(V + 1e-8)
    Im = np.where(cond, I, I_i)

    Im_img = Image.fromarray(np.clip(Im, 0, 255).astype(np.uint8))
    Im_name = f"{img_id}_Im.png"
    Im_img.save(os.path.join(UPLOAD_DIR, Im_name))
    step_paths["Im"] = Im_name

    # Step 5: Io = I - s * Im
    I_o = (I - final_weight * Im).copy()

    final_img = Image.fromarray(np.clip(I_o, 0, 255).astype(np.uint8))
    final_name = f"{img_id}_final.png"
    final_img.save(os.path.join(UPLOAD_DIR, final_name))
    step_paths["final"] = final_name

    # Step 6: (Paper: rescale using m ± 2.5σ)
    result = I_o.copy()

    first_ten = np.percentile(I_o, 0.5)
    last_ten = np.percentile(I_o, 100 - 0.5)
    result[result < first_ten] = first_ten
    result[result > last_ten] = last_ten    

    result = (result - result.min()) / (result.max() - result.min()) * 255
    result = result.astype(np.uint8)
    result_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    result_name = f"{img_id}_result.png"
    result_img.save(os.path.join(UPLOAD_DIR, result_name))
    step_paths["result"] = result_name
    return step_paths



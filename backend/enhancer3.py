import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def run_guided_filter(input_path, img_id, radius=8, eps=0.01):
    img = cv2.imread(input_path)  
    guide = img.copy()
    src = img.copy()

    guide_f = guide.astype(np.float32) / 255.0
    src_f = src.astype(np.float32) / 255.0

    if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'guidedFilter'):
        raise RuntimeError("OpenCV ximgproc module with guidedFilter is required")

    guided = cv2.ximgproc.guidedFilter(
        guide=guide_f,
        src=src_f,
        radius=radius,
        eps=eps
    )

    out = np.clip(guided * 255.0, 0, 255).astype(np.uint8)
    result_name = f"{img_id}_guided.png"
    result_path = os.path.join(UPLOAD_DIR, result_name)
    cv2.imwrite(result_path, out)

    return {"guided": result_name}

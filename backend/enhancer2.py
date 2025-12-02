import cv2
import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def run_bilateral_enhancement(input_path, img_id, sigma_color=25.0, sigma_space=3.0):
    """
    Áp dụng bilateral filter OpenCV cho ảnh xám
    input_path: path ảnh gốc
    img_id: string id để lưu file
    sigma_color: tương ứng sigma_r
    sigma_space: tương ứng sigma_d
    """
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    filtered = cv2.bilateralFilter(
        src=gray,
        d=0,               
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    result_name = f"{img_id}_bilateral.png"
    result_path = os.path.join(UPLOAD_DIR, result_name)
    cv2.imwrite(result_path, filtered)

    return {"bilateral": result_name}

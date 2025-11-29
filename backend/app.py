from flask import Flask, request, jsonify, send_from_directory
from enhancer import run_enhancement
import os
import uuid

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ===============================
# 1) Trang chủ -> serve index.html
# ===============================
@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')


# ===============================
# 2) Serve file tĩnh (CSS, JS)
# ===============================
@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('../frontend', filename)


# ===============================
# 3) API upload
# ===============================
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img = request.files['image']
    img_id = str(uuid.uuid4())
    input_filename = f"{img_id}_input.png"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    img.save(input_path)

    # ======= Read parameters from form (with safe casting & defaults) =======
    def _get_float(name, default):
        v = request.form.get(name, None)
        try:
            return float(v) if v is not None else default
        except:
            return default

    def _get_int(name, default):
        v = request.form.get(name, None)
        try:
            return int(v) if v is not None else default
        except:
            return default

    params = {
        "alpha": _get_float("alpha", 0.5),             
        "kernel_size": _get_int("kernel_size", 15),  
        "k": _get_int("iterations", 5),
        "final_weight": _get_float("final_weight", 0.5)
    }

    # Optional: enforce safe ranges
    params["alpha"] = max(0.0, min(params["alpha"], 10.0))
    params["kernel_size"] = max(1, min(params["kernel_size"], 50))
    params["k"] = max(1, min(params["k"], 10))
    params["final_weight"] = max(0.0, min(params["final_weight"], 2.0))

    # run enhancement with params
    result_files = run_enhancement(input_path, img_id, **params)
    return jsonify({'steps': result_files, 'params_used': params})


# ===============================
# 4) Serve output images
# ===============================
@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)

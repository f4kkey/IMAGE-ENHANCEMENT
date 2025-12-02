from flask import Flask, request, jsonify, send_from_directory
from enhancer import run_enhancement
from enhancer2 import run_bilateral_enhancement  
from enhancer3 import run_guided_filter 
from flask_cors import CORS
import os
import uuid

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
CORS(app)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('../frontend', filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img = request.files['image']
    img_id = str(uuid.uuid4())
    input_filename = f"{img_id}_input.png"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    img.save(input_path)

    method = request.form.get("method", "default")

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

    if method == "bilateral":
        sigma_color = _get_float("sigma_color", 25.0)
        sigma_space = _get_float("sigma_space", 3.0)
        result_files = run_bilateral_enhancement(input_path, img_id, sigma_color, sigma_space)
        params_used = {"sigma_color": sigma_color, "sigma_space": sigma_space, "method": "bilateral"}
    
    elif method == "guided":
        radius = _get_int("radius", 8)
        eps = _get_float("eps", 0.01)
        result_files = run_guided_filter(input_path, img_id, radius, eps)
        params_used = {"radius": radius, "eps": eps, "method": "guided"}
    else:
        params = {
            "alpha": _get_float("alpha", 0.5),             
            "kernel_size": _get_int("kernel_size", 15),  
            "k": _get_int("iterations", 5),
            "final_weight": _get_float("final_weight", 0.5)
        }

        params["alpha"] = max(0.0, min(params["alpha"], 10.0))
        params["kernel_size"] = max(1, min(params["kernel_size"], 50))
        params["k"] = max(1, min(params["k"], 10))
        params["final_weight"] = max(0.0, min(params["final_weight"], 2.0))

        result_files = run_enhancement(input_path, img_id, **params)
        params_used = params
        params_used["method"] = "default"

    return jsonify({'steps': result_files, 'params_used': params_used})

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

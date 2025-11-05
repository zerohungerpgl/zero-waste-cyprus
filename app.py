# app.py
import os
import io
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import random

# TensorFlow / Keras imports
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
THUMB_FOLDER = UPLOAD_FOLDER / "thumbs"
DB_PATH = BASE_DIR / "predictions.db"
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMB_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB limit

# Load MobileNetV2 (ImageNet) once on startup
model = MobileNetV2(weights="imagenet")

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Database helpers ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        confidence REAL,
        weight TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_record(filename, prediction, confidence, weight):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
      INSERT INTO predictions (filename, prediction, confidence, weight, timestamp)
      VALUES (?, ?, ?, ?, ?)
    """, (filename, prediction, float(confidence), weight, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_stats(limit=100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, prediction, confidence, weight, timestamp FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

# --- Utilities ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def blur_faces_in_cv2(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = img_cv[y:y+h, x:x+w]
        # strong blur
        k = max(99, (w//3)|1)
        blurred = cv2.GaussianBlur(face, (k, k), 30)
        img_cv[y:y+h, x:x+w] = blurred
    return img_cv

def predict_food_from_path(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    label = decode_predictions(preds, top=1)[0][0]
    # decode_predictions returns (class_id, class_name, score)
    return label[1], float(label[2])

def make_thumbnail(in_path, out_path, size=(300,300)):
    with Image.open(in_path) as im:
        im.thumbnail(size)
        im.save(out_path)

# Initialize DB on startup
init_db()

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    # Render template; template checks for 'prediction' variable
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form post from the frontend.
    Accepts: 'image' file, optional 'weight' and 'weight_mode'.
    Returns: Renders index.html with prediction inserted.
    Also saves record to SQLite and stores blurred image on disk.
    """
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        return "Αρχείο δεν επιτρέπεται. Επιλογές: png/jpg/jpeg/gif", 400

    # secure filename + unique suffix
    filename = secure_filename(file.filename)
    name_root, ext = os.path.splitext(filename)
    unique_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    saved_name = f"{name_root}_{unique_suffix}{ext}"
    saved_path = UPLOAD_FOLDER / saved_name

    # Save incoming file
    file.save(saved_path)

    # Read with OpenCV, blur faces, save a blurred copy
    img_cv = cv2.imdecode(np.fromfile(str(saved_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_cv is None:
        # fallback via PIL
        pil_im = Image.open(saved_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    img_blurred = blur_faces_in_cv2(img_cv)
    blurred_name = f"blurred_{saved_name}"
    blurred_path = UPLOAD_FOLDER / blurred_name
    # Use imwrite with unicode-safe path
    cv2.imencode(ext, img_blurred)[1].tofile(str(blurred_path))

    # Prediction
    try:
        pred_label, confidence = predict_food_from_path(str(blurred_path))
    except Exception as e:
        # If prediction fails, return a friendly message
        pred_label, confidence = "unknown", 0.0

    # Weight handling
    weight_mode = request.form.get("weight_mode", "manual")
    if weight_mode == "manual":
        weight_val = request.form.get("weight", "").strip() or "unknown"
    else:
        weight_val = f"{random.uniform(50, 100):.1f} kg"  # simulate

    # Save thumbnail for displaying on frontend
    thumb_path = THUMB_FOLDER / blurred_name
    try:
        make_thumbnail(str(blurred_path), str(thumb_path))
    except Exception:
        # ignore thumbnail errors
        pass

    # Save to DB
    save_record(blurred_name, pred_label, confidence, weight_val)

    # Render the same index page with the prediction result
    prediction_text = f"{pred_label} ({confidence*100:.2f}%) — Βάρος: {weight_val}"
    return render_template("index.html", prediction=prediction_text, image_url=f"/uploads/thumbs/{blurred_name}")

# Serve uploads (thumbnails and images)
@app.route("/uploads/<path:filename>")
def uploads_static(filename):
    # safe serve from uploads folder
    return send_from_directory(str(UPLOAD_FOLDER), filename, as_attachment=False)

@app.route("/uploads/thumbs/<path:filename>")
def thumbs_static(filename):
    return send_from_directory(str(THUMB_FOLDER), filename, as_attachment=False)

# --- API endpoints ---
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error":"no image provided"}), 400
    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error":"invalid file"}), 400

    filename = secure_filename(file.filename)
    name_root, ext = os.path.splitext(filename)
    unique_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    saved_name = f"{name_root}_{unique_suffix}{ext}"
    saved_path = UPLOAD_FOLDER / saved_name
    file.save(saved_path)

    img_cv = cv2.imdecode(np.fromfile(str(saved_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_cv is None:
        pil_im = Image.open(saved_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    img_blurred = blur_faces_in_cv2(img_cv)
    blurred_name = f"blurred_{saved_name}"
    blurred_path = UPLOAD_FOLDER / blurred_name
    cv2.imencode(ext, img_blurred)[1].tofile(str(blurred_path))

    try:
        pred_label, confidence = predict_food_from_path(str(blurred_path))
    except Exception:
        pred_label, confidence = "unknown", 0.0

    # weight: check form data or generate
    weight = request.form.get("weight")
    if not weight:
        weight = f"{random.uniform(50,100):.1f} kg"

    save_record(blurred_name, pred_label, confidence, weight)

    return jsonify({
        "food": pred_label,
        "confidence": confidence,
        "weight": weight,
        "image": f"/uploads/{blurred_name}"
    })

@app.route("/api/stats", methods=["GET"])
def api_stats():
    limit = int(request.args.get("limit", 100))
    rows = get_stats(limit)
    data = []
    for r in rows:
        data.append({
            "id": r[0],
            "filename": r[1],
            "prediction": r[2],
            "confidence": r[3],
            "weight": r[4],
            "timestamp": r[5]
        })
    return jsonify({"rows": data})

if __name__ == "__main__":
    # For development only. Use a WSGI server (gunicorn) in production.
    app.run(host="0.0.0.0", port=5000, debug=True)

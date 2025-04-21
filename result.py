import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import uuid
from flask_cors import CORS
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

MODEL_PATH = "efficientnet_balanced_finetuned.h5"
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["cardboard", "compost", "glass", "metal", "paper", "plastic", "trash"]

IMG_SIZE = 224
WINDOW_SIZE = 224
STRIDE = 100
CONFIDENCE_THRESHOLD = 0.5

def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    boxes = []

    for y in range(0, height, STRIDE):
        for x in range(0, width, STRIDE):
            crop = image[y:min(y+WINDOW_SIZE, height), x:min(x+WINDOW_SIZE, width)]
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            input_tensor = tf.expand_dims(resized, axis=0)
            input_tensor = preprocess_input(input_tensor)

            preds = model.predict(input_tensor, verbose=0)[0]
            class_idx = np.argmax(preds)
            confidence = preds[class_idx]

            if confidence > CONFIDENCE_THRESHOLD:
                boxes.append({
                    "label": CLASS_NAMES[class_idx],
                    "confidence": float(round(confidence, 2)),
                    "box": [int(x), int(y), int(min(x+WINDOW_SIZE, width)), int(min(y+WINDOW_SIZE, height))]
                })
    return boxes

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = f"temp_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    results = detect_objects(filepath)
    os.remove(filepath)
    return jsonify({"detections": results})

if __name__ == '__main__':
    app.run(debug=True)

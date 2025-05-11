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

MODEL_PATH = "efficientnet_balanced_finetuned1.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
CLASS_NAMES = ["cardboard", "compost", "glass", "metal", "paper", "plastic", "trash"]

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

def classify_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    input_tensor = tf.expand_dims(image, axis=0)
    input_tensor = preprocess_input(input_tensor)

    preds = model.predict(input_tensor, verbose=0)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    if confidence > CONFIDENCE_THRESHOLD:
        return {
            "label": CLASS_NAMES[class_idx],
            "confidence": float(round(confidence, 2))
        }
    else:
        return {
            "label": "uncertain",
            "confidence": float(round(confidence, 2))
        }

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = f"temp_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    result = classify_image(filepath)
    os.remove(filepath)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

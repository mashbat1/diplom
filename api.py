import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import os
from flask_cors import CORS  # Import CORS
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)  # Enable CORS globally

# Load the best available model
MODEL_PATH = "efficientnet_balanced_finetuned.h5"  # Change if needed
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
class_names = ["cardboard", "compost", "glass", "metal", "paper", "plastic", "trash"]  # ✅ зөв

# Function to process and predict image

def predict_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)  # ✅ яг model-той адил хэлбэр
    img_array = tf.expand_dims(img_array, 0)

    print(f"Input image shape: {img_array.shape}")
    predictions = model.predict(img_array)
    print(f"Raw predictions: {predictions}")

    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)
    return predicted_class, confidence



# Flask API for Image Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    predicted_class, confidence = predict_image(file_path, model)
    return jsonify({'class': predicted_class, 'confidence': f'{confidence:.2f}%'})

if __name__ == '__main__':
    app.run(debug=True)

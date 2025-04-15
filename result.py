from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
WEIGHTS_PATH = "vgg16_final_model.h5"

# Check if weights file exists
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"❌ ERROR: Weights file '{WEIGHTS_PATH}' not found!")

# Rebuild model architecture
base_model = VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights=None)
model = Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu", kernel_regularizer="l2"),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

# Build the model before loading weights
model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, 3))

# Load weights
try:
    model.load_weights(WEIGHTS_PATH)
    print("✅ Weights successfully loaded from:", WEIGHTS_PATH)
except Exception as e:
    print(f"❌ ERROR loading weights: {e}")
    exit(1)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Function to process and predict image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)
    return predicted_class, confidence

# API endpoint for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = "uploaded_image.jpg"
    file.save(file_path)
    
    predicted_class, confidence = predict_image(file_path)
    
    return jsonify({'class': predicted_class, 'confidence': f'{confidence:.2f}%'})

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)

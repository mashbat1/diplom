import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import os
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS globally

# Load the best available model
MODEL_PATH = "waste_classification_model.keras"  # Change if needed
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
class_names = ["cardboard", "compost", "glass", "metal", "paper", "plastic", "trash"]  # Adjust if necessary

# Function to process and predict image
def predict_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(244, 244))  # Change to 244x244
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Rescale

    # Debugging: Print the shape of input data
    print(f"Input image shape: {img_array.shape}")

    predictions = model.predict(img_array)

    # Debugging: Print raw predictions
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

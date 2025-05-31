import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
import logging
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained model
logger.info("Loading model...")
try:
    model = load_model('forest_fire_model.h5', compile=False)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image):
    # Resize image to match model's expected sizing
    image = cv2.resize(image, (250, 250))
    # Normalize the image
    image = image.astype('float32') / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    fire_probability = float(prediction[0][1])
    
    # Determine the result
    result = "Fire Detected" if fire_probability > 0.5 else "No Fire Detected"
    confidence = fire_probability if fire_probability > 0.5 else 1 - fire_probability
    
    return jsonify({
        'result': result,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    # Try to get port from environment variable, default to 10000
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Attempting to start server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
print("Loading model...")
model = load_model('forest_fire_model.h5')
print("Model loaded successfully!")

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
    app.run(debug=True)


import os
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from flask_cors import CORS # Import CORS to allow cross-origin requests

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define the path to your Keras model file using a relative path.
# This assumes 'coffee_disease.h5' is in the same directory as this script.
MODEL_PATH = 'coffee_disease.h5'

# Load the trained Keras model
# We use a try/except block to handle cases where the model file might not be found.
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# A mapping of class indices to human-readable labels
# You should update these labels to match the classes in your model.
CLASS_LABELS = ['Healthy', 'Coffee Leaf Rust', 'Miner', 'Cercospora']

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive an image and return a prediction.
    """
    # Check if the model was loaded successfully
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    # Ensure an image file was sent in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']

    # Ensure the file has a filename and is a valid image
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if not file.content_type.startswith('image'):
        return jsonify({'error': 'File is not a supported image type.'}), 400

    try:
        # Read the image file and convert it to a PIL Image object
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image to match the model's input requirements
        # Your model might have different input requirements, so adjust this section as needed.
        # This example assumes a 224x224 input size and normalization.
        img = img.resize((224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values to 0-1

        # Make a prediction using the model
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index]) * 100
        predicted_class_label = CLASS_LABELS[predicted_class_index]
        
        # Format the response
        response = {
            'predicted_class': predicted_class_label,
            'confidence': f'{confidence:.2f}%'
        }
        
        return jsonify(response)

    except Exception as e:
        # Log the full error for debugging purposes
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal server error occurred during prediction.'}), 500

if __name__ == '__main__':
    # The debug=True option provides helpful error messages during development.
    # When deploying to Render, the service will handle running the app,
    # so this section is mainly for local testing.
    app.run(host='127.0.0.1', port=5000, debug=True)

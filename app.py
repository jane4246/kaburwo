import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define a path for the model and load it
MODEL_PATH = os.path.join(os.getcwd(), 'coffee_disease.h5')
model = load_model(MODEL_PATH)
labels = ['Healthy', 'Leaf Rust'] # Make sure these labels match your model's output classes

@app.route('/')
def serve_index():
    """
    Serves the index.html file from the current directory.
    This is the new route that fixes the "Not Found" error.
    """
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    """
    Handles image prediction requests.
    """
    # Check if an image was provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # Open and preprocess the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction using the loaded model
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app locally on a specific port.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

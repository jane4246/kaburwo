import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.getcwd(), 'coffee_disease.h5')

print(f"🔍 Looking for model at: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    labels = ['Healthy', 'Leaf Rust']
    print("✅ Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    labels = []

@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if model is None:
        print("❌ No model loaded in memory")
        return jsonify({'error': 'Machine learning model is not available.'}), 503

    if 'image' not in request.files:
        print("❌ No image file received in request")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    print(f"📂 Received file: {file.filename}")

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Ensure this matches model.input_shape
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print(f"📏 Preprocessed image shape: {img_array.shape}")

        prediction = model.predict(img_array)
        print(f"📊 Raw prediction: {prediction}")

        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

        print(f"✅ Predicted: {predicted_class} ({confidence:.2f})")

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

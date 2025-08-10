import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tflite_runtime.interpreter as tflite

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------
# Model Setup
# ---------------------------
MODEL_PATH = os.path.join(os.getcwd(), 'coffee_disease.tflite')
labels = ['Healthy', 'Leaf Rust']  # Make sure this matches your training labels
interpreter = None
input_index = None
output_index = None

try:
    print(f"🔍 Loading TFLite model from {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    print("✅ Model loaded successfully")
    print(f"Model input shape: {input_details[0]['shape']}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    interpreter = None

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def serve_index():
    """Serve index.html from current directory."""
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    """Handle image prediction requests."""
    if interpreter is None:
        print("❌ Model not loaded")
        return jsonify({'error': 'Machine learning model is not available.'}), 503

    if 'image' not in request.files:
        print("❌ No image file provided in request")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    print(f"📂 Received file: {file.filename}")

    try:
        # ---------------------------
        # Image Preprocessing
        # ---------------------------
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Must match model training size
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print(f"📏 Preprocessed image shape: {img_array.shape}")

        # ---------------------------
        # Prediction
        # ---------------------------
        interpreter.set_tensor(input_index, img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)

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

# ---------------------------
# App Entry
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

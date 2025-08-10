import os
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load TFLite model
MODEL_PATH = os.path.join(os.getcwd(), "coffee_disease.tflite")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    labels = [
        "coffee leaf trunk",
        "coffee rust",
        "healthy",
        "leaf spot",
        "nematodes",
        "phoma",
        "re_spider_mite"
    ]  # Must match your model's classes
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None
    labels = []

@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if interpreter is None:
        return jsonify({'error': 'Machine learning model is not available.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # Load and preprocess the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        # Crash-proof check
        if prediction.size == 0:
            return jsonify({'error': 'Model returned empty prediction'}), 500

        predicted_class_index = int(np.argmax(prediction))

        if 0 <= predicted_class_index < len(labels):
            predicted_class = labels[predicted_class_index]
        else:
            predicted_class = f"Unknown class {predicted_class_index}"

        confidence = float(prediction[predicted_class_index]) if 0 <= predicted_class_index < len(prediction) else 0.0

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        print(f"Prediction failed with exception: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

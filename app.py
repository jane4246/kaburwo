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
        "cescospora_leaf_sport",
        "coffee_canker",
        "coffee_leaf_rust",
        "healthy",
        "nematodes",
        "phoma_leaf_sport",
        "red_spider_mite"
    ]  # Must match your model's classes
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None
    labels = []

# This function is now shared by the two routes
def process_single_image(image_file):
    """
    Processes a single image file and returns its prediction.
    """
    try:
        # Load and preprocess the image
        img = Image.open(io.BytesIO(image_file.read()))
        img = img.convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, 128, 128, 3)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        # Crash-proof check
        if prediction.size == 0:
            return {'error': 'Model returned empty prediction'}

        predicted_class_index = int(np.argmax(prediction))
        if 0 <= predicted_class_index < len(labels):
            predicted_class = labels[predicted_class_index]
        else:
            predicted_class = f"Unknown class {predicted_class_index}"

        confidence = float(prediction[predicted_class_index]) if 0 <= predicted_class_index < len(prediction) else 0.0

        return {
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2f}"
        }

    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single_image():
    if interpreter is None:
        return jsonify({'error': 'Machine learning model is not available.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    result = process_single_image(file)

    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch_images():
    if interpreter is None:
        return jsonify({'error': 'Machine learning model is not available.'}), 503

    files = request.files.getlist('images')

    if not files:
        return jsonify({'error': 'No image files provided'}), 400
    
    results = []
    for file in files:
        if file.filename == '':
            continue
        
        prediction_result = process_single_image(file)
        results.append({'filename': file.filename, 'prediction': prediction_result})
    
    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

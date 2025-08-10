import tensorflow as tf

h5_path = "coffee_disease.h5"
tflite_path = "coffee_disease.tflite"

print("Loading .h5 model from:", h5_path)
model = tf.keras.models.load_model(h5_path)
print("Model loaded. Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: enable optimizations (size/speed) — uncomment if desired
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("Conversion complete. Saved:", tflite_path)

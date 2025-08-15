import os
import io
import numpy as np
import sqlite3
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from datetime import datetime

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load TFLite model
MODEL_PATH = os.path.join(os.getcwd(), "coffee_disease.tflite")

# Initialize Vikoba database
def init_vikoba_db():
    conn = sqlite3.connect('vikoba.db')
    c = conn.cursor()
    
    # Create groups table
    c.execute('''CREATE TABLE IF NOT EXISTS groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                admin_id INTEGER)''')
    
    # Create group members table
    c.execute('''CREATE TABLE IF NOT EXISTS group_members (
                group_id INTEGER,
                member_id INTEGER,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (group_id, member_id))''')
    
    # Create group messages table
    c.execute('''CREATE TABLE IF NOT EXISTS group_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER,
                user_id INTEGER,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create detection shares table
    c.execute('''CREATE TABLE IF NOT EXISTS detection_shares (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER,
                detection_data TEXT,
                shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_vikoba_db()

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
    ]
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None
    labels = []

# Function to process a single image
def process_single_image(image_file):
    try:
        img = Image.open(io.BytesIO(image_file.read()))
        img = img.convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        if prediction.size == 0:
            return {'error': 'Model returned empty prediction'}

        predicted_class_index = int(np.argmax(prediction))
        predicted_class = labels[predicted_class_index] if 0 <= predicted_class_index < len(labels) else f"Unknown class {predicted_class_index}"
        confidence = float(prediction[predicted_class_index]) if 0 <= predicted_class_index < len(prediction) else 0.0

        return {
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2f}"
        }

    except Exception as e:
        return {'error': str(e)}

# Serve index page
@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')

# Unified prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'Machine learning model is not available.'}), 503

    files = request.files.getlist('images')  
    if not files or all(f.filename == '' for f in files):
        if 'image' in request.files:
            files = [request.files['image']]
        else:
            return jsonify({'error': 'No image file(s) provided'}), 400

    results = []
    for file in files:
        if file.filename == '':
            continue
        prediction_result = process_single_image(file)
        results.append({
            'filename': file.filename,
            'prediction': prediction_result
        })

    if len(results) == 1:
        return jsonify(results[0])
    return jsonify({'results': results})

# Vikoba group routes
@app.route('/vikoba')
def vikoba_dashboard():
    # In a real app, you'd get user_id from session
    user_id = session.get('user_id', 1)
    
    conn = sqlite3.connect('vikoba.db')
    c = conn.cursor()
    
    # Get user's groups
    c.execute("""
        SELECT g.id, g.name, COUNT(gm.member_id) as member_count 
        FROM groups g
        JOIN group_members gm ON g.id = gm.group_id
        WHERE gm.member_id = ?
        GROUP BY g.id
    """, (user_id,))
    user_groups = c.fetchall()
    
    # Get all groups for discovery
    c.execute("""
        SELECT g.id, g.name, COUNT(gm.member_id) as member_count 
        FROM groups g
        LEFT JOIN group_members gm ON g.id = gm.group_id
        GROUP BY g.id
    """)
    all_groups = c.fetchall()
    
    conn.close()
    
    return render_template('vikoba.html', user_groups=user_groups, all_groups=all_groups)

@app.route('/vikoba/create', methods=['GET', 'POST'])
def create_vikoba():
    if request.method == 'POST':
        group_name = request.form['group_name']
        user_id = session.get('user_id', 1)
        
        conn = sqlite3.connect('vikoba.db')
        c = conn.cursor()
        
        # Create new group
        c.execute("INSERT INTO groups (name, admin_id) VALUES (?, ?)", 
                 (group_name, user_id))
        group_id = c.lastrowid
        
        # Add creator as first member
        c.execute("INSERT INTO group_members (group_id, member_id) VALUES (?, ?)", 
                 (group_id, user_id))
        
        conn.commit()
        conn.close()
        
        return redirect(url_for('vikoba_group', group_id=group_id))
    
    return render_template('create_vikoba.html')

@app.route('/vikoba/<int:group_id>')
def vikoba_group(group_id):
    conn = sqlite3.connect('vikoba.db')
    c = conn.cursor()
    
    # Get group info
    c.execute("SELECT * FROM groups WHERE id = ?", (group_id,))
    group = c.fetchone()
    
    # Get members
    c.execute("""
        SELECT member_id 
        FROM group_members 
        WHERE group_id = ?
    """, (group_id,))
    members = c.fetchall()
    
    # Get latest messages
    c.execute("""
        SELECT * 
        FROM group_messages 
        WHERE group_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 5
    """, (group_id,))
    messages = c.fetchall()
    
    # Get shared detections
    c.execute("""
        SELECT * 
        FROM detection_shares 
        WHERE group_id = ? 
        ORDER BY shared_at DESC 
        LIMIT 3
    """, (group_id,))
    shares = c.fetchall()
    
    conn.close()
    
    return render_template('vikoba_group.html', 
                          group=group, 
                          members=members,
                          messages=messages,
                          shares=shares)

@app.route('/vikoba/<int:group_id>/chat', methods=['GET', 'POST'])
def vikoba_chat(group_id):
    if request.method == 'POST':
        message = request.form['message']
        user_id = session.get('user_id', 1)
        
        conn = sqlite3.connect('vikoba.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO group_messages (group_id, user_id, message) 
            VALUES (?, ?, ?)
        """, (group_id, user_id, message))
        conn.commit()
        conn.close()
        
        return redirect(url_for('vikoba_chat', group_id=group_id))
    
    conn = sqlite3.connect('vikoba.db')
    c = conn.cursor()
    
    # Get group name
    c.execute("SELECT name FROM groups WHERE id = ?", (group_id,))
    group_name = c.fetchone()[0]
    
    # Get chat history
    c.execute("""
        SELECT * 
        FROM group_messages 
        WHERE group_id = ? 
        ORDER BY timestamp ASC
    """, (group_id,))
    messages = c.fetchall()
    
    conn.close()
    
    return render_template('vikoba_chat.html', 
                          group_id=group_id,
                          group_name=group_name,
                          messages=messages)

@app.route('/share_detection', methods=['POST'])
def share_detection():
    group_id = request.form['group_id']
    detection_data = request.form['detection_data']
    user_id = session.get('user_id', 1)
    
    conn = sqlite3.connect('vikoba.db')
    c = conn.cursor()
    c.execute("""
        INSERT INTO detection_shares (group_id, detection_data) 
        VALUES (?, ?)
    """, (group_id, detection_data))
    conn.commit()
    conn.close()
    
    # Also post to group chat
    conn = sqlite3.connect('vikoba.db')
    c = conn.cursor()
    c.execute("""
        INSERT INTO group_messages (group_id, user_id, message) 
        VALUES (?, ?, ?)
    """, (group_id, user_id, f"Shared disease detection: {detection_data}"))
    conn.commit()
    conn.close()
    
    return jsonify({'status': 'success', 'message': 'Detection shared with group'})

# Health check endpoint for Render cold starts
@app.route('/health', methods=['GET'])
def health_check():
    if interpreter is None:
        return jsonify({
            "status": "unavailable",
            "message": "Model not loaded",
            "classes": []
        }), 503
    
    return jsonify({
        "status": "available",
        "message": "Model loaded and ready",
        "classes": labels
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

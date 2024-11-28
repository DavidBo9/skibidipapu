# app.py
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import json
import cv2
from tweet_analyzer import TweetAnalyzer
from batch_classifier import BatchImageClassifier

# Configure application
app = Flask(__name__)

# Configure file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# App configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model paths
TWEET_MODEL_PATH = os.path.join(MODELS_FOLDER, 'analisis_tweets.h5')
EMOTION_MODEL_PATH = os.path.join(MODELS_FOLDER, 'emotion_classifier_v3.h5')
FACE_MODEL_PATH = os.path.join(MODELS_FOLDER, 'emotion_region_classifier.h5')

# Initialize analyzers with error handling
try:
    tweet_analyzer = TweetAnalyzer()
    print("Tweet analyzer initialized successfully")
except Exception as e:
    print(f"Error initializing tweet analyzer: {str(e)}")
    tweet_analyzer = None

try:
    if os.path.exists(FACE_MODEL_PATH):
        face_classifier = BatchImageClassifier(FACE_MODEL_PATH)
        print("Face classifier initialized successfully")
    else:
        print(f"Face model not found at: {FACE_MODEL_PATH}")
        face_classifier = None
except Exception as e:
    print(f"Error initializing face classifier: {str(e)}")
    face_classifier = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html', 
                         tweet_analyzer_status=tweet_analyzer is not None,
                         face_classifier_status=face_classifier is not None)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    if tweet_analyzer is None:
        return jsonify({'error': 'Tweet analyzer not initialized'}), 500

    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        results = tweet_analyzer.analyze_tweet(text)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_images', methods=['POST'])
def analyze_images():
    if face_classifier is None:
        return jsonify({'error': 'Face classifier not initialized'}), 500

    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files[]')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                frame = cv2.imread(filepath)
                
                if frame is None:
                    raise ValueError("Failed to load image")
                
                faces = face_classifier.detect_faces(frame)
                face_results = []
                
                for face_box in faces:
                    class_idx, confidence = face_classifier.process_face(frame, face_box)
                    if class_idx is not None:
                        face_results.append({
                            'emotion': face_classifier.class_names[class_idx],
                            'confidence': float(confidence)
                        })
                
                results.append({
                    'image': filename,
                    'faces_detected': len(faces),
                    'emotions': face_results
                })
                
            except Exception as e:
                results.append({
                    'image': filename,
                    'error': str(e)
                })
            
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)

    return jsonify(results)

if __name__ == '__main__':
    print(f"\nModel paths:")
    print(f"Tweet model path: {TWEET_MODEL_PATH}")
    print(f"Emotion model path: {EMOTION_MODEL_PATH}")
    print(f"Face model path: {FACE_MODEL_PATH}")
    
    app.run(debug=True)
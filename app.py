# app.py
from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import json
import cv2
import base64
import tensorflow as tf
from openai import OpenAI
from dotenv import load_dotenv
from tweet_analyzer import TweetAnalyzer
from batch_classifier import BatchImageClassifier

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Initialize Flask and OpenAI
app = Flask(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Configure file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# App configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB max file size

# Model paths
TWEET_MODEL_PATH = os.path.join(MODELS_FOLDER, 'analisis_tweets.h5')
EMOTION_MODEL_PATH = os.path.join(MODELS_FOLDER, 'emotion_classifier_v3.h5')
FACE_MODEL_PATH = os.path.join(MODELS_FOLDER, 'emotion_region_classifier.h5')

# Initialize analyzers
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
MAX_IMAGE_SIZE = 800

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image):
    height, width = image.shape[:2]
    if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
        if height > width:
            new_height = MAX_IMAGE_SIZE
            new_width = int(width * (MAX_IMAGE_SIZE / height))
        else:
            new_width = MAX_IMAGE_SIZE
            new_height = int(height * (MAX_IMAGE_SIZE / width))
        image = cv2.resize(image, (new_width, new_height))
    return image

def get_ai_recommendations(text_results, image_results):
    try:
        text_sentiment = text_results.get('sentiment', 'unknown')
        text_emotion = text_results.get('emotion', 'unknown')
        
        face_emotion = 'unknown'
        if image_results and 'emotions' in image_results and image_results['emotions']:
            face_emotion = image_results['emotions'][0].get('emotion', 'unknown')

        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an empathetic AI consultant analyzing emotional states from text and facial expressions.
                    Provide personalized recommendations based on the emotional analysis results.
                    Include:
                    1. A brief analysis of the emotional state
                    2. 2-3 specific activity recommendations
                    3. A positive, encouraging message
                    
                    Keep responses concise but meaningful, about 3-4 sentences.
                    Focus on constructive, helpful suggestions that could improve the person's emotional well-being.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Text Analysis Results:
                    - Sentiment: {text_sentiment}
                    - Primary Text Emotion: {text_emotion}
                    - Facial Expression: {face_emotion}
                    
                    Please provide a personalized analysis and recommendations.
                    """
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

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
    if len(files) > 5:
        return jsonify({'error': 'Maximum 5 images allowed'}), 400

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
                
                frame = resize_image(frame)
                faces = face_classifier.detect_faces(frame)
                face_results = []
                
                for face_box in faces:
                    class_idx, confidence = face_classifier.process_face(frame, face_box)
                    if class_idx is not None:
                        frame = face_classifier.draw_prediction(frame, class_idx, confidence, face_box)
                        face_results.append({
                            'emotion': face_classifier.class_names[class_idx],
                            'confidence': float(confidence)
                        })
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                results.append({
                    'image': filename,
                    'faces_detected': len(faces),
                    'emotions': face_results,
                    'analyzed_image': img_base64
                })
                
            except Exception as e:
                results.append({
                    'image': filename,
                    'error': str(e)
                })
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                
        tf.keras.backend.clear_session()

    return jsonify(results)

@app.route('/analyze_combined', methods=['POST'])
def analyze_combined():
    if 'text_results' not in request.json or 'image_results' not in request.json:
        return jsonify({'error': 'Missing required data'}), 400
        
    try:
        text_results = request.json['text_results']
        image_results = request.json['image_results']
        
        recommendations = get_ai_recommendations(text_results, image_results)
        
        return jsonify({
            'text_analysis': text_results,
            'image_analysis': image_results,
            'ai_recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
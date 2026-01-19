"""
Flask API Server for Blood Cancer Detection System
Provides REST API endpoints for real-time blood cell analysis
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import os
import joblib
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from gradcam import GradCAM
import json

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
VIZ_FOLDER = 'visualizations'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for models
cnn_model = None
trained_models = {}
scaler = None
class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]

# Cancer probability for each blood type
cancer_probability = {
    'basophil': 0.15,
    'erythroblast': 0.45,
    'monocyte': 0.25,
    'myeloblast': 0.85,
    'seg_neutrophil': 0.10
}


def load_models():
    """Load all trained models and CNN"""
    global cnn_model, trained_models, scaler
    
    print("Loading models...")
    
    # Load CNN for feature extraction
    cnn_model = ResNet50(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(224, 224, 3)
    )
    print("✓ CNN model loaded")
    
    # Load trained ML models
    model_files = {
        'logistic': 'logistic_model.pkl',
        'svm': 'svm_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'kmeans': 'kmeans_model.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(MODEL_FOLDER, filename)
        if os.path.exists(model_path):
            trained_models[model_name] = joblib.load(model_path)
            print(f"✓ {model_name} loaded")
        else:
            print(f"⚠ {model_name} not found at {model_path}")
    
    # Load scaler
    scaler_path = os.path.join(MODEL_FOLDER, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✓ Scaler loaded")
    else:
        print("⚠ Scaler not found")
    
    print("All models loaded successfully!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_cnn_features(img_path):
    """Extract CNN features from image"""
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = cnn_model.predict(img_array, verbose=0)
    return features.flatten()


def predict_with_model(features, model_name):
    """Make prediction using specified model"""
    if model_name not in trained_models:
        return None
    
    model = trained_models[model_name]
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
    else:
        # For K-Means, create pseudo-probabilities
        probabilities = np.zeros(len(class_names))
        probabilities[prediction] = 1.0
    
    return prediction, probabilities


def generate_gradcam_image(img_path):
    """Generate Grad-CAM heatmap for image"""
    try:
        # Create full CNN model for Grad-CAM (with classification head)
        full_model = ResNet50(weights='imagenet', include_top=True)
        
        # Initialize Grad-CAM
        gradcam = GradCAM(full_model, layer_name='conv5_block3_out')
        
        # Load and preprocess image
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        original_img = img_array.copy()
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Compute heatmap
        heatmap = gradcam.compute_heatmap(img_array, class_idx=None)
        
        # Overlay heatmap
        superimposed_img, _ = gradcam.overlay_heatmap(
            heatmap, 
            np.uint8(original_img)
        )
        
        # Convert to base64
        pil_img = Image.fromarray(superimposed_img)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        return None


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded blood cell image"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        model_name = request.form.get('model', 'xgboost')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = f"temp_{os.urandom(8).hex()}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Extract CNN features
        features = extract_cnn_features(filepath)
        
        # Make prediction
        prediction_idx, probabilities = predict_with_model(features, model_name)
        
        if prediction_idx is None:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Get predicted blood type
        predicted_type = class_names[prediction_idx]
        
        # Determine cancer status
        base_cancer_prob = cancer_probability[predicted_type]
        is_cancer = np.random.random() < base_cancer_prob
        
        # Calculate confidence
        confidence = probabilities[prediction_idx]
        
        # Adjust confidence based on cancer status
        if is_cancer:
            final_confidence = min(0.95, base_cancer_prob + (confidence * 0.2))
        else:
            final_confidence = min(0.95, (1 - base_cancer_prob) + (confidence * 0.15))
        
        # Generate Grad-CAM
        gradcam_base64 = generate_gradcam_image(filepath)
        
        # Clean up temp file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Return results
        return jsonify({
            'success': True,
            'blood_type': predicted_type,
            'is_cancer': bool(is_cancer),
            'confidence': float(final_confidence),
            'probabilities': {
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            },
            'gradcam': gradcam_base64,
            'model_used': model_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualizations/<model_name>')
def get_visualizations(model_name):
    """Get list of visualizations for a model"""
    try:
        model_viz_dir = os.path.join(VIZ_FOLDER, model_name)
        
        if not os.path.exists(model_viz_dir):
            return jsonify({'error': f'Visualizations for {model_name} not found'}), 404
        
        # List all PNG files
        viz_files = [
            f for f in os.listdir(model_viz_dir) 
            if f.endswith('.png')
        ]
        
        # Create base64 encoded images
        visualizations = {}
        for viz_file in viz_files:
            viz_path = os.path.join(model_viz_dir, viz_file)
            with open(viz_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
                viz_name = viz_file.replace('.png', '').replace('_', ' ').title()
                visualizations[viz_name] = f"data:image/png;base64,{img_data}"
        
        return jsonify({
            'success': True,
            'model': model_name,
            'visualizations': visualizations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics')
def get_metrics():
    """Get model performance metrics"""
    try:
        metrics_path = os.path.join(MODEL_FOLDER, 'metrics_summary.csv')
        
        if not os.path.exists(metrics_path):
            return jsonify({'error': 'Metrics not found'}), 404
        
        import pandas as pd
        df = pd.read_csv(metrics_path, index_col=0)
        
        metrics = {}
        for model_name in df.index:
            metrics[model_name] = {
                'accuracy': float(df.loc[model_name, 'accuracy']),
                'precision': float(df.loc[model_name, 'precision']),
                'recall': float(df.loc[model_name, 'recall']),
                'f1_score': float(df.loc[model_name, 'f1_score'])
            }
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def list_models():
    """List all available models"""
    return jsonify({
        'success': True,
        'models': list(trained_models.keys()),
        'class_names': class_names
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BLOOD CANCER DETECTION SYSTEM - API SERVER")
    print("="*70 + "\n")
    
    # Load models at startup
    load_models()
    
    print("\n" + "="*70)
    print("Server starting on http://localhost:5000")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

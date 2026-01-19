"""
Enhanced Flask App with YOLO Support
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
from gradcam import GradCAM
import uuid
from datetime import datetime

# Try to import YOLO (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available. Install: pip install ultralytics")

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
GRADCAM_FOLDER = 'static/gradcam'
MODELS_FOLDER = 'models'
YOLO_FOLDER = 'models/yolo'
VISUALIZATIONS_FOLDER = 'visualizations'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Blood cell types
CLASS_NAMES = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]

# Cancer risk mapping
CANCER_RISK = {
    "basophil": 0.15,
    "erythroblast": 0.45,
    "monocyte": 0.25,
    "myeloblast": 0.85,
    "seg_neutrophil": 0.10
}

# Load models at startup
print("Loading models...")

try:
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    print("✓ ResNet50 loaded")
except Exception as e:
    print(f"✗ Error loading ResNet50: {e}")
    resnet_model = None

try:
    scaler = joblib.load(os.path.join(MODELS_FOLDER, 'scaler.pkl'))
    print("✓ Scaler loaded")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")
    scaler = None

# Load ML models
ml_models = {}
model_files = {
    'logistic': 'logistic_model.pkl',
    'svm': 'svm_model.pkl',
    'random_forest': 'random_forest_model.pkl',
    'xgboost': 'xgboost_model.pkl',
    'kmeans': 'kmeans_model.pkl'
}

for model_name, model_file in model_files.items():
    try:
        model_path = os.path.join(MODELS_FOLDER, model_file)
        ml_models[model_name] = joblib.load(model_path)
        print(f"✓ {model_name} loaded")
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")

# Load YOLO model
yolo_model = None
if YOLO_AVAILABLE:
    try:
        yolo_model_path = os.path.join(YOLO_FOLDER, 'yolo_best.pt')
        if os.path.exists(yolo_model_path):
            yolo_model = YOLO(yolo_model_path)
            print("✓ YOLO model loaded")
        else:
            print(f"⚠️  YOLO model not found at: {yolo_model_path}")
    except Exception as e:
        print(f"✗ Error loading YOLO: {e}")

# Initialize Grad-CAM
try:
    gradcam = GradCAM(resnet_model)
    print("✓ Grad-CAM initialized")
except Exception as e:
    print(f"✗ Error initializing Grad-CAM: {e}")
    gradcam = None

print("Server initialization complete!")


def extract_cnn_features(img_array):
    """Extract 2048-dimensional feature vector using ResNet50"""
    if resnet_model is None:
        raise Exception("ResNet50 model not loaded")
    
    img_array = np.expand_dims(img_array, axis=0)
    features = resnet_model.predict(img_array, verbose=0)
    return features.flatten()


def preprocess_image(image_file):
    """Preprocess uploaded image for CNN"""
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array, img


def generate_gradcam(img_array, predicted_class_idx):
    """Generate Grad-CAM heatmap for the prediction"""
    if gradcam is None:
        return None
    
    try:
        heatmap_img = gradcam.generate_heatmap(img_array, predicted_class_idx)
        filename = f"gradcam_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(GRADCAM_FOLDER, filename)
        cv2.imwrite(filepath, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        return f"/static/gradcam/{filename}"
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None


def predict_with_yolo(image_path):
    """
    Make prediction using YOLO object detection
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with YOLO prediction results
    """
    if yolo_model is None:
        raise Exception("YOLO model not loaded")
    
    try:
        # Run YOLO prediction
        results = yolo_model.predict(image_path, save=False, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return {
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'num_detections': 0,
                'detections': []
            }
        
        # Get the detection with highest confidence
        result = results[0]
        boxes = result.boxes
        
        # Get class with highest confidence
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        
        max_conf_idx = np.argmax(confidences)
        predicted_class_idx = classes[max_conf_idx]
        confidence = float(confidences[max_conf_idx])
        
        # Get predicted class name
        predicted_class = CLASS_NAMES[predicted_class_idx] if predicted_class_idx < len(CLASS_NAMES) else 'unknown'
        
        # Get all detections
        detections = []
        for i in range(len(boxes)):
            detections.append({
                'class': CLASS_NAMES[int(classes[i])] if int(classes[i]) < len(CLASS_NAMES) else 'unknown',
                'confidence': float(confidences[i]),
                'box': boxes.xyxy[i].cpu().numpy().tolist()
            })
        
        # Generate annotated image
        annotated_img = result.plot()
        
        # Save annotated image
        filename = f"yolo_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(GRADCAM_FOLDER, filename)  # Reuse gradcam folder for simplicity
        cv2.imwrite(filepath, annotated_img)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'num_detections': len(boxes),
            'detections': detections,
            'annotated_image_path': f"/static/gradcam/{filename}"
        }
        
    except Exception as e:
        print(f"YOLO prediction error: {e}")
        raise


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expected form data:
        - image: Image file
        - model: Model name (logistic, svm, random_forest, xgboost, kmeans, yolo)
        
    Returns:
        JSON with prediction results and Grad-CAM/YOLO visualization path
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        model_name = request.form.get('model', 'xgboost')
        
        # Save uploaded image temporarily for YOLO
        temp_image_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex[:8]}.jpg")
        image_file.save(temp_image_path)
        
        # Handle YOLO separately
        if model_name == 'yolo':
            if yolo_model is None:
                return jsonify({
                    'error': 'YOLO model not available',
                    'message': 'YOLO model not loaded. Check if yolo_best.pt exists in models/yolo/'
                }), 400
            
            try:
                yolo_results = predict_with_yolo(temp_image_path)
                
                predicted_class = yolo_results['predicted_class']
                confidence = yolo_results['confidence']
                cancer_risk = CANCER_RISK.get(predicted_class, 0.5)
                is_cancer = cancer_risk > 0.5
                
                response = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'is_cancer': is_cancer,
                    'cancer_risk': cancer_risk,
                    'model_used': 'yolo',
                    'num_detections': yolo_results['num_detections'],
                    'detections': yolo_results['detections'],
                    'gradcam_path': yolo_results['annotated_image_path']  # Use annotated image as "gradcam"
                }
                
                # Cleanup
                os.remove(temp_image_path)
                return jsonify(response)
                
            except Exception as e:
                os.remove(temp_image_path)
                return jsonify({'error': f'YOLO prediction failed: {str(e)}'}), 500
        
        # For ML models, use CNN features
        image_file.seek(0)  # Reset file pointer
        img_array, original_img = preprocess_image(image_file)
        
        # Extract CNN features
        features = extract_cnn_features(img_array)
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Get the selected model
        if model_name not in ml_models:
            os.remove(temp_image_path)
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        model = ml_models[model_name]
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 0.75
        
        # Get predicted class name
        predicted_class = CLASS_NAMES[prediction]
        
        # Determine cancer risk
        cancer_risk = CANCER_RISK[predicted_class]
        is_cancer = cancer_risk > 0.5
        
        # Generate Grad-CAM
        gradcam_path = generate_gradcam(img_array, prediction)
        
        # Prepare response
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_cancer': is_cancer,
            'cancer_risk': cancer_risk,
            'model_used': model_name,
            'gradcam_path': gradcam_path,
            'all_probabilities': {
                CLASS_NAMES[i]: float(probabilities[i]) if hasattr(model, 'predict_proba') else 0.0
                for i in range(len(CLASS_NAMES))
            }
        }
        
        # Cleanup
        os.remove(temp_image_path)
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return jsonify({'error': str(e)}), 500


@app.route('/visualizations/<model_name>/<filename>')
def get_visualization(model_name, filename):
    """Serve training visualization images"""
    viz_path = os.path.join(VISUALIZATIONS_FOLDER, model_name)
    
    if not os.path.exists(os.path.join(viz_path, filename)):
        return jsonify({'error': 'Visualization not found'}), 404
    
    return send_from_directory(viz_path, filename)


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'resnet_loaded': resnet_model is not None,
        'scaler_loaded': scaler is not None,
        'models_loaded': list(ml_models.keys()),
        'gradcam_available': gradcam is not None,
        'yolo_available': yolo_model is not None
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BLOOD CANCER DETECTION SYSTEM - WEB SERVER")
    print("="*70)
    print(f"\nServer starting on http://localhost:5000")
    print(f"ML Models loaded: {len(ml_models)}")
    print(f"ResNet50 loaded: {resnet_model is not None}")
    print(f"Grad-CAM available: {gradcam is not None}")
    print(f"YOLO available: {yolo_model is not None}")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

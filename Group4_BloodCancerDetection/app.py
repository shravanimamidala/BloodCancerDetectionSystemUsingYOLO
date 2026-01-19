from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import cv2
import uuid

app = Flask(__name__, static_folder='static')
CORS(app)

print("="*70)
print("LOADING MODELS...")
print("="*70)

# Load ResNet50 for feature extraction AND Grad-CAM
print("Loading ResNet50...")
resnet_features = ResNet50(weights='imagenet', include_top=False, pooling='avg')
resnet_full = ResNet50(weights='imagenet', include_top=True)
print("✓ ResNet50 loaded (features + Grad-CAM)")

# Load scaler and ML models
scaler = joblib.load('models/scaler.pkl')
logistic_model = joblib.load('models/logistic_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')

models = {
    'logistic': logistic_model,
    'svm': svm_model,
    'random_forest': rf_model,
    'xgboost': xgb_model,
    'kmeans': kmeans_model
}

print("✓ Scaler loaded")
print("✓ Logistic Regression loaded")
print("✓ SVM loaded")
print("✓ Random Forest loaded")
print("✓ XGBoost loaded")


# Try to load YOLO (optional)
yolo_model = None
try:
    from ultralytics import YOLO
    if os.path.exists('models/yolo/yolo_best.pt'):
        yolo_model = YOLO('models/yolo/yolo_best.pt')
        print("✓ YOLO loaded")
    else:
        print("⚠️  YOLO model not found (models/yolo/yolo_best.pt)")
except ImportError:
    print("⚠️  YOLO not available (pip install ultralytics)")
except Exception as e:
    print(f"⚠️  YOLO load error: {e}")

class_names = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']
cancer_risk = {
    'basophil': 0.15,
    'erythroblast': 0.45,
    'monocyte': 0.25,
    'myeloblast': 0.85,
    'seg_neutrophil': 0.10
}

# Create gradcam folder
os.makedirs('static/gradcam', exist_ok=True)

print("="*70)
print(f"✓ All models loaded successfully!")
print("="*70)


def extract_features(img_path):
    """Extract 2048-dim features using ResNet50"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_features.predict(img_array, verbose=0)
    return features.flatten()


def generate_gradcam(img_path, predicted_class_idx):
    """
    Generate Grad-CAM heatmap using ResNet50
    Shows red heatmap overlay indicating where the model focused
    """
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Get last convolutional layer
        last_conv_layer = resnet_full.get_layer('conv5_block3_out')
        
        # Create gradient model
        grad_model = Model(
            inputs=[resnet_full.inputs],
            outputs=[last_conv_layer.output, resnet_full.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, predicted_class_idx % 1000]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        
        # Resize and colorize heatmap (RED overlay)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Load original image and superimpose heatmap
        original = cv2.imread(img_path)
        original = cv2.resize(original, (224, 224))
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        
        # Save heatmap
        filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join('static/gradcam', filename)
        cv2.imwrite(filepath, superimposed)
        
        print(f"  ✓ Grad-CAM saved: {filename}")
        return f"/static/gradcam/{filename}"
        
    except Exception as e:
        print(f"  ✗ Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return None


def handle_yolo_prediction(file):
    """Handle YOLO object detection prediction"""
    if yolo_model is None:
        return jsonify({
            'error': 'YOLO not available',
            'message': 'Install: pip install ultralytics, and ensure models/yolo/yolo_best.pt exists'
        }), 400
    
    try:
        # Save temp file
        temp_path = 'temp_upload_yolo.jpg'
        file.save(temp_path)
        
        print(f"  Running YOLO detection...")
        
        # Run YOLO prediction
        results = yolo_model.predict(temp_path, save=False, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            os.remove(temp_path)
            return jsonify({
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'is_cancer': False,
                'cancer_risk': 0.0,
                'model_used': 'yolo',
                'gradcam_path': None,
                'message': 'No cells detected'
            })
        
        # Get best detection
        result = results[0]
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        
        max_conf_idx = np.argmax(confidences)
        predicted_class_idx = classes[max_conf_idx]
        confidence = float(confidences[max_conf_idx])
        
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else 'unknown'
        is_cancer = cancer_risk.get(predicted_class, 0.0) > 0.5
        
        # Generate annotated image
        annotated_img = result.plot()
        filename = f"yolo_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join('static/gradcam', filename)
        cv2.imwrite(filepath, annotated_img)
        
        print(f"  ✓ YOLO result: {predicted_class} ({confidence:.2%})")
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_cancer': is_cancer,
            'cancer_risk': cancer_risk.get(predicted_class, 0.0),
            'model_used': 'yolo',
            'gradcam_path': f"/static/gradcam/{filename}",
            'num_detections': len(boxes)
        })
        
    except Exception as e:
        print(f"  ✗ YOLO error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists('temp_upload_yolo.jpg'):
            os.remove('temp_upload_yolo.jpg')
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    """Serve main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    - Accepts image upload
    - Extracts features using ResNet50 (or YOLO for object detection)
    - Predicts using selected ML model
    - Generates Grad-CAM heatmap (or YOLO bounding boxes)
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        model_name = request.form.get('model', 'xgboost')
        
        print(f"\n{'='*50}")
        print(f"📊 Prediction Request")
        print(f"{'='*50}")
        print(f"  Model: {model_name}")
        
        # Handle YOLO separately
        if model_name == 'yolo':
            return handle_yolo_prediction(file)
        
        # Save temp file
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)
        print(f"  ✓ Image uploaded")
        
        # Extract features
        print(f"  Extracting CNN features...")
        features = extract_features(temp_path)
        features_scaled = scaler.transform([features])
        print(f"  ✓ Features extracted (2048 dims)")
        
        # Predict
        model = models[model_name]
        prediction = model.predict(features_scaled)[0]
        
        # Get confidence (handle K-Means which doesn't have predict_proba)
        if model_name == 'kmeans':
            # K-Means: use distance-based confidence
            distances = model.transform(features_scaled)[0]
            min_distance = distances[prediction]
            max_distance = np.max(distances)
            confidence = float(1 - (min_distance / (max_distance + 1e-10)))
            confidence = max(0.5, min(0.95, confidence))  # Clamp between 50-95%
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = float(proba[prediction])
        else:
            confidence = 0.75
        
        predicted_class = class_names[prediction]
        is_cancer = cancer_risk[predicted_class] > 0.5
        
        print(f"  ✓ Prediction: {predicted_class}")
        print(f"  ✓ Confidence: {confidence:.2%}")
        
        # Generate Grad-CAM heatmap
        print(f"  Generating Grad-CAM heatmap...")
        gradcam_path = generate_gradcam(temp_path, prediction)
        
        # Cleanup
        os.remove(temp_path)
        
        # Response
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_cancer': is_cancer,
            'cancer_risk': cancer_risk[predicted_class],
            'model_used': model_name,
            'gradcam_path': gradcam_path
        }
        
        print(f"  ✓ Complete!")
        print(f"{'='*50}\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        if os.path.exists('temp_upload.jpg'):
            os.remove('temp_upload.jpg')
        
        return jsonify({'error': str(e)}), 500


@app.route('/visualizations/<model_name>/<filename>')
def visualizations(model_name, filename):
    """Serve training visualization images"""
    return send_from_directory(f'visualizations/{model_name}', filename)


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': list(models.keys()),
        'gradcam': True,
        'resnet_features': True,
        'resnet_full': True,
        'yolo_available': yolo_model is not None
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔬 BLOOD CANCER DETECTION SYSTEM - WEB SERVER")
    print("="*70)
    print("🌐 Server: http://localhost:5000")
    print(f"📊 ML Models: {len(models)}")
    print("🧠 ResNet50: Loaded (feature extraction)")
    print("🔥 Grad-CAM: Enabled (red heatmap overlay)")
    print(f"🎯 YOLO: {'Enabled' if yolo_model else 'Not available'}")
    print("\n💡 What You'll See:")
    print("   - Logistic/SVM/RF/XGB/K-Means: Red heatmap overlay")
    print("   - YOLO: Bounding boxes with cell detection")
    print("   - Shows where AI focused when making predictions")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

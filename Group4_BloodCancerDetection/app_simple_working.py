from flask import Flask, request, jsonify, send_from_directory
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

# Load ResNet50 for feature extraction AND Grad-CAM
print("Loading ResNet50...")
resnet_features = ResNet50(weights='imagenet', include_top=False, pooling='avg')
resnet_full = ResNet50(weights='imagenet', include_top=True)
print("✓ ResNet50 loaded")

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

def extract_features(img_path):
    """Extract 2048-dim features using ResNet50"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_features.predict(img_array, verbose=0)
    return features.flatten()

def generate_gradcam(img_path, predicted_class_idx):
    """Generate Grad-CAM heatmap using ResNet50"""
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
            class_channel = predictions[:, predicted_class_idx % 1000]  # ImageNet has 1000 classes
        
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
        
        # Resize and colorize heatmap
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
        
        print(f"✓ Grad-CAM saved: {filename}")
        return f"/static/gradcam/{filename}"
        
    except Exception as e:
        print(f"✗ Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400
        
        file = request.files['image']
        model_name = request.form.get('model', 'xgboost')
        
        print(f"\nPrediction: model={model_name}")
        
        # Save temp file
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)
        
        # Extract features
        features = extract_features(temp_path)
        features_scaled = scaler.transform([features])
        
        # Predict
        model = models[model_name]
        prediction = model.predict(features_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            confidence = float(proba[prediction])
        else:
            confidence = 0.75
        
        predicted_class = class_names[prediction]
        is_cancer = cancer_risk[predicted_class] > 0.5
        
        # Generate Grad-CAM heatmap
        print("Generating Grad-CAM...")
        gradcam_path = generate_gradcam(temp_path, prediction)
        
        # Cleanup
        os.remove(temp_path)
        
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_cancer': is_cancer,
            'cancer_risk': cancer_risk[predicted_class],
            'model_used': model_name,
            'gradcam_path': gradcam_path
        }
        
        print(f"✓ Result: {predicted_class} ({confidence:.2%})")
        print(f"✓ Grad-CAM: {gradcam_path}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists('temp_upload.jpg'):
            os.remove('temp_upload.jpg')
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations/<model_name>/<filename>')
def visualizations(model_name, filename):
    return send_from_directory(f'visualizations/{model_name}', filename)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models': list(models.keys()),
        'gradcam': True
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔬 BLOOD CANCER DETECTION SYSTEM")
    print("="*60)
    print("🌐 Server: http://localhost:5000")
    print("📊 Models loaded:", len(models))
    print("🔥 Grad-CAM: Enabled")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

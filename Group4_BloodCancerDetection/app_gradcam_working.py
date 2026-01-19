from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load models
cnn_model = load_model('models/cnn_feature_extractor.h5')
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
    """Extract CNN features from image"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    features = cnn_model.predict(img_array)
    return features.flatten()

def generate_gradcam(img_path, predicted_class_idx):
    """Generate Grad-CAM heatmap"""
    try:
        from tensorflow.keras.models import Model
        import tensorflow as tf
        
        # Load image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get last conv layer
        last_conv_layer = cnn_model.get_layer('conv5_block3_out')
        
        # Create gradient model
        grad_model = Model(
            inputs=[cnn_model.inputs],
            outputs=[last_conv_layer.output, cnn_model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, predicted_class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        
        # Resize and colorize
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Load original and superimpose
        original = cv2.imread(img_path)
        original = cv2.resize(original, (224, 224))
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        
        # Save
        import uuid
        filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join('static/gradcam', filename)
        cv2.imwrite(filepath, superimposed)
        
        return f"/static/gradcam/{filename}"
    except Exception as e:
        print(f"Grad-CAM error: {e}")
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
        
        # Generate Grad-CAM
        gradcam_path = generate_gradcam(temp_path, prediction)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_cancer': is_cancer,
            'cancer_risk': cancer_risk[predicted_class],
            'model_used': model_name,
            'gradcam_path': gradcam_path
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations/<model_name>/<filename>')
def visualizations(model_name, filename):
    return send_from_directory(f'visualizations/{model_name}', filename)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

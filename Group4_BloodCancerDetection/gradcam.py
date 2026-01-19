"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Generates heatmaps showing which regions of the image influenced the prediction
"""

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input


class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Keras model
            layer_name: Name of target conv layer (default: last conv layer)
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:  # Conv layer has 4D output
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"Using layer: {layer_name}")
        
        # Create gradient model
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    
    def compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image array (1, H, W, 3)
            class_idx: Target class index (None = predicted class)
            eps: Small value for numerical stability
            
        Returns:
            heatmap: 2D array with activation values
        """
        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            # Get conv layer output and predictions
            conv_outputs, predictions = self.grad_model(img_array)
            
            # If class_idx not specified, use predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of class score with respect to conv layer output
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by computed gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Multiply each channel by corresponding gradient
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all channels
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Apply ReLU (only positive activations)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: 2D heatmap array
            original_img: Original image (H, W, 3) or path
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap
            
        Returns:
            superimposed_img: Image with heatmap overlay
        """
        # Load image if path provided
        if isinstance(original_img, str):
            original_img = cv2.imread(original_img)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Convert heatmap to 0-255 range
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap_colored * alpha + original_img * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)
        
        return superimposed_img, heatmap_colored
    
    def generate_gradcam(self, img_path, class_idx=None, save_path=None):
        """
        Complete Grad-CAM pipeline: load image, compute heatmap, overlay
        
        Args:
            img_path: Path to input image
            class_idx: Target class index (None = predicted class)
            save_path: Path to save output image
            
        Returns:
            superimposed_img: Image with Grad-CAM overlay
            heatmap: Raw heatmap
            predicted_class: Predicted class index
        """
        # Load and preprocess image
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        original_img = img_array.copy()
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        
        # Compute heatmap
        heatmap = self.compute_heatmap(img_array, class_idx)
        
        # Overlay heatmap on original image
        superimposed_img, heatmap_colored = self.overlay_heatmap(
            heatmap, 
            np.uint8(original_img)
        )
        
        # Save if path provided
        if save_path:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(np.uint8(original_img))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap_colored)
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # Superimposed
            axes[2].imshow(superimposed_img)
            axes[2].set_title(f'Grad-CAM Overlay (Class: {predicted_class})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Grad-CAM visualization saved to: {save_path}")
        
        return superimposed_img, heatmap, predicted_class


def create_gradcam_for_classification(cnn_model, img_path, class_names, save_path=None):
    """
    Helper function to create Grad-CAM for blood cell classification
    
    Args:
        cnn_model: Trained CNN model
        img_path: Path to blood cell image
        class_names: List of class names
        save_path: Optional path to save visualization
        
    Returns:
        gradcam_img: Grad-CAM visualization
        predicted_class_name: Name of predicted class
    """
    # Create Grad-CAM object
    gradcam = GradCAM(cnn_model)
    
    # Generate Grad-CAM
    superimposed_img, heatmap, predicted_class = gradcam.generate_gradcam(
        img_path, save_path=save_path
    )
    
    predicted_class_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
    
    print(f"Predicted class: {predicted_class_name} (index: {predicted_class})")
    
    return superimposed_img, predicted_class_name


if __name__ == "__main__":
    from tensorflow.keras.applications import ResNet50
    
    # Example usage
    print("Loading model...")
    model = ResNet50(weights='imagenet', include_top=True)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, layer_name='conv5_block3_out')
    
    # Generate Grad-CAM for an image
    img_path = "test_image.jpg"
    gradcam_img, heatmap, pred_class = gradcam.generate_gradcam(
        img_path, 
        save_path="gradcam_output.png"
    )
    
    print(f"Predicted class: {pred_class}")

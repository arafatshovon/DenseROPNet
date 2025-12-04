"""
Explainability Module for ROP Detection
========================================

This module provides Explainable AI (XAI) techniques for interpreting
model predictions, including Grad-CAM and LIME.

Grad-CAM (Gradient-weighted Class Activation Mapping):
    Produces a heatmap showing which regions of the input image most
    influenced the model's prediction.

LIME (Local Interpretable Model-Agnostic Explanations):
    Identifies superpixels that contribute most to the prediction.

References:
    - Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep
      Networks via Gradient-based Localization"
    - Ribeiro et al. (2016). "Why Should I Trust You?" Explaining the
      Predictions of Any Classifier"
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from lime import lime_image
from skimage.segmentation import mark_boundaries


def get_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a given image.
    
    Grad-CAM uses the gradients of the target class flowing into the
    final convolutional layer to produce a coarse localization map
    highlighting important regions in the image.
    
    Args:
        model: Trained Keras model
        image: Input image (H, W, C) or (1, H, W, C)
        class_index: Target class index (uses predicted class if None)
        last_conv_layer_name: Name of last conv layer (auto-detected if None)
    
    Returns:
        Heatmap as numpy array (H, W) with values in [0, 1]
    """
    # Ensure batch dimension
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    
    # Find last convolutional layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        raise ValueError("Could not find a convolutional layer in the model")
    
    # Create gradient model
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        
        loss = predictions[:, class_index]
        tape.watch(conv_outputs)
    
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute channel-wise mean of gradients
    pooled_grads = tf.reduce_mean(grads[0], axis=(0, 1))
    
    # Weight feature maps by gradient importance
    conv_output = conv_outputs[0]
    heatmap = conv_output * pooled_grads
    heatmap = tf.reduce_sum(heatmap, axis=-1)
    
    # Apply ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: Original image (H, W, C) in RGB, values in [0, 1] or [0, 255]
        heatmap: Grad-CAM heatmap (H, W) with values in [0, 1]
        alpha: Opacity of heatmap overlay
        colormap: OpenCV colormap for heatmap
    
    Returns:
        Overlaid image in RGB format
    """
    # Ensure image is in proper format
    if image.max() <= 1.0:
        image_uint8 = np.uint8(image * 255)
    else:
        image_uint8 = np.uint8(image)
    
    # Resize heatmap to match image size
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay


def plot_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_names: List[str],
    true_label: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Generate and plot Grad-CAM visualization.
    
    Args:
        model: Trained Keras model
        image: Input image (H, W, C)
        class_names: List of class names
        true_label: Ground truth label index (optional)
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    # Get prediction
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_label = class_names[pred_idx]
    confidence = predictions[0][pred_idx]
    
    # Generate heatmap
    heatmap = get_gradcam_heatmap(model, image, class_index=pred_idx)
    overlay = overlay_gradcam(image, heatmap)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    ax1 = axes[0]
    display_img = image if image.max() <= 1.0 else image / 255.0
    ax1.imshow(display_img)
    title = f"Original Image"
    if true_label is not None:
        title += f"\nTrue: {class_names[true_label]}"
    ax1.set_title(title, fontsize=11)
    ax1.axis("off")
    
    # Heatmap
    ax2 = axes[1]
    ax2.imshow(heatmap, cmap="jet")
    ax2.set_title("Grad-CAM Heatmap", fontsize=11)
    ax2.axis("off")
    
    # Overlay
    ax3 = axes[2]
    ax3.imshow(overlay)
    ax3.set_title(f"Grad-CAM Overlay\nPred: {pred_label} ({confidence:.2%})", fontsize=11)
    ax3.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grad-CAM visualization saved to: {save_path}")
    
    return fig


def explain_with_lime(
    model: tf.keras.Model,
    image: np.ndarray,
    class_names: List[str],
    num_samples: int = 1000,
    num_features: int = 5,
    true_label: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Generate LIME explanation for a prediction.
    
    LIME creates interpretable explanations by learning a simpler
    model locally around the prediction to identify which parts
    of the image are most important.
    
    Args:
        model: Trained Keras model
        image: Input image (H, W, C) normalized to [0, 1]
        class_names: List of class names
        num_samples: Number of samples for LIME
        num_features: Number of superpixels to highlight
        true_label: Ground truth label index (optional)
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Prediction function for LIME
    def predict_fn(images):
        images = np.array(images)
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        return model.predict(images, verbose=0)
    
    # Get prediction
    predictions = predict_fn(image)
    pred_idx = np.argmax(predictions[0])
    pred_label = class_names[pred_idx]
    confidence = predictions[0][pred_idx]
    
    # Create LIME explainer and generate explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )
    
    # Get image and mask for the top predicted class
    explained_class = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        explained_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False,
    )
    
    # Create visualization with boundaries
    lime_img = mark_boundaries(temp, mask, color=(0, 1, 0))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    ax1 = axes[0]
    ax1.imshow(image)
    title = "Original Image"
    if true_label is not None:
        title += f"\nTrue: {class_names[true_label]}"
    ax1.set_title(title, fontsize=11)
    ax1.axis("off")
    
    # LIME explanation
    ax2 = axes[1]
    ax2.imshow(lime_img)
    ax2.set_title(f"LIME Explanation\nPred: {pred_label} ({confidence:.2%})", fontsize=11)
    ax2.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"LIME explanation saved to: {save_path}")
    
    return fig


def explain_prediction(
    model: tf.keras.Model,
    image: np.ndarray,
    class_names: List[str],
    true_label: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Generate combined Grad-CAM and LIME explanations.
    
    Args:
        model: Trained Keras model
        image: Input image (H, W, C)
        class_names: List of class names
        true_label: Ground truth label index (optional)
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    # Ensure image is normalized
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Get prediction
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_label = class_names[pred_idx]
    confidence = predictions[0][pred_idx]
    
    # Generate Grad-CAM
    heatmap = get_gradcam_heatmap(model, image, class_index=pred_idx)
    overlay = overlay_gradcam(image, heatmap)
    
    # Generate LIME
    def predict_fn(images):
        return model.predict(np.array(images), verbose=0)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image, predict_fn, top_labels=1, hide_color=0, num_samples=500
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    lime_img = mark_boundaries(temp, mask, color=(0, 1, 0))
    
    # Plot combined
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original
    axes[0].imshow(image)
    title = "Original"
    if true_label is not None:
        title += f"\nTrue: {class_names[true_label]}"
    axes[0].set_title(title, fontsize=10)
    axes[0].axis("off")
    
    # Grad-CAM heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[1].axis("off")
    
    # Grad-CAM overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay", fontsize=10)
    axes[2].axis("off")
    
    # LIME
    axes[3].imshow(lime_img)
    axes[3].set_title(f"LIME\nPred: {pred_label} ({confidence:.2%})", fontsize=10)
    axes[3].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Explanation saved to: {save_path}")
    
    return fig


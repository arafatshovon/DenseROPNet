#!/usr/bin/env python3
"""
Inference Script for ROP Detection Model
=========================================

This script performs inference on new fundus images using a trained
ROP detection model.

Usage:
    python predict.py --model checkpoints/best_model.keras --image path/to/image.jpg
    python predict.py --model checkpoints/best_model.keras --image_dir path/to/images/

Author: Arafat
"""

import os
import sys
import argparse
import json
import numpy as np
from glob import glob

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from src.data.preprocessing import preprocess_image
from src.utils.explainability import plot_gradcam


# Default class names
DEFAULT_CLASSES = ["Normal", "Stage1", "Stage2", "Stage3"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ROP Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.keras file)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class names",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions (JSON)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate Grad-CAM visualization",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use",
    )
    
    return parser.parse_args()


def predict_single(
    model: tf.keras.Model,
    image_path: str,
    classes: list,
    target_size: tuple = (299, 299),
) -> dict:
    """
    Predict ROP stage for a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        classes: List of class names
        target_size: Target image size
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    image = preprocess_image(image_path, target_size=target_size)
    
    # Get prediction
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch, verbose=0)[0]
    
    pred_idx = int(np.argmax(predictions))
    pred_label = classes[pred_idx]
    confidence = float(predictions[pred_idx])
    
    # Create probability dictionary
    probabilities = {cls: float(prob) for cls, prob in zip(classes, predictions)}
    
    return {
        "image": image_path,
        "predicted_class": pred_label,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": probabilities,
        "preprocessed_image": image,
    }


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Validate inputs
    if args.image is None and args.image_dir is None:
        print("Error: Please provide --image or --image_dir")
        sys.exit(1)
    
    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(glob(os.path.join(args.image_dir, ext)))
            image_paths.extend(glob(os.path.join(args.image_dir, ext.upper())))
    
    if not image_paths:
        print("Error: No images found")
        sys.exit(1)
    
    print("=" * 60)
    print("ROP Detection Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Images: {len(image_paths)}")
    print(f"Classes: {args.classes}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model(args.model)
    
    # Process images
    print("\nProcessing images...")
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(image_path)}")
        
        try:
            result = predict_single(model, image_path, args.classes)
            
            print(f"    Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
            
            # Generate explanation if requested
            if args.explain:
                explain_path = os.path.splitext(image_path)[0] + "_gradcam.png"
                plot_gradcam(
                    model=model,
                    image=result["preprocessed_image"],
                    class_names=args.classes,
                    save_path=explain_path,
                )
                print(f"    Explanation saved to: {explain_path}")
            
            # Remove preprocessed image from results (not JSON serializable)
            del result["preprocessed_image"]
            results.append(result)
            
        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                "image": image_path,
                "error": str(e),
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if "error" not in r]
    if successful:
        for cls in args.classes:
            count = sum(1 for r in successful if r["predicted_class"] == cls)
            print(f"  {cls}: {count}")
    
    print(f"\n  Total: {len(successful)} successful, {len(results) - len(successful)} failed")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()


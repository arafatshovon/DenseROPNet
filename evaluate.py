#!/usr/bin/env python3
"""
Evaluation Script for ROP Detection Model
==========================================

This script evaluates a trained ROP detection model on test data,
generating metrics, visualizations, and explainability outputs.

Usage:
    python evaluate.py --model checkpoints/best_model.keras --data_dir /path/to/data
    python evaluate.py --config configs/config.yaml

Author: Arafat
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from src.data.dataloader import ROPDataLoader, create_data_generator
from src.data.preprocessing import preprocess_image
from src.utils.metrics import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.utils.visualization import plot_sample_predictions
from src.utils.explainability import plot_gradcam, explain_with_lime, explain_prediction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ROP Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.keras file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to test dataset directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate explainability visualizations",
    )
    parser.add_argument(
        "--num_explain",
        type=int,
        default=5,
        help="Number of samples for explainability",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_gpu(gpu_ids: str):
    """Setup GPU configuration."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def evaluate_model(
    model: tf.keras.Model,
    X_test: list,
    y_test: np.ndarray,
    classes: list,
    batch_size: int,
    target_size: tuple,
) -> dict:
    """
    Evaluate model on test data.
    
    Returns dictionary with predictions and metrics.
    """
    # Create test generator
    test_gen = create_data_generator(
        X_test,
        y_test.tolist(),
        batch_size=batch_size,
        target_size=target_size,
        shuffle=False,
        augment=False,
        infinite=False,
    )
    
    # Collect predictions
    all_true = []
    all_pred_proba = []
    all_images = []
    
    for batch_images, batch_labels in test_gen:
        pred_proba = model.predict(batch_images, verbose=0)
        all_true.extend(batch_labels)
        all_pred_proba.extend(pred_proba)
        all_images.extend(batch_images)
    
    y_true = np.array(all_true)
    y_pred_proba = np.array(all_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    images = np.array(all_images)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_pred_proba, classes)
    
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "images": images,
        "metrics": metrics,
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup GPU
    setup_gpu(args.gpu)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    print("=" * 60)
    print("ROP Detection Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = tf.keras.models.load_model(args.model)
    print(f"Model loaded: {model.name}")
    
    # Load data
    print("\n[2/4] Loading test data...")
    classes = config["data"]["classes"]
    target_size = tuple(config["preprocessing"]["image_size"])
    
    # Check for saved test data
    test_data_path = os.path.join(os.path.dirname(args.model), "..", "results", "test_data.json")
    if os.path.exists(test_data_path):
        with open(test_data_path, "r") as f:
            test_data = json.load(f)
        X_test = test_data["X_test"]
        y_test = np.array(test_data["y_test"])
        print(f"Loaded saved test data: {len(X_test)} samples")
    else:
        # Load full dataset and split
        loader = ROPDataLoader(
            data_dir=config["data"]["data_dir"],
            classes=classes,
        )
        _, _, X_test, _, _, y_test = loader.split_data(
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"],
            random_state=config["data"]["random_state"],
        )
    
    print(f"Test samples: {len(X_test)}")
    
    # Evaluate model
    print("\n[3/4] Evaluating model...")
    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
        batch_size=args.batch_size,
        target_size=target_size,
    )
    
    # Print and save results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Macro F1-Score: {results['metrics']['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {results['metrics']['weighted_f1']:.4f}")
    
    if "macro_auc" in results["metrics"]:
        print(f"Macro AUC: {results['metrics']['macro_auc']:.4f}")
    
    # Classification report
    print("\n")
    report = print_classification_report(results["y_true"], results["y_pred"], classes)
    
    # Save classification report
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        classes,
        save_path=os.path.join(args.output_dir, "figures", "confusion_matrix.png"),
    )
    
    # Normalized confusion matrix
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        classes,
        normalize=True,
        save_path=os.path.join(args.output_dir, "figures", "confusion_matrix_normalized.png"),
    )
    
    # Plot ROC curves
    print("Generating ROC curves...")
    plot_roc_curves(
        results["y_true"],
        results["y_pred_proba"],
        classes,
        save_path=os.path.join(args.output_dir, "figures", "roc_curves.png"),
    )
    
    # Plot sample predictions
    print("Generating sample predictions...")
    plot_sample_predictions(
        results["images"],
        results["y_true"],
        results["y_pred"],
        classes,
        num_samples=9,
        save_path=os.path.join(args.output_dir, "figures", "sample_predictions.png"),
    )
    
    # Explainability analysis
    if args.explain:
        print("\n[4/4] Generating explainability visualizations...")
        explain_dir = os.path.join(args.output_dir, "figures", "explainability")
        os.makedirs(explain_dir, exist_ok=True)
        
        # Select random samples
        num_explain = min(args.num_explain, len(results["images"]))
        indices = np.random.choice(len(results["images"]), num_explain, replace=False)
        
        for i, idx in enumerate(indices):
            print(f"  Explaining sample {i+1}/{num_explain}...")
            
            explain_prediction(
                model=model,
                image=results["images"][idx],
                class_names=classes,
                true_label=results["y_true"][idx],
                save_path=os.path.join(explain_dir, f"explanation_{i+1}.png"),
            )
    else:
        print("\n[4/4] Skipping explainability (use --explain to enable)")
    
    # Save metrics to JSON
    metrics_to_save = {
        "accuracy": float(results["metrics"]["accuracy"]),
        "macro_precision": float(results["metrics"]["macro_precision"]),
        "macro_recall": float(results["metrics"]["macro_recall"]),
        "macro_f1": float(results["metrics"]["macro_f1"]),
        "weighted_precision": float(results["metrics"]["weighted_precision"]),
        "weighted_recall": float(results["metrics"]["weighted_recall"]),
        "weighted_f1": float(results["metrics"]["weighted_f1"]),
        "per_class_precision": results["metrics"]["precision"].tolist(),
        "per_class_recall": results["metrics"]["recall"].tolist(),
        "per_class_f1": results["metrics"]["f1_score"].tolist(),
        "confusion_matrix": results["metrics"]["confusion_matrix"].tolist(),
    }
    
    if "macro_auc" in results["metrics"]:
        metrics_to_save["macro_auc"] = float(results["metrics"]["macro_auc"])
        metrics_to_save["per_class_auc"] = {
            classes[k]: float(v) for k, v in results["metrics"]["auc_per_class"].items()
        }
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Training Script for ROP Detection Model
========================================

This script trains the DenseNet121-SE-RANB-DualPool model for
Retinopathy of Prematurity (ROP) classification.

Usage:
    python train.py --config configs/config.yaml
    python train.py --data_dir /path/to/data --epochs 50 --batch_size 32

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
from src.data.dataloader import ROPDataLoader, create_data_generators
from src.models.densenet_se import create_rop_model
from src.utils.callbacks import get_callbacks
from src.utils.visualization import plot_training_history, plot_class_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ROP Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Path to dataset directory (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID(s) to use",
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
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU(s): {gpu_ids}")
            print(f"Available GPUs: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available, using CPU")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup GPU
    setup_gpu(args.gpu)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["optimizer"]["learning_rate"] = args.learning_rate
    if args.output_dir:
        config["output"]["model_dir"] = os.path.join(args.output_dir, "checkpoints")
        config["output"]["log_dir"] = os.path.join(args.output_dir, "logs")
        config["output"]["eval_dir"] = os.path.join(args.output_dir, "results")
    
    # Create output directories
    os.makedirs(config["output"]["model_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)
    os.makedirs(config["output"]["eval_dir"], exist_ok=True)
    
    print("=" * 60)
    print("ROP Detection Model Training")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Data directory: {config['data']['data_dir']}")
    print(f"Classes: {config['data']['classes']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['optimizer']['learning_rate']}")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    loader = ROPDataLoader(
        data_dir=config["data"]["data_dir"],
        classes=config["data"]["classes"],
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=config["data"]["random_state"],
    )
    
    # Print class distributions
    print("\nClass distributions:")
    loader.get_class_distribution(y_train, "Training")
    loader.get_class_distribution(y_val, "Validation")
    loader.get_class_distribution(y_test, "Test")
    
    # Save class distribution plot
    fig = plot_class_distribution(
        y_train,
        config["data"]["classes"],
        title="Training Set Class Distribution",
        save_path=os.path.join(config["output"]["eval_dir"], "class_distribution.png"),
    )
    
    # Create data generators
    print("\n[2/5] Creating data generators...")
    train_gen, val_gen, steps_per_epoch, validation_steps = create_data_generators(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=config["training"]["batch_size"],
        target_size=tuple(config["preprocessing"]["image_size"]),
        augment_train=config["preprocessing"]["augmentation"]["enabled"],
    )
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Create model
    print("\n[3/5] Building model...")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = tf.keras.models.load_model(args.resume)
    else:
        model = create_rop_model(
            input_shape=tuple(config["model"]["input_shape"]),
            num_classes=config["model"]["num_classes"],
            backbone_weights=config["model"]["backbone"]["weights"],
            freeze_backbone=config["model"]["backbone"]["freeze"],
            dropout_rate=config["model"]["head"]["dropout_rate"],
            l2_regularization=config["model"]["head"]["l2_regularization"],
        )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"]["optimizer"]["learning_rate"]
        ),
        loss=config["training"]["loss"],
        metrics=config["training"]["metrics"],
    )
    
    print(f"Model: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Setup callbacks
    print("\n[4/5] Setting up callbacks...")
    model_save_path = os.path.join(
        config["output"]["model_dir"],
        config["output"]["best_model_name"],
    )
    
    callbacks = get_callbacks(
        model_save_path=model_save_path,
        log_dir=config["output"]["log_dir"],
        csv_log_path=os.path.join(
            config["output"]["eval_dir"],
            config["output"]["history_csv"],
        ),
        early_stopping_patience=config["training"]["callbacks"]["early_stopping"]["patience"],
        reduce_lr_patience=config["training"]["callbacks"]["reduce_lr"]["patience"],
        reduce_lr_factor=config["training"]["callbacks"]["reduce_lr"]["factor"],
        min_lr=config["training"]["callbacks"]["reduce_lr"]["min_lr"],
    )
    
    # Train model
    print("\n[5/5] Starting training...")
    print("-" * 60)
    
    start_time = datetime.now()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=config["training"]["epochs"],
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    
    training_time = datetime.now() - start_time
    
    print("-" * 60)
    print(f"\nTraining completed in: {training_time}")
    
    # Save training history
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(os.path.join(config["output"]["eval_dir"], "training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=2)
    
    # Plot training history
    fig = plot_training_history(
        history,
        save_path=os.path.join(config["output"]["eval_dir"], "training_curves.png"),
    )
    
    # Print best results
    best_epoch = np.argmin(history.history["val_loss"])
    print(f"\nBest Model (Epoch {best_epoch + 1}):")
    print(f"  Training Loss: {history.history['loss'][best_epoch]:.4f}")
    print(f"  Training Accuracy: {history.history['accuracy'][best_epoch]:.4f}")
    print(f"  Validation Loss: {history.history['val_loss'][best_epoch]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    
    # Save test data paths for evaluation
    test_data = {
        "X_test": X_test,
        "y_test": y_test.tolist(),
        "classes": config["data"]["classes"],
    }
    with open(os.path.join(config["output"]["eval_dir"], "test_data.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Logs saved to: {config['output']['log_dir']}")
    print(f"Results saved to: {config['output']['eval_dir']}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


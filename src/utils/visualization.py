"""
Visualization Utilities for ROP Detection
==========================================

This module provides visualization functions for training history,
sample predictions, and data exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
import os


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot training history showing loss and accuracy curves.
    
    Args:
        history: Training history dictionary (from model.fit())
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    # Handle both Keras History object and dict
    if hasattr(history, "history"):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot Loss
    ax1 = axes[0]
    epochs = range(1, len(history["loss"]) + 1)
    
    ax1.plot(epochs, history["loss"], "b-", linewidth=2, label="Training Loss")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Validation Loss")
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2 = axes[1]
    
    ax2.plot(epochs, history["accuracy"], "b-", linewidth=2, label="Training Accuracy")
    if "val_accuracy" in history:
        ax2.plot(epochs, history["val_accuracy"], "r-", linewidth=2, label="Validation Accuracy")
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to: {save_path}")
    
    return fig


def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    num_samples: int = 9,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
) -> plt.Figure:
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        images: Array of images
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to plot
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    num_samples = min(num_samples, len(images))
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        img = images[i]
        
        # Handle different image formats
        if img.max() <= 1.0:
            img = img  # Already normalized
        else:
            img = img / 255.0
        
        ax.imshow(img)
        
        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        
        # Color based on correct/incorrect prediction
        color = "green" if y_true[i] == y_pred[i] else "red"
        
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}",
            fontsize=10,
            color=color,
        )
        ax.axis("off")
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")
    
    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sample predictions saved to: {save_path}")
    
    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot class distribution as a bar chart.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    from collections import Counter
    
    counts = Counter(labels)
    classes = [class_names[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(classes, values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(classes))))
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            str(val),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Class distribution saved to: {save_path}")
    
    return fig


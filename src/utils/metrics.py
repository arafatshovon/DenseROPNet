"""
Evaluation Metrics and Visualization for ROP Detection
========================================================

This module provides functions for computing and visualizing evaluation
metrics including confusion matrix, ROC curves, and classification reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Tuple, Optional
from itertools import cycle


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels (integer indices)
        y_pred: Predicted labels (integer indices)
        y_pred_proba: Prediction probabilities for each class
        class_names: List of class names
    
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1
    metrics["support"] = support
    
    # Macro and weighted averages
    for avg in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
        metrics[f"{avg}_precision"] = p
        metrics[f"{avg}_recall"] = r
        metrics[f"{avg}_f1"] = f
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        num_classes = y_pred_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        
        auc_scores = {}
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc_scores[i] = auc(fpr, tpr)
        
        metrics["auc_per_class"] = auc_scores
        metrics["macro_auc"] = np.mean(list(auc_scores.values()))
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Print and return classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print("=" * 60)
    print(report)
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
        cmap: Colormap for heatmap
        normalize: Whether to normalize the confusion matrix
    
    Returns:
        Matplotlib Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Prediction probabilities for each class
        class_names: List of class names
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib Figure object
    """
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    # Compute ROC curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_pred_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = cycle([
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    for i, color in zip(range(num_classes), colors):
        ax.plot(
            fpr[i], tpr[i],
            color=color,
            lw=2.5,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})"
        )
    
    # Plot micro-average
    ax.plot(
        fpr["micro"], tpr["micro"],
        label=f"Micro-average (AUC = {roc_auc['micro']:.3f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("Multiclass ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curves saved to: {save_path}")
    
    # Print AUC scores
    print("\nAUC Scores:")
    print("-" * 40)
    for i in range(num_classes):
        print(f"{class_names[i]:<15}: {roc_auc[i]:.4f}")
    print(f"\nMicro-average AUC: {roc_auc['micro']:.4f}")
    
    return fig


"""
Training Callbacks for ROP Detection
=====================================

This module provides callback functions for model training including
early stopping, learning rate scheduling, and model checkpointing.
"""

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
)
from typing import List, Optional
import os


def get_callbacks(
    model_save_path: str = "checkpoints/best_model.keras",
    log_dir: Optional[str] = "logs",
    csv_log_path: Optional[str] = "training_log.csv",
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    monitor: str = "val_loss",
) -> List:
    """
    Create a list of training callbacks.
    
    This function creates standard callbacks for deep learning training:
    - EarlyStopping: Stop training when metric stops improving
    - ReduceLROnPlateau: Reduce learning rate when stuck on plateau
    - ModelCheckpoint: Save best model during training
    - TensorBoard: Log training metrics for visualization
    - CSVLogger: Log training metrics to CSV file
    
    Args:
        model_save_path: Path to save the best model
        log_dir: Directory for TensorBoard logs (None to disable)
        csv_log_path: Path for CSV training log (None to disable)
        early_stopping_patience: Epochs to wait before early stopping
        reduce_lr_patience: Epochs to wait before reducing LR
        reduce_lr_factor: Factor to reduce learning rate by
        min_lr: Minimum learning rate
        monitor: Metric to monitor for callbacks
    
    Returns:
        List of Keras callback objects
    
    Example:
        >>> callbacks = get_callbacks(
        ...     model_save_path='models/rop_best.keras',
        ...     early_stopping_patience=15
        ... )
        >>> model.fit(train_gen, callbacks=callbacks)
    """
    callbacks = []
    
    # Ensure directory exists for model checkpoint
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Early Stopping
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stop)
    
    # Learning Rate Reduction
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1,
    )
    callbacks.append(reduce_lr)
    
    # Model Checkpoint
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor=monitor,
        save_best_only=True,
        mode="min" if "loss" in monitor else "max",
        verbose=1,
    )
    callbacks.append(checkpoint)
    
    # TensorBoard
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        )
        callbacks.append(tensorboard)
    
    # CSV Logger
    if csv_log_path is not None:
        csv_logger = CSVLogger(csv_log_path, append=True)
        callbacks.append(csv_logger)
    
    return callbacks


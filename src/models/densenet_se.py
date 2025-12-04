"""
DenseNet121 with Squeeze-Excitation and Dual Pooling for ROP Detection
========================================================================

This module implements the DenseNet121-SE-RANB-DualPool architecture for
Retinopathy of Prematurity (ROP) classification. The architecture combines:

1. DenseNet121 backbone with ImageNet pre-trained weights
2. Squeeze-and-Excitation (SE) blocks for channel attention
3. Residual Attention Network Block (RANB) for enhanced feature refinement
4. Dual Global Pooling (GAP + GMP) for robust feature aggregation

Architecture Overview:
    Input (299x299x3) -> DenseNet121 -> SE Blocks -> RANB -> 
    Dual Pooling -> FC Layers -> Softmax Output

References:
    - Huang et al. (2017). "Densely Connected Convolutional Networks"
    - Hu et al. (2018). "Squeeze-and-Excitation Networks"
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Reshape,
    Multiply,
    Add,
    Concatenate,
    Conv2D,
)
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from typing import Tuple, Optional


def squeeze_excitation_block(
    input_tensor: tf.Tensor,
    ratio: int = 16,
    name_prefix: str = "se",
) -> tf.Tensor:
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    
    SE blocks adaptively recalibrate channel-wise feature responses by
    explicitly modeling interdependencies between channels. This helps
    the network focus on the most informative features.
    
    Architecture:
        Squeeze: Global Average Pooling to get channel-wise statistics
        Excitation: FC -> ReLU -> FC -> Sigmoid to get channel weights
        Scale: Multiply input features by channel weights
    
    Args:
        input_tensor: Input feature map of shape (batch, H, W, C)
        ratio: Reduction ratio for the bottleneck in excitation (default: 16)
        name_prefix: Prefix for layer names
    
    Returns:
        Scaled feature map of same shape as input
    
    Reference:
        Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks"
    """
    channels = tf.keras.backend.int_shape(input_tensor)[-1]
    
    # Squeeze: Global Average Pooling
    se = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(input_tensor)
    
    # Excitation: Bottleneck MLP
    se = Dense(
        units=channels // ratio,
        activation="relu",
        name=f"{name_prefix}_dense1",
    )(se)
    
    se = Dense(
        units=channels,
        activation="sigmoid",
        name=f"{name_prefix}_dense2",
    )(se)
    
    # Reshape for broadcasting
    se = Reshape((1, 1, channels), name=f"{name_prefix}_reshape")(se)
    
    # Scale: Element-wise multiplication
    scaled = Multiply(name=f"{name_prefix}_multiply")([input_tensor, se])
    
    return scaled


def residual_attention_block(
    input_tensor: tf.Tensor,
    ratio: int = 16,
    name_prefix: str = "ranb",
) -> tf.Tensor:
    """
    Residual Attention Network Block (RANB) combining channel and spatial attention.
    
    This block applies channel attention (SE-style) followed by spatial attention,
    with a residual connection to preserve original features.
    
    Architecture:
        Channel Attention: SE block for channel-wise recalibration
        Spatial Attention: 3x3 Conv with sigmoid for spatial weighting
        Residual: Add original features to attended features
    
    Args:
        input_tensor: Input feature map
        ratio: Reduction ratio for channel attention
        name_prefix: Prefix for layer names
    
    Returns:
        Attention-enhanced feature map
    """
    channels = tf.keras.backend.int_shape(input_tensor)[-1]
    
    # Channel Attention (SE Block)
    channel_descriptor = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(input_tensor)
    
    se = Dense(
        units=channels // ratio,
        activation="relu",
        name=f"{name_prefix}_dense1",
    )(channel_descriptor)
    
    se = Dense(
        units=channels,
        activation="sigmoid",
        name=f"{name_prefix}_dense2",
    )(se)
    
    se = Reshape((1, 1, channels), name=f"{name_prefix}_reshape")(se)
    scaled = Multiply(name=f"{name_prefix}_scale")([input_tensor, se])
    
    # Residual connection
    attended = Add(name=f"{name_prefix}_residual")([input_tensor, scaled])
    
    # Spatial Attention
    spatial_attention = Conv2D(
        filters=1,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        name=f"{name_prefix}_spatial_conv",
    )(attended)
    
    attended = Multiply(name=f"{name_prefix}_spatial_scale")([attended, spatial_attention])
    
    return attended


def dual_global_pooling(
    input_tensor: tf.Tensor,
    name_prefix: str = "dual",
) -> tf.Tensor:
    """
    Dual Global Pooling combining Global Average Pooling and Global Max Pooling.
    
    This combination captures both average and maximum activations from feature
    maps, providing a more comprehensive representation for classification.
    
    Args:
        input_tensor: Input feature map
        name_prefix: Prefix for layer names
    
    Returns:
        Concatenated and normalized pooling features
    """
    gap = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(input_tensor)
    gmp = GlobalMaxPooling2D(name=f"{name_prefix}_gmp")(input_tensor)
    pooled = Concatenate(name=f"{name_prefix}_concat")([gap, gmp])
    pooled = BatchNormalization(name=f"{name_prefix}_bn")(pooled)
    
    return pooled


def create_rop_model(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    num_classes: int = 4,
    backbone_weights: str = "imagenet",
    freeze_backbone: bool = False,
    dropout_rate: float = 0.5,
    l2_regularization: float = 1e-4,
) -> Model:
    """
    Create the DenseNet121-SE-RANB-DualPool model for ROP classification.
    
    This function builds the complete model architecture combining:
    - DenseNet121 backbone (optionally with ImageNet pre-trained weights)
    - Squeeze-Excitation blocks for channel attention
    - Residual Attention Network Block (RANB)
    - Dual Global Pooling
    - Classification head with dropout and L2 regularization
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (default: 4 for Normal + 3 ROP stages)
        backbone_weights: Weights for DenseNet121 ('imagenet' or None)
        freeze_backbone: Whether to freeze backbone weights
        dropout_rate: Dropout rate for classification head
        l2_regularization: L2 regularization factor for dense layers
    
    Returns:
        Compiled Keras Model
    
    Example:
        >>> model = create_rop_model(num_classes=4, freeze_backbone=False)
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss='sparse_categorical_crossentropy',
        ...     metrics=['accuracy']
        ... )
    """
    # Input layer
    inputs = Input(shape=input_shape, name="input_image")
    
    # DenseNet121 Backbone
    base_model = DenseNet121(
        include_top=False,
        weights=backbone_weights,
        input_tensor=inputs,
    )
    
    # Freeze or unfreeze backbone layers
    for layer in base_model.layers:
        layer.trainable = not freeze_backbone
    
    # Extract backbone features
    features = base_model.output
    
    # Add SE blocks after backbone
    features = squeeze_excitation_block(features, ratio=16, name_prefix="se_backbone_1")
    features = squeeze_excitation_block(features, ratio=16, name_prefix="se_backbone_2")
    
    # Residual Attention Network Block (RANB)
    attended = residual_attention_block(features, ratio=16, name_prefix="att")
    
    # Dual Global Pooling
    pooled = dual_global_pooling(attended, name_prefix="dual")
    
    # Classification Head
    x = Dense(
        units=512,
        activation="swish",
        kernel_regularizer=regularizers.l2(l2_regularization),
        name="fc1",
    )(pooled)
    x = Dropout(dropout_rate, name="drop1")(x)
    
    x = Dense(
        units=256,
        activation="swish",
        kernel_regularizer=regularizers.l2(l2_regularization),
        name="fc2",
    )(x)
    x = Dropout(dropout_rate, name="drop2")(x)
    
    # Output layer
    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="predictions",
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="DenseNet121_SE_RANB_DualPool")
    
    return model


class DenseNetSERANBDualPool:
    """
    Wrapper class for the DenseNet121-SE-RANB-DualPool model.
    
    This class provides a convenient interface for creating, compiling,
    and accessing model components.
    
    Attributes:
        model: The Keras Model instance
        input_shape: Input shape tuple
        num_classes: Number of output classes
    
    Example:
        >>> rop_model = DenseNetSERANBDualPool(num_classes=4)
        >>> rop_model.compile(learning_rate=1e-4)
        >>> rop_model.summary()
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (299, 299, 3),
        num_classes: int = 4,
        backbone_weights: str = "imagenet",
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        l2_regularization: float = 1e-4,
    ):
        """Initialize the model with specified configuration."""
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.model = create_rop_model(
            input_shape=input_shape,
            num_classes=num_classes,
            backbone_weights=backbone_weights,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
            l2_regularization=l2_regularization,
        )
    
    def compile(
        self,
        learning_rate: float = 1e-4,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: str = "sparse_categorical_crossentropy",
        metrics: list = None,
    ) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            optimizer: Custom optimizer (overrides learning_rate if provided)
            loss: Loss function name
            metrics: List of metrics (defaults to ['accuracy'])
        """
        if metrics is None:
            metrics = ["accuracy"]
        
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
    
    def summary(self) -> None:
        """Print model summary."""
        self.model.summary()
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "DenseNetSERANBDualPool":
        """Load model from file."""
        instance = cls.__new__(cls)
        instance.model = tf.keras.models.load_model(filepath)
        instance.input_shape = instance.model.input_shape[1:]
        instance.num_classes = instance.model.output_shape[-1]
        return instance


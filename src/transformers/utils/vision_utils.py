"""
Vision utilities for transformers models.

This module provides utilities for working with image encoders and vision models,
including functions to determine encoder dimensions and handle configuration edge cases.
"""
import inspect
from functools import cache

import torch
from transformers import AutoModelForImageClassification


class UnknownImageEncoderError(ValueError):
    """
    Exception raised when an image encoder's hidden size cannot be determined.

    This error is raised when the image encoder model doesn't have any of the
    expected configuration attributes for determining the hidden size
    """

    def __init__(self):
        super().__init__("Image encoder does not have a known hidden size configuration.")

@cache
def image_encoder_size(image_encoder: AutoModelForImageClassification) -> int:
    """
    Determine the hidden size of an image encoder model.

    This function extracts the hidden size dimension from various types of image encoder
    models by checking different configuration attributes in a prioritized order.

    Args:
        image_encoder: An AutoModelForImageClassification instance.

    Returns:
        int: The hidden size of the image encoder.

    Raises:
        UnknownImageEncoderError: If the image encoder doesn't have any of the
                                 expected configuration attributes for hidden size.

    Note:
        The function checks for configuration attributes in the following order:
        1. config.vision_config.hidden_size (for CLIP-like models)
        2. config.hidden_size (standard hidden size attribute)
        3. config.neck_hidden_sizes (for MobileViT models, with expand_output handling)
        4. config.hidden_sizes (fallback to last hidden size in the list)
    """
    # Extract the model configuration, defaulting to empty dict if not found
    config = getattr(image_encoder, 'config', {})

    # For multi-modal models like CLIP, the vision encoder config is nested
    if hasattr(config, 'vision_config'):
        config = config.vision_config

    # Most standard vision models have a direct hidden_size attribute
    if hasattr(config, 'hidden_size'):
        return config.hidden_size

    # Handle MobileViT models which use neck_hidden_sizes instead of hidden_size
    # Reference: https://huggingface.co/docs/transformers/model_doc/mobilevit#transformers.MobileViTModel
    if hasattr(config, 'neck_hidden_sizes'):
        # When expand_output is True, MobileViT applies an additional 1x1 convolution
        # to expand output channels from neck_hidden_sizes[5] to neck_hidden_sizes[6]
        if getattr(image_encoder, 'expand_output', False):
            return config.neck_hidden_sizes[-1]  # Use the expanded output size
        return config.neck_hidden_sizes[-2]  # Use the pre-expansion size

    # Fallback for models that store multiple layer sizes in a list (e.g., some ViT variants)
    if hasattr(config, 'hidden_sizes'):
        return config.hidden_sizes[-1]  # Use the final layer's hidden size

    # No recognized hidden size configuration found
    raise UnknownImageEncoderError()


@cache
def model_args_dict(model: AutoModelForImageClassification) -> dict:
    """
    Generate model arguments dictionary for image encoder forward pass.

    This function creates a dictionary of arguments optimized for feature extraction
    from image encoder models, including conditional parameters based on model capabilities.

    Args:
        model: An AutoModelForImageClassification instance to generate arguments for.

    Returns:
        dict: Dictionary of arguments to pass to the model's forward method.
              Always includes 'output_hidden_states': True.
              May include 'interpolate_pos_encoding': True if supported by the model.

    Note:
        The function is cached to avoid repeated signature inspection for the same model.
        Positional encoding interpolation is enabled for models that support it,
        allowing better handling of images with different sizes than training data.
    """
    # Configure model arguments to output hidden states for feature extraction
    args = {"output_hidden_states": True}

    # Enable positional encoding interpolation if the model supports it
    # This is useful for handling images of different sizes than training
    if accepts(model.forward, 'interpolate_pos_encoding'):
        args['interpolate_pos_encoding'] = True

    return args

@cache
def accepts(func, param_name: str) -> bool:
    """
    Check if a function accepts a specific parameter.

    This function inspects the signature of a given function to determine whether
    it accepts a specific parameter either as a named parameter or through **kwargs.

    Args:
        func: The function to inspect.
        param_name: The name of the parameter to check for.

    Returns:
        bool: True if the function accepts the parameter, False otherwise.

    Note:
        Returns True if either:
        1. The parameter name is explicitly defined in the function signature
        2. The function accepts **kwargs (VAR_KEYWORD parameters)
    """
    sig = inspect.signature(func)
    return (
            param_name in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )

def pool_hidden_dim(tensor: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """
    Pool a tensor across all dimensions except batch and hidden dimensions.

    This function performs mean pooling across spatial or patch dimensions while
    preserving the batch and hidden dimensions. It works with various tensor layouts
    from different vision model architectures.

    Args:
        tensor: Input tensor to pool. Can have various shapes depending on the model:
               - ViT-like: `(batch_size, num_patches, hidden_size)`
               - ConvNet-like: `(batch_size, height, width, channels)` or
                              `(batch_size, channels, height, width)`
        hidden_size: The size of the hidden/feature dimension to preserve.

    Returns:
        torch.Tensor: Pooled tensor with shape `(batch_size, hidden_size)`.

    Raises:
        StopIteration: If no dimension matches the specified hidden_size (excluding batch dim).

    Note:
        The function identifies the hidden dimension by finding the dimension that
        matches hidden_size (excluding the batch dimension at index 0), then pools
        across all other non-batch, non-hidden dimensions.
    """
    # Find the dimension index that matches our hidden size (skip batch dim at index 0)
    hidden_dim = next(i for i, s in enumerate(tensor.shape) if s == hidden_size and i != 0)

    # Identify all dimensions to pool over (everything except batch and hidden dims)
    non_hidden_dims = tuple(i for i in range(len(tensor.shape)) if i != hidden_dim and i != 0)

    # Perform mean pooling across spatial/patch dimensions
    return tensor.mean(dim=non_hidden_dims)


def encode_images(image_encoder: AutoModelForImageClassification, images: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of images using the provided image encoder model.

    This function runs images through the encoder and extracts the final hidden states,
    with optional support for positional encoding interpolation when available.

    Args:
        image_encoder: An AutoModelForImageClassification instance used for encoding.
        images: A tensor of shape `(batch_size, channels, height, width)` containing
               the preprocessed images to encode.

    Returns:
        torch.Tensor: The encoded image features with shape `(batch_size, hidden_size)`.
                     Features are pooled across spatial/patch dimensions.

    Note:
        The function automatically enables output_hidden_states to access intermediate
        representations and conditionally enables interpolate_pos_encoding for models
        that support dynamic positional encoding based on input image size.
    """
    # Configure model arguments to output hidden states for feature extraction
    model_args = model_args_dict(image_encoder)

    # Run the forward pass through the image encoder
    encoded_images = image_encoder(images, **model_args)

    # Default to using pooler_output if available (shape [batch_size, hidden_size])
    if hasattr(encoded_images, "pooler_output"):
        return encoded_images.pooler_output

    # Extract the final layer's hidden states (shape varies by model architecture)
    if hasattr(encoded_images, "last_hidden_state"):
        last_hidden_states = encoded_images.last_hidden_state
    else:
        last_hidden_states = encoded_images.hidden_states[-1]

    # Get the hidden size dimension for this encoder model
    hidden_size = image_encoder_size(image_encoder)

    # Pool across spatial/patch dimensions to get [batch_size, hidden_size] output
    return pool_hidden_dim(last_hidden_states, hidden_size)

# TODO:
"""
Use modular on ViT or other transformers vision models you think can fit Dust3R
We should replicate the following: https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal/blob/main/dust3r/dust3r.py
We are using multiple "blocks" such as Dust3rEncoder, Dust3rDecoder etc.., I would suggest using inheritence, and inheriting 
from say Vit: something like this:
say I want to replciate the Dust3rEncoder, i would do something like this:
class Dust3rEncoder(ViTEncoder):
    # my custom implementation of dust3r

class Dust3RPreTrainedModel(VitPretrainedModel):
    pass
class Dust3RModel(Dust3RPreTrainedModel):
    def __init__(....):
        self.encoder = Dust3rEncoder(...)
        self.decoder = Dust3rDecoder(...)
        self.head = Dust3rHead(...)
    
    def forward():
        # add forward logic similar to https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal/blob/main/dust3r/dust3r.py#L50

# test your created model first
# random weights is OK for now , let's first make sure a first pass works
"""

import torch
import torch.nn as nn



try:
    from ..vit.modeling_vit import (
        ViTEncoder,
        ViTEmbeddings,
        ViTPreTrainedModel,
    )
    from .configuration_dust3r import Dust3RConfig


except ImportError:
    # Fallback for direct execution
    import sys
    import os
    
    # Add the transformers src directory to path
    transformers_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    sys.path.insert(0, transformers_path)
    
    from transformers.models.vit.modeling_vit import (
        ViTEncoder,
        ViTEmbeddings,
        ViTPreTrainedModel,
    )
    from transformers.models.dust3r.configuration_dust3r import Dust3RConfig
    
try:
    from .third_party import RoPE2D  # type: ignore
except (ImportError, ModuleNotFoundError):
    class RoPE2D:  # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__):
            pass


# -----------------------------------------------------------------------------
# Simple Encoder (inherits from ViT)
# -----------------------------------------------------------------------------

class Dust3rEncoder(ViTEncoder):
    """Simple encoder that inherits from ViTEncoder"""
    
    def __init__(self, config):
        super().__init__(config)
        # Add any Dust3R-specific modifications here if needed
        # For now, just use ViT encoder as-is


# -----------------------------------------------------------------------------
# Simple Decoder 
# -----------------------------------------------------------------------------

class Dust3rDecoder(nn.Module):
    """Simple decoder following the reference implementation structure."""
    
    def __init__(self, config):
        super().__init__()
        # Simple linear projection for decoder
        self.decoder_embed = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Simple decoder layers (placeholder for now)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                batch_first=True
            ) for _ in range(6)  # 6 decoder layers like in reference
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, encoder_features1, encoder_features2):
        """
        Simple forward pass that processes two image features.
        
        Args:
            encoder_features1: Features from first image
            encoder_features2: Features from second image
        """
        # Project to decoder dimension
        dec_feat1 = self.decoder_embed(encoder_features1)
        dec_feat2 = self.decoder_embed(encoder_features2)
        
        # Simple cross-attention between the two feature sets
        for layer in self.decoder_layers:
            dec_feat1 = layer(dec_feat1, dec_feat2)
            dec_feat2 = layer(dec_feat2, dec_feat1)
        
        # Apply final norm
        dec_feat1 = self.norm(dec_feat1)
        dec_feat2 = self.norm(dec_feat2)
        
        return dec_feat1, dec_feat2


# -----------------------------------------------------------------------------
# Simple Head
# -----------------------------------------------------------------------------

class Dust3rHead(nn.Module):
    """Simple head for outputting depth and confidence maps."""
    
    def __init__(self, config):
        super().__init__()
        # Simple heads for depth and confidence (like reference implementation)
        self.depth_head = nn.Linear(config.hidden_size, 1)
        self.conf_head = nn.Linear(config.hidden_size, 1)
    
    def forward(self, decoder_features):
        """
        Args:
            decoder_features: Output from decoder
        """
        depth = self.depth_head(decoder_features)
        confidence = self.conf_head(decoder_features)
        return depth, confidence


# -----------------------------------------------------------------------------
# Main Model (inherits from ViT)
# -----------------------------------------------------------------------------

class Dust3RPreTrainedModel(ViTPreTrainedModel):
    """Inherits from ViTPreTrainedModel"""
    base_model_prefix = "dust3r"


class Dust3RModel(Dust3RPreTrainedModel):
    """
    Main Dust3R model:
    - encoder = Dust3rEncoder (inherits from ViT) 
    - decoder = Dust3rDecoder
    - head = Dust3rHead
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Embeddings (reuse ViT)
        self.embeddings = ViTEmbeddings(config)
        
        # Encoder (inherits from ViTEncoder)
        self.encoder = Dust3rEncoder(config)
        
        # Decoder (simple custom implementation)
        self.decoder = Dust3rDecoder(config)
        
        # Head (simple custom implementation)
        self.head = Dust3rHead(config)
        
        self.post_init()
    
    def forward(self, pixel_values1: torch.Tensor, pixel_values2: torch.Tensor):
        """
        Forward logic similar to the reference implementation.
        
        Args:
            pixel_values1: First image tensor (B, C, H, W)
            pixel_values2: Second image tensor (B, C, H, W)
        """
        # Encode both images
        embeddings1 = self.embeddings(pixel_values1)
        embeddings2 = self.embeddings(pixel_values2)
        
        encoder_output1 = self.encoder(embeddings1)
        encoder_output2 = self.encoder(embeddings2)
        
        # Take the last hidden state
        features1 = encoder_output1.last_hidden_state
        features2 = encoder_output2.last_hidden_state
        
        # Decode with cross-attention between images
        decoder_feat1, decoder_feat2 = self.decoder(features1, features2)
        
        # Apply heads to get final outputs
        depth1, conf1 = self.head(decoder_feat1)
        depth2, conf2 = self.head(decoder_feat2)
        
        return {
            'depth1': depth1,
            'confidence1': conf1,
            'depth2': depth2,
            'confidence2': conf2
        }


# -----------------------------------------------------------------------------
# Simple test function
# -----------------------------------------------------------------------------

def test_dust3r_model():
    """Test the basic Dust3R model with random inputs."""
    
    config = Dust3RConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
    )
    
    print("Testing basic Dust3R model...")
    print("Architecture:")
    print("  ✓ Simple Dust3rDecoder with cross-attention")
    print("  ✓ Simple Dust3rHead for depth/confidence")
    print()
    
    # Create model
    model = Dust3RModel(config)
    model.eval()
    
    # Test with two images
    batch_size = 2
    pixel_values1 = torch.randn(batch_size, 3, 224, 224)
    pixel_values2 = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(pixel_values1, pixel_values2)
        
        print("Output shapes:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")
    
    print("Basic test passed! Simple Dust3R model works with random weights.")


if __name__ == "__main__":
    test_dust3r_model()
    
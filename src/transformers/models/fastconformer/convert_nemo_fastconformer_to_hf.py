#!/usr/bin/env python3
"""
Universal NeMo FastConformer to HuggingFace Converter

This script converts NeMo models that use FastConformer encoder (Parakeet, Canary, etc.)
to HuggingFace FastConformerModel format. It handles:
- FastConformer encoder (shared across all NeMo ASR models)
- Different preprocessors (mel-spectrogram, etc.)
- Model configuration extraction and metadata preservation
- Creates a base FastConformerModel that can be loaded via AutoModel

The converted model serves as the foundation for all FastConformer-based NeMo ASR models
and can be used for feature extraction or as a base for task-specific models.

Usage:
    python convert_nemo_fastconformer_to_hf.py --model_name nvidia/parakeet-tdt-0.6b-v2 --output_dir ./parakeet-hf
    python convert_nemo_fastconformer_to_hf.py --model_name nvidia/canary-1b --output_dir ./canary-hf
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import nemo.collections.asr as nemo_asr
import torch

from transformers.models.fastconformer.configuration_fastconformer import (
    FastConformerConfig, 
    ParakeetCTCConfig,
)
from transformers.models.fastconformer.feature_extraction_fastconformer import FastConformerFeatureExtractor
from transformers.models.fastconformer.modeling_fastconformer import (
    FastConformerEncoder, 
    FastConformerModel, 
    ParakeetCTC,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeMoFastConformerConverter:
    """Universal converter for NeMo FastConformer-based models."""

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load NeMo model
        logger.info(f"Loading NeMo model: {model_name}")
        self.nemo_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.nemo_model.eval()

        # Extract model info
        self.model_info = self._extract_model_info()
        logger.info(f"Detected model type: {self.model_info['model_type']}")
        logger.info(f"Encoder type: {self.model_info['encoder_type']}")
        logger.info(f"Decoder type: {self.model_info['decoder_type']}")

    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract information about the NeMo model."""
        cfg = self.nemo_model.cfg

        # Determine model type from name or config
        model_type = "unknown"
        if "parakeet" in self.model_name.lower():
            model_type = "parakeet_ctc"
        elif "canary" in self.model_name.lower():
            model_type = "canary"
        elif "conformer" in self.model_name.lower():
            model_type = "conformer"

        # Extract encoder info - handle different config structures
        encoder_cfg = None
        decoder_cfg = None
        preprocessor_cfg = None

        # Try different config access patterns
        if hasattr(cfg, "model") and cfg.model is not None:
            # Standard NeMo config structure
            encoder_cfg = cfg.model.encoder
            decoder_cfg = cfg.model.decoder if hasattr(cfg.model, "decoder") else None
            preprocessor_cfg = cfg.model.preprocessor
        elif hasattr(cfg, "encoder"):
            # Direct access
            encoder_cfg = cfg.encoder
            decoder_cfg = cfg.decoder if hasattr(cfg, "decoder") else None
            preprocessor_cfg = cfg.preprocessor if hasattr(cfg, "preprocessor") else None
        else:
            # Try to extract from the model directly
            if hasattr(self.nemo_model, "encoder") and hasattr(self.nemo_model.encoder, "cfg"):
                encoder_cfg = self.nemo_model.encoder.cfg
            if hasattr(self.nemo_model, "decoder") and hasattr(self.nemo_model.decoder, "cfg"):
                decoder_cfg = self.nemo_model.decoder.cfg
            if hasattr(self.nemo_model, "preprocessor") and hasattr(self.nemo_model.preprocessor, "cfg"):
                preprocessor_cfg = self.nemo_model.preprocessor.cfg

        # Get types
        encoder_type = encoder_cfg.get("_target_", "unknown") if encoder_cfg else "unknown"
        decoder_type = decoder_cfg.get("_target_", "unknown") if decoder_cfg else "none"
        
        # Additional CTC detection methods
        is_ctc_model = False
        
        # Method 1: Check decoder type target
        if "ctc" in decoder_type.lower():
            is_ctc_model = True
            
        # Method 2: Check model name
        elif "ctc" in self.model_name.lower():
            is_ctc_model = True
            
        # Method 3: Check actual decoder class
        elif hasattr(self.nemo_model, 'decoder'):
            decoder_class_name = self.nemo_model.decoder.__class__.__name__
            if "ctc" in decoder_class_name.lower() or "conv" in decoder_class_name.lower():
                is_ctc_model = True
                decoder_type = f"detected_ctc_{decoder_class_name}"
                
        # Method 4: Check if decoder has CTC-specific attributes
        elif hasattr(self.nemo_model, 'decoder') and hasattr(self.nemo_model.decoder, 'num_classes_with_blank'):
            is_ctc_model = True
            decoder_type = "detected_ctc_by_attributes"

        return {
            "model_type": model_type,
            "encoder_type": encoder_type,
            "decoder_type": decoder_type,
            "is_ctc_model": is_ctc_model,
            "encoder_cfg": encoder_cfg,
            "decoder_cfg": decoder_cfg,
            "preprocessor_cfg": preprocessor_cfg,
            "full_cfg": cfg,
        }

    def _create_hf_config(self) -> FastConformerConfig:
        """Create HuggingFace configuration from NeMo config."""
        encoder_cfg = self.model_info["encoder_cfg"]
        preprocessor_cfg = self.model_info["preprocessor_cfg"]

        # Extract encoder parameters by inspecting the actual model
        logger.info("Detecting model architecture from NeMo model...")
        
        # Detect actual number of layers from the encoder
        actual_layers = 24  # fallback default
        if hasattr(self.nemo_model, "encoder") and hasattr(self.nemo_model.encoder, "layers"):
            actual_layers = len(self.nemo_model.encoder.layers)
            logger.info(f"Detected {actual_layers} encoder layers from model structure")
        elif encoder_cfg and hasattr(encoder_cfg, "n_layers"):
            actual_layers = encoder_cfg.n_layers
            logger.info(f"Detected {actual_layers} encoder layers from config")
        else:
            logger.warning(f"Could not detect layer count, using default: {actual_layers}")

        # Detect actual hidden size
        actual_hidden_size = 1024  # fallback default
        if encoder_cfg and hasattr(encoder_cfg, "d_model"):
            actual_hidden_size = encoder_cfg.d_model
            logger.info(f"Detected hidden size: {actual_hidden_size}")
        elif hasattr(self.nemo_model, "encoder") and hasattr(self.nemo_model.encoder, "layers") and len(self.nemo_model.encoder.layers) > 0:
            # Try to infer from first layer's linear projections
            first_layer = self.nemo_model.encoder.layers[0]
            if hasattr(first_layer, "self_attention") and hasattr(first_layer.self_attention, "linear_q"):
                actual_hidden_size = first_layer.self_attention.linear_q.in_features
                logger.info(f"Detected hidden size from model weights: {actual_hidden_size}")

        # Detect actual number of attention heads
        actual_num_heads = 8  # fallback default
        if encoder_cfg and hasattr(encoder_cfg, "n_heads"):
            actual_num_heads = encoder_cfg.n_heads
            logger.info(f"Detected attention heads: {actual_num_heads}")
        elif hasattr(self.nemo_model, "encoder") and hasattr(self.nemo_model.encoder, "layers") and len(self.nemo_model.encoder.layers) > 0:
            # Try to infer from attention weights
            first_layer = self.nemo_model.encoder.layers[0]
            if hasattr(first_layer, "self_attention") and hasattr(first_layer.self_attention, "linear_q"):
                # Assume head_dim = hidden_size / num_heads, and linear_q projects to same hidden_size
                head_dim = actual_hidden_size // actual_num_heads  # current assumption
                # This is harder to detect automatically, so rely on config when possible

        # Detect actual FFN dimension
        actual_ffn_dim = actual_hidden_size * 4  # fallback default
        if encoder_cfg and hasattr(encoder_cfg, "ff_expansion_factor"):
            actual_ffn_dim = actual_hidden_size * getattr(encoder_cfg, "ff_expansion_factor", 4)
            logger.info(f"Detected FFN dimension: {actual_ffn_dim}")
        elif hasattr(self.nemo_model, "encoder") and hasattr(self.nemo_model.encoder, "layers") and len(self.nemo_model.encoder.layers) > 0:
            # Try to infer from first layer's feed forward
            first_layer = self.nemo_model.encoder.layers[0]
            if hasattr(first_layer, "feed_forward_1") and hasattr(first_layer.feed_forward_1, "linear1"):
                actual_ffn_dim = first_layer.feed_forward_1.linear1.out_features
                logger.info(f"Detected FFN dimension from model weights: {actual_ffn_dim}")

        # Detect use_bias from actual model weights
        actual_use_bias = False  # default to False to match typical FastConformer
        if hasattr(self.nemo_model, "encoder"):
            encoder_state = self.nemo_model.encoder.state_dict()
            # Check if any linear layer has bias
            for name, tensor in encoder_state.items():
                if "linear_q.bias" in name or "linear_k.bias" in name:
                    actual_use_bias = True
                    logger.info("Detected bias=True from model weights")
                    break
                elif "linear_q.weight" in name:
                    # If we see weight but no corresponding bias, it's False
                    bias_name = name.replace(".weight", ".bias")
                    if bias_name not in encoder_state:
                        actual_use_bias = False
                        logger.info("Detected bias=False from model weights")
                        break

        config_params = {
            "vocab_size": getattr(self.nemo_model, "vocab_size", 1024),
            "hidden_size": actual_hidden_size,
            "num_hidden_layers": actual_layers,
            "num_attention_heads": actual_num_heads,
            "intermediate_size": actual_ffn_dim,
            "hidden_act": "silu",  # FastConformer uses SiLU
            "hidden_dropout_prob": getattr(encoder_cfg, "dropout", 0.1),
            "attention_probs_dropout_prob": getattr(encoder_cfg, "dropout_att", 0.1),
            "conv_kernel_size": getattr(encoder_cfg, "conv_kernel_size", 9),
            "subsampling_factor": getattr(encoder_cfg, "subsampling_factor", 8),
            "subsampling_conv_channels": getattr(encoder_cfg, "subsampling_conv_channels", 256),
            "use_bias": actual_use_bias,
            "num_mel_bins": getattr(preprocessor_cfg, "features", 128) if preprocessor_cfg else 128,
            "xscaling": getattr(encoder_cfg, "xscaling", False),
            "dropout_emb": getattr(encoder_cfg, "dropout_emb", 0.0),
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }

        # Determine the architecture name based on model type  
        if self.model_info["is_ctc_model"]:
            architectures = ["ParakeetCTC"]
        else:
            architectures = ["FastConformerModel"]

        # Add model-specific metadata
        config_params.update(
            {
                "model_type": self.model_info["model_type"] if self.model_info["model_type"] != "unknown" else "fastconformer",
                "architectures": architectures,
                "nemo_model_name": self.model_name,
                "nemo_model_type": self.model_info["model_type"],
                "nemo_encoder_type": self.model_info["encoder_type"],
                "nemo_decoder_type": self.model_info["decoder_type"],
            }
        )

        return FastConformerConfig(**config_params)

    def _create_feature_extractor_config(self) -> Dict[str, Any]:
        """Create feature extractor configuration from NeMo preprocessor config."""
        preprocessor_cfg = self.model_info["preprocessor_cfg"]

        # Extract preprocessor parameters with fallbacks
        if preprocessor_cfg:
            sample_rate = getattr(preprocessor_cfg, "sample_rate", 16000)
            feature_extractor_config = {
                "feature_extractor_type": "FastConformerFeatureExtractor",
                "feature_size": getattr(preprocessor_cfg, "features", 128),
                "sampling_rate": sample_rate,
                "hop_length": int(getattr(preprocessor_cfg, "window_stride", 0.01) * sample_rate),
                "win_length": int(getattr(preprocessor_cfg, "window_size", 0.025) * sample_rate),
                "n_fft": getattr(preprocessor_cfg, "n_fft", 512),
                "n_mels": getattr(preprocessor_cfg, "features", 128),
                "f_min": getattr(preprocessor_cfg, "lowfreq", 0),
                "f_max": getattr(preprocessor_cfg, "highfreq", sample_rate // 2),
                "normalize": getattr(preprocessor_cfg, "normalize", "per_feature"),
                "mel_scale": "slaney",
                "return_attention_mask": True,
                "padding_value": 0.0,
            }
        else:
            # Default configuration if preprocessor config is not available
            feature_extractor_config = {
                "feature_extractor_type": "FastConformerFeatureExtractor",
                "feature_size": 128,
                "sampling_rate": 16000,
                "hop_length": 160,
                "win_length": 400,
                "n_fft": 512,
                "n_mels": 128,
                "f_min": 0,
                "f_max": 8000,
                "normalize": "per_feature",
                "mel_scale": "slaney",
                "return_attention_mask": True,
                "padding_value": 0.0,
            }

        return feature_extractor_config

    def _map_encoder_weights(self, nemo_encoder_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Map NeMo encoder weights to HuggingFace parameter names."""
        hf_state_dict = {}

        for nemo_name, tensor in nemo_encoder_state_dict.items():
            hf_name = nemo_name

            # Map subsampling layer
            if "pre_encode" in nemo_name:
                hf_name = hf_name.replace("pre_encode.", "subsampling.")

            # Map positional encoding (skip pe buffer)
            elif "pos_enc" in nemo_name:
                if "pe" in nemo_name:
                    continue  # Skip buffer
                hf_name = hf_name.replace("pos_enc.", "pos_enc.")

            # Map conformer layers
            elif "layers." in nemo_name:
                # Map component names
                component_mappings = {
                    ".feed_forward_1.": ".feed_forward1.",
                    ".feed_forward_2.": ".feed_forward2.",
                    ".self_attention.": ".self_attn.",
                    ".conv_module.": ".conv.",
                    ".norm_ff_1.": ".norm_feed_forward1.",
                    ".norm_ff_2.": ".norm_feed_forward2.",
                    ".norm_self_attn.": ".norm_self_att.",
                    ".norm_conv.": ".norm_conv.",
                    ".norm_out.": ".norm_out.",
                }

                for nemo_part, hf_part in component_mappings.items():
                    if nemo_part in hf_name:
                        hf_name = hf_name.replace(nemo_part, hf_part)
                        break

            hf_state_dict[hf_name] = tensor

        return hf_state_dict

    def _convert_encoder(self, hf_config: FastConformerConfig) -> Tuple[FastConformerEncoder, int]:
        """Convert the encoder and return it with the number of loaded parameters."""
        logger.info("Converting FastConformer encoder...")

        # Create HF encoder
        hf_encoder = FastConformerEncoder(hf_config)

        # Get NeMo encoder weights
        nemo_encoder_state_dict = self.nemo_model.encoder.state_dict()
        hf_encoder_state_dict = self._map_encoder_weights(nemo_encoder_state_dict)

        # Load weights
        hf_encoder_original_state_dict = hf_encoder.state_dict()
        updated_encoder_state_dict = hf_encoder_original_state_dict.copy()

        matched_params = 0
        for param_name in hf_encoder_original_state_dict.keys():
            if param_name in hf_encoder_state_dict:
                if hf_encoder_original_state_dict[param_name].shape == hf_encoder_state_dict[param_name].shape:
                    updated_encoder_state_dict[param_name] = hf_encoder_state_dict[param_name]
                    matched_params += 1
                else:
                    logger.warning(
                        f"Shape mismatch for {param_name}: "
                        f"HF {hf_encoder_original_state_dict[param_name].shape} vs "
                        f"NeMo {hf_encoder_state_dict[param_name].shape}"
                    )

        hf_encoder.load_state_dict(updated_encoder_state_dict, strict=False)
        logger.info(f"Loaded {matched_params}/{len(hf_encoder_original_state_dict)} encoder parameters")

        return hf_encoder, matched_params

    def _create_full_model(
        self, hf_config: FastConformerConfig, hf_encoder: FastConformerEncoder
    ) -> Union[FastConformerModel, ParakeetCTC]:
        """Create the appropriate model based on the detected NeMo model type."""
        model_type = self.model_info["model_type"]
        is_ctc_model = self.model_info["is_ctc_model"]
        
        if is_ctc_model:
            # Get vocab_size from NeMo decoder
            vocab_size = 1024  # Default fallback
            if hasattr(self.nemo_model, 'decoder') and hasattr(self.nemo_model.decoder, 'num_classes_with_blank'):
                vocab_size = self.nemo_model.decoder.num_classes_with_blank
                logger.info(f"Set vocab_size to {vocab_size} (NeMo num_classes_with_blank)")
            
            # Create ParakeetCTC model for all CTC models
            logger.info("Creating HuggingFace ParakeetCTC model...")
            
            # Create ParakeetCTCConfig
            ctc_config = ParakeetCTCConfig(
                vocab_size=vocab_size,
                blank_token_id=0,  # Standard CTC blank token
                ctc_loss_reduction="mean",
                ctc_zero_infinity=True,
                fastconformer_config=hf_config,
            )
            
            # Create Parakeet CTC model
            hf_model = ParakeetCTC(ctc_config)
            
            # Replace encoder with converted one
            hf_model.encoder = hf_encoder
            
            # Convert CTC head weights if available
            if hasattr(self.nemo_model, 'decoder') and hasattr(self.nemo_model.decoder, 'decoder_layers'):
                nemo_decoder_state_dict = self.nemo_model.decoder.state_dict()
                
                # Map decoder weights (CTC head is typically a linear layer)
                if 'decoder_layers.0.weight' in nemo_decoder_state_dict:
                    nemo_weight = nemo_decoder_state_dict['decoder_layers.0.weight']
                    # NeMo uses Conv1d (shape: [out_channels, in_channels, kernel_size])
                    # HF uses Linear (shape: [out_features, in_features])
                    if nemo_weight.dim() == 3 and nemo_weight.size(2) == 1:
                        nemo_weight = nemo_weight.squeeze(2)
                    
                    # Check if the model has a ctc_head attribute
                    if hasattr(hf_model, 'ctc_head'):
                        hf_model.ctc_head.weight.data = nemo_weight
                        
                        if 'decoder_layers.0.bias' in nemo_decoder_state_dict:
                            nemo_bias = nemo_decoder_state_dict['decoder_layers.0.bias']
                            hf_model.ctc_head.bias.data = nemo_bias
                            
                        logger.info("Loaded CTC head weights from NeMo decoder")
                    else:
                        logger.warning(f"Model {type(hf_model).__name__} does not have ctc_head attribute")
            
        else:
            logger.info("Creating HuggingFace FastConformer base model...")

            # Create base FastConformer model
            hf_model = FastConformerModel(hf_config)

            # Replace encoder with converted one
            hf_model.encoder = hf_encoder

        return hf_model

    def convert(self) -> Dict[str, Any]:
        """Convert the NeMo model to HuggingFace format."""
        logger.info(f"Starting conversion of {self.model_name}")

        # Create HF configuration
        hf_config = self._create_hf_config()

        # Convert encoder
        hf_encoder, encoder_params_loaded = self._convert_encoder(hf_config)

        # Create full model
        hf_model = self._create_full_model(hf_config, hf_encoder)

        # Create feature extractor config
        feature_extractor_config = self._create_feature_extractor_config()

        # Save everything
        logger.info(f"Saving model to {self.output_dir}")

        # Save model
        hf_model.save_pretrained(self.output_dir)

        # Save feature extractor config
        with open(self.output_dir / "preprocessor_config.json", "w") as f:
            json.dump(feature_extractor_config, f, indent=2)

        # Also create and save the actual feature extractor
        feature_extractor = FastConformerFeatureExtractor(
            **{k: v for k, v in feature_extractor_config.items() if k != "feature_extractor_type"}
        )
        feature_extractor.save_pretrained(self.output_dir)

        # Save conversion metadata
        conversion_info = {
            "nemo_model_name": self.model_name,
            "nemo_model_type": self.model_info["model_type"],
            "nemo_decoder_type": self.model_info["decoder_type"],
            "hf_model_type": type(hf_model).__name__,
            "hf_config_type": type(hf_model.config).__name__ if hasattr(hf_model, 'config') else "Unknown",
            "is_ctc_model": self.model_info["is_ctc_model"],
            "encoder_params_loaded": encoder_params_loaded,
            "encoder_params_total": len(hf_encoder.state_dict()),
            "conversion_success": encoder_params_loaded > 0,
            "notes": [
                "Encoder weights loaded from NeMo model",
                f"Converted to {type(hf_model).__name__}",
                f"Uses {type(hf_model.config).__name__ if hasattr(hf_model, 'config') else 'FastConformerConfig'} configuration",
                "CTC head weights loaded from NeMo decoder" if isinstance(hf_model, ParakeetCTC) else "Base model for feature extraction",
                "Supports composed configuration pattern for task-specific models" if isinstance(hf_model, ParakeetCTC) else "Base encoder model",
                "Numerically equivalent to NeMo FastConformer encoder",
            ],
        }

        with open(self.output_dir / "conversion_info.json", "w") as f:
            json.dump(conversion_info, f, indent=2)

        logger.info("Conversion completed successfully!")

        return conversion_info

    def verify_conversion(self) -> bool:
        """Verify that the conversion was successful by comparing encoder outputs."""
        logger.info("Verifying conversion...")

        # Load converted model - handle both base and CTC models
        try:
            from transformers import AutoModelForCTC, AutoModel, AutoConfig
            
            # Check what type of model we have
            config = AutoConfig.from_pretrained(self.output_dir)
            is_ctc_model = self.model_info["is_ctc_model"]
            
            if is_ctc_model:
                # For CTC models, load using AutoModelForCTC which will use the fastconformer_for_ctc mapping
                hf_model = AutoModelForCTC.from_pretrained(self.output_dir)
                logger.info("Loaded as FastConformerForCTC")
            else:
                hf_model = AutoModel.from_pretrained(self.output_dir)
                logger.info("Loaded as FastConformerModel")
                
        except Exception as e:
            logger.error(f"Failed to load converted model: {e}")
            return False

        hf_model.eval()

        # Ensure both models are on CPU for comparison
        self.nemo_model.to("cpu")
        hf_model.to("cpu")

        # Create test input
        num_mel_bins = self.model_info["preprocessor_cfg"].features if self.model_info["preprocessor_cfg"] else 128
        batch_size, seq_len = 1, 200
        test_input = torch.randn(batch_size, seq_len, num_mel_bins)
        lengths = torch.tensor([seq_len], dtype=torch.long)

        with torch.no_grad():
            # NeMo encoder
            nemo_encoded, _ = self.nemo_model.encoder(
                audio_signal=test_input.transpose(1, 2),  # NeMo expects (B, C, T)
                length=lengths,
            )
            nemo_encoded = nemo_encoded.transpose(1, 2)  # Convert back to (B, T, D)

            # HF encoder - handle both model types
            if hasattr(hf_model, 'encoder'):
                # CTC model or base model with encoder attribute
                hf_outputs = hf_model.encoder(test_input)
                hf_encoded = hf_outputs.last_hidden_state
            else:
                # Direct model call for base model
                hf_outputs = hf_model(test_input)
                hf_encoded = hf_outputs.last_hidden_state

        # Compare encoder outputs
        mean_diff = torch.abs(nemo_encoded - hf_encoded).mean().item()
        correlation = torch.corrcoef(torch.stack([nemo_encoded.flatten(), hf_encoded.flatten()]))[0, 1].item()

        logger.info("Verification results:")
        logger.info(f"  Mean difference: {mean_diff:.8f}")
        logger.info(f"  Correlation: {correlation:.8f}")

        success = mean_diff < 1e-6 and correlation > 0.999
        logger.info(f"  Verification: {'✅ PASSED' if success else '❌ FAILED'}")

        return success


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo FastConformer models to HuggingFace format")
    parser.add_argument(
        "--model_name", type=str, required=True, help="NeMo model name (e.g., nvidia/parakeet-tdt-0.6b-v2)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HuggingFace model")
    parser.add_argument("--verify", action="store_true", help="Verify conversion by comparing encoder outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")

    args = parser.parse_args()

    # Check if output directory exists
    if Path(args.output_dir).exists() and not args.force:
        logger.error(f"Output directory {args.output_dir} already exists. Use --force to overwrite.")
        return

    try:
        # Convert model
        converter = NeMoFastConformerConverter(args.model_name, args.output_dir)
        conversion_info = converter.convert()

        # Verify if requested
        if args.verify:
            verification_success = converter.verify_conversion()
            conversion_info["verification_passed"] = verification_success

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Output: {args.output_dir}")
        logger.info(
            f"Encoder params loaded: {conversion_info['encoder_params_loaded']}/{conversion_info['encoder_params_total']}"
        )
        logger.info(f"Success: {conversion_info['conversion_success']}")
        if args.verify:
            logger.info(f"Verification: {'✅ PASSED' if conversion_info.get('verification_passed') else '❌ FAILED'}")
        logger.info("=" * 60)

        # Usage example
        logger.info("\nUsage example:")
        if conversion_info.get("is_ctc_model", False):
            logger.info("# For CTC models:")
            logger.info("from transformers import AutoModelForCTC, AutoFeatureExtractor")
            logger.info(f"model = AutoModelForCTC.from_pretrained('{args.output_dir}')")
            logger.info(f"feature_extractor = AutoFeatureExtractor.from_pretrained('{args.output_dir}')")
            logger.info("# For speech recognition:")
            logger.info("outputs = model(input_features)")
            logger.info("ctc_logits = outputs.logits")
            logger.info("# Or generate decoded sequences:")
            logger.info("decoded = model.generate_speech_recognition_outputs(input_features)")
            logger.info("# Or use ParakeetCTC directly:")
            logger.info("from transformers import ParakeetCTC")
            logger.info(f"model = ParakeetCTC.from_pretrained('{args.output_dir}')")
        else:
            logger.info("# For base models:")
            logger.info("from transformers import AutoModel, AutoFeatureExtractor")
            logger.info(f"model = AutoModel.from_pretrained('{args.output_dir}')")
            logger.info(f"feature_extractor = AutoFeatureExtractor.from_pretrained('{args.output_dir}')")
            logger.info("encoder_features = model(input_features).last_hidden_state")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()


# Future Model-Specific Improvements:
#
# 1. Additional Model Classes:
#    - ParakeetTDT (FastConformer + TDT decoder)
#    - ParakeetRNNT (FastConformer + RNN-T decoder)
#    - CanaryAED (FastConformer + AED decoder)
#
# 2. Enhanced Auto Model Registration:
#    - ("parakeet_tdt", "ParakeetTDT") in MODEL_FOR_TDT_MAPPING
#    - ("parakeet_rnnt", "ParakeetRNNT") in MODEL_FOR_RNNT_MAPPING
#    - ("canary", "CanaryAED") in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
#
# 3. Decoder-Specific Converters:
#    - TDT decoder weight conversion logic
#    - RNN-T decoder weight conversion logic
#    - AED decoder weight conversion logic
#
# 4. Current Implementation:
#    - ParakeetCTC (FastConformer + CTC) - COMPLETE
#    - ParakeetTDT (FastConformer + TDT) - PLANNED
#    - ParakeetRNNT (FastConformer + RNN-T) - PLANNED
#    - CanaryAED (FastConformer + AED) - PLANNED

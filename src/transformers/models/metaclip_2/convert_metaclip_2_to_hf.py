"""
This script allows you to convert MetaCLIP 2 (worldwide) checkpoints from the
original repository to the Hugging Face format.

URL: https://github.com/facebookresearch/MetaCLIP

To convert:
1. git clone the MetaCLIP repository
2. place it in the same directory as this script
3. move the conversion script to the MetaCLIP repository.

Then run the script with:

```bash
cd MetaCLIP
python convert_metaclip_2_to_hf.py --checkpoint_path /path/to/checkpoint --model_name ViT-H-14-quickgelu-worldwide
```
"""

import argparse
import os
from typing import Optional

import torch
from PIL import Image

# Import MetaCLIP modules
from src.mini_clip.factory import create_model_and_transforms
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPProcessor,
    MetaClip2Config,
    MetaClip2Model,
)


def load_metaclip2_checkpoint(checkpoint_path: str, model_name: str) -> torch.nn.Module:
    """Load MetaCLIP 2 model from checkpoint."""
    print(f"Loading MetaCLIP 2 model: {model_name}")

    # For worldwide models, use WorldWideCLIP class
    model_name_with_class = model_name
    if "worldwide" in model_name.lower():
        model_name_with_class = f"{model_name}@WorldWideCLIP"
        print("Using WorldWideCLIP class for worldwide model")

    # Create model using the factory
    model, _, preprocess = create_model_and_transforms(model_name_with_class, pretrained=checkpoint_path, device="cpu")
    model.eval()
    return model, preprocess


def create_hf_config(tokenizer: AutoTokenizer, model_name: str) -> tuple[MetaClip2Config, int]:
    """Create Hugging Face MetaClip2Config from MetaCLIP model.

    This is based on the configs found at https://github.com/facebookresearch/MetaCLIP/tree/main/src/mini_clip/model_configs.
    """
    print("Creating Hugging Face config...")

    # Vision config
    vision_configs = {
        "ViT-H-14-quickgelu-worldwide": {
            "image_size": 224,
            "patch_size": 14,
            "hidden_size": 1280,
            "intermediate_size": 1280 * 4,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "hidden_act": "quick_gelu",
            "projection_dim": 1024,
        },
        "ViT-H-14-378-worldwide": {
            "image_size": 378,
            "patch_size": 14,
            "hidden_size": 1280,
            "intermediate_size": 1280 * 4,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "hidden_act": "gelu",
            "projection_dim": 1024,
        },
        "ViT-bigG-14-worldwide": {
            "image_size": 224,
            "patch_size": 14,
            "hidden_size": 1664,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "hidden_act": "gelu",
            "projection_dim": 1280,
        },
        "ViT-bigG-14-378-worldwide": {
            "image_size": 378,
            "patch_size": 14,
            "hidden_size": 1664,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "hidden_act": "gelu",
            "projection_dim": 1280,
        },
    }

    vision_config = vision_configs[model_name]
    image_size = vision_config["image_size"]

    # Text config
    text_configs = {
        "ViT-H-14-quickgelu-worldwide": {
            "hidden_size": 1024,
            "intermediate_size": 1024 * 4,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "max_position_embeddings": 77,
            "vocab_size": 901629,
            "eos_token_id": tokenizer.eos_token_id,
            "hidden_act": "quick_gelu",
            "projection_dim": 1024,
        },
        "ViT-H-14-378-worldwide": {
            "hidden_size": 1024,
            "intermediate_size": 1024 * 4,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "max_position_embeddings": 77,
            "vocab_size": 901629,
            "eos_token_id": tokenizer.eos_token_id,
            "hidden_act": "gelu",
            "projection_dim": 1024,
        },
        "ViT-bigG-14-worldwide": {
            "hidden_size": 1280,
            "intermediate_size": 1280 * 4,
            "num_attention_heads": 20,
            "num_hidden_layers": 32,
            "max_position_embeddings": 77,
            "vocab_size": 901629,
            "eos_token_id": tokenizer.eos_token_id,
            "hidden_act": "gelu",
            "projection_dim": 1280,
        },
        "ViT-bigG-14-378-worldwide": {
            "hidden_size": 1280,
            "intermediate_size": 1280 * 4,
            "num_attention_heads": 20,
            "num_hidden_layers": 32,
            "max_position_embeddings": 77,
            "vocab_size": 901629,
            "eos_token_id": tokenizer.eos_token_id,
            "hidden_act": "gelu",
            "projection_dim": 1280,
        },
    }

    text_config = text_configs[model_name]
    projection_dim = text_config["projection_dim"]

    # Create config
    config = MetaClip2Config(
        vision_config=vision_config,
        text_config=text_config,
        projection_dim=projection_dim,
    )

    return config, image_size


def convert_state_dict(metaclip_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert MetaCLIP state dict to Hugging Face format."""
    print("Converting state dict...")

    hf_state_dict = {}

    for key, value in metaclip_state_dict.items():
        new_key = key

        # Handle specific mappings first before general prefix replacements
        if key == "visual.proj":
            new_key = "visual_projection.weight"
            # Don't transpose! MetaCLIP: x @ proj, HF: Linear(x) = x @ weight.T
            # So we want weight.T = proj, which means weight = proj.T
            # But since we're storing proj as weight, we need proj.T
            value = value.T  # This gives us the correct orientation for Linear layer
        elif key == "text_projection":
            new_key = "text_projection.weight"
            # Same logic as visual projection
            value = value.T
        elif key == "token_embedding.weight":
            new_key = "text_model.embeddings.token_embedding.weight"
        elif key == "positional_embedding":
            new_key = "text_model.embeddings.position_embedding.weight"
        elif key == "ln_final.weight":
            new_key = "text_model.final_layer_norm.weight"
        elif key == "ln_final.bias":
            new_key = "text_model.final_layer_norm.bias"
        # Vision encoder mappings
        elif key.startswith("visual."):
            new_key = key.replace("visual.", "vision_model.")

            # Handle specific vision model components
            if "conv1" in new_key:
                new_key = new_key.replace("conv1", "embeddings.patch_embedding")
            elif "class_embedding" in new_key:
                new_key = new_key.replace("class_embedding", "embeddings.class_embedding")
            elif "positional_embedding" in new_key:
                new_key = new_key.replace("positional_embedding", "embeddings.position_embedding.weight")
            elif "ln_pre" in new_key:
                new_key = new_key.replace("ln_pre", "pre_layrnorm")
            elif "ln_post" in new_key:
                new_key = new_key.replace("ln_post", "post_layernorm")
            elif "transformer.resblocks" in new_key:
                new_key = new_key.replace("transformer.resblocks", "encoder.layers")
                # Handle attention and MLP mappings within transformer blocks
                if "attn.in_proj" in new_key:
                    # Split the in_proj into q, k, v projections
                    if "weight" in new_key:
                        # We'll handle this later in a special case
                        continue
                    elif "bias" in new_key:
                        continue
                elif "attn.out_proj" in new_key:
                    new_key = new_key.replace("attn.out_proj", "self_attn.out_proj")
                elif "ln_1" in new_key:
                    new_key = new_key.replace("ln_1", "layer_norm1")
                elif "ln_2" in new_key:
                    new_key = new_key.replace("ln_2", "layer_norm2")
                elif "mlp.c_fc" in new_key:
                    new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
                elif "mlp.c_proj" in new_key:
                    new_key = new_key.replace("mlp.c_proj", "mlp.fc2")

        # Text encoder mappings
        elif key.startswith("transformer."):
            new_key = key.replace("transformer.", "text_model.encoder.")

            if "resblocks" in new_key:
                new_key = new_key.replace("resblocks", "layers")
                # Similar mappings as vision transformer
                if "attn.in_proj" in new_key:
                    continue  # Handle separately
                elif "attn.out_proj" in new_key:
                    new_key = new_key.replace("attn.out_proj", "self_attn.out_proj")
                elif "ln_1" in new_key:
                    new_key = new_key.replace("ln_1", "layer_norm1")
                elif "ln_2" in new_key:
                    new_key = new_key.replace("ln_2", "layer_norm2")
                elif "mlp.c_fc" in new_key:
                    new_key = new_key.replace("mlp.c_fc", "mlp.fc1")
                elif "mlp.c_proj" in new_key:
                    new_key = new_key.replace("mlp.c_proj", "mlp.fc2")

        hf_state_dict[new_key] = value

    # Handle in_proj weights separately (split into q, k, v)
    for key, value in metaclip_state_dict.items():
        if "attn.in_proj_weight" in key:
            # Split the combined qkv weight into separate q, k, v weights
            dim = value.shape[0] // 3
            q_weight = value[:dim]
            k_weight = value[dim : 2 * dim]
            v_weight = value[2 * dim :]

            base_key = key.replace("attn.in_proj_weight", "")
            if key.startswith("visual."):
                base_key = base_key.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
            else:
                base_key = base_key.replace("transformer.resblocks", "text_model.encoder.layers")

            hf_state_dict[f"{base_key}self_attn.q_proj.weight"] = q_weight
            hf_state_dict[f"{base_key}self_attn.k_proj.weight"] = k_weight
            hf_state_dict[f"{base_key}self_attn.v_proj.weight"] = v_weight

        elif "attn.in_proj_bias" in key:
            # Split the combined qkv bias into separate q, k, v biases
            dim = value.shape[0] // 3
            q_bias = value[:dim]
            k_bias = value[dim : 2 * dim]
            v_bias = value[2 * dim :]

            base_key = key.replace("attn.in_proj_bias", "")
            if key.startswith("visual."):
                base_key = base_key.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
            else:
                base_key = base_key.replace("transformer.resblocks", "text_model.encoder.layers")

            hf_state_dict[f"{base_key}self_attn.q_proj.bias"] = q_bias
            hf_state_dict[f"{base_key}self_attn.k_proj.bias"] = k_bias
            hf_state_dict[f"{base_key}self_attn.v_proj.bias"] = v_bias

    return hf_state_dict


def verify_conversion(
    original_model, hf_model, preprocess, image_processor, tokenizer, test_image_path: Optional[str] = None
) -> bool:
    """Verify that the conversion produces the same outputs."""
    print("Verifying conversion...")

    # Create test image
    if test_image_path and os.path.exists(test_image_path):
        image = Image.open(test_image_path)
    else:
        # Create a dummy image
        image = Image.new("RGB", (224, 224), color="red")

    # Verify image processor
    processed_image = preprocess(image).unsqueeze(0)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    print("Shape of pixel_values:", pixel_values.shape)
    print("Shape of processed_image:", processed_image.shape)
    assert torch.allclose(pixel_values, processed_image)

    # Use tokenizer to get input_ids
    texts = ["a cat", "a dog", "a bird"]
    token_inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    input_ids = token_inputs.input_ids

    print(f"Processed text shape: {input_ids.shape}")
    print(f"Processed image shape: {processed_image.shape}")

    with torch.no_grad():
        # Original model outputs
        orig_image_features = original_model.encode_image(processed_image)
        orig_text_features = original_model.encode_text(input_ids)

        # Normalize and compute logits
        orig_image_features = orig_image_features / orig_image_features.norm(dim=-1, keepdim=True)
        orig_text_features = orig_text_features / orig_text_features.norm(dim=-1, keepdim=True)
        orig_logits = original_model.logit_scale.exp() * orig_image_features @ orig_text_features.T

        print(f"Original text features: {orig_text_features[0][:5].tolist()}")
        print(f"Original image features: {orig_image_features[0][:5].tolist()}")

    with torch.no_grad():
        hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values)
        hf_logits = hf_outputs.logits_per_image

        # Debug: Check HF model features
        print(f"HF text features: {hf_outputs.text_embeds[0][:5].tolist()}")
        print(f"HF image features: {hf_outputs.image_embeds[0][:5].tolist()}")
        print(f"HF model EOS token ID: {hf_model.config.text_config.eos_token_id}")

    # Compare outputs
    print(f"Original logits: {orig_logits}")
    print(f"HF logits: {hf_logits}")
    print(f"Logit scale - Original: {original_model.logit_scale.exp():.6f}, HF: {hf_model.logit_scale.exp():.6f}")

    # Check if they're close
    if orig_logits.shape == hf_logits.shape and torch.allclose(orig_logits, hf_logits, atol=1e-4):
        print("✅ Conversion verified! Outputs match.")
        return True
    else:
        print("❌ Conversion failed! Outputs don't match.")
        if orig_logits.numel() > 0 and hf_logits.numel() > 0:
            print(f"Max difference: {(orig_logits - hf_logits).abs().max()}")
        return False


def push_to_hub(hf_model: MetaClip2Model, processor: CLIPProcessor, repo_name: str):
    """Push the converted model to Hugging Face Hub."""
    print(f"Pushing to hub: {repo_name}")

    try:
        hf_model.push_to_hub(repo_name)
        processor.push_to_hub(repo_name)
        print(f"✅ Successfully pushed to {repo_name}")
    except Exception as e:
        print(f"❌ Failed to push to hub: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert MetaCLIP 2 to Hugging Face format")
    parser.add_argument("--checkpoint_path", required=True, help="Path to MetaCLIP 2 checkpoint")
    parser.add_argument("--model_name", required=True, help="MetaCLIP model name (e.g., ViT-H-14-quickgelu-worldwide)")
    parser.add_argument("--output_dir", default="./converted_models", help="Output directory for converted model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--hub_repo_name", help="Hub repository name")
    parser.add_argument("--test_image", help="Path to test image for verification")

    args = parser.parse_args()

    # Load original model
    original_model, preprocess = load_metaclip2_checkpoint(args.checkpoint_path, args.model_name)

    # Create HF config
    # Requires the tokenizer for the eos token id
    tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-v-base")
    config, image_size = create_hf_config(tokenizer=tokenizer, model_name=args.model_name)

    # Create processor
    image_processor = CLIPImageProcessor(
        size={"height": image_size, "width": image_size}, crop_size={"height": image_size, "width": image_size}
    )
    processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Create HF model
    hf_model = MetaClip2Model(config)

    # Convert state dict
    converted_state_dict = convert_state_dict(original_model.state_dict())

    for name, param in hf_model.named_parameters():
        print(name, param.shape)

    # Load converted weights
    hf_model.load_state_dict(converted_state_dict)

    # Verify conversion
    if not verify_conversion(original_model, hf_model, preprocess, image_processor, tokenizer, args.test_image):
        print("Conversion verification failed. Please check the conversion logic.")
        return

    # Save model locally
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        hf_model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)

    # Push to hub if requested
    if args.push_to_hub and args.hub_repo_name:
        push_to_hub(hf_model, processor, args.hub_repo_name)


if __name__ == "__main__":
    main()

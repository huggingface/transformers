import argparse
import os
import torch
import copy
from transformers import VibeVoiceConfig, VibeVoiceForConditionalGeneration, VibeVoiceTokenizer


def convert_vibevoice_checkpoint(checkpoint_path, pytorch_dump_folder_path, model_size="1.5B"):
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Check if state_dict is nested (e.g. under "state_dict" key)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Define generic config
    config = VibeVoiceConfig()
    # Adjust config based on model size if necessary
    # For now utilizing default values which match 1.5B usually

    print("Initializing HF model...")
    model = VibeVoiceForConditionalGeneration(config)

    new_state_dict = {}

    # Mapping logic
    # 1. LLM Backbone (Qwen2) splitting
    # Original likely has "llm.layers.0..." up to 24 (or similar)
    # We split into language_model (e.g. 0-15??) and tts_language_model (16-24??)
    # Need to check config.tts_backbone_num_hidden_layers default

    tts_layers = config.tts_backbone_num_hidden_layers
    total_layers = getattr(config.decoder_config, "num_hidden_layers", 24)
    lm_layers = total_layers - tts_layers

    print(f"Splitting LLM weights: {lm_layers} text layers, {tts_layers} tts layers.")

    # Helper to map Qwen2 keys
    def map_qwen_layer(old_key, new_prefix, layer_mapping=None):
        # old_key: llm.layers.0.self_attn.q_proj.weight
        # new_prefix: model.language_model
        # layer_mapping: {0:0, 1:1, ... 15:0, 16:1...}

        parts = old_key.split(".")
        # Assuming format: llm.model.layers.X... or llm.layers.X...
        # Let's clean up prefix "llm." or "llm.model."
        if old_key.startswith("llm.model."):
            remaining = old_key[10:]
        elif old_key.startswith("llm."):
            remaining = old_key[4:]
        else:
            return None, None

        if remaining.startswith("layers."):
            layer_idx = int(remaining.split(".")[1])
            suffix = ".".join(remaining.split(".")[2:])

            if layer_idx < lm_layers:
                # Belongs to text LM
                new_key = f"model.language_model.layers.{layer_idx}.{suffix}"
                return new_key, state_dict[old_key]
            else:
                # Belongs to TTS LM
                new_layer_idx = layer_idx - lm_layers
                new_key = f"model.tts_language_model.layers.{new_layer_idx}.{suffix}"
                return new_key, state_dict[old_key]

        elif remaining.startswith("embed_tokens."):
            # Both share embeddings? Or just language_model has it?
            # AutoModel usually has embed_tokens.
            # In VibeVoiceModel init:
            # self.language_model = AutoModel...
            # self.tts_language_model = AutoModel...
            # If they share embeddings, we might need to duplicate or it's tied.
            # VibeVoiceModel logic doesn't explicitly tie them yet in my code?
            # Usually we duplicate weight for initialization.
            new_key_lm = f"model.language_model.embed_tokens.weight"
            new_key_tts = f"model.tts_language_model.embed_tokens.weight"
            return [new_key_lm, new_key_tts], state_dict[old_key]

        elif remaining.startswith("norm."):
            # Final norm usually belongs to the last part (TTS LM)?
            # Or text LM has a norm too?
            # Code: self.language_model.norm = nn.Identity()
            # Code: self.tts_language_model.norm = RMSNorm... (from Qwen)
            # So "norm" goes to tts_language_model.norm
            new_key = f"model.tts_language_model.norm.{remaining.split('.', 1)[1]}"  # weight
            return new_key, state_dict[old_key]

        return None, None

    # Iterate over all keys
    for key in state_dict.keys():
        if key.startswith("llm."):
            new_key, value = map_qwen_layer(key, "")
            if new_key:
                if isinstance(new_key, list):
                    for nk in new_key:
                        new_state_dict[nk] = value
                else:
                    new_state_dict[new_key] = value

        elif key.startswith("vae."):
            # Map to model.acoustic_tokenizer
            # key: vae.encoder... -> model.acoustic_tokenizer.encoder...
            # BUT: VibeVoiceAcousticTokenizerModel only has `decoder` (TokenizerDecoder) in my code?
            # Wait, `modeling_vibevoice.py` `VibeVoiceAcousticTokenizerModel` has `self.decoder = TokenizerDecoder`.
            # Does it have encoder?
            # Conf VibeVoiceAcousticTokenizerModel: `__init__` only initialized `decoder`.
            # If default VibeVoice uses just decoder (for generation), we might ignore encoder?
            # BUT `VibeVoiceConfig` has `acoustic_tokenizer_config`.
            # If the checkpoint has `vae.decoder...`, map to `model.acoustic_tokenizer.decoder...`.
            suffix = key[4:]
            new_key = f"model.acoustic_tokenizer.{suffix}"
            new_state_dict[new_key] = state_dict[key]

        elif key.startswith("diffusion."):
            # Map to model.prediction_head
            suffix = key[10:]
            new_key = f"model.prediction_head.{suffix}"
            new_state_dict[new_key] = state_dict[key]

        elif key.startswith("text_encoder."):
            # If separate text encoder?
            # Probably covered by llm splitting.
            pass

    # Handle specific VibeVoice keys
    # speech_connector
    if "speech_connector.fc1.weight" in state_dict:
        new_state_dict["model.acoustic_connector.fc1.weight"] = state_dict["speech_connector.fc1.weight"]
        new_state_dict["model.acoustic_connector.fc1.bias"] = state_dict["speech_connector.fc1.bias"]
        # ... others

    print(f"Saving HF model to {pytorch_dump_folder_path}")
    model.load_state_dict(new_state_dict, strict=False)
    model.save_pretrained(pytorch_dump_folder_path)

    # Save tokenizer
    print("Saving tokenizer...")
    try:
        tokenizer = VibeVoiceTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")  # Base
        tokenizer.save_pretrained(pytorch_dump_folder_path)
    except Exception as e:
        print(f"Could not load/save base tokenizer: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to original pt checkpoint")
    parser.add_argument("--dump_path", type=str, required=True, help="Path to output folder")
    parser.add_argument("--model_size", type=str, default="1.5B", help="Model size")
    args = parser.parse_args()

    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    convert_vibevoice_checkpoint(args.checkpoint_path, args.dump_path, args.model_size)

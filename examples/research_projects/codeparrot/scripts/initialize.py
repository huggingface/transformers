from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# Load codeparrot tokenizer trained for Python code tokenization
tokenizer_name = "transformersbook/codeparrot"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Set model name
model_name = "codeparrot"

# Config: "scale_attn_by_layer_idx" and "reorder_and_upcast_attn" are Mistral stability tweaks
config_kwargs = {"vocab_size": len(tokenizer), "scale_attn_by_layer_idx": True, "reorder_and_upcast_attn": True}

# Load model config (GPT-2 large in this case)
config = AutoConfig.from_pretrained("gpt2-large", **config_kwargs)

# Initialize new model with config
model = AutoModelForCausalLM(config)

# Save model and tokenizer to the hub
model.save_pretrained(model_name, push_to_hub=True)
tokenizer.save_pretrained(model_name, push_to_hub=True)

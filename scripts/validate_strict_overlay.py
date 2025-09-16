"""End-to-end validation of strict overlay functionality."""

import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from assist_strict.assisted import assisted_generate_strict, ConfigDriftError


# Test configuration
MODEL_NAME = "microsoft/DialoGPT-small"
ASSISTANT_NAME = "microsoft/DialoGPT-small"

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """Validate strict overlay functionality end-to-end."""
    logger.info("Loading models for strict overlay validation...")

    # Load models and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    assistant_model = AutoModelForCausalLM.from_pretrained(ASSISTANT_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Ensure tokenizer has pad token - required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare simple input
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Capture original assistant config for comparison
    original_config = assistant_model.generation_config.to_dict()

    logger.info("Performing strict assisted generation...")

    # Execute strict assisted generation
    result = assisted_generate_strict(
        model=model,
        inputs=inputs.input_ids,
        assistant_model=assistant_model,
        num_assistant_tokens=5,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    # Validate config unchanged
    post_call_config = assistant_model.generation_config.to_dict()
    if original_config != post_call_config:
        raise ConfigDriftError("Assistant config was modified during generation")

    # Validate successful generation
    assert result is not None, "Generation returned None"
    assert hasattr(result, 'shape'), "Expected tensor result with shape attribute"

    logger.info("✓ Strict overlay validation successful!")
    logger.info(f"✓ Assistant config preserved: {len(original_config)} parameters unchanged")
    logger.info(f"✓ Generation completed with output shape: {result.shape}")


if __name__ == "__main__":
    main()

"""
Examples of using declarative LogitProcessor configuration in Transformers.

This demonstrates the new logit_processors configuration feature that allows
users to specify logit processing strategies via JSON configuration instead
of manual LogitsProcessor instantiation.
"""

import json
import torch  # noqa: F401 (used in custom processor example)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    LogitProcessorRegistry,
    LogitsProcessor,
)


def example_basic_usage():
    """Basic example of using configured logit processors."""
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Define logit processors via configuration
    logit_config = [
        {"type": "TemperatureLogitsWarper", "temperature": 0.8},
        {"type": "TopKLogitsWarper", "top_k": 50},
        {"type": "RepetitionPenaltyLogitsProcessor", "penalty": 1.2}
    ]
    
    # Create GenerationConfig with logit processors
    generation_config = GenerationConfig(
        max_length=100,
        do_sample=True,
        logit_processors=logit_config
    )
    
    # Generate text
    input_ids = tokenizer.encode("The future of AI is", return_tensors="pt")
    outputs = model.generate(input_ids, generation_config=generation_config)
    
    print("Generated text:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def example_json_string_config():
    """Example using JSON string configuration."""
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Define as JSON string (useful for config files)
    logit_config_json = json.dumps([
        {"type": "TemperatureLogitsWarper", "temperature": 0.9},
        {"type": "TopPLogitsWarper", "top_p": 0.95}
    ])
    
    generation_config = GenerationConfig(
        max_length=50,
        do_sample=True,
        logit_processors=logit_config_json
    )
    
    input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
    outputs = model.generate(input_ids, generation_config=generation_config)
    
    print("Generated text with JSON config:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def example_custom_processor():
    """Example with custom LogitProcessor."""

    @LogitProcessorRegistry.register
    class WordBanLogitsProcessor(LogitsProcessor):  # type: ignore
        """Custom processor that bans specific words."""

        def __init__(self, banned_words, tokenizer_vocab):
            self.banned_token_ids = [tokenizer_vocab[w] for w in banned_words if w in tokenizer_vocab]

        def __call__(self, input_ids, scores):  # type: ignore
            for token_id in self.banned_token_ids:
                scores[:, token_id] = float('-inf')
            return scores
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Use custom processor in configuration
    logit_config = [
        {"type": "TemperatureLogitsWarper", "temperature": 0.8},
        {
            "type": "WordBanLogitsProcessor", 
            "banned_words": ["bad", "terrible"],
            "tokenizer_vocab": tokenizer.get_vocab()
        }
    ]
    
    generation_config = GenerationConfig(
        max_length=50,
        do_sample=True,
        logit_processors=logit_config
    )
    
    input_ids = tokenizer.encode("The weather today is", return_tensors="pt")
    outputs = model.generate(input_ids, generation_config=generation_config)
    
    print("Generated text with custom word banning:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def example_config_file():
    """Example of loading configuration from file."""
    
    # Save config to file
    config = {
        "max_length": 100,
        "do_sample": True,
        "logit_processors": [
            {"type": "TemperatureLogitsWarper", "temperature": 0.7},
            {"type": "TopKLogitsWarper", "top_k": 40},
            {"type": "NoRepeatNGramLogitsProcessor", "ngram_size": 3}
        ]
    }
    
    with open("generation_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load config from file
    with open("generation_config.json", "r") as f:
        loaded_config = json.load(f)
    
    generation_config = GenerationConfig(**loaded_config)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    input_ids = tokenizer.encode("In the year 2050", return_tensors="pt")
    outputs = model.generate(input_ids, generation_config=generation_config)
    
    print("Generated text from config file:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    print("=== Basic Usage Example ===")
    example_basic_usage()
    
    print("\n=== JSON String Config Example ===")
    example_json_string_config()
    
    print("\n=== Custom Processor Example ===")
    example_custom_processor()
    
    print("\n=== Config File Example ===")
    example_config_file()
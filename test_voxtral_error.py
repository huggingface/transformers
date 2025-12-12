"""
This script reproduces the confusing error message for Voxtral AutoTokenizer.
Only for Transformers Issue #41553.
"""

from transformers import AutoTokenizer

def main():
    model_name = "Qwen/Qwen2.5-VL-Voxtral"

    print(f"Attempting to load AutoTokenizer for model: {model_name}")

    # This should trigger the original confusing error
    tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    main()

import argparse
import os
from tokenizers import processors
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def convert_glm4_tokenizer(input_dir, output_dir, use_post_processor=False):
    """
    Convert GLM4 tokenizer with optional post-processing and save as tokenizer.json

    Args:
        input_dir (str): Directory containing original tokenizer files
        output_dir (str): Directory to save converted tokenizer
        use_post_processor (bool): Whether to apply GLM4-specific post processing
    """
    # Load the original tokenizer
    fast_tok = PreTrainedTokenizerFast.from_pretrained(
        input_dir,
        trust_remote_code=True,
    )

    # Configure post processor based on option
    if use_post_processor:
        # GLM4-specific template processing with special tokens
        fast_tok._tokenizer.post_processor = processors.Sequence([
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single="[gMASK]:0 <sop>:0 $A:0",
                pair="[gMASK]:0 <sop>:0 $A:0 $B:1",
                special_tokens=[("[gMASK]", 151331), ("<sop>", 151333)],
            ),
        ])
    else:
        # Basic byte-level processing only
        fast_tok._tokenizer.post_processor = processors.Sequence([
            processors.ByteLevel(trim_offsets=False)
        ])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the converted tokenizer
    fast_tok.save_pretrained(output_dir)

    print(f"Tokenizer converted and saved to: {output_dir}")
    print(f"Post processor enabled: {use_post_processor}")

    return fast_tok


def main():
    parser = argparse.ArgumentParser(description="Convert GLM4 tokenizer")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing tokenizer files (tokenizer.model, tokenizer_config.json, etc.)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save converted tokenizer.json"
    )
    parser.add_argument(
        "--use_post_processor",
        action="store_true",
        help="Apply GLM4-specific post processing with [gMASK] and <sop> tokens"
    )

    args = parser.parse_args()

    # Convert the tokenizer
    convert_glm4_tokenizer(args.input_dir, args.output_dir, args.use_post_processor)


if __name__ == "__main__":
    main()
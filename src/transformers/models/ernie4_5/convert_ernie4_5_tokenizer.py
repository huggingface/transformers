import argparse

from transformers import Ernie4_5Tokenizer, Ernie4_5TokenizerFast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_name",
        help="Name of the repo where the tokenizer is located at.",
        default="baidu/ERNIE-4.5-0.3B-Base-PT",
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write the tokenizer",
    )
    args = parser.parse_args()

    hf_tok = Ernie4_5Tokenizer.from_pretrained(args.repo_name)
    hf_tok.model_max_length = 131072
    hf_tok.init_kwargs.pop("auto_map", None)
    # special tokens which we need to map as additional special tokens instead
    hf_tok.init_kwargs.pop("header_start_token", None)
    hf_tok.init_kwargs.pop("header_end_token", None)
    hf_tok.init_kwargs.pop("sys_start_token", None)
    hf_tok.init_kwargs.pop("sys_end_token", None)
    for token in [
        "<mask:4>",
        "<mask:5>",
        "<mask:6>",
        "<mask:7>",
    ]:
        hf_tok.add_tokens([token], special_tokens=True)

    # save slow model and convert on load time
    hf_tok.save_pretrained("/tmp/ernie4_5_tokenizer")
    hf_tok_fast = Ernie4_5TokenizerFast.from_pretrained("/tmp/ernie4_5_tokenizer", from_slow=True)
    hf_tok_fast.save_pretrained(args.output_dir, push_to_hub=args.push_to_hub)

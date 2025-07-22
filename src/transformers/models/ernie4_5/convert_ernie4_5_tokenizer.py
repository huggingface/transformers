# Copyright (c) 2025 HuggingFace Inc. team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from transformers import LlamaTokenizer, LlamaTokenizerFast


DEFAULT_CHAT_TEMPLATE = '{%- if not add_generation_prompt is defined -%}\n    {%- set add_generation_prompt = true -%}\n{%- endif -%}\n{%- if not cls_token is defined -%}\n    {%- set cls_token = "<|begin_of_sentence|>" -%}\n{%- endif -%}\n{%- if not sep_token is defined -%}\n    {%- set sep_token = "<|end_of_sentence|>" -%}\n{%- endif -%}\n{{- cls_token -}}\n{%- for message in messages -%}\n    {%- if message["role"] == "user" -%}\n        {{- "User: " + message["content"] + "\n" -}}\n    {%- elif message["role"] == "assistant" -%}\n        {{- "Assistant: " + message["content"] + sep_token -}}\n    {%- elif message["role"] == "system" -%}\n        {{- message["content"] + "\n" -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- "Assistant: " -}}\n{%- endif -%}'
DEFAULT_TEXT_ADD_TOKENS = [
    "<mask:4>",
    "<mask:5>",
    "<mask:6>",
    "<mask:7>",
]


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

    hf_tok = LlamaTokenizer.from_pretrained(
        args.repo_name,
        pad_token="<unk>",
        cls_token="<|begin_of_sentence|>",
        sep_token="<|end_of_sentence|>",
        mask_token="<mask:1>",
        add_bos_token=False,
        add_prefix_space=False,
        chat_template=DEFAULT_CHAT_TEMPLATE,
        legacy=True,
    )
    hf_tok.model_max_length = 131072
    hf_tok.init_kwargs.pop("auto_map", None)
    # special tokens which we need to map as additional special tokens instead
    hf_tok.init_kwargs.pop("header_start_token", None)
    hf_tok.init_kwargs.pop("header_end_token", None)
    hf_tok.init_kwargs.pop("sys_start_token", None)
    hf_tok.init_kwargs.pop("sys_end_token", None)
    for token in DEFAULT_TEXT_ADD_TOKENS:
        hf_tok.add_tokens([token], special_tokens=True)

    # save slow model and convert on load time
    hf_tok.save_pretrained("/tmp/ernie4_5_tokenizer")
    hf_tok_fast = LlamaTokenizerFast.from_pretrained("/tmp/ernie4_5_tokenizer", from_slow=True)
    hf_tok_fast.save_pretrained(args.output_dir, push_to_hub=args.push_to_hub)

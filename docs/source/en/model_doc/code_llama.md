<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CodeLlama

## Overview

The CodeLlama model was proposed in [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) by Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve.

The abstract from the paper is the following:

*We release Code Llama, a family of large language models for code based on Llama 2 providing state-of-the-art performance among open models, infilling capabilities, support for large input contexts, and zero-shot instruction following ability for programming tasks. We provide multiple flavors to cover a wide range of applications: foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models (Code Llama - Instruct) with 7B, 13B and 34B parameters each. All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens. 7B and 13B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama reaches state-of-the-art performance among open models on several code benchmarks, with scores of up to 53% and 55% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every other publicly available model on MultiPL-E. We release Code Llama under a permissive license that allows for both research and commercial use.*

Checkout all CodeLlama models [here](https://huggingface.co/models?search=code_llama)

Tips:

- These models have the same architcture as the `Llama2` models
- The infilling task is supported out of the box. You should be using the `tokenizer.fill_token` where you want your input to be filled.
- The model conversion script is the same as for the `Llama2` familly:

Here is a sample usage
```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- After conversion, the model and tokenizer can be loaded via:

```python
>>> from transformers import LlamaForCausalLM, CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
>>> generated_ids = model.generate(input_ids, max_new_tokens=128)

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
```

If you only want the infilled part:
```python
>>> from transformers import pipeline
>>> import torch

>>> pipeline = pipeline("text-generation",model="codellama/CodeLlama-7b-hf",torch_dtype=torch.float16, device_map="auto")
>>> pipeline('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens = 128, return_type = 1)
```
Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM). For the 75B model, it's thus 145GB of RAM needed.

- The LLaMA tokenizer is a BPE model based on [sentencepiece](https://github.com/google/sentencepiece). One quirk of sentencepiece is that when decoding a sequence, if the first token is the start of the word (e.g. "Banana"), the tokenizer does not prepend the prefix space to the string.

This model was contributed by [ArthurZucker](https://huggingface.co/ArthurZ). The original code of the authors can be found [here](https://github.com/facebookresearch/llama).


## CodeLlamaTokenizer

[[autodoc]] CodeLlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeLlamaTokenizerFast

[[autodoc]] CodeLlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary
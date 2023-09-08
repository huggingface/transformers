<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Persimmon

## Overview

The Persimmon model was created by [ADEPT](https://www.adept.ai/blog/persimmon-8b), and authored by Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, Arushi Somani.

The authors introduced Persimmon-8B, a decoder model based on the classic transformers architecture, with query and key normalization. Persimmon-8B is a fully permissively-licensed model with approximately 8 billion parameters, released under the Apache license.  Some of the key attributes of Persimmon-8B are long context size (16K), performance, and capabilities for multimodal extensions.

The authors showcase their approach to model evaluation, focusing on practical text generation, mirroring how users interact with language models. The work also includes a comparative analysis, pitting Persimmon-8B against other prominent models (MPT 7B Instruct and Llama 2 Base 7B 1-Shot), across various evaluation tasks. The results demonstrate Persimmon-8B's competitive performance, even with limited training data.

In terms of model details, the work outlines the architecture and training methodology of Persimmon-8B, providing insights into its design choices, sequence length, and dataset composition. The authors present a fast inference code that outperforms traditional implementations through operator fusion and CUDA graph utilization while maintaining code coherence. They express their anticipation of how the community will leverage this contribution to drive innovation, hinting at further upcoming releases as part of an ongoing series of developments.


<Tip warning={true}>

The `Persimmon` models were trained using `bfloat16`, but the original inference uses `float16` The checkpoints uploaded on the hub use `torch_dtype = 'float16'` which will be
used by the `AutoModel` API to cast the checkpoints from `torch.float32` to `torch.float16`. 

The `dtype` of the online weights is mostly irrelevant, unless you are using `torch_dtype="auto"` when initializing a model using `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`. The reason is that the model will first be downloaded ( using the `dtype` of the checkpoints online) then it will be cast to the default `dtype` of `torch` (becomes `torch.float32`). Users should specify the `torch_dtype` they want, and if they don't it will be `torch.float32`.

Finetuning the model in `float16` is not recommended and known to produce `nan`, as such the model should be fine-tuned in `bfloat16`.

</Tip>


Tips:

- To convert the model, you need to clone the original repository using `git clone https://github.com/persimmon-ai-labs/adept-inference`, then get the checkpoints:

```bash
git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
tar -xvf 8b_base_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path \
    --pt_model_path /path/to/8b_chat_model_release/iter_0001251/mp_rank_00/model_optim_rng.pt
    --ada_lib_path /path/to/adept-inference
```

For the chat model:
```bash
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```

Thereafter, models can be loaded via:

```py
from transformers import PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained("/output/path")
tokenizer = PersimmonTokenizer.from_pretrained("/output/path")
```

This model was contributed by [ArthurZ](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/persimmon-ai-labs/adept-inference).

- Perismmon uses a `sentencepiece` based tokenizer, with a `Unigram` model. It supports bytefallback, which is only available in `tokenizers==0.14.0` for the fast tokenizer.
The `LlamaTokenizer` is used as it is a standard wrapper around sentencepiece. The `chat` template will be updated with the templating functions in a follow up PR!

- The authors suggest to use the following prompt format for the chat mode: `f"human: {prompt}\n\nadept:"`


## PersimmonConfig

[[autodoc]] PersimmonConfig

## PersimmonModel

[[autodoc]] PersimmonModel
    - forward

## PersimmonForCausalLM

[[autodoc]] PersimmonForCausalLM
    - forward

## PersimmonForSequenceClassification

[[autodoc]] PersimmonForSequenceClassification
    - forward

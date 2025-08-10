<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Evolla

## Overview

The Evolla model was proposed in [Decoding the Molecular Language of Proteins with Evolla](https://doi.org/10.1101/2025.01.05.630192) by [Zhou et al.](https://doi.org/10.1101/2025.01.05.630192).

Evolla is an advanced 80-billion-parameter protein-language generative model designed to decode the molecular language of proteins. It integrates information from protein sequences, structures, and user queries to generate precise and contextually nuanced insights into protein function. Trained on an unprecedented AI-generated dataset of 546 million protein question-answer pairs and 150 billion word tokens, Evolla significantly advances research in proteomics and functional genomics, providing expert-level insights and shedding light on the molecular logic encoded in proteins.

The abstract from the paper is the following:

*Proteins, nature’s intricate molecular machines, are the products of billions of years of evolution and play fundamental roles in sustaining life. Yet, deciphering their molecular language - that is, understanding how protein sequences and structures encode and determine biological functions - remains a corner-stone challenge in modern biology. Here, we introduce Evolla, an 80 billion frontier protein-language generative model designed to decode the molecular language of proteins. By integrating information from protein sequences, structures, and user queries, Evolla generates precise and contextually nuanced insights into protein function. A key innovation of Evolla lies in its training on an unprecedented AI-generated dataset: 546 million protein question-answer pairs and 150 billion word tokens, designed to reflect the immense complexity and functional diversity of proteins. Post-pretraining, Evolla integrates Direct Preference Optimization (DPO) to refine the model based on preference signals and Retrieval-Augmented Generation (RAG) for external knowledge incorporation, improving response quality and relevance. To evaluate its performance, we propose a novel framework, Instructional Response Space (IRS), demonstrating that Evolla delivers expert-level insights, advancing research in proteomics and functional genomics while shedding light on the molecular logic encoded in proteins. The online demo is available at http://www.chat-protein.com/.*

Examples:

```python
processor = EvollaProcessor.from_pretrained("westlake-repl/Evolla-10B-DPO-hf")
model = EvollaForProteinText2Text.from_pretrained("westlake-repl/Evolla-10B-DPO-hf")
# aa_seq should have same length as foldseek
protein_inputs = [
    {
        
        "aa_seq": "MATGGRRG...",
        "foldseek": "###lqpfd...", # hashtag means the low-confidence foldseek tokens
    },
    {
        "aa_seq": "MLPGLALL...",
        "foldseek": "dfwwkwad...",
    }
]
message_list = [
    [
        {
            "role": "system",
            "content": "You are an AI expert that can answer any questions about protein.",
        },
        {"role": "user", "content": "What is the function of this protein?"},
    ],
    [
        {
            "role": "system",
            "content": "You are an AI expert that can answer any questions about protein.",
        },
        {"role": "user", "content": "What is the function of this protein?"},
    ]
]
input_dict = processor(
    protein_informations, messages_list, return_tensors="pt", text_max_length=512, protein_max_length=1024
)
with torch.no_grad():
    generated_ids = hf_model.generate(**input_dict)
generated_texts = processor.batch_decode(
    generated_ids, skip_special_tokens=True
)
```

Tips:

- This model was contributed by [Xibin Bayes Zhou](https://huggingface.co/XibinBayesZhou).
- The original code can be found [here](https://github.com/westlake-repl/Evolla).


## EvollaConfig

[[autodoc]] EvollaConfig

## EvollaModel

[[autodoc]] EvollaModel
    - forward

## EvollaForProteinText2Text

[[autodoc]] EvollaForProteinText2Text
    - forward

## EvollaProcessor

[[autodoc]] EvollaProcessor
    - __call__

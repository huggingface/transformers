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

*This model was released on 2025-01-05 and added to Hugging Face Transformers on 2025-07-26 and contributed by [XibinBayesZhou](https://huggingface.co/XibinBayesZhou).*

# Evolla

[Evolla](https://huggingface.co/papers/2025.01.05.630192) is an 80-billion-parameter protein-language generative model aimed at decoding the molecular language of proteins. It leverages protein sequences, structures, and user queries to provide precise insights into protein function. Trained on a large AI-generated dataset of 546 million protein question-answer pairs and 150 billion word tokens, Evolla incorporates Direct Preference Optimization (DPO) and Retrieval-Augmented Generation (RAG) to enhance response quality. The model evaluates its performance using a novel Instructional Response Space (IRS) framework, demonstrating expert-level insights in proteomics and functional genomics.

<hfoptions id="usage">
<hfoption id="EvollaForProteinText2Text">

```py
import torch
from transformers import AutoProcessor, EvollaForProteinText2Text

processor = AutoProcessor.from_pretrained("westlake-repl/Evolla-10B-DPO-hf")
model = EvollaForProteinText2Text.from_pretrained("westlake-repl/Evolla-10B-DPO-hf", dtype="auto")
protein_inputs = [
    {
        
        "aa_seq": "MATGGRRG...",
        "foldseek": "###lqpfd...",
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
    protein_inputs, messages_list, return_tensors="pt", text_max_length=512, protein_max_length=1024
)
with torch.no_grad():
    generated_ids = hf_model.generate(**input_dict)
print(processor.batch_decode(generated_ids, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

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


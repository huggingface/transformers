<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# HHEMv2

## Overview

HHEM-2.1-open is a major upgrade to [HHEM-1.0-Open](https://huggingface.co/vectara/hallucination_evaluation_model/tree/hhem-1.0-open) created by [Vectara](https://vectara.com) in November 2023. The HHEM model series are designed for detecting hallucinations in LLMs. They are particularly useful in the context of building retrieval-augmented-generation (RAG) applications where a set of facts is summarized by an LLM, and HHEM can be used to measure the extent to which this summary is factually consistent with the facts.

## Usage Examples

The model takes a list of pairs of (premise, hypothesis) as the input and returns a score between 0 and 1 for each parir where 0 means that the hypothesis is not evidenced at all by the premise and 1 means the hypothesis is fully supported by the premise.


```python
from transformers import HHEMv2Model
pairs = [ # Test data, List[Tuple[str, str]]
    ("The capital of France is Berlin.", "The capital of France is Paris."), # factual but hallucinated
    ('I am in California', 'I am in United States.'), # Consistent
    ('I am in United States', 'I am in California.'), # Hallucinated
    ("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."),
    ("A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a red bridge"),
    ("A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond man wearing a brown shirt is reading a book."),
    ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg.")
]
# Step 1: Load the model
model = HHEMv2Model.from_pretrained('vectara/hallucination_evaluation_model')
# Step 2: Use the model to predict
model.predict(pairs) # note the predict() method. Do not do model(pairs). 
# tensor([0.0111, 0.6474, 0.1290, 0.8969, 0.1846, 0.0050, 0.0543])
```

## HHEMv2Config

[[autodoc]] HHEMv2Config


## HHEMv2Model

[[autodoc]] HHEMv2Model
    - forward
    - predict

</pt>
<tf>

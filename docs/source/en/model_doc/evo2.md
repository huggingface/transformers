<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->


# Evo2

## Overview

The Evo2 model was proposed in [Genome modeling and design across all domains of life with Evo 2](https://www.biorxiv.org/content/10.1101/2024.02.27.582234v1) by Garyk Brixi, Matthew G. Durrant, Jerome Ku, Michael Poli, et al.
It is a biological foundation model trained on 9.3 trillion DNA base pairs from a curated genomic atlas spanning all domains of life.

The abstract from the paper is the following:

Evo 2 is a biological foundation model trained on 9.3 trillion DNA base pairs from a curated genomic atlas spanning all domains of life. The model features 7B and 40B parameter architectures and can process sequences up to 1 million base pairs at nucleotide-level resolution. It learns from DNA sequences alone to accurately predict the functional impacts of genetic variation, including noncoding pathogenic mutations and clinically significant BRCA1 variants, without task-specific finetuning. Mechanistic interpretability analyses reveal that Evo 2 autonomously learns a breadth of biological features, such as exon-intron boundaries, transcription factor binding sites, protein structural elements, and prophage genomic regions. Beyond its predictive capabilities, Evo 2 can generate mitochondrial, prokaryotic, and eukaryotic sequences at genome scale with greater naturalness and coherence than previous methods. Guiding Evo 2 via inference-time search enables controllable generation of epigenomic structure, for which the first inference-time scaling results in biology are demonstrated. The project makes Evo 2 fully open, including model parameters, training code, inference code, and the OpenGenome2 dataset, to accelerate the exploration and design of biological complexity.

Tips:

- Evo 2 is a genomic foundation model, meaning it is designed to process and generate DNA sequences.
- It uses the StripedHyena architecture, which combines attention with Hyena filters to handle long contexts efficiently.
- The model is trained on a massive dataset of 9.3 trillion base pairs.
- It can handle context lengths up to 1 million base pairs (though this specific implementation may be limited by available memory).

This model was contributed by [arcinstitute](https://huggingface.co/arcinstitute).
The original code can be found [here](https://github.com/ArcInstitute/evo2).
The model was converted to Hugging Face format by [McClain Thiel](mailto:mcclain.thiel@gmail.com).

## Usage examples

```python
from transformers import Evo2Config, Evo2ForCausalLM, Evo2Tokenizer

# Initialize model and tokenizer
config = Evo2Config()
model = Evo2ForCausalLM(config)
tokenizer = Evo2Tokenizer()

# Encode input DNA sequence
sequence = "ACGTACGT"
input_ids = tokenizer.encode(sequence, return_tensors="pt")

# Generate
output = model.generate(input_ids, max_length=20)
generated_sequence = tokenizer.decode(output[0])
print(generated_sequence)
```

## Evo2Config

[[autodoc]] Evo2Config

## Evo2ForCausalLM

[[autodoc]] Evo2ForCausalLM


## Evo2Model

[[autodoc]] Evo2Model
    - forward

## Evo2PreTrainedModel

[[autodoc]] Evo2PreTrainedModel
    - forward

<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# QuasarV4

## Overview

The QuasarV4 model is a transformer-based architecture with an innovative token temperature mechanism for natural language processing tasks. It was developed by SILX AI Labs and the model weights are available at [silx-ai/QuasarV4-600M-Transformer](https://huggingface.co/silx-ai/QuasarV4-600M-Transformer).

The transformer-based implementation is compatible with the Hugging Face Transformers library and can be used for causal language modeling tasks.

### Model Architecture

The QuasarV4 transformer-based model extends the standard transformer architecture with several innovative components:

1. **Token Temperature Mechanism**: A new approach that dynamically adjusts the importance of tokens based on their contextual significance. This mechanism allows the model to focus more computational resources on critical tokens and less on filler words, improving generation quality.

2. **Temperature Aggregation Plus**: A multi-layer network that processes token temperatures across the sequence to create a global temperature focus, enabling better long-range dependencies.

3. **Cross-token Temperature Attention**: Applies attention mechanisms specifically to temperature values, allowing tokens to influence each other's importance based on semantic relationships.

4. **DenseNet-style Residual Connections**: Incorporates connections from earlier layers (at 1/3 and 2/3 depth) directly to the output, creating richer gradient flows and enabling better information preservation through the network depth.

The architecture includes:
- Multi-headed attention with separate query, key, and value projections
- MLP blocks with SiLU activation functions
- Token temperature layers that dynamically adjust the importance of different tokens
- Global temperature scaling for context-aware processing



### Model Details

- **Model Type**: Causal Language Model
- **Implementation**: Transformer-based architecture with token temperature mechanisms
- **Language(s)**: English
- **License**: Apache 2.0
- **Resources for more information**:
  - [Model Card](https://huggingface.co/silx-ai/QuasarV4-600M-Transformer)

## Usage

The QuasarV4 model can be loaded using the `AutoModelForCausalLM` class:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("silx-ai/QuasarV4-600M-Transformer")

# Load model
model = AutoModelForCausalLM.from_pretrained("silx-ai/QuasarV4-600M-Transformer")

# Generate text
input_text = "The capital of France is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

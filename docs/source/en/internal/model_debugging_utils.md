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

# Model debugging toolboxes

This page lists all the debugging and model adding tools used by the library, as well as the utility functions it provides for it.

Most of those are only useful if you are adding new models in the library.


## Model addition debuggers


### Model addition debugger - context manager for model adders

This context manager is a power user tool intended for model adders.
It tracks all forward calls within a model forward and logs a slice of each input and output on a nested Json.
To note, this context manager enforces `torch.inference_mode()`.

### Rationale

Because when porting models to transformers, even from python to python, model adders often have to do a lot of manual operations, involving saving and loading tensors, comparing dtypes, etc. This small tool can hopefully shave off some time.

### Usage

Add this context manager as follows to debug a model:

```python
import torch
from PIL import Image
import requests
from transformers import LlavaProcessor, LlavaForConditionalGeneration
torch.random.manual_seed(673)

# load pretrained model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True)

# create random image input
random_image = Image.fromarray(torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy())

# prompt
prompt = "<image>Describe this image."

# process inputs
inputs = processor(text=prompt, images=random_image, return_tensors="pt")

# call forward method (not .generate!)
with model_addition_debugger_context(model, "optional_path_to_your_output_file.json"):
    output = model.forward(**inputs)

```


[[autodoc]] model_addition_debugger

[[autodoc]] model_addition_debugger_context

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

This page lists all the debugging and model adding tools used by the library, as well as the utility functions it
provides for it.

Most of those are only useful if you are adding new models in the library.


## Model addition debuggers


### Model addition debugger - context manager for model adders

This context manager is a power user tool intended for model adders. It tracks all forward calls within a model forward
and logs a slice of each input and output on a nested JSON. To note, this context manager enforces `torch.no_grad()`.

### Rationale

When porting models to transformers, even from python to python, model adders often have to do a lot of manual
operations, involving saving and loading tensors, comparing dtypes, etc. This small tool can hopefully shave off some
time.

### Usage

Add this context manager as follows to debug a model:

```python
import torch
from PIL import Image
import requests
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from transformers.model_debugging_utils import model_addition_debugger_context
torch.random.manual_seed(673)

# load pretrained model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id)

# create random image input
random_image = Image.fromarray(torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy())

# prompt
prompt = "<image>Describe this image."

# process inputs
inputs = processor(text=prompt, images=random_image, return_tensors="pt")

# call forward method (not .generate!)
with model_addition_debugger_context(
    model,
    debug_path="optional_path_to_your_directory",
    do_prune_layers=False # This will output ALL the layers of a model.
):
    output = model.forward(**inputs)

```


### Reading results

The debugger generates two files from the forward call, both with the same base name, but ending either with
`_SUMMARY.json` or with `_FULL_TENSORS.json`.

The first one will contain a summary of each module's _input_ and _output_ tensor values and shapes.

```json
{
  "module_path": "MolmoForConditionalGeneration",
  "inputs": {
    "args": [],
    "kwargs": {
      "input_ids": {
        "shape": "torch.Size([1, 589])",
        "dtype": "torch.int64"
      },
      "attention_mask": {
        "shape": "torch.Size([1, 589])",
        "dtype": "torch.int64"
      },
      "pixel_values": {
        "shape": "torch.Size([1, 5, 576, 588])",
        "dtype": "torch.float32",
        "mean": "tensor(-8.9514e-01, device='cuda:0')",
        "std": "tensor(9.2586e-01, device='cuda:0')",
        "min": "tensor(-1.7923e+00, device='cuda:0')",
        "max": "tensor(1.8899e+00, device='cuda:0')"
    }
  },
  "children": [
    {
      "module_path": "MolmoForConditionalGeneration.language_model.model.embed_tokens",
      "inputs": {
        "args": [
          {
            "shape": "torch.Size([1, 589])",
            "dtype": "torch.int64"
          }
        ]
      },
      "outputs": {
        "shape": "torch.Size([1, 589, 3584])",
        "dtype": "torch.float32",
        "mean": "tensor(6.5460e-06, device='cuda:0')",
        "std": "tensor(2.3807e-02, device='cuda:0')",
        "min": "tensor(-3.3398e-01, device='cuda:0')",
        "max": "tensor(3.9453e-01, device='cuda:0')"
      }
    },
    {
      "module_path": "MolmoForConditionalGeneration.vision_tower",
      "inputs": {
        "args": [
          {
            "shape": "torch.Size([5, 1, 576, 588])",
            "dtype": "torch.float32",
            "mean": "tensor(-8.9514e-01, device='cuda:0')",
            "std": "tensor(9.2586e-01, device='cuda:0')",
            "min": "tensor(-1.7923e+00, device='cuda:0')",
            "max": "tensor(1.8899e+00, device='cuda:0')"
          }
        ],
        "kwargs": {
          "output_hidden_states": "True"
        }
      },
      "children": [
        { ... and so on
```

The `_FULL_TENSORS.json` file will display a full view of all tensors, which is useful for comparing two files.

```json
      "pixel_values": {
        "shape": "torch.Size([1, 5, 576, 588])",
        "dtype": "torch.float32",
        "value": [
          "tensor([[[[-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          ...,",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00]],",
          "",
          "         [[-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          ...,",
          "          [-1.4857e+00, -1.4820e+00, -1.2100e+00,  ..., -6.0979e-01, -5.9650e-01, -3.8527e-01],",
          "          [-1.6755e+00, -1.7221e+00, -1.4518e+00,  ..., -7.5577e-01, -7.4658e-01, -5.5592e-01],",
          "          [-7.9957e-01, -8.2162e-01, -5.7014e-01,  ..., -1.3689e+00, -1.3169e+00, -1.0678e+00]],",
          "",
          "         [[-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          ...,",
          "          [-3.0322e-01, -5.0645e-01, -5.8436e-01,  ..., -6.2439e-01, -7.9160e-01, -8.1188e-01],",
          "          [-4.4921e-01, -6.5653e-01, -7.2656e-01,  ..., -3.4702e-01, -5.2146e-01, -5.1326e-01],",
          "          [-3.4702e-01, -5.3647e-01, -5.4170e-01,  ..., -1.0915e+00, -1.1968e+00, -1.0252e+00]],",
          "",
          "         [[-1.1207e+00, -1.2718e+00, -1.0678e+00,  ..., 1.2013e-01, -1.3126e-01, -1.7197e-01],",
          "          [-6.9738e-01, -9.1166e-01, -8.5454e-01,  ..., -5.5050e-02, -2.8134e-01, -4.2793e-01],",
          "          [-3.4702e-01, -5.5148e-01, -5.8436e-01,  ..., 1.9312e-01, -8.6235e-02, -2.1463e-01],",
          "          ...,",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00]],",
          "",
          "         [[-1.0039e+00, -9.5669e-01, -6.5546e-01,  ..., -1.4711e+00, -1.4219e+00, -1.1389e+00],",
          "          [-1.0039e+00, -9.5669e-01, -6.5546e-01,  ..., -1.7193e+00, -1.6771e+00, -1.4091e+00],",
          "          [-1.6317e+00, -1.6020e+00, -1.2669e+00,  ..., -1.2667e+00, -1.2268e+00, -8.9720e-01],",
          "          ...,",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00],",
          "          [-1.7923e+00, -1.7521e+00, -1.4802e+00,  ..., -1.7923e+00, -1.7521e+00, -1.4802e+00]]]], device='cuda:0')"
        ],
        "mean": "tensor(-8.9514e-01, device='cuda:0')",
        "std": "tensor(9.2586e-01, device='cuda:0')",
        "min": "tensor(-1.7923e+00, device='cuda:0')",
        "max": "tensor(1.8899e+00, device='cuda:0')"
      },
```

#### Saving tensors to disk

Some model adders may benefit from logging full tensor values to disk to support, for example, numerical analysis
across implementations.

Set `use_repr=False` to write tensors to disk using [SafeTensors](https://huggingface.co/docs/safetensors/en/index).

```python
with model_addition_debugger_context(
    model,
    debug_path="optional_path_to_your_directory",
    do_prune_layers=False,
    use_repr=False,   # Defaults to True
):
    output = model.forward(**inputs)
```

When using `use_repr=False`, tensors are written to the same disk location as the `_SUMMARY.json` and
`_FULL_TENSORS.json` files. The `value` property of entries in the `_FULL_TENSORS.json` file will contain a relative
path reference to the associated `.safetensors` file. Each tensor is written to its own file as the `data` property of
the state dictionary. File names are constructed using the `module_path` as a prefix with a few possible postfixes that
are built recursively.

*   Module inputs are denoted with the `_inputs` and outputs by `_outputs`.
*   `list` and `tuple` instances, such as `args` or function return values, will be postfixed with `_{index}`.
*   `dict` instances will be postfixed with `_{key}`.

### Comparing between implementations

Once the forward passes of two models have been traced by the debugger, one can compare the `json` output files. See
below: we can see slight differences between these two implementations' key projection layer. Inputs are mostly
identical, but not quite. Looking through the file differences makes it easier to pinpoint which layer is wrong.


![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/files_difference_debugging.png)


### Limitations and scope

This feature will only work for torch-based models, and would require more work and case-by-case approach for say
`jax`-based models that are usually compiled. Models relying heavily on external kernel calls may work, but trace will
probably miss some things. Regardless, any python implementation that aims at mimicking another implementation can be
traced once instead of reran N times with breakpoints.

If you pass `do_prune_layers=False` to your model debugger, ALL the layers will be outputted to `json`. Else, only the
first and last layer will be shown. This is useful when some layers (typically cross-attention) appear only after N
layers.

[[autodoc]] model_addition_debugger_context

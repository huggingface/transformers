<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Instantiate a big model

A barrier to accessing very large pretrained models is the amount of memory required. When loading a pretrained PyTorch model, you usually:

1. Create a model with random weights.
2. Load your pretrained weights.
3. Put those pretrained weights in the model.

The first two steps both require a full version of the model in memory and if the model weighs several GBs, you may not have enough memory for two copies of it. This problem is amplified in distributed training environments because each process loads a pretrained model and stores two copies in memory.

> [!TIP]
> The randomly created model is initialized with "empty" tensors, which take space in memory without filling it. The random values are whatever was in this chunk of memory at the time. To improve loading speed, the [`_fast_init`](https://github.com/huggingface/transformers/blob/c9f6e5e35156e068b227dd9b15521767f6afd4d2/src/transformers/modeling_utils.py#L2710) parameter is set to `True` by default to skip the random initialization for all weights that are correctly loaded.

This guide will show you how Transformers can help you load large pretrained models despite their memory requirements.

## Sharded checkpoints

From Transformers v4.18.0, a checkpoint larger than 10GB is automatically sharded by the [`~PreTrainedModel.save_pretrained`] method. It is split into several smaller partial checkpoints and creates an index file that maps parameter names to the files they're stored in.

The maximum shard size is controlled with the `max_shard_size` parameter, but by default it is 5GB, because it is easier to run on free-tier GPU instances without running out of memory.

For example, let's shard [BioMistral/BioMistral-7B](https://hf.co/BioMistral/BioMistral-7B).

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'generation_config.json', 'model-00001-of-00006.safetensors', 'model-00002-of-00006.safetensors', 'model-00003-of-00006.safetensors', 'model-00004-of-00006.safetensors', 'model-00005-of-00006.safetensors', 'model-00006-of-00006.safetensors', 'model.safetensors.index.json']
```

The sharded checkpoint is reloaded with the [`~PreTrainedModel.from_pretrained`] method.

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

The main advantage of sharded checkpoints for big models is that each shard is loaded after the previous one, which caps the memory usage to only the model size and the largest shard size.

You could also directly load a sharded checkpoint inside a model without the [`~PreTrainedModel.from_pretrained`] method (similar to PyTorch's `load_state_dict()` method for a full checkpoint). In this case, use the [`~modeling_utils.load_sharded_checkpoint`] method.

```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     load_sharded_checkpoint(model, tmp_dir)
```

### Shard metadata

The index file determines which keys are in the checkpoint and where the corresponding weights are stored. This file is loaded like any other JSON file and you can get a dictionary from it.

```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

The `metadata` key provides the total model size.

```py
>>> index["metadata"]
{'total_size': 28966928384}
```

The `weight_map` key maps each parameter name (typically `state_dict` in a PyTorch model) to the shard it's stored in.

```py
>>> index["weight_map"]
{'lm_head.weight': 'model-00006-of-00006.safetensors',
 'model.embed_tokens.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.input_layernorm.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.mlp.down_proj.weight': 'model-00001-of-00006.safetensors',
 ...
}
```

## Accelerate's Big Model Inference

> [!TIP]
> Make sure you have Accelerate v0.9.0 or later and PyTorch v1.9.0 or later installed.

From Transformers v4.20.0, the [`~PreTrainedModel.from_pretrained`] method is supercharged with Accelerate's [Big Model Inference](https://hf.co/docs/accelerate/usage_guides/big_modeling) feature to efficiently handle really big models! Big Model Inference creates a *model skeleton* on PyTorch's [**meta**](https://pytorch.org/docs/main/meta.html) device. The randomly initialized parameters are only created when the pretrained weights are loaded. This way, you aren't keeping two copies of the model in memory at the same time (one for the randomly initialized model and one for the pretrained weights), and the maximum memory consumed is only the full model size.

To enable Big Model Inference in Transformers, set `low_cpu_mem_usage=True` in the [`~PreTrainedModel.from_pretrained`] method.

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", low_cpu_mem_usage=True)
```

Accelerate automatically dispatches the model weights across all available devices, starting with the fastest device (GPU) first and then offloading to the slower devices (CPU and even hard drive). This is enabled by setting `device_map="auto"` in the [`~PreTrainedModel.from_pretrained`] method. When you pass the `device_map` parameter, `low_cpu_mem_usage` is automatically set to `True` so you don't need to specify it.

```py
from transformers import AutoModelForCausalLM

# these loading methods are equivalent
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto", low_cpu_mem_usage=True)
```

You can also write your own `device_map` by mapping each layer to a device. It should map all model parameters to a device, but you don't have to detail where all the submodules of a layer go if the entire layer is on the same device.

```python
device_map = {"model.layers.1": 0, "model.layers.14": 1, "model.layers.31": "cpu", "lm_head": "disk"}
```

Access `hf_device_map` attribute to see how Accelerate split the model across devices.

```py
gemma.hf_device_map
```

```python out
{'model.embed_tokens': 0,
 'model.layers.0': 0,
 'model.layers.1': 0,
 'model.layers.2': 0,
 'model.layers.3': 0,
 'model.layers.4': 0,
 'model.layers.5': 0,
 'model.layers.6': 0,
 'model.layers.7': 0,
 'model.layers.8': 0,
 'model.layers.9': 0,
 'model.layers.10': 0,
 'model.layers.11': 0,
 'model.layers.12': 0,
 'model.layers.13': 0,
 'model.layers.14': 'cpu',
 'model.layers.15': 'cpu',
 'model.layers.16': 'cpu',
 'model.layers.17': 'cpu',
 'model.layers.18': 'cpu',
 'model.layers.19': 'cpu',
 'model.layers.20': 'cpu',
 'model.layers.21': 'cpu',
 'model.layers.22': 'cpu',
 'model.layers.23': 'cpu',
 'model.layers.24': 'cpu',
 'model.layers.25': 'cpu',
 'model.layers.26': 'cpu',
 'model.layers.27': 'cpu',
 'model.layers.28': 'cpu',
 'model.layers.29': 'cpu',
 'model.layers.30': 'cpu',
 'model.layers.31': 'cpu',
 'model.norm': 'cpu',
 'lm_head': 'cpu'}
```

## Model data type

PyTorch model weights are normally instantiated as torch.float32 and it can be an issue if you try to load a model as a different data type. For example, you'd need twice as much memory to load the weights in torch.float32 and then again to load them in your desired data type, like torch.float16.

> [!WARNING]
> Due to how PyTorch is designed, the `torch_dtype` parameter only supports floating data types.

To avoid wasting memory like this, explicitly set the `torch_dtype` parameter to the desired data type or set `torch_dtype="auto"` to load the weights with the most optimal memory pattern (the data type is automatically derived from the model weights).

<hfoptions id="dtype">
<hfoption id="specific dtype">

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype=torch.float16)
```

</hfoption>
<hfoption id="auto dtype">

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype="auto")
```

</hfoption>
</hfoptions>

You can also set the data type to use for models instantiated from scratch.

```python
import torch
from transformers import AutoConfig, AutoModel

my_config = AutoConfig.from_pretrained("google/gemma-2b", torch_dtype=torch.float16)
model = AutoModel.from_config(my_config)
```

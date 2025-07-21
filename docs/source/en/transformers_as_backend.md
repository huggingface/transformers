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

# Using Transformers as a Backend for Inference Servers

## What’s an inference backend, and why should you care?

An inference backend is the part of your system that actually runs the model and returns outputs. Think of it as the engine behind APIs, chatbots, and anything else using generative Language Model in production.

Instead of each server implementing its own model logic, many of them now rely on Transformers to do the heavy lifting. That means if your model works in 🤗 Transformers, it can also work **out of the box** in any inference server that uses Transformers as a backend.

No need to duplicate code. You write your model once, and it works across multiple inference engines with consistent behavior and makes general maintaining much easier. No extra effort to get server support, as long as your model follows Transformers recommended standards as outlined in below sections.

You can find the list of currently supported backends below. This list is still growing, and if there's a backend you'd like to see supported, feel free to open an issue.

### vLLM

https://docs.vllm.ai/en/latest/models/supported_models.html#transformers

* How vLLM integrates with Transformers backend
* Any known caveats or limitations
* Example usage / deployment snippet

[vLLM](https://github.com/vllm-project/vllm) is a high-performance inference engine optimized for serving LLMs at scale. It supports many models implemented in the 🤗 Transformers library through its transformers backend, including all decoder-only LLMs and several vision-language models. For VLMs, currently only image inputs are supported, supporting video inputs is planned. 

vLLM automatically selects the best backend. If the model isn’t natively supported, it falls back to Transformers. You can also force the use of the Transformers backend by setting `model_impl="transformers"`.

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.2-1B", model_impl="transformers")
```

Refer to the official [vLLM docs](https://docs.vllm.ai/en/latest/models/transformers_backend.html) to see more usage examples and tips with Transformers backend.


### SGLang

[SGLang](https://github.com/InternLM/sglang) is a high-performance, OpenAI-compatible server and runtime designed for chat-based LLMs. It offers fast inference, role-based conversation handling, and support for custom pipelines, making it great for building real-world LLM apps. With Transformers as a backend you can run any compatible model without waiting for native support, including custom and Hub-hosted models.

SGLang will automatically fall back to the Transformers backend if a model isn’t natively supported. You can also set it explicitly:

```python
import sglang as sgl

llm = sgl.Engine("meta-llama/Llama-3.2-1B-Instruct", impl="transformers")
print(llm.generate(["The capital of France is"], {"max_new_tokens": 20})[0])
```

Or launch as an OpenAI-compatible server:

```bash
python3 -m sglang.launch_server \
  --model-path kyutai/helium-1-preview-2b \
  --impl transformers \
  --host 0.0.0.0 \
  --port 30000
```

For more, refer to [SGLang's official docs](https://github.com/InternLM/sglang).

## TGI

[TGI](https://huggingface.co/docs/text-generation-inference/index) can serve models that aren't [natively implemented](https://huggingface.co/docs/text-generation-inference/supported_models) by falling back on the Transformers implementation of the model. Some of TGIs high-performance features aren't available in the Transformers implementation, but other features like continuous batching and streaming are still supported.

> [!TIP]
> Refer to the [Non-core model serving](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models) guide for more details.

Serve a Transformers implementation the same way you'd serve a TGI model.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2
```

Add `--trust-remote_code` to the command to serve a custom Transformers model.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <CUSTOM_MODEL_ID> --trust-remote-code
```

## Making Your Model Compatible Once and For All Backends

To make your custom model work out of the box with backends like vLLM and SGLang, it needs to follow some conventions, mainly to ensure smooth integration and optimized inference.


### General Requirements

For a model to be supported via the Transformers backend:

1. It must be Transformers-compatible following [custom model guidelines](https://huggingface.co/docs/transformers/model_sharing#custom-code). That means the model has to be supported in the core library or with custom code in the Hub. You can contribute your model to the core library following [these rules](https://huggingface.co/docs/transformers/en/add_new_model) or available as a [custom model](https://huggingface.co/docs/transformers/v4.53.2/en/custom_models) in the Hub. Make sure that the model has a valid `config.json` in its directory and a valid `auto_map` field pointing to the model class in the config.

2. The model's attention module needs to be backend configurable to benefit from performance features of various inference servers. For that the model needs to support the new [AttentionInterface](https://huggingface.co/docs/transformers/en/attention_interface) which allows anyone to register their custom and optimized attention functions to be used in the model. All you have to do is to use `ALL_ATTENTION_FUNCTIONS` when defining the attention layer and propagate `**kwargs` all the way through your base `MyModel` class to the attention layers, because some custom attention fucntions require a new arg in its `forward`. And don't forget to set `_supports_attention_backend = True` in you `MyPreTrainedModel` class.

<details>
<summary>modeling_my_model.py</summary>

```python

from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):

    def forward(self, hidden_states, **kwargs):
        ...
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )
        ...

class MyModel(PreTrainedModel):
    _supports_attention_backend = True
```

</details>

3. Optionally, if you want the model to support tensor parallel and/or pipeline parallel features, you can the following keys in the config file: 
    * `base_model_tp_plan` for [tensor parallelism](https://huggingface.co/docs/transformers/perf_infer_gpu_multi) - a dict that maps fully qualified layer name patterns to tensor parallel styles (currently only "colwise" and "rowwise" are supported).
    * `base_model_pp_plan` for pipeline parallelism - a dict that maps direct child layer names to tuples of lists of strs.The list in the first element of the tuple contains the names of the input arguments. The list in the last element of the tuple contains the names of the variables the layer outputs to in your modeling code

<details>
<summary>configuration_my_model.py</summary>

```python

from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```
</details>


### Multimodal Requirements

To enable seamless support for vision-language models in inference servers, your model needs to follow a few extra conventions on top of the general ones. These rules ensure that your model integrates properly with multimodal data handling.


1. Your model must have a base class (e.g. MyMultimodalModel) that handles multimodal fusion without an LM head and a separate generative class that adds the language modeling head and inherits from `GenerationMixin`. The base model must implement a get_image_features() method that takes in image pixel values and returns the vision tower’s encoded outputs. These will later be merged with language embeddings. The shape of returned features should match the number of input images. If your encoder returns variable-length outputs (e.g., patch-based), return a list of 2D tensors of shape `(image_seq_len, image_dim)` - one per image.

 
<details>
<summary>modeling_my_multimodal_model.py</summary>

```python
from transformers.generation import GenerationMixin

class MyMultimodalModel(MyMultimodalPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multimodal_projection = nn.Linear(vision_dim, text_dim)
    
    def get_image_features(self, pixel_values):
        return self.vision_tower(pixel_values).last_hidden_states
    
    def forward(self, inpit_ids, pixel_values, **kwargs):
        # process your inputs
        return MyModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            image_hidden_states=image_features,
            [...]
        )

class MyMultimodalModelForConditionalGeneration(MyMultimodalPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyMultimodalModel(config)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
```
</details>


2. Your config must be nested, with the minimal set of the following fields:

    * text_config: decoder config
    * vision_config: vision encoder config
    * image_token_id: ID of the image placeholder token used in input to indicate image position

<details>
<summary>configuration_my_multimodal_model.py</summary>

```python
from transformers import PretrainedConfig

class MyMultimodalConfig(PretrainedConfig):
    def __init__(self, text_config, vision_config, image_token_id, **kwargs):
        self.image_token_id = image_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)
```
</details>

3. The model's processing class must have `self.image_token` and `self.image_token_ids` attributes. These are the placeholder tokens used to indicate image positions in the input. Note that it is the same tokens used by users when constructing a input prompt and the model when merging image features with text features. Additionally, the class needs a `self._get_num_multimodal_tokens()` helper method that computes the number of placeholder tokens needed for multimodal inputs with given sizes and returns a `MultiModalData` object. Note, that placeholder for row and column tokens are not counted as image placholders, only tokens tha will be replaced by image features are considered.

Finally, when `return_mm_token_type_ids=True`, the class must return `mm_token_type_ids` indicating whether each position is a text token (`0`) or image placeholder token (`1`). Each image's token type IDs must be contiguous with no breaks between consecutive ones.

<details>
<summary>processing_my_multimodal_model.py</summary>

```python
class MyMultimodalProcessor(ProcessorMixin):

    def __call__(self, images=None, text=None, **kwargs):
        if return_mm_token_type_ids:
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        vision_data = {}
        if image_sizes is not None:
            num_image_tokens = [256] * len(image_sizes) # 256 placeholder tokens for each image always
            num_image_patches = [1] * len(image_sizes) # no patching, thus each image is processed as a single base image
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)
```
</details>

## Additional Resources

Refer to the blog posts below to know more about how each inference server was integrated.

* [Blog Post in vLLM](https://blog.vllm.ai/2025/04/11/transformers-backend.html)
* [Blog Post in SGLang](https://huggingface.co/blog/transformers-backend-sglang)

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

# Inference server backends

Transformers' models are compatible with different inference servers like vLLM and SGLang. Instead of implementing a model for each inference server, you only need one model, which can be plugged into any inference server. It simplifies maintenance and makes it easy for users to use different inference servers for different use cases.

With Transformers as a backend, you can also serve any model - including custom and Hub-hosted models - without waiting for native support.

This guide shows how to use Transformers' models as a backend to some popular inference servers and how to build a model that supports all inference servers.

## vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-performance inference engine optimized for serving LLMs at scale. It supports many Transformers' models, including all decoder-only LLMs and several vision-language models (VLMs). VLMs currently support image inputs only, with video support planned.

vLLM automatically selects the best backend, and if a model isn't natively supported, it falls back to the Transformers model. To explicitly use a Transformers' model, set `model_impl="transformers"`.

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.2-1B", model_impl="transformers")
```

Add `--model-impl transformers` to `vllm serve` to launch a server with a Transformers' model.

```bash
vllm serve meta-llama/Llama-3.2-1B \
    --task generate \
    --model-impl transformers
```

Refer to the [vLLM docs](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers) for more usage examples and tips on using a Transformers as the backend.

## SGLang

[SGLang](https://github.com/InternLM/sglang) is a high-performance, OpenAI-compatible server and runtime designed for chat-based LLMs. It offers fast inference, role-based conversation handling, and support for custom pipelines, making it great for building real-world LLM apps.

SGLang automatically falls back to the Transformers backend if a model isn't natively supported. To explicitly use a Transformers' model, set `impl="transformers"`.

```python
import sglang as sgl

llm = sgl.Engine("meta-llama/Llama-3.2-1B-Instruct", impl="transformers")
print(llm.generate(["The capital of France is"], {"max_new_tokens": 20})[0])
```

Add `impl transformers` to `sglang.launch_server` to launch a server with a Transformers' model.

```bash
python3 -m sglang.launch_server \
  --model-path kyutai/helium-1-preview-2b \
  --impl transformers \
  --host 0.0.0.0 \
  --port 30000
```

Refer to the [SGLang docs](https://docs.sglang.ai/supported_models/transformers_fallback.html) for more usage examples and tips on using a Transformers as the backend.

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

## Building a compatible model backend

To ensure a model is compatible as a backend to any inference server, make sure it is compatible with Transformers and supports the [AttentionInterface](./attention_interface) class.

1. A model must be Transformers-compatible following the model [contribution guidelines](./add_new_model) or the [custom model contribution guidelines](./custom_models). Make sure the model has a valid `config.json` in its directory and a valid `auto_map` field pointing to the model class in the config.

2. A model's attentions needs to be configurable with the [AttentionInterface](./attention_interface) to allow custom and optimized attention functions. This is important for enabling the performance features of the different inference servers.
   Use `ALL_ATTENTION_FUNCTIONS` when defining the attention layer and propagate `**kwargs**` from the base `MyModel` class to the attention layers. Set `_supports_attention_backend` to `True` in [`PreTrainedModel`]. Expand the code below for an example.

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

3. This step is optional, but if you want to support tensor parallel and/or pipeline parallel features, add the following keys to the config.
    * `base_model_tp_plan` enables [tensor parallelism](./perf_infer_gpu_multi) by mapping fully qualified layer name patterns to tensor parallel styles. Only the `"colwise"` and `"rowwise"` partitioning strategies are currently supported.
    * `base_model_pp_plan` enables pipeline parallelism by mapping direct child layer names to tuples of lists of strings. The list in the first element of the tuple contains the names of the input arguments. The list in the last element of the tuple contains the names of the variables the layer outputs to in the modeling code.

 Expand the code below for an example.

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

### Multimodal models

For multimodal models, you need to include a few more changes on top of the general recommendations. These rules ensure that your model integrates properly with multimodal data.

1. A multimodal model requires a base `MyMultiModalModel` class to handle multimodal fusion without a language modeling head and a separate generative class that adds a head.

    The base model needs to implement the `get_image_features()` method to accept image pixel values and return encoded outputs. These are later merged with the language embeddings and don't require any postprocessing. The shape of the returned features must match the number of input images. If a vision encoder returns variable-length outputs (patch-based), return a list of 2D tensors of size `(image_seq_len, image_dim)` for each image.

Expand the code below for an example.

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
    
    def forward(self, input_ids, pixel_values, **kwargs):
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

2. A multimodal model config must be nested with the following fields.
    * text_config: decoder language model config
    * vision_config: vision encoder config
    * image_token_id: ID of the image placeholder token used in the input to indicate image position

3. A multimodal model's processing class must have the `self.image_token` and `self.image_token_ids` attributes. These are placeholder tokens used to indicate image positions in the input. The placeholder token is the same token used in the input prompt and to mask scatter image features.

   The processing class also needs `self._get_num_multimodal_tokens` method to compute the number of placeholder tokens needed for multimodal inputs with given sizes and to return a [`MultiModalData`] object. The placeholder for row and column tokens don't count as image placeholders. Only the tokens that are actually replaced by image features are computed.

Finally, when `return_mm_token_type_ids=True`, the class has to return `mm_token_type_ids` to indicate whether each position is a text token (`0`) or image placeholder token (`1`). Each image's token type IDs must be contiguous with no breaks between consecutive ones.

Expand the code below for an example.

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

## Resources

* Read the [Transformers backend integration in vLLM](https://blog.vllm.ai/2025/04/11/transformers-backend.html) blog post for more details about the Transformers backend in vLLM.
* Read the [Transformers backend integration in SGLang](https://huggingface.co/blog/transformers-backend-sglang) blog post for more details about the Transformers backend in SGLang.

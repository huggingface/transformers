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

# Inference backends

Optimized inference engines like vLLM and SGLang have their own native model implementations. They also support Transformers models on the backend. Transformers models let you avoid reimplementing models for each inference engine. Use models that aren't natively supported by either inference engine. Choose the inference engine that fits your deployment.

## vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference engine for serving LLMs at scale.

Set `model_impl="transformers"` to use a Transformers model for offline inference.

```py
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.2-1B", model_impl="transformers")
print(llm.generate(["The capital of France is"]))
```

Add `--model-impl transformers` to the `vllm serve` command for online serving.

```bash
vllm serve meta-llama/Llama-3.2-1B \
    --task generate \
    --model-impl transformers
```

Refer to the [vLLM docs](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers) for more usage examples and tips for using a Transformers model as the backend.

## SGLang

[SGLang](https://docs.sglang.ai) is a low-latency, high-throughput inference engine with a frontend language for building and running LLM applications like agentic workflows.

Set `impl="transformers"` to use a Transformers model for offline inference.

```py
import sglang as sgl

llm = sgl.Engine("meta-llama/Llama-3.2-1B-Instruct", impl="transformers")
print(llm.generate(["The capital of France is"], {"max_new_tokens": 20})[0])
```

Add `impl transformers` to the `sglang.launch_server` command for online serving.

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --impl transformers \
  --host 0.0.0.0 \
  --port 30000
```

Refer to the [SGLang docs](https://docs.sglang.ai/supported_models/transformers_fallback.html) for more usage examples and tips for using a Transformers model as the backend.

## TGI

[Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) is a Rust-based production inference server. It includes continuous batching, tensor parallelism, FlashAttention, and deployment observability.

Run the Docker command below for online serving.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id meta-llama/Llama-3.2-1B-Instruct
```

Add `--trust-remote_code` to serve a custom Transformers model from the Hub.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <CUSTOM_MODEL_ID> --trust-remote-code
```

Refer to the [Non-core model serving](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models) guide for more details.

## Building a compatible model backend

A model must be Transformers-compatible and support the [`AttentionInterface`] class to be used as a backend to any inference engine.

1. Follow the model [contribution guidelines](./add_new_model) or the [custom model contribution guidelines](./custom_models). The model must have a valid `config.json` in its directory and a valid `auto_map` field pointing to the model class in the config.

2. Use the [`AttentionInterface`] class for custom and optimized attention functions. This enables the performance features of each inference engine. 

   Use `ALL_ATTENTION_FUNCTIONS` when defining the attention layer and propagate `**kwargs**` from the base `MyModel` class to the attention layers. Set `_supports_attention_backend` to `True` in [`PreTrainedModel`].
   
   Expand the code below for an example.

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

3. Enable optional tensor or pipeline parallelism by adding the following keys to [`PreTrainedConfig`].

    * `base_model_tp_plan` enables [tensor parallelism](./perf_infer_gpu_multi) by mapping fully qualified layer name patterns to tensor parallel styles. Supports only the `"colwise"` and `"rowwise"` partitioning strategies.
    * `base_model_pp_plan` enables pipeline parallelism by mapping direct child layer names to tuples of lists of strings. The first element of the tuple contains the names of the input arguments. The last element contains the variable names of the layer outputs in the modeling code.

    Expand the code below for an example.

    <details>
    <summary>configuration_my_model.py</summary>

    ```python

    from transformers import PreTrainedConfig

    class MyConfig(PreTrainedConfig):
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

Multimodal models like vision language models require additional changes beyond the [vision language model contribution checklist](./contributing#vision-language-model-contribution-checklist). These changes ensure multimodal inputs are properly processed.

1. The [`ProcessorMixin`] class must include the `self.image_token` and `self.image_token_ids` attributes. These placeholder tokens indicate image positions in the input. The same token appears in the input prompt for images and in the model code to scatter image features.

2. The [`ProcessorMixin`] class must include a `self._get_num_multimodal_tokens` method. This method computes the number of placeholder tokens required for multimodal inputs with given sizes. It returns a [`MultiModalData`] object. Placeholders between `<image>` tokens, such as row or column tokens, don't count as image placeholders. Count only tokens replaced by image features later in the modeling code.

3. The [`ProcessorMixin`] class must check the value of `return_mm_token_type_ids` and return `mm_token_type_ids`. This indicates whether each position is a text token (`0`), image placeholder token (`1`), or a video placeholder token (`2`). Multimodal token type id sequences must be contiguous with no breaks between consecutive tokens. Treat special tokens for beginning, ending, row, and column tokens as placeholders.

Expand the code below for an example.

<details>
<summary>modeling_my_multimodal_model.py</summary>

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

* Read the [Transformers backend integration in vLLM](https://blog.vllm.ai/2025/04/11/transformers-backend.html) blog post for more details.
* Read the [Transformers backend integration in SGLang](https://huggingface.co/blog/transformers-backend-sglang) blog post for more details.

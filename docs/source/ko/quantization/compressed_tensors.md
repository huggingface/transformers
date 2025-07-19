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

# compressed-tensors[[compressed-tensors]]

[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)는 [safetensors](https://github.com/huggingface/safetensors) 파일을 압축된 텐서 데이터 타입으로 확장해서, dense, int-quantized (int8), float-quantized (fp8), pack-quantized (int32로 패킹된 int4나 int8 가중치 양자화) 같은 다양한 양자화와 sparse 형식을 저장하고 불러올 수 있는 통합 체크포인트 형식을 제공해 줍니다.

compressed-tensors는 [PEFT](https://huggingface.co/docs/peft)를 사용한 파인튜닝을 지원하고, 다음과 같은 기능들도 함께 제공합니다.

- fp8, int4, int8 가중치와 활성화 정밀도를 지원합니다.
- [tensor, channel, group, block, token](https://github.com/neuralmagic/compressed-tensors/blob/83b2e7a969d70606421a76b9a3d112646077c8de/src/compressed_tensors/quantization/quant_args.py#L43-L52)별로 양자화 스케일과 영점 전략을 설정할 수 있습니다.
- 동적 토큰별 활성화 양자화 (또는 정적 전략)를 사용할 수 있습니다.
- 가중치 sparsity (구조화되지 않은 형태나 2:4 같은 반구조화 형태)를 양자화와 함께 사용해서 더욱 강력한 압축이 가능합니다.
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 모듈뿐만 아니라 어떤 모듈이든 양자화할 수 있습니다.
- 이름이나 클래스로 특정 모듈만 선택해서 지원합니다.

[PyPI](https://pypi.org/project/compressed-tensors)에서 compressed-tensors를 설치해서 최신 안정 버전을 받으시거나 (추천), 소스에서 설치해서 최신 기능을 사용하실 수 있습니다.

<hfoptions id="install">
<hfoption id="PyPI">

```bash
pip install compressed-tensors
```

</hfoption>
<hfoption id="source code">

```bash
git clone https://github.com/neuralmagic/compressed-tensors
cd compressed-tensors
pip install -e .
```

</hfoption>
</hfoptions>

Hugging Face Hub에서 호환되는 모델을 찾으시려면 compressed-tensors [태그](https://huggingface.co/models?other=compressed-tensors)로 검색해 보세요.

현재는 이미 양자화된 모델만 불러올 수 있고, 모델을 불러온 후에는 저장할 수 없습니다. 모델을 compressed-tensors 형식으로 양자화하고 싶으시다면 [llm-compressor](https://github.com/vllm-project/llm-compressor)를 참고해 주세요. 또는 모델을 따로 만들어서 compressed-tensors 설정으로 직렬화하실 수도 있습니다.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf", device_map="auto")

# 메모리 사용량 측정하기
mem_params = sum([param.nelement()*param.element_size() for param in ct_model.parameters()])
print(f"{mem_params/2**30:.4f} GB")
# 8.4575 GB
```

## 모델 체크포인트[[model-checkpoint]]

compressed-tensor 모델은 설정 항목을 통해 정의됩니다. 다음 예시는 [nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf/blob/main/config.json) `config.json` 파일에서 가져왔습니다.

압축하는 동안과 압축 후에 유연하게 표현할 수 있도록 많은 항목들이 있지만, 로딩과 추론을 위한 항목들은 주요 항목 몇 개만 보시면 충분합니다.

```yaml
"quantization_config": {
  "config_groups": {
    "group_0": {
      "input_activations": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      },
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      }
    }
  },
  "format": "naive-quantized",
  "ignore": ["lm_head"],
  "quant_method": "compressed-tensors",
  "quantization_status": "frozen"
},
```

설정 파일은 설정 그룹(`group_0`)의 양자화를 지정하는데, 정적 텐서별 전략으로 fp8에 가중치와 활성화 양자화를 포함합니다. `lm_head` 모듈은 `ignore` 키에 나와 있듯이 양자화하지 않습니다.

모델 가중치를 더 자세히 보고 싶으시다면, 모델 카드의 [safetensors 뷰어](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf?show_file_info=model.safetensors.index.json)를 사용해서 모든 [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 모듈의 양자화된 가중치, 입력 스케일, 가중치 스케일을 확인하실 수 있습니다.

| 텐서 | 형태 |	정밀도 |
| ------- | ----- | --------- |
model.layers.0.input_layernorm.weight	| [4 096]	| BF16 
model.layers.0.mlp.down_proj.input_scale	| [1]	| BF16 
model.layers.0.mlp.down_proj.weight	| [4 096, 14 336] |	F8_E4M3 
model.layers.0.mlp.down_proj.weight_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.input_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.weight	| [14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.gate_proj.weight_scale	| [1] |	BF16 
model.layers.0.mlp.up_proj.input_scale|	[1]	|BF16 
model.layers.0.mlp.up_proj.weight |	[14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.up_proj.weight_scale | [1]	| BF16 
model.layers.0.post_attention_layernorm.weight |	[4 096]	|BF16 
model.layers.0.self_attn.k_proj.input_scale |	[1]	|  BF16
model.layers.0.self_attn.k_proj.weight |	[1 024, 4 096]|	F8_E4M3
model.layers.0.self_attn.k_proj.weight_scale |[1]	| BF16 
model.layers.0.self_attn.o_proj.input_scale	| [1]	| BF16
model.layers.0.self_attn.o_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.o_proj.weight_scale | [1]	| BF16 
model.layers.0.self_attn.q_proj.input_scale	| [1]	| BF16 
model.layers.0.self_attn.q_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.q_proj.weight_scale |	[1] | BF16 
model.layers.0.self_attn.v_proj.input_scale	| [1] | BF16 
model.layers.0.self_attn.v_proj.weight |	[1 024, 4 096]	| F8_E4M3 
model.layers.0.self_attn.v_proj.weight_scale |	[1] |	BF16 

[`~quantizers.HFQuantizer`] 통합으로 compressed-tensors 모델을 불러올 때, 양자화 설정에 지정된 모든 [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 모듈은 압축된 가중치와 추론을 위한 순전파를 관리하는 [CompressedLinear](https://github.com/neuralmagic/compressed-tensors/blob/975cb223b19fcac2b98a4271d17668462d4d6e1d/src/compressed_tensors/linear/compressed_linear.py#L30) 모듈로 바뀝니다. `lm_head` 모듈은 여전히 양자화되지 않은 nn.Linear 모듈로 남아 있습니다.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf")
print(ct_model)
"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (k_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (v_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (o_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (up_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (down_proj): CompressedLinear(
            in_features=14336, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""
```

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

[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)는 [safetensors](https://github.com/huggingface/safetensors) 파일을 압축된 텐서 데이터 타입으로 확장해서, dense, int 양자화(int8), float 양자화(fp8), pack 양자화(int32로 패킹된 int4나 int8 가중치 양자화) 등 다양한 양자화·sparse 형식을 하나의 체크포인트 형식으로 저장하고 불러올 수 있게 합니다.

compressed-tensors는 [PEFT](https://huggingface.co/docs/peft)를 사용한 파인튜닝을 지원하며, 다음과 같은 기능들을 제공합니다.

- fp8, int4, int8 가중치 및 활성화 정밀도.
- [tensor, channel, group, block, token](https://github.com/neuralmagic/compressed-tensors/blob/83b2e7a969d70606421a76b9a3d112646077c8de/src/compressed_tensors/quantization/quant_args.py#L43-L52) 수준의 양자화 스케일과 영점 전략을 제공합니다.
- 토큰별 동적 활성화 양자화(또는 정적 전략)를 지원합니다.
- 구조화되지 않은 형태 또는 2:4와 같은 반구조화된 형태의 가중치 희소성을 양자화와 결합하여 극한의 압축을 달성할 수 있습니다.
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 모듈뿐만 아니라 어떤 모듈이든 양자화가 가능합니다.
- 모듈 이름 또는 클래스별 양자화 대상을 지정할 수 있습니다.

최신 안정 버전은 [PyPI](https://pypi.org/project/compressed-tensors)에서 설치할 수 있습니다. 안정화되지 않은 최신 기능을 사용하려면 소스 코드를 이용해 설치하실 수 있습니다.

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

compressed-tensors [태그](https://huggingface.co/models?other=compressed-tensors)를 사용하여 Hugging Face Hub에서 양자화된 모델을 찾을 수 있습니다. 

현재는 이미 양자화된 모델만 불러올 수 있고, 불러온 모델은 다시 저장할 수 없습니다. compressed-tensors 형식으로 모델을 양자화하려면 [llm-compressor](https://github.com/vllm-project/llm-compressor)를 참고해 주세요. 또는 모델을 직접 생성하고 compressed-tensors 설정으로 직렬화할 수도 있습니다.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf", device_map="auto")

# 메모리 사용량 측정하기
mem_params = sum([param.nelement()*param.element_size() for param in ct_model.parameters()])
print(f"{mem_params/2**30:.4f} GB")
# 8.4575 GB
```

## 모델 체크포인트[[model-checkpoint]]

compressed-tensor 모델은 구성 항목을 통해 정의됩니다. 다음 예시는 [nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf/blob/main/config.json) `config.json` 파일에서 가져온 것입니다.

압축 전후의 유연한 표현을 위해 많은 항목이 존재하지만, 모델 불러오기와 추론에는 핵심 항목 몇 가지만 알아도 됩니다.

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

구성 파일은 구성 그룹(`group_0`)의 양자화를 지정하며, 정적 per-tensor 전략으로 가중치와 활성화를 fp8로 양자화합니다. `ignore` 키에 명시된 것처럼 `lm_head` 모듈은 양자화되지 않습니다.

모델 가중치를 더 자세히 보려면, 모델 카드의 [safetensors 뷰어](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf?show_file_info=model.safetensors.index.json)를 사용하여 모든 [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 모듈의 양자화된 가중치, 입력 스케일, 가중치 스케일을 확인할 수 있습니다.

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

compressed-tensors 모델을 [`~quantizers.HFQuantizer`] 통합으로 불러오면, 양자화 설정에 지정된 모든 [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) 모듈이 [CompressedLinear](https://github.com/neuralmagic/compressed-tensors/blob/975cb223b19fcac2b98a4271d17668462d4d6e1d/src/compressed_tensors/linear/compressed_linear.py#L30) 모듈로 대체되어 압축 가중치와 순전파를 관리합니다. `lm_head` 모듈은 여전히 양자화되지 않은 nn.Linear 모듈로 유지됩니다.

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

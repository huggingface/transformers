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

# 새로운 양자화 기법에 기여하기 [[contribute-new-quantization-method]]

Transformers는 QLoRA, GPTQ, LLM.int8, AWQ와 같은 다양한 양자화 방법을 지원하고 통합합니다. 그러나 아직 통합되지 않은 다른 양자화 방법들도 있습니다. [`HfQuantizer`] 클래스를 사용하면 이러한 양자화 방법을 Transformers 모델에 쉽게 추가하고 사용할 수 있습니다. [`HfQuantizer`]는 모든 PyTorch 모듈에 적용하는 것이 아니라, 양자화 방법을 추가하기 위한 내부 헬퍼 클래스로 설계되었습니다.

이 가이드에서는 [`HfQuantizer`] 클래스를 사용하여 새로운 양자화 방법을 통합하는 방법을 소개합니다.

## 요구사항 [[requirements]]

새로운 양자화 방법을 Transformers에 통합하기 전에, 추가하려는 방법이 다음의 조건을 충족하는지 확인하세요. 현재는 PyTorch 모듈로 실행할 수 있는 양자화 방법만 지원됩니다.

- 누구나 pip로 설치할 수 있는 Python 패키지로 제공되어야 합니다(소스에서만 설치할 수 있어도 괜찮습니다). 이상적으로는 pip 패키지에 사전 컴파일된 커널이 포함되는 것이 바람직합니다. 
- CPU, GPU 등과 같이 일반적으로 사용되는 하드웨어에서 실행될 수 있어야 합니다.
- `Linear8bitLt`, `Linear4bit`과 같이 양자회된 선형 레이어는 `nn.Module`로 감싸져야 하고, 이러한 레이어는 다음과 같이 정의되어야 합니다:

```py
class Linear4bit(nn.Module):
    def __init__(self, ...):
        ...
    
    def forward(self, x):
        return my_4bit_kernel(x, self.weight, self.bias)
```

이렇게 하면, `nn.Linear`의 일부 인스턴스를 대상 클래스(target class)로 교체하여 Transformers의 모델을 쉽게 양자화할 수 있습니다.

- 양자화된 가중치를 로컬에 저장하거나 Hub에 푸시할 수 있도록 양자화 방법은 직렬화 가능해야 합니다.
- 빈번한 호환성 변경을 방지하기 위해, 양자화 커널이나 프리미티브를 포함하는 패키지가 안정적인지 확인하세요.

AWQ와 같은 일부 양자화 방법은 데이터 보정을 통해 모델을 "사전 양자화"해야 할 수도 있습니다. 이런 경우, 추론에는 Transformers를 사용하고 모델 양자화는 ML 커뮤니티에서 관리하는 다른 라이브러리를 사용하는 것을 권장합니다.

## 새로운 HFQuantizer 클래스 구축하기 [[build-a-new-hfquantizer-class]]

1. [src/transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/quantization_config.py) 내부에 새로운 양자화 구성 클래스를 생성하고 and [src/transformers/__init__.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py)의 [`_import_structure`](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py#L1088)객체에 새로운 양자화 구성을 추가하여 Transformers 메인 `init`에서 해당 구성에 접근할 수 있습니다.

2. [src/transformers/quantizers/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers) 디렉터리 내부에 `quantizer_your_method.py`라는 새로운 파일을 생성하고, [src/transformers/quantizers/base.py::HfQuantizer](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/base.py#L28)을 상속 받도록 하세요. 새로운 양자화기(quantizer)와 양자화 구성을 [src/transformers/quantizers/auto.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/auto.py)의 양자화 자동 매핑(quantization auto-mapping)에 추가하는 것도 잊지 마세요.

3. 여러분의 양자화 방법에 대해 다음의 클래스 속성 또는 속성 메소드들을 정의하세요:

* `requires_calibration`: 양자화 방법이 데이터 보정 과정을 요구하는지 여부를 나타내는 속성입니다. 이 속성이 `True`로 설정되면, 양자화된 가중치를 사용한 추론만 지원할 수 있으며, 추론과 양자화 모두를 지원할 수는 없습니다. 
* `required_packages`: 양자화된 가중치를 사용하는 데 필요한 패키지들의 문자열 리스트입니다. 필요하다면 [transformers/src/utils/import_utils.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/import_utils.py)의 `is_auto_awq_available`과 같은 새로운 유틸리티 메소드를 정의해야 할 수도 있습니다.
* `requires_parameters_quantization`: 양자화 방법이 기본 `nn.Parameter` 객체에 특별한 처리가 필요한 경우에만 사용됩니다. 예를 들어, bitsandbytes 방식은 `Params4bit`와 `Int8Param`을 사용하여 양자화된 파라미터들에 대한 특별한 처리를 요구하기 때문에 해당 플래그를 `True`로 설정해야 합니다. 하지만 최근의 대부분의 양자화 방법은 `torch.uint8` 내에 int2/int4 가중치를 포함시키므로, 해당 플래그는 일반적으로 필요하지 않습니다(기본값은 `False`로 설정). 
* `is_serializable`: 해당 방법이 직렬화 가능한지를 결정하는 속성 메소드입니다.
* `is_trainable`:  PEFT 사용 여부에 상관없이 양자화 방법 위에서 모델을 미세 조정할 수 있는지를 결정하는 속성 메소드입니다.

4. `validate_environment`와 `update_torch_dtype` 메소드를 작성하세요. 이 메소드들은 양자화된 모델을 생성하기 전에 호출되어 사용자가 올바른 구성을 사용하고 있는지 확인합니다. 다른 양자화 기법에서 이 작업이 어떻게 이루어지는지 참고할 수 있습니다.

5. `_process_model_before_weight_loading` 메소드를 작성하세요. Transformers에서는, 양자회된 모델들은 가중치를 불러오기전에 `"meta"` 디바이스에서 초기화 됩니다. 이 과정에서 `_process_model_before_weight_loading` 메소드는 모델의 뼈대를 조작하여 `nn.Linear`와 같은 일부 모듈을 양자화 모듈로 교체합니다. 모듈 교체 로직이나 기타 유틸리티 메소드를 정의하려면[transformers/src/integrations/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/integrations) 디렉터리내에 새로운 파일을 생성하고 해당 폴더의 `__init__.py` 파일에서 관련 메소드를 등록하여 사용할 수 있도록 해야 합니다. [quantizer_awq.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/quantizer_awq.py)와 같은 다른 양자화 방법을 먼저 참고하시는 것이 도움이 될 것입니다.

6. `_process_model_after_weight_loading` 메소드를 작성하세요. 해당 메소드는 가중치를 로드한 후 모델을 조작해야 하는 추가 기능을 구현할 수 있게 해줍니다.

7. 모든 내용을 문서화하세요! 여러분의 양자화 방법이 잘 문서화 되도록 `docs/source/en/quantization` 디렉터리 아래에 새로운 파일을 추가하고 `docs/source/en/quantization/overview.md` 파일의 테이블에 새로운 행을 추가해주세요.

8. 테스트를 추가하세요! 먼저 `docker/transformers-quantization-latest-gpu` 안의 nightly Dockerfile에 해당 패키지를 추가하고, `tests/quantization/xxx`에 새로운 테스트 파일을 추가해야 합니다. 다른 양자화 방법들은 테스트가 어떻게 구현되어 있는지 참고해보세요. 

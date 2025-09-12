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

# 모델 문서화[[documenting-a-model]]

Transformers의 `@auto_docstring` 데코레이터는 모델 클래스 및 해당 메소드에 일관된 docstring을 생성합니다. 이는 표준 인자 설명을 자동으로 포함하면서도 새로운 또는 사용자 지정 인자를 추가하기 위한 재정의를 허용하여 상용구 코드를 줄입니다. 표준 docstring을 수동으로 추가할 필요 없이 새 인자 문서화에만 집중할 수 있어 [새 모델 기여하기](./modular_transformers)가 더 쉬워집니다.

이 가이드에서는 `@auto_docstring` 데코레이터 사용 방법과 그 작동 방식을 설명합니다.

## @auto_docstring[[autodocstring]]

모델링 파일(`modular_model.py` 또는 `modeling_model.py`)에서 데코레이터를 가져오는 것으로 시작합니다.

```python
from ...utils import auto_docstring
```

`@auto_docstring` 를 클래스 혹은 함수에 적용할지 선택하여 아래 예시처럼 사용하세요.

<hfoptions id="type">
<hfoption id="classes">

`@auto_docstring`을 클래스 정의 바로 위에 배치합니다. 이 데코레이터는 `__init__` 메소드의 시그니처와 docstring에서 매개변수 설명을 추출합니다.

```python
from transformers.modeling_utils import PreTrainedModel
from ...utils import auto_docstring

@auto_docstring
class MyAwesomeModel(PreTrainedModel):
    def __init__(self, config, custom_parameter: int = 10, another_custom_arg: str = "default"):
        r"""
        custom_parameter (`int`, *optional*, defaults to 10):
            MyAwesomeModel에 대한 custom_parameter 설명입니다.
        another_custom_arg (`str`, *optional*, defaults to "default"):
            다른 고유 인자에 대한 문서입니다.
        """
        super().__init__(config)
        self.custom_parameter = custom_parameter
        self.another_custom_arg = another_custom_arg
        # ... ... 나머지 초기화 코드(init)

    # ... 기타 메소드
```

보다 세밀한 제어를 위해 인자를 `@auto_docstring` 에 직접 전달할 수도 있습니다. `custom_intro` 파라미터를 사용하여 인자 목록에 대한 설명을, `custom_args` 파라미터를 사용하여 개별 인자들을 설명할 수 있습니다.

```python
@auto_docstring(
    custom_intro="""이 모델은 특정 시너지 작업을 수행합니다. 
    표준 Transformer 아키텍처를 기반으로 고유한 수정 사항을 포함합니다.""",
    custom_args="""
    custom_parameter (`type`, *optional*, defaults to `default_value`):
        `args_doc.py`에 정의되지 않았거나 재정의하는 경우 custom_parameter에 대한 간결한 설명입니다.
    internal_helper_arg (`type`, *optional*, defaults to `default_value`):
        `args_doc.py`에 정의되지 않았거나 재정의하는 경우 internal_helper_arg에 대한 간결한 설명입니다.
    """
)
class MySpecialModel(PreTrainedModel):
    def __init__(self, config: ConfigType, custom_parameter: "type" = "default_value", internal_helper_arg=None):
        # ...
```

`custom_intro`만 사용하고 사용자 지정 인자를 클래스에 직접 정의할 수도 있습니다.

```python
@auto_docstring(
    custom_intro="""이 모델은 특정 시너지 작업을 수행합니다.
    표준 Transformer 아키텍처를 기반으로 고유한 수정 사항을 포함합니다.""",
)
class MySpecialModel(PreTrainedModel):
    def __init__(self, config: ConfigType, custom_parameter: "type" = "default_value", internal_helper_arg=None):
        r"""
        custom_parameter (`type`, *optional*, defaults to `default_value`):
            `args_doc.py`에 정의되지 않았거나 재정의하는 경우 custom_parameter에 대한 간결한 설명입니다.
        internal_helper_arg (`type`, *optional*, defaults to `default_value`):
            `args_doc.py`에 정의되지 않았거나 재정의하는 경우 internal_helper_arg에 대한 간결한 설명입니다.
        """
        # ...
```

</hfoption>
<hfoption id="functions">

`@auto_docstring`을 메소드 정의 바로 위에 배치합니다. 이 데코레이터는 함수 시그니처에서 매개변수 설명을 추출합니다.

```python
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        new_custom_argument: Optional[torch.Tensor] = None,
        arg_documented_in_args_doc: Optional[torch.Tensor] = None,
        # ... 다른 인자들
    ) -> Union[Tuple, ModelOutput]: # 반환 값에 대한 설명은 ModelOutput 클래스 docstring에서 자동으로 생성됩니다.
        r"""
        new_custom_argument (`torch.Tensor`, *optional*):
            Description of this new custom argument and its expected shape or type.
        """
        # ...
```

제어를 강화하기 위해 인자들을 `@auto_docstring`에 직접 전달할 수도 있습니다. `custom_intro` 매개변수는 인자들에 대한 전반적인 설명을 작성할 때 사용하고, `custom_args` 매개변수는 각 인자별 설명을 작성할 때 사용합니다.

docstring의 `Returns` 및 `Examples` 부분도 수동으로 지정할 수 있습니다.


```python
MODEL_COMMON_CUSTOM_ARGS = r"""
    common_arg_1 (`torch.Tensor`, *optional*, defaults to `default_value`):
        common_arg_1에 대한 설명
    common_arg_2 (`torch.Tensor`, *optional*, defaults to `default_value`):
        common_arg_2에 대한 설명
    ...
"""

class MyModel(PreTrainedModel):
    # ...
    @auto_docstring(
        custom_intro="""
        This is a custom introduction for the function.
        """
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        common_arg_1: Optional[torch.Tensor] = None,
        common_arg_2: Optional[torch.Tensor] = None,
        #...
        function_specific_argument: Optional[torch.Tensor] = None,
        # ... 다른 인자들
    ) -> torch.Tensor:
        r"""
        function_specific_argument (`torch.Tensor`, *optional*):
            이 함수에 특정한 인자에 대한 설명입니다.

        Returns:
            `torch.Tensor`: 일반적인 타입을 반환하는 함수에 대해 사용자 지정 "Returns" 섹션을 지정할 수 있습니다.

        Example:

        (기본 예시를 사용자 지정 예시로 재정의하거나, pipeline이 없는 모델 클래스에 대한 예시를 추가하는 경우)

        ```python
        ...
        ```
        """
        # ...
```

</hfoption>
</hfoptions>

## 인자 문서화[[documenting-arguments]]

다양한 유형의 인자를 문서화하기 위한 규칙을 아래서 확인할 수 있습니다.

- 표준 인자(`input_ids`, `attention_mask`, `pixel_values` 등)는 `args_doc.py`에서 정의되어 있으며, 해당 파일에서 불러와 사용합니다. 이는 표준 인자에 대한 단일 정보 출처이므로, 인자의 설명과 형태가 `args_doc.py`의 인자와 동일한 경우에는 로컬에서 재정의해서는 안 됩니다.

    표준 인자가 모델에서 다르게 동작하는 경우, `r""" """` 블록 내에서 로컬로 재정의할 수 있습니다. 이 로컬 정의는 더 높은 우선순위를 가집니다. 예를 들어, labels 인자는 종종 모델별로 사용자 정의되며 일반적으로 재정의가 필요합니다.


- 새 인자나 사용자 정의 인자는 함수 시그니처 뒤의 `r""" """` 블록에, 클래스의 경우 `__init__` 메소드의 docstring에 문서화해야 합니다.

    ```py
    argument_name (`type`, *optional*, defaults to `X`):
        인자에 대한 설명입니다.
        Explain its purpose, expected shape/type if complex, and default behavior.
        여러 줄에 걸쳐 작성할 수 있습니다.
    ```

    * `type`은 백틱으로 감싸세요.
    * 인자가 필수가 아니거나 기본값이 있는 경우 *optional*을 추가하세요.
    * 기본값이 있는 경우 "defaults to X"를 추가하세요. 기본값이 None인 경우 "defaults to None"을 추가할 필요는 없습니다.

    이러한 인자들은 `custom_args` 인자로 `@auto_docstring`에 전달할 수도 있습니다. 이는 모델링 파일의 여러 위치에서 새로운 인자들의 docstring 블록이 반복되는 경우 한 번만 정의하는 데 사용됩니다.

    ```py
    class MyModel(PreTrainedModel):
    # ...
    @auto_docstring(
        custom_intro="""
        이 함수에 대한 사용자 정의 도입부입니다.
        """
        custom_args=r"""
        common_arg_1 (`torch.Tensor`, *optional*, defaults to `default_value`):
            common_arg_1에 대한 설명
        """
    )
    ```

## docstring 검사[[checking-the-docstrings]]

Transformers는 Pull Request에서 CI(continuous intergration, 지속적 통합) 검사를 트리거할 때 docstring 유효성을 확인하는 유틸리티 스크립트를 포함합니다. 스크립트는 다음 기준을 검사합니다.

* `@auto_docstring` 이 관련 모드 클래스 및 공개 메소드에 적용되었는지 확인합니다.
* 인자가 완전하고 일관적인지 확인합니다. 문서화된 인자가 시그니처에 존재하는지 확인하고, docstring의 타입 및 기본값이 시그니처와 일치하는지 검증합니다. 알려진 표준 인자가 아니거나 로컬 설명이 없는 인자는 플래그가 지정됩니다.
* `<fill_type>` 및 `<fill_docstring>` 같은 자리 표시자(placeholder)를 잊지 않고 채우도록 도와줍니다.
* docstring이 예상되는 docstring 스타일에 따라 형식이 지정되었는지 확인합니다.

이러한 검사를 커밋하기 전에 로컬에서 실행할 수 있습니다. 다음 명령어를 실행하세요.

```bash
make fix-copies
```

`make fix-copies` 는 다른 여러 검사도 실행합니다. 해당 검사가 필요하지 않은 경우, docstring 및 자동 docstring 검사만 수행하려면 아래 명령어를 실행하세요.

```bash
python utils/check_docstrings.py # diff에 포함된 파일만 확인하고 수정하지 않습니다.
# python utils/check_docstrings.py --fix_and_overwrite # diff의 파일을 수정하고 덮어씁니다.
# python utils/check_docstrings.py --fix_and_overwrite --check_all # 모든 파일을 수정하고 덮어씁니다.
```

## modular_model.py 파일[[modularmodelpy-files]]

모듈화된 파일(`modular_model.py`) 작업 시 `@auto_docstring` 적용에 대한 다음 지침을 따르세요.

- 모듈화된 파일의 독립 실행형 모델의 경우, `modeling_model.py` 파일에서와 같이 `@auto_docstring`을 적용합니다. 
- 다른 라이브러리 모델을 상속하는 모델의 경우, `@auto_docstring`은 생성된 모델링 파일로 자동으로 전달됩니다. 모듈화된 파일에 `@auto_docstring`을 추가할 필요가 없습니다.

    `@auto_docstring` 동작을 수정해야 하는 경우, 모듈화된 파일에 사용자 지정 데코레이터를 적용합니다. 원본 함수 또는 클래스에 있는 **다른 모든 데코레이터를 반드시 포함**해야 합니다.

> [!WARNING]
> 모듈화된 파일에서 데코레이터를 재정의할 때, 부모 모델의 해당 함수 또는 클래스에 적용된 **모든** 데코레이터를 포함해야 합니다. 일부 데코레이터만 재정의하면 다른 데코레이터는 생성된 모델링 파일에 포함되지 않습니다.

## 작동 방식[[how-it-works]]

`@auto_docstring` 데코레이터는 다음을 통해 docstring을 자동으로 생성합니다.

1. 데코레이트된 클래스의 `__init__` 메소드 또는 데코레이트된 함수의 시그니처(인자, 타입, 기본값)를 검사합니다. 
2. 일반적인 인자 (`input_ids`, `attention_mask`, 등)에 대해 미리 정의된 docstring을 [`ModelArgs`], [`ImageProcessorArgs`] 및 `args_doc.py` 파일과 같은 내부 라이브러리 소스에서 검색합니다. 
3. 아래와 같이 두 가지 방법 중 하나로 인자 설명을 추가합니다.

    | 방법 | 설명 | 사용법 |
    |---|---|---|
    | `r""" """` | 메소드 시그니처에 직접 또는 `__init__` docstring 내에 사용자 지정 docstring 내용을 추가합니다. | 새로운 인자를 문서화하거나 표준 설명을 재정의합니다. |
    | `custom_args` | `@auto_docstring`에 특정 인자에 대한 사용자 지정 Docstring을 직접 추가합니다. | 모델링 파일의 여러 위치에서 반복되는 경우 새로운 인자에 대한 docstring을 한 번만 정의합니다. |

4. 클래스 및 함수 설명을 추가합니다. `ModelForCausalLM`과 같은 표준 명명 패턴을 가진 모델 클래스 또는 파이프라인에 속하는 경우, `@auto_docstring`은 `args_doc.py`의 `ClassDocstring`을 사용하여 적절한 설명을 자동으로 생성합니다. 

    또한, `@auto_docstring`은 클래스 또는 함수를 설명하는 `custom_intro` 인자를 허용합니다.

5. 템플릿 시스템을 사용해 이미 정의된 docstring에 Transformers [auto_modules](https://github.com/huggingface/transformers/tree/main/src/transformers/models/auto)의 정보(`{{processor_class}}` , `{{config_class}}` 등)를 동적으로 삽입합니다. 

6. 모델의 작업 또는 pipeline 호환성을 기반으로 적절한 사용 예시를 찾습니다. 모델의 구성 클래스에서 체크포인트 정보를 추출하여 실제 모델 식별자를 포함한 구체적인 예시를 제공합니다. 

7. docstring에 반환 값을 추가합니다. `forward`와 같은 메소드의 경우, 데코레이터는 메소드의 반환 타입 주석을 기반으로 docstring에 `Returns` 필드를 자동으로 생성합니다.

    예를 들어, 메소드가 [`~transformers.utils.ModelOutput`] 서브클래스를 반환하는 경우, `@auto_docstring`은 클래스의 Docstring에서 필드 설명을 추출하여 포괄적인 반환 값 설명을 생성합니다. 함수의 docstring에서 사용자 지정 `Returns` 필드를 수동으로 지정할 수도 있습니다.

8. unpack 연산자로 타입이 지정된 kwargs를 풀어서 설명합니다. 특정 메소드(`UNROLL_KWARGS_METHODS`에 정의됨) 또는 클래스(`UNROLL_KWARGS_CLASSES`에 정의됨)의 경우, 데코레이터는 `Unpack[KwargsTypedDict]`로 타입이 지정된 `**kwargs` 파라미터를 처리합니다. `TypedDict`에서 문서를 추출하여 각 파라미터를 함수의 docstring에 추가합니다. 

    현재 [`FastImageProcessorKwargs`]만 지원합니다. 

## 모범 사례[[best-practices]]

Transformers의 일관되고 유익한 문서를 유지하는 데 도움이 되도록 다음 모범 사례를 따르세요!

* 새로운 PyTorch 모델 클래스 ([`PreTrainedModel`] 서브클래스) 및 `forward` 또는 `get_text_features`와 같은 기본 메소드에 `@auto_docstring`을 사용하세요. 
* 클래스의 경우, `@auto_docstring`은 `__init__` 메소드의 docstring에서 파라미터 설명을 가져온다는 점을 기억하세요.
* 표준 docstring을 우선 사용하고, 모델의 동작이 다른 경우에만 공통 인자를 재정의하세요.
* 새로운 인자나 사용자 지정 인자는 명확하게 문서화하세요.
* 변경 사항을 커밋하기 전에 `check_docstrings` 를 로컬에서 반복적으로 실행하여 확인하세요.

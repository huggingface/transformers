# Jan: 로컬 LLM 제공자로서 서빙 API 사용하기 [[jan-using-the-serving-api-as-a-local-llm-provider]]

이 예제는 [Jan](https://jan.ai/) 앱의 로컬 LLM 제공자로 `transformers serve`를 사용하는 방법을 보여줍니다. Jan은 완전히 사용자의 기기에서 실행되는 ChatGPT 대안 그래픽 인터페이스입니다. `transformers serve`에 대한 요청은 로컬 앱에서 직접 오며 - 이 섹션은 Jan에 중점을 두고 있지만, 로컬 요청을 수행하는 다른 앱에 대한 지침으로 확장할 수 있습니다.

## 로컬에서 모델 실행하기 [[running-models-locally]]

`transformers serve`를 Jan과 연결하려면 새로운 모델 제공자를 설정해야 합니다("설정" > "모델 제공자"). "제공자 추가"를 클릭하고 새 이름을 설정하세요. 새 모델 제공자 페이지에서 "기본 URL"을 다음 패턴으로 설정하기만 하면 됩니다:

```shell
http://[host]:[port]/v1
```

여기서 `host`와 `port`는 `transformers serve` CLI 매개변수입니다(기본값은 `localhost:8000`). 이 설정 후, "새로고침"을 클릭하면 "모델" 섹션에서 일부 모델을 볼 수 있어야 합니다. "API 키" 텍스트 필드에도 일부 텍스트를 추가해야 합니다 - 이 데이터는 실제로 사용되지 않지만, 필드는 비어 있을 수 없습니다. 사용자 정의 모델 제공자 페이지는 다음과 같이 보여야 합니다:

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_jan_model_providers.png"/>
</h3>

이제 채팅할 준비가 되었습니다!

> [!TIP]
> `transformers serve`를 통해 Jan에 `transformers`와 호환되는 모든 모델을 추가할 수 있습니다. 생성한 사용자 정의 모델 제공자에서 "모델" 섹션의 "+" 버튼을 클릭하고 해당 허브 저장소 이름을 추가하세요. 예: `Qwen/Qwen3-4B`.

## 별도 머신에서 모델 실행하기 [[running-models-on-a-separate-machine]]

이 예제를 마무리하기 위해, 좀 더 고급 사용 사례를 살펴보겠습니다. 모델을 제공할 강력한 머신이 있지만 다른 기기에서 Jan을 사용하는 것을 선호한다면, 포트 포워딩을 추가해야 합니다. Jan 머신에서 서버로의 `ssh` 액세스가 있다면, Jan 머신의 터미널에 다음을 입력하여 이를 수행할 수 있습니다

```
ssh -N -f -L 8000:localhost:8000 your_server_account@your_server_IP -p port_to_ssh_into_your_server
```

포트 포워딩은 Jan에만 국한되지 않습니다: 다른 머신에서 실행 중인 `transformers serve`를 선택한 앱과 연결하는 데 사용할 수 있습니다.
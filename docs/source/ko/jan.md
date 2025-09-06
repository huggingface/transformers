# Jan: 로컬 LLM 제공자로 서빙 API 사용하기

이 예제는 [Jan](https://jan.ai/) 앱을 위한 로컬 LLM 제공자로 `transformers serve`를 사용하는 방법을 보여줍니다. Jan은 완전히 사용자의 기기에서 실행되는 ChatGPT 대안 그래픽 인터페이스입니다. `transformers serve`에 대한 요청은 로컬 앱에서 직접 이루어집니다 -- 이 섹션은 Jan에 초점을 맞추고 있지만, 로컬 요청을 수행하는 다른 앱에도 일부 지침을 확장할 수 있습니다.

## 로컬에서 모델 실행하기

`transformers serve`를 Jan과 연결하려면, 새 모델 제공자를 설정해야 합니다("설정" > "모델 제공자"). "제공자 추가"를 클릭하고, 새 이름을 설정하세요. 새 모델 제공자 페이지에서, 다음 패턴으로 "기본 URL"만 설정하면 됩니다:

```shell
http://[host]:[port]/v1
```

여기서 `host`와 `port`는 `transformers serve` CLI 매개변수입니다(기본값은 `localhost:8000`). 이를 설정한 후, "새로고침"을 누르면 "모델" 섹션에서 일부 모델을 볼 수 있어야 합니다. "API 키" 텍스트 필드에 텍스트를 추가하세요 -- 이 데이터는 실제로 사용되지 않지만, 필드는 비어 있을 수 없습니다. 맞춤 설정된 모델 제공자 페이지는 다음과 같이 보여야 합니다:

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_serve_jan_model_providers.png"/>
</h3>

이제 채팅할 준비가 되었습니다!

> [!TIP]
> `transformers serve`를 통해 Jan에 `transformers`와 호환되는 모든 모델을 추가할 수 있습니다. 생성한 맞춤 모델 제공자에서, "모델" 섹션에 있는 "+" 버튼을 클릭하고 Hub 저장소 이름(예: `Qwen/Qwen3-4B`)을 추가하세요.

## 별도 기기에서 모델 실행하기

이 예제를 마무리하기 위해, 더 고급 사용 사례를 살펴보겠습니다. 모델을 서빙할 강력한 기계가 있지만 다른 장치에서 Jan을 사용하고 싶다면, 포트 포워딩을 추가해야 합니다. Jan 기기에서 서버로 `ssh` 접속이 가능하다면, Jan 기기의 터미널에 다음을 입력하여 이를 수행할 수 있습니다

```
ssh -N -f -L 8000:localhost:8000 your_server_account@your_server_IP -p port_to_ssh_into_your_server
```

포트 포워딩은 Jan에만 특화된 것이 아닙니다: 다른 기기에서 실행 중인 `transformers serve`를 원하는 앱과 연결하는 데 사용할 수 있습니다.
```
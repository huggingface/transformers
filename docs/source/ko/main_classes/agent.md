<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 에이전트 & 도구 [[agents-tools]]

<Tip warning={true}>

Transformers Agent는 실험 중인 API이므로 언제든지 변경될 수 있습니다. 
API나 기반 모델이 자주 업데이트되므로, 에이전트가 제공하는 결과물은 달라질 수 있습니다.

</Tip>

에이전트와 도구에 대해 더 알아보려면 [소개 가이드](../transformers_agents)를 꼭 읽어보세요. 
이 페이지에는 기본 클래스에 대한 API 문서가 포함되어 있습니다.

## 에이전트 [[agents]]

우리는 기본 [`Agent`] 클래스를 기반으로 두 가지 유형의 에이전트를 제공합니다:
- [`CodeAgent`]는 한 번에 동작합니다. 작업을 해결하기 위해 코드를 생성한 다음, 바로 실행합니다.
- [`ReactAgent`]는 단계별로 동작하며, 각 단계는 하나의 생각, 하나의 도구 호출 및 실행으로 구성됩니다. 이 에이전트에는 두 가지 클래스가 있습니다:
  - [`ReactJsonAgent`]는 도구 호출을 JSON으로 작성합니다.
  - [`ReactCodeAgent`]는 도구 호출을 Python 코드로 작성합니다.

### Agent [[agent]]

[[autodoc]] Agent

### CodeAgent [[codeagent]]

[[autodoc]] CodeAgent

### React agents [[react-agents]]

[[autodoc]] ReactAgent

[[autodoc]] ReactJsonAgent

[[autodoc]] ReactCodeAgent

## Tools [[tools]]

### load_tool [[loadtool]]

[[autodoc]] load_tool

### Tool [[tool]]

[[autodoc]] Tool

### Toolbox [[toolbox]]

[[autodoc]] Toolbox

### PipelineTool [[pipelinetool]]

[[autodoc]] PipelineTool

### launch_gradio_demo [[launchgradiodemo]]

[[autodoc]] launch_gradio_demo

### ToolCollection [[toolcollection]]

[[autodoc]] ToolCollection

## 엔진 [[engines]]

에이전트 프레임워크에서 사용할 수 있는 엔진을 자유롭게 만들고 사용할 수 있습니다.
이 엔진들은 다음과 같은 사양을 가지고 있습니다:
1. 입력(`List[Dict[str, str]]`)에 대한 [메시지 형식](../chat_templating.md)을 따르고 문자열을 반환해야 합니다.
2. 인수 `stop_sequences`에 시퀀스가 전달되기 *전에* 출력을 생성하는 것을 중지해야 합니다.

### HfApiEngine [[HfApiEngine]]

편의를 위해, 위의 사항을 구현하고 대규모 언어 모델 실행을 위해 추론 엔드포인트를 사용하는 `HfApiEngine`을 추가했습니다.

```python
>>> from transformers import HfApiEngine

>>> messages = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "No need to help, take it easy."},
... ]

>>> HfApiEngine()(messages, stop_sequences=["conversation"])

"That's very kind of you to say! It's always nice to have a relaxed "
```

[[autodoc]] HfApiEngine


## 에이전트 유형 [[agent-types]]

에이전트는 도구 간의 모든 유형의 객체를 처리할 수 있습니다; 도구는 완전히 멀티모달이므로 텍스트, 이미지, 오디오, 비디오 등 다양한 유형을 수락하고 반환할 수 있습니다. 
도구 간의 호환성을 높이고 ipython (jupyter, colab, ipython 노트북, ...)에서 이러한 
반환 값을 올바르게 렌더링하기 위해 이러한 유형을 중심으로 래퍼 클래스를 
구현합니다.

래핑된 객체는 처음과 동일하게 작동해야 합니다; 텍스트 객체는 여전히 문자열로 작동해야 하며, 
이미지 객체는 여전히 `PIL.Image`로 작동해야 합니다.

이러한 유형에는 세 가지 특정 목적이 있습니다:

- `to_raw`를 호출하면 기본 객체가 반환되어야 합니다.
- `to_string`을 호출하면 객체가 문자열로 반환되어야 합니다: 
`AgentText`의 경우 문자열이 될 수 있지만, 다른 경우에는 객체의 직렬화된 버전의 경로일 수 있습니다.
- ipython 커널에서 표시할 때 객체가 올바르게 표시되어야 합니다.

### AgentText [[agenttext]]

[[autodoc]] transformers.agents.agent_types.AgentText

### AgentImage [[agentimage]]

[[autodoc]] transformers.agents.agent_types.AgentImage

### AgentAudio [[agentaudio]]

[[autodoc]] transformers.agents.agent_types.AgentAudio

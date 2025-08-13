### `tiny-agents` CLI 및 MCP 도구[[tiny-agents-cli-and-mcp-tools]]

MCP 도구의 사용을 보여주기 위해 [`tiny-agents`](https://huggingface.co/blog/python-tiny-agents) CLI와 `transformers serve` 서버를 연동하는 방법을 살펴보겠습니다.

> [!TIP]
> 이 예시처럼 많은 Hugging Face Spaces를 MCP 서버로 활용할 수 있습니다. 호환 가능한 모든 Spaces는 [여기](https://huggingface.co/spaces?filter=mcp-server)에서 찾을 수 있습니다.

MCP 도구를 사용하려면 먼저 모델에 사용 가능한 도구를 알려야 합니다. 예를 들어, [이미지 생성 MCP 서버](https://evalstate-flux1-schnell.hf.space/)를 참조하는 `tiny-agents` 설정 파일을 살펴보겠습니다.

```json
{
  "model": "Menlo/Jan-nano",
  "endpointUrl": "http://localhost:8000",
  "servers": [
    {
      "type": "sse",
      "url": "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/sse"
    }
  ]
}
```

그런 다음 아래 명령어로 `tiny-agents` 채팅 인터페이스를 실행할 수 있습니다.

```bash
tiny-agents run path/to/your/config.json
```

백그라운드에서 `transformers serve`가 실행 중이라면, 이제 로컬 모델에서 MCP 도구를 사용할 수 있습니다. 다음은 `tiny-agents`와의 채팅 세션 예시입니다.

```bash
Agent loaded with 1 tools:
 • flux1_schnell_infer
»  Generate an image of a cat on the moon
<Tool req_0_tool_call>flux1_schnell_infer {"prompt": "a cat on the moon", "seed": 42, "randomize_seed": true, "width": 1024, "height": 1024, "num_inference_steps": 4}

Tool req_0_tool_call
[Binary Content: Image image/webp, 57732 bytes]
The task is complete and the content accessible to the User
Image URL: https://evalstate-flux1-schnell.hf.space/gradio_api/file=/tmp/gradio/3dbddc0e53b5a865ed56a4e3dbdd30f3f61cf3b8aabf1b456f43e5241bd968b8/image.webp
380576952

Flux 1 Schnell 이미지 생성기를 사용하여 달 위의 고양이 이미지를 생성했습니다. 이미지는 1024x1024 픽셀이며 4번의 추론 단계를 거쳐 생성되었습니다. 변경 사항이 필요하거나 추가 도움이 필요하시면 알려주세요!
```

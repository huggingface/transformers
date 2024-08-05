<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 웹 서버를 위한 파이프라인 사용하기[[using_pipelines_for_a_webserver]]

<Tip>
추론 엔진을 만드는 것은 복잡한 주제이며, "최선의" 솔루션은 문제 공간에 따라 달라질 가능성이 높습니다. CPU 또는 GPU를 사용하는지에 따라 다르고 낮은 지연 시간을 원하는지, 높은 처리량을 원하는지, 다양한 모델을 지원할 수 있길 원하는지, 하나의 특정 모델을 고도로 최적화하길 원하는지 등에 따라 달라집니다. 이 주제를 해결하는 방법에는 여러 가지가 있으므로, 이 장에서 제시하는 것은 처음 시도해 보기에 좋은 출발점일 수는 있지만, 이 장을 읽는 여러분이 필요로 하는 최적의 솔루션은 아닐 수 있습니다.
</Tip>

핵심적으로 이해해야 할 점은 [dataset](pipeline_tutorial#using-pipelines-on-a-dataset)를 다룰 때와 마찬가지로 반복자를 사용 가능하다는 것입니다. 왜냐하면, 웹 서버는 기본적으로 요청을 기다리고 들어오는 대로 처리하는 시스템이기 때문입니다.

보통 웹 서버는 다양한 요청을 동시에 다루기 위해 매우 다중화된 구조(멀티 스레딩, 비동기 등)를 지니고 있습니다. 반면에, 파이프라인(대부분 파이프라인 안에 있는 모델)은 병렬처리에 그다지 좋지 않습니다. 왜냐하면 파이프라인은 많은 RAM을 차지하기 때문입니다. 따라서, 파이프라인이 실행 중이거나 계산 집약적인 작업 중일 때 모든 사용 가능한 리소스를 제공하는 것이 가장 좋습니다.

이 문제를 우리는 웹 서버가 요청을 받고 보내는 가벼운 부하를 처리하고, 실제 작업을 처리하는 단일 스레드를 갖는 방법으로 해결할 것입니다. 이 예제는 `starlette` 라이브러리를 사용합니다.
실제 프레임워크는 중요하지 않지만, 다른 프레임워크를 사용하는 경우 동일한 효과를 보기 위해선 코드를 조정하거나 변경해야 할 수 있습니다.

`server.py`를 생성하세요:

```py
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio


async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    pipe = pipeline(model="google-bert/bert-base-uncased")
    while True:
        (string, response_q) = await q.get()
        out = pipe(string)
        await response_q.put(out)


app = Starlette(
    routes=[
        Route("/", homepage, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
```

이제 다음 명령어로 실행시킬 수 있습니다:

```bash
uvicorn server:app
```

이제 쿼리를 날려볼 수 있습니다:

```bash
curl -X POST -d "test [MASK]" http://localhost:8000/
#[{"score":0.7742936015129089,"token":1012,"token_str":".","sequence":"test."},...]
```

자, 이제 웹 서버를 만드는 방법에 대한 좋은 개념을 알게 되었습니다!

중요한 점은 모델을 **한 번만** 가져온다는 것입니다. 따라서 웹 서버에는 모델의 사본이 없습니다. 이런 방식은 불필요한 RAM이 사용되지 않습니다. 그런 다음 큐 메커니즘을 사용하면, 다음과 같은
동적 배치를 사용하기 위해 추론 전 단계에 몇 개의 항목을 축적하는 것과 같은 멋진 작업을 할 수 있습니다:

<Tip warning={true}>
코드는 의도적으로 가독성을 위해 의사 코드처럼 작성되었습니다!
아래 코드를 작동시키기 전에 시스템 자원이 충분한지 확인하세요!
</Tip>

```py
(string, rq) = await q.get()
strings = []
queues = []
while True:
    try:
        (string, rq) = await asyncio.wait_for(q.get(), timeout=0.001)  # 1ms
    except asyncio.exceptions.TimeoutError:
        break
    strings.append(string)
    queues.append(rq)
strings
outs = pipe(strings, batch_size=len(strings))
for rq, out in zip(queues, outs):
    await rq.put(out)
```

다시 말씀 드리자면, 제안된 코드는 가독성을 위해 최적화되었으며, 최상의 코드는 아닙니다.
첫째, 배치 크기 제한이 없으며 이는 일반적으로 좋은 방식이 아닙니다.
둘째, 모든 큐 가져오기에서 타임아웃이 재설정되므로 추론을 실행하기 전에 1ms보다 훨씬 오래 기다릴 수 있습니다(첫 번째 요청을 그만큼 지연시킴).

단일 1ms 길이의 데드라인을 두는 편이 더 좋습니다.

이 방식을 사용하면 큐가 비어 있어도 항상 1ms를 기다리게 될 것입니다. 
큐에 아무것도 없을 때 추론을 원하는 경우에는 최선의 방법이 아닐 수 있습니다.
하지만 배치 작업이 사용례에 따라 정말로 중요하다면 의미가 있을 수도 있습니다. 
다시 말하지만, 최상의 솔루션은 없습니다.

## 고려해야 할 몇 가지 사항[[few_things_you_might want_to_consider]]

### 에러 확인[[error_checking]]

프로덕션 환경에서는 문제가 발생할 여지가 많습니다. 
메모리가 모자라거나, 공간이 부족하거나, 모델을 가져오는 데에 실패하거나, 쿼리가 잘못되었거나, 쿼리는 정확해도 모델 설정이 잘못되어 실행에 실패하는 등등 많은 경우가 존재합니다.

일반적으로 서버가 사용자에게 오류를 출력하는 것이 좋으므로
오류를 표시하기 위해 `try...except` 문을 많이 추가하는 것이 좋습니다. 
하지만 보안 상황에 따라 모든 오류를 표시하는 것은 보안상 위험할 수도 있다는 점을 명심해야합니다.

### 서킷 브레이킹[[circuit_breaking]]

웹 서버는 일반적으로 서킷 브레이킹을 수행할 때 더 나은 상황에 직면합니다.
즉, 이는 서버가 쿼리를 무기한 기다리는 대신 과부하 상태일 때 적절한 오류를 반환하는 것을 의미합니다.
서버가 매우 오랜 시간 동안 대기하거나 적당한 시간이 지난 후에 504 에러를 반환하는 대신 503 에러를 빠르게 반환하게 하는 것입니다.

제안된 코드에는 단일 큐가 있으므로 구현하기가 비교적 쉽습니다.
큐 크기를 확인하는 것은 웹 서버가 과부하 상항 하에 있을 때 에러를 반환하기 위한 가장 기초적인 작업입니다.

### 메인 쓰레드 차단[[blocking_the_main_thread]]

현재 PyTorch는 비동기 처리를 지원하지 않으며, 실행 중에는 메인 스레드가 차단됩니다. 
따라서 PyTorch를 별도의 스레드/프로세스에서 실행하도록 강제하는 것이 좋습니다.
여기서는 이 작업이 수행되지 않았습니다. 왜냐하면 코드가 훨씬 더 복잡하기 때문입니다(주로 스레드, 비동기 처리, 큐가 서로 잘 맞지 않기 때문입니다).
하지만 궁극적으로는 같은 작업을 수행하는 것입니다.

단일 항목의 추론이 오래 걸린다면 (> 1초), 메인 쓰레드를 차단하는 것은 중요할 수 있습니다. 왜냐하면 이 경우 추론 중 모든 쿼리는 오류를 받기 전에 1초를 기다려야 하기 때문입니다.

### 동적 배치[[dynamic_batching]]

일반적으로, 배치 처리가 1개 항목을 한 번에 전달하는 것에 비해 반드시 성능 향상이 있는 것은 아닙니다(자세한 내용은 [`batching details`](./main_classes/pipelines#pipeline-batching)을 참고하세요).
하지만 올바른 설정에서 사용하면 매우 효과적일 수 있습니다.
API에는 기본적으로 속도 저하의 가능성이 매우 높기 때문에 동적 배치 처리가 없습니다.
하지만 매우 큰 모델인 BLOOM 추론의 경우 동적 배치 처리는 모든 사람에게 적절한 경험을 제공하는 데 **필수**입니다.

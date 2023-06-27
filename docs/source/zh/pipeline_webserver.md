<!--⚠️ 请注意，此文件采用Markdown格式，但包含我们文档生成器（类似于MDX）的特定语法，可能无法正确地在您的Markdown查看器中显示。-->

# 使用流水线进行网络服务器

<Tip> 

创建推理引擎是一个复杂的话题，"最佳" 解决方案很可能取决于您的问题空间。您是否使用 CPU 还是 GPU？您是否希望最低延迟、最高吞吐量、对多个模型的支持，还是只优化 1 个特定模型？有很多解决此问题的方法，因此我们将介绍一个很好的默认解决方案，这可能不一定是最优的解决方案。

</Tip>


关键是我们可以使用迭代器，就像您在 [数据集上](pipeline_tutorial#using-pipelines-on-a-dataset) 一样，因为网络服务器基本上是一个等待请求并按照它们的顺序处理的系统。

通常，网络服务器是多路复用（多线程、异步等）来同时处理各种请求。另一方面，流水线（以及大多数底层模型）在并行计算方面并不是特别出色；它们占用了大量的 RAM，因此最好在运行时为它们提供所有可用的资源，或者这是一个计算密集型的任务。我们将通过让网络服务器处理接收和发送请求的轻负载，并使用单个线程处理实际工作来解决这个问题。

此示例将使用 `starlette`。

实际框架并不是真正重要的，但如果您使用另一个框架，则可能需要调整或更改代码以实现相同的效果。创建 `server.py` 文件：



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
    pipe = pipeline(model="bert-base-uncased")
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

现在可以通过以下命令启动它：```bash
uvicorn server:app
```

然后可以查询它：```bash
curl -X POST -d "test [MASK]" http://localhost:8000/
#[{"score":0.7742936015129089,"token":1012,"token_str":".","sequence":"test."},...]
```

以上就是如何创建网络服务器的简要概述！

真正重要的是我们仅 **一次** 加载模型，因此在网络服务器上没有模型的多个副本。

这样，就不会浪费不必要的 RAM。然后，排队机制允许您执行一些高级操作，例如在推断之前积累一些项以使用动态批处理：

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

<Tip  warning={true}> 

在激活此功能之前，请确保检查其是否适用于您的负载！
</Tip>

所提供的代码的优化重点在于可读性，而不是成为最佳代码。

首先，没有批处理大小限制，这通常不是一个好主意。

接下来，超时在每次获取队列时被重置，这意味着在运行推断之前可能会等待比 1 毫秒更长的时间（延迟第一个请求）。




即使队列为空，这将始终等待 1 毫秒，这可能不是最佳选择，因为如果队列中没有内容，您可能希望开始执行推断。



但是，如果批处理对于您的用例确实非常重要，那么也许这是有意义的。再次强调，没有一个最佳解决方案。

## 您可能想要考虑的几件事

### 错误检查 Error checking

在生产环境中可能会出现很多问题：内存不足、空间不足，加载模型可能会失败，查询可能出错，查询可能因为模型配置错误而无法运行等等。


通常，如果服务器将错误输出给用户，添加许多 `try..except` 语句以显示这些错误是一个好主意。但请记住，根据您的安全上下文，这可能也是一种安全风险。


### 断路器 Circuit breaking

与其无限期地等待查询，不如在服务器超载时返回适当的错误。超载时返回 503 错误，而不是等待很长时间或在长时间后返回 504 错误。

在所提供的代码中，实现断路器相对容易，因为只有一个队列。通过查看队列大小，可以基本上开始在负载过高之前返回错误。


### 阻塞主线程 Blocking the main thread

目前，PyTorch 不支持异步操作，计算会阻塞主线程。这意味着如果强制 PyTorch 在自己的线程/进程上运行，效果会更好。这里没有这样做是因为代码更复杂（主要是因为线程、异步和队列之间无法很好地协同工作）。

但是，从根本上讲，它们执行的是相同的操作。
如果单个项的推理时间很长（> 1 秒），这将非常重要，因为在这种情况下，这意味着推断期间的每个查询在接收错误之前都必须等待 1 秒。


### 动态批处理 Dynamic batching

总体而言，与一次传递 1 个项目相比，批处理不一定是一种改进（有关更多信息，请参阅 [批处理细节](./main_classes/pipelines#pipeline-batching)）。但在正确的场景中使用它可能非常有效。在 API 中，默认情况下没有动态批处理（太多机会造成减速）。

但对于 BLOOM 推理来说，这是一款非常庞大的模型，动态批处理是提供良好体验的 **重要** 方法。

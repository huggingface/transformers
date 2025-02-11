<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Using pipelines for a webserver

<Tip>
Creating an inference engine is a complex topic, and the "best" solution 
will most likely depend on your problem space. Are you on CPU or GPU? Do
you want the lowest latency, the highest throughput, support for
many models, or just highly optimize 1 specific model?
There are many ways to tackle this topic, so what we are going to present is a good default
to get started which may not necessarily be the most optimal solution for you.
</Tip>


The key thing to understand is that we can use an iterator, just like you would [on a
dataset](pipeline_tutorial#using-pipelines-on-a-dataset), since a webserver is basically a system that waits for requests and
treats them as they come in.

Usually webservers are multiplexed (multithreaded, async, etc..) to handle various
requests concurrently. Pipelines on the other hand (and mostly the underlying models)
are not really great for parallelism; they take up a lot of RAM, so it's best to give them all the available resources when they are running or it's a compute-intensive job.

We are going to solve that by having the webserver handle the light load of receiving
and sending requests, and having a single thread handling the actual work.
This example is going to use `starlette`. The actual framework is not really
important, but you might have to tune or change the code if you are using another
one to achieve the same effect.

Create `server.py`:

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

Now you can start it with:
```bash
uvicorn server:app
```

And you can query it:
```bash
curl -X POST -d "test [MASK]" http://localhost:8000/
#[{"score":0.7742936015129089,"token":1012,"token_str":".","sequence":"test."},...]
```

And there you go, now you have a good idea of how to create a webserver!

What is really important is that we load the model only **once**, so there are no copies
of the model on the webserver. This way, no unnecessary RAM is being used.
Then the queuing mechanism allows you to do fancy stuff like maybe accumulating a few
items before inferring to use dynamic batching:

<Tip warning={true}>

The code sample below is intentionally written like pseudo-code for readability.
Do not run this without checking if it makes sense for your system resources!

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

Again, the proposed code is optimized for readability, not for being the best code.
First of all, there's no batch size limit which is usually not a 
great idea. Next, the timeout is reset on every queue fetch, meaning you could
wait much more than 1ms before running the inference (delaying the first request 
by that much). 

It would be better to have a single 1ms deadline.

This will always wait for 1ms even if the queue is empty, which might not be the
best since you probably want to start doing inference if there's nothing in the queue.
But maybe it does make sense if batching is really crucial for your use case.
Again, there's really no one best solution.


## Few things you might want to consider

### Error checking

There's a lot that can go wrong in production: out of memory, out of space,
loading the model might fail, the query might be wrong, the query might be
correct but still fail to run because of a model misconfiguration, and so on.

Generally, it's good if the server outputs the errors to the user, so
adding a lot of `try..except` statements to show those errors is a good
idea. But keep in mind it may also be a security risk to reveal all those errors depending 
on your security context.

### Circuit breaking

Webservers usually look better when they do circuit breaking. It means they 
return proper errors when they're overloaded instead of just waiting for the query indefinitely. Return a 503 error instead of waiting for a super long time or a 504 after a long time.

This is relatively easy to implement in the proposed code since there is a single queue.
Looking at the queue size is a basic way to start returning errors before your 
webserver fails under load.

### Blocking the main thread

Currently PyTorch is not async aware, and computation will block the main
thread while running. That means it would be better if PyTorch was forced to run
on its own thread/process. This wasn't done here because the code is a lot more
complex (mostly because threads and async and queues don't play nice together).
But ultimately it does the same thing.

This would be important if the inference of single items were long (> 1s) because 
in this case, it means every query during inference would have to wait for 1s before
even receiving an error.

### Dynamic batching

In general, batching is not necessarily an improvement over passing 1 item at 
a time (see [batching details](./main_classes/pipelines#pipeline-batching) for more information). But it can be very effective
when used in the correct setting. In the API, there is no dynamic
batching by default (too much opportunity for a slowdown). But for BLOOM inference -
which is a very large model - dynamic batching is **essential** to provide a decent experience for everyone.

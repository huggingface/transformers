<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Webサーバー用のパイプラインの使用

<Tip>
推論エンジンの作成は複雑なトピックであり、"最適な"ソリューションはおそらく問題の領域に依存するでしょう。CPUまたはGPUを使用していますか？最低のレイテンシ、最高のスループット、多くのモデルのサポート、または特定のモデルの高度な最適化を望んでいますか？
このトピックに取り組むための多くの方法があり、私たちが紹介するのは、おそらく最適なソリューションではないかもしれないが、始めるための良いデフォルトです。
</Tip>

重要なことは、Webサーバーはリクエストを待機し、受信したように扱うシステムであるため、[データセット](pipeline_tutorial#using-pipelines-on-a-dataset)のように、イテレータを使用できることです。

通常、Webサーバーは並列処理（マルチスレッド、非同期など）されて、さまざまなリクエストを同時に処理します。一方、パイプライン（および主にその基礎となるモデル）は並列処理にはあまり適していません。それらは多くのRAMを使用するため、実行中に利用可能なリソースをすべて提供するか、計算集約型のジョブである場合に最適です。

Webサーバーは受信と送信の軽い負荷を処理し、実際の作業を1つのスレッドで処理するようにします。この例では`starlette`を使用します。実際のフレームワークはあまり重要ではありませんが、別のフレームワークを使用している場合は、同じ効果を得るためにコードを調整または変更する必要があるかもしれません。

`server.py`を作成してください：


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

ここから始めることができます：
```bash
uvicorn server:app
```

そして、次のようにクエリできます：
```bash
curl -X POST -d "test [MASK]" http://localhost:8000/
#[{"score":0.7742936015129089,"token":1012,"token_str":".","sequence":"test."},...]
```



そして、これでウェブサーバーを作成する方法の良いアイデアを持っています！

本当に重要なのは、モデルを**一度だけ**ロードすることです。これにより、ウェブサーバー上にモデルのコピーがないため、不必要なRAMが使用されなくなります。
その後、キューイングメカニズムを使用して、動的バッチ処理を行うなど、いくつかのアイテムを蓄積してから推論を行うなど、高度な処理を行うことができます：

<Tip warning={true}>

以下のコードサンプルは、可読性のために擬似コードのように書かれています。システムリソースに合理的かどうかを確認せずに実行しないでください！

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

まず第一に、通常はあまり良いアイデアではないバッチサイズの制限がありません。次に、タイムアウトはキューの取得ごとにリセットされるため、推論を実行する前に1ms以上待つ可能性があります（最初のリクエストの遅延に1ms分遅れが生じます）。

1msの締め切りを1回だけ持つのが良いでしょう。

これは、キューに何もない場合でも常に1ms待機しますが、キューに何もない場合に推論を開始したい場合は適していないかもしれません。ただし、バッチ処理が本当に重要な場合には意味があるかもしれません。再度、1つの最適な解決策は存在しません。

## Few things you might want to consider

### Error checking

本番環境では多くの問題が発生する可能性があります：メモリ不足、スペース不足、モデルの読み込みが失敗するかもしれません、クエリが誤っているかもしれません、クエリが正しい場合でもモデルの構成エラーのために実行に失敗するかもしれませんなど。

一般的には、サーバーがエラーをユーザーに出力すると良いため、これらのエラーを表示するための多くの`try..except`ステートメントを追加することは良いアイデアです。ただし、セキュリティコンテキストに応じてこれらのエラーをすべて表示することはセキュリティリスクになる可能性があることに注意してください。

### Circuit breaking

Webサーバーは通常、過負荷時に正しいエラーを返す方が良いです。クエリを無期限に待つ代わりに適切なエラーを返します。長時間待つ代わりに503エラーを返すか、長時間待ってから504エラーを返すかです。

提案されたコードでは単一のキューがあるため、キューサイズを見ることは、Webサーバーが負荷に耐える前にエラーを返すための基本的な方法です。

### Blocking the main thread

現在、PyTorchは非同期を認識していないため、計算はメインスレッドをブロックします。つまり、PyTorchが独自のスレッド/プロセスで実行されるようにすると良いでしょう。提案されたコードは、スレッドと非同期とキューがうまく連携しないため、これは行われていませんが、最終的には同じことを行います。

これは、単一のアイテムの推論が長い場合（>1秒）に重要です。この場合、推論中にすべてのクエリが1秒待たなければならないことを意味します。

### Dynamic batching

一般的に、バッチ処理は1回のアイテムを1回渡すよりも改善されることは必ずしもありません（詳細は[バッチ処理の詳細](./main_classes/pipelines#pipeline-batching)を参照）。しかし、正しい設定で使用すると非常に効果的です。APIではデフォルトで動的バッチ処理は行われません（遅延の機会が多すぎます）。しかし、非常に大規模なモデルであるBLOOM推論の場合、動的バッチ処理は**重要**です。これにより、すべてのユーザーにとってまともなエクスペリエンスを提供できます。

以上が、提供されたテキストのMarkdown形式の翻訳です。

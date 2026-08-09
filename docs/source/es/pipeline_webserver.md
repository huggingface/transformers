<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Uso de un flujo de trabajo para un servidor web

<Tip>
Crear un motor de inferencia es un tema complejo, y la "mejor" solución probablemente dependerá de tu caso de uso. ¿Estás en CPU o en GPU? ¿Quieres la latencia más baja, el rendimiento más alto, soporte para muchos modelos o simplemente optimizar altamente un modelo específico? Hay muchas formas de abordar este tema, así que lo que vamos a presentar es un buen valor predeterminado para comenzar, que no necesariamente será la solución más óptima para ti.
</Tip>


Lo fundamental para entender es que podemos usar un iterador, tal como [en un conjunto de datos](pipeline_tutorial#uso-de-pipelines-en-un-conjunto-de-datos), ya que un servidor web es básicamente un sistema que espera solicitudes y las trata a medida que llegan.

Por lo general, los servidores web están multiplexados (multihilo, asíncrono, etc.) para manejar varias solicitudes simultáneamente. Por otro lado, los flujos de trabajo (y principalmente los modelos subyacentes) no son realmente ideales para el paralelismo; consumen mucha RAM, por lo que es mejor darles todos los recursos disponibles cuando se están ejecutando o es un trabajo intensivo en cómputo.

Vamos a resolver esto haciendo que el servidor web maneje la carga ligera de recibir y enviar solicitudes, y que un único hilo maneje el trabajo real. Este ejemplo va a utilizar `starlette`. El marco de trabajo no es realmente importante, pero es posible que debas ajustar o cambiar el código si estás utilizando otro para lograr el mismo efecto.

Crear `server.py`:

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

Ahora puedes empezar con:
```bash
uvicorn server:app
```

Y puedes consultarlo con:
```bash
curl -X POST -d "test [MASK]" http://localhost:8000/
#[{"score":0.7742936015129089,"token":1012,"token_str":".","sequence":"test."},...]
```

¡Y listo, ahora tienes una buena idea de cómo crear un servidor web!

Lo realmente importante es cargar el modelo solo **una vez**, de modo que no haya copias del modelo en el servidor web. De esta manera, no se utiliza RAM innecesariamente. Luego, el mecanismo de queuing (colas) te permite hacer cosas sofisticadas como acumular algunos elementos antes de inferir para usar el agrupamiento dinámico:

<Tip warning={true}>

El ejemplo de código a continuación está escrito intencionalmente como pseudocódigo para facilitar la lectura.
¡No lo ejecutes sin verificar si tiene sentido para los recursos de tu sistema!

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

Nuevamente, el código propuesto está optimizado para la legibilidad, no para ser el mejor código.
En primer lugar, no hay límite de tamaño de lote, lo cual generalmente no es una buena idea. Luego, el tiempo de espera se restablece en cada obtención de la cola, lo que significa que podrías esperar mucho más de 1ms antes de ejecutar la inferencia (retrasando la primera solicitud en esa cantidad).

Sería mejor tener un único plazo de 1ms.

Esto siempre esperará 1ms incluso si la cola está vacía, lo que podría no ser lo mejor ya que probablemente quieras comenzar a hacer inferencias si no hay nada en la cola. Pero tal vez tenga sentido si el agrupamiento es realmente crucial para tu caso de uso. Nuevamente, no hay una solución única y mejor.


## Algunas cosas que podrías considerar

### Comprobación de errores

Hay muchas cosas que pueden salir mal en producción: falta de memoria, falta de espacio, cargar el modelo podría fallar, la consulta podría ser incorrecta, la consulta podría ser correcta pero aún así fallar debido a una mala configuración del modelo, y así sucesivamente.

Generalmente, es bueno que el servidor muestre los errores al usuario, por lo que agregar muchos bloques `try..except` para mostrar esos errores es una buena idea. Pero ten en cuenta que también puede ser un riesgo de seguridad revelar todos esos errores dependiendo de tu contexto de seguridad.

### Interrupción de circuito

Los servidores web suelen verse mejor cuando hacen interrupciones de circuitos. Significa que devuelven errores adecuados cuando están sobrecargados en lugar de simplemente esperar la consulta indefinidamente. Devolver un error 503 en lugar de esperar un tiempo muy largo o un error 504 después de mucho tiempo.

Esto es relativamente fácil de implementar en el código propuesto ya que hay una sola cola. Mirar el tamaño de la cola es una forma básica de empezar a devolver errores antes de que tu servidor web falle bajo carga.

### Bloqueo del hilo principal

Actualmente, PyTorch no es consciente de la asincronía, y el cálculo bloqueará el hilo principal mientras se ejecuta. Esto significa que sería mejor si PyTorch se viera obligado a ejecutarse en su propio hilo/proceso. Esto no se hizo aquí porque el código es mucho más complejo (principalmente porque los hilos, la asincronía y las colas no se llevan bien juntos). Pero en última instancia, hace lo mismo.

Esto sería importante si la inferencia de elementos individuales fuera larga (> 1s) porque en este caso, significa que cada consulta durante la inferencia tendría que esperar 1s antes de recibir incluso un error.

### Procesamiento por lotes dinámico

En general, el procesamiento por lotes no es necesariamente una mejora respecto a pasar 1 elemento a la vez (ver [procesamiento por lotes](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching) para más información). Pero puede ser muy efectivo cuando se usa en el entorno correcto. En la API, no hay procesamiento por lotes dinámico por defecto (demasiada oportunidad para una desaceleración). Pero para la inferencia de BLOOM - que es un modelo muy grande - el procesamiento por lotes dinámico es **esencial** para proporcionar una experiencia decente para todos.

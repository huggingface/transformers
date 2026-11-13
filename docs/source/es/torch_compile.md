<!---Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# torch.compile

[torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) compila código de PyTorch en kernels fusionados para que se ejecute más rápido. Durante el entrenamiento, traza juntas la pasada hacia delante y la pasada hacia atrás y las compila en kernels optimizados, reduciendo la sobrecarga de lanzar operaciones por separado y fusionando operaciones para recortar el uso del ancho de banda de memoria.

Pon `torch_compile=True` en [`TrainingArguments`] para activarlo. El entrenamiento compila tanto la pasada hacia delante como la de hacia atrás, a diferencia de la inferencia, que solo compila la pasada hacia delante. La compilación ocurre en el primer paso de entrenamiento, así que es normal que sea bastante más lento que los pasos siguientes.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    torch_compile=True,
    torch_compile_backend="inductor",
    torch_compile_mode="reduce-overhead",
)
```

## Backend

El backend por defecto es `inductor`, que compila a kernels de Triton con AOTAutograd. Es la opción adecuada para la mayoría de las cargas de entrenamiento. Usa `cudagraphs` para entradas de forma fija, o `ipex` para entrenamiento en CPU de Intel.

## Modo de compilación

Usa la siguiente tabla como ayuda para elegir un modo de `torch.compile`.

| modo | descripción |
|---|---|
| default | equilibrio entre el tiempo de compilación y el de ejecución |
| reduce-overhead | reduce la sobrecarga de Python/CPU usando CUDA graphs a costa de algo más de memoria |
| max-autotune | prueba varias implementaciones de kernels durante la compilación y elige la más rápida (compilación más larga) |
| max-autotune-no-cudagraphs | igual que max-autotune pero sin CUDA graphs |

## Próximos pasos

- Consulta la guía [torch.compile para inferencia](./perf_torch_compile) para más detalles sobre la compilación fullgraph y los benchmarks de inferencia.

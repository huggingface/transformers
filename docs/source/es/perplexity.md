<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Perplejidad de modelos de longitud fija

[[open-in-colab]]

La perplejidad, perplexity en inglés (PPL), es una de las métricas más comunes para evaluar modelos de lenguaje. Antes de sumergirnos, debemos tener en cuenta que esta métrica se aplica específicamente a modelos de lenguaje clásicos (a veces llamados modelos autorregresivos o causales) y no está bien definida para modelos de lenguaje enmascarados como BERT (ver [resumen del modelo](model_summary)).

La perplejidad se define como el logaritmo exponenciado del promedio de la log-verosimilitud negativa de una secuencia. Si tenemos una secuencia tokenizada \(X = (x_0, x_1, \dots, x_t)\), entonces la perplejidad de \(X\) es,

$$\text{PPL}(X) = \exp \left\{ {-\frac{1}{t}\sum_i^t \log p_\theta (x_i|x_{<i}) } \right\}$$

donde \\(\log p_\theta (x_i|x_{<i})\\) es el logaritmo de la verosimilitud del token i-ésimo condicionado a los tokens precedentes \\(x_{<i}\\) según nuestro modelo. De manera intuitiva, se puede pensar en esto como una evaluación de la capacidad del modelo para predecir de manera uniforme entre el conjunto de tokens especificados en un corpus. Es importante destacar que el procedimiento de tokenización tiene un impacto directo en la perplejidad de un modelo, lo cual siempre debe tenerse en cuenta al comparar diferentes modelos.

Esto también es equivalente a la exponenciación de la entropía cruzada entre los datos y las predicciones del modelo. Para obtener más intuición sobre la perplejidad y su relación con los Bits Por Carácter (BPC) y la compresión de datos, echa un vistazo a esta [fantástica publicación en el blog de "The Gradient"](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/).

## Cálculo de PPL con modelos de longitud fija

Si no estuviéramos limitados por el tamaño del contexto de un modelo, evaluaríamos la perplejidad (PPL) del modelo auto regresivamente factorizando una secuencia y condicionándonos en toda la subsecuencia precedente en cada paso, como se muestra a continuación.

<img width="600" alt="Full decomposition of a sequence with unlimited context length" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_full.gif"/>

Sin embargo, al trabajar con modelos aproximados, generalmente tenemos una restricción en la cantidad de tokens que el modelo puede procesar. La versión más grande de [GPT-2](model_doc/gpt2), por ejemplo, tiene una longitud fija de 1024 tokens, por lo que no podemos calcular \\(p_\theta(x_t|x_{<t})\\) directamente cuando \\(t\\) es mayor que 1024.

En cambio, la secuencia se divide típicamente en subsecuencias iguales al tamaño máximo de entrada del modelo. Si el tamaño máximo de entrada, de un modelo es \\(k\\), entonces aproximamos la probabilidad de un token \\(x_t\\) condicionándonos solo en los \\(k-1\\) tokens que lo preceden en lugar de todo el contexto. Al evaluar la perplejidad del modelo en una secuencia, un enfoque tentador pero sub óptimo es dividir la secuencia en fragmentos independientes y sumar los logaritmos de la verosimilitud descompuestas de cada segmento de manera independiente.

<img width="600" alt="Suboptimal PPL not taking advantage of full available context" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_chunked.gif"/>

Esto es rápido de calcular, ya que la perplejidad de cada segmento se puede calcular en un solo pase hacia adelante, pero sirve como una aproximación pobre de la perplejidad completamente factorizada y generalmente dará como resultado una PPL más alta (peor) porque el modelo tendrá menos contexto en la mayoría de los pasos de predicción.

En cambio, la PPL de modelos de longitud fija debería evaluarse con una estrategia de ventana deslizante. Esto implica deslizar repetidamente la ventana de contexto para que el modelo tenga más contexto al hacer cada predicción.

<img width="600" alt="Sliding window PPL taking advantage of all available context" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_sliding.gif"/>

Esta es una aproximación más cercana a la verdadera descomposición de la probabilidad de la secuencia y generalmente dará como resultado una puntuación más favorable. La desventaja es que requiere un pase hacia adelante separado para cada token en el corpus. Un buen compromiso práctico es emplear una ventana deslizante estratificada, moviendo el contexto con pasos más grandes en lugar de deslizarse de 1 token a la vez. Esto permite que la computación avance mucho más rápido, mientras le da al modelo un contexto amplio para hacer
predicciones en cada paso.

## Example: Calculating perplexity with GPT-2 in 🤗 Transformers

Let's demonstrate this process with GPT-2.

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
```

We'll load in the WikiText-2 dataset and evaluate the perplexity using a few different sliding-window strategies. Since
this dataset is small and we're just doing one forward pass over the set, we can just load and encode the entire
dataset in memory.

```python
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
```

With 🤗 Transformers, we can simply pass the `input_ids` as the `labels` to our model, and the average negative
log-likelihood for each token is returned as the loss. With our sliding window approach, however, there is overlap in
the tokens we pass to the model at each iteration. We don't want the log-likelihood for the tokens we're just treating
as context to be included in our loss, so we can set these targets to `-100` so that they are ignored. The following
is an example of how we could do this with a stride of `512`. This means that the model will have at least 512 tokens
for context when calculating the conditional likelihood of any one token (provided there are 512 preceding tokens
available to condition on).

```python
import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
```

Running this with the stride length equal to the max input length is equivalent to the suboptimal, non-sliding-window
strategy we discussed above. The smaller the stride, the more context the model will have in making each prediction,
and the better the reported perplexity will typically be.

When we run the above with `stride = 1024`, i.e. no overlap, the resulting PPL is `19.44`, which is about the same
as the `19.93` reported in the GPT-2 paper. By using `stride = 512` and thereby employing our striding window
strategy, this jumps down to `16.45`. This is not only a more favorable score, but is calculated in a way that is
closer to the true autoregressive decomposition of a sequence likelihood.

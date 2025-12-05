<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Pipelines para inferência

Um [pipeline] simplifica o uso dos modelos no [Model Hub](https://huggingface.co/models) para a inferência de uma diversidade de tarefas,
como a geração de texto, a segmentação de imagens e a classificação de áudio.
Inclusive, se não tem experiência com alguma modalidade específica ou não compreende o código que forma os modelos,
pode usar eles mesmo assim com o [pipeline]! Este tutorial te ensinará a:

* Utilizar um [`pipeline`] para inferência.
* Utilizar um tokenizador ou model específico.
* Utilizar um [`pipeline`] para tarefas de áudio e visão computacional.

<Tip>

    Acesse a documentação do [`pipeline`] para obter uma lista completa de tarefas possíveis.

</Tip>

## Uso do pipeline

Mesmo que cada tarefa tenha um [`pipeline`] associado, é mais simples usar a abstração geral do [`pipeline`] que
contém todos os pipelines das tarefas mais específicas.
O [`pipeline`] carrega automaticamenta um modelo predeterminado e um tokenizador com capacidade de inferência para sua
tarefa.

1. Comece carregando um [`pipeline`] e especifique uma tarefa de inferência:

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="text-generation")
```

2. Passe seu dado de entrada, no caso um texto, ao [`pipeline`]:

```py
>>> generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Iron-priests at the door to the east, and thirteen for the Lord Kings at the end of the mountain'}]
```

Se tiver mais de uma entrada, passe-a como uma lista:

```py
>>> generator(
...     [
...         "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
...         "Nine for Mortal Men, doomed to die, One for the Dark Lord on his dark throne",
...     ]
... )
```

Qualquer parâmetro adicional para a sua tarefa também pode ser incluído no [`pipeline`]. A tarefa `text-generation` tem um método
[`~generation.GenerationMixin.generate`] com vários parâmetros para controlar a saída.
Por exemplo, se quiser gerar mais de uma saída, defina-a no parâmetro `num_return_sequences`:

```py
>>> generator(
...     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
...     num_return_sequences=2,
... )
```

### Selecionando um modelo e um tokenizador

O [`pipeline`] aceita qualquer modelo do [Model Hub](https://huggingface.co/models). Há rótulos adicionais no Model Hub
que te permitem filtrar pelo modelo que gostaria de usar para sua tarefa. Uma vez que tiver escolhido o modelo apropriado,
carregue-o com as classes `AutoModelFor` e [`AutoTokenizer`] correspondentes. Por exemplo, carregue a classe [`AutoModelForCausalLM`]
para uma tarefa de modelagem de linguagem causal:

```py
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

Crie uma [`pipeline`] para a sua tarefa e especifíque o modelo e o tokenizador que foram carregados:

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
```

Passe seu texto de entrada ao [`pipeline`] para gerar algum texto:

```py
>>> generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Dragon-lords (for them to rule in a world ruled by their rulers, and all who live within the realm'}]
```

## Pipeline de audio

A flexibilidade do [`pipeline`] significa que também pode-se extender às tarefas de áudio.
La flexibilidad de [`pipeline`] significa que también se puede extender a tareas de audio.

Por exemplo, classifiquemos a emoção de um breve fragmento do famoso discurso de John F. Kennedy /home/rzimmerdev/dev/transformers/docs/source/pt/pipeline_tutorial.md
Encontre um modelo de [audio classification](https://huggingface.co/models?pipeline_tag=audio-classification) para
reconhecimento de emoções no Model Hub e carregue-o usando o [`pipeline`]:

```py
>>> from transformers import pipeline

>>> audio_classifier = pipeline(
...     task="audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

Passe o arquivo de áudio ao [`pipeline`]:

```py
>>> audio_classifier("jfk_moon_speech.wav")
[{'label': 'calm', 'score': 0.13856211304664612},
 {'label': 'disgust', 'score': 0.13148026168346405},
 {'label': 'happy', 'score': 0.12635163962841034},
 {'label': 'angry', 'score': 0.12439591437578201},
 {'label': 'fearful', 'score': 0.12404385954141617}]
```

## Pipeline de visão computacional

Finalmente, utilizar um [`pipeline`] para tarefas de visão é praticamente a mesma coisa.
Especifique a sua tarefa de visão e passe a sua imagem ao classificador.
A imagem pode ser um link ou uma rota local à imagem. Por exemplo, que espécie de gato está presente na imagem?

![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

```py
>>> from transformers import pipeline

>>> vision_classifier = pipeline(task="image-classification")
>>> vision_classifier(
...     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
[{'label': 'lynx, catamount', 'score': 0.4403027892112732},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.03433405980467796},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.032148055732250214},
 {'label': 'Egyptian cat', 'score': 0.02353910356760025},
 {'label': 'tiger cat', 'score': 0.023034192621707916}]
```

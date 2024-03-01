<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Guia de Instalação

Neste guia poderá encontrar informações para a instalação do 🤗 Transformers para qualquer biblioteca de
Machine Learning com a qual esteja a trabalhar. Além disso, poderá encontrar informações sobre como gerar cachês e
configurar o 🤗 Transformers para execução em modo offline (opcional).

🤗 Transformers foi testado com Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, e Flax. Para instalar a biblioteca de
deep learning com que deseja trabalhar, siga as instruções correspondentes listadas a seguir:

* [PyTorch](https://pytorch.org/get-started/locally/)
* [TensorFlow 2.0](https://www.tensorflow.org/install/pip)
* [Flax](https://flax.readthedocs.io/en/latest/)

## Instalação pelo Pip

É sugerido instalar o 🤗 Transformers num [ambiente virtual](https://docs.python.org/3/library/venv.html). Se precisar
de mais informações sobre ambientes virtuais em Python, consulte este [guia](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
Um ambiente virtual facilitará a manipulação e organização de projetos e evita problemas de compatibilidade entre dependências.

Comece criando um ambiente virtual no diretório do seu projeto:

```bash
python -m venv .env
```

E para ativar o ambiente virtual:

```bash
source .env/bin/activate
```

Agora É possível instalar o 🤗 Transformers com o comando a seguir:

```bash
pip install transformers
```

Somente para a CPU, é possível instalar o 🤗 Transformers e a biblioteca de deep learning respectiva apenas numa linha.

Por exemplo, para instalar o 🤗 Transformers e o PyTorch, digite:

```bash
pip install transformers[torch]
```

🤗 Transformers e TensorFlow 2.0:

```bash
pip install transformers[tf-cpu]
```

🤗 Transformers e Flax:

```bash
pip install transformers[flax]
```

Por último, verifique se o 🤗 Transformers foi instalado com sucesso usando o seguinte comando para baixar um modelo pré-treinado:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

Em seguida, imprima um rótulo e sua pontuação:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## Instalação usando a fonte

Para instalar o 🤗 Transformers a partir da fonte use o seguinte comando:

```bash
pip install git+https://github.com/huggingface/transformers
```

O comando acima instalará a versão `master` mais atual em vez da última versão estável. A versão `master` é útil para
utilizar os últimos updates contidos em 🤗 Transformers. Por exemplo, um erro recente pode ter sido corrigido somente
após a última versão estável, antes que houvesse um novo lançamento. No entanto, há a possibilidade que a versão `master` não esteja estável.
A equipa trata de mantér a versão `master` operacional e a maioria dos erros são resolvidos em poucas horas ou dias.
Se encontrar quaisquer problemas, por favor abra um [Issue](https://github.com/huggingface/transformers/issues) para que o
mesmo possa ser corrigido o mais rápido possível.

Verifique que o 🤗 Transformers está instalado corretamente usando o seguinte comando:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## Instalação editável

Uma instalação editável será necessária caso desejas um dos seguintes:
* Usar a versão `master` do código fonte.
* Contribuir ao 🤗 Transformers e precisa testar mudanças ao código.

Para tal, clone o repositório e instale o 🤗 Transformers com os seguintes comandos:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Estes comandos vão ligar o diretório para o qual foi clonado o repositório ao caminho de bibliotecas do Python.
O Python agora buscará dentro dos arquivos que foram clonados além dos caminhos normais da biblioteca.
Por exemplo, se os pacotes do Python se encontram instalados no caminho `~/anaconda3/envs/main/lib/python3.7/site-packages/`,
o Python também buscará módulos no diretório onde clonamos o repositório `~/transformers/`.

<Tip warning={true}>

É necessário manter o diretório `transformers` se desejas continuar usando a biblioteca.

</Tip>

Assim, É possível atualizar sua cópia local para com a última versão do 🤗 Transformers com o seguinte comando:

```bash
cd ~/transformers/
git pull
```

O ambiente de Python que foi criado para a instalação do 🤗 Transformers encontrará a versão `master` em execuções seguintes.

## Instalação usando o Conda

É possível instalar o 🤗 Transformers a partir do canal conda `conda-forge` com o seguinte comando:

```bash
conda install conda-forge::transformers
```

## Configuração do Cachê

Os modelos pré-treinados são baixados e armazenados no cachê local, encontrado em `~/.cache/huggingface/transformers/`.
Este é o diretório padrão determinado pela variável `TRANSFORMERS_CACHE` dentro do shell.
No Windows, este diretório pré-definido é dado por `C:\Users\username\.cache\huggingface\transformers`.
É possível mudar as variáveis dentro do shell em ordem de prioridade para especificar um diretório de cachê diferente:

1. Variável de ambiente do shell (por padrão): `TRANSFORMERS_CACHE`.
2. Variável de ambiente do shell:`HF_HOME` + `transformers/`.
3. Variável de ambiente do shell: `XDG_CACHE_HOME` + `/huggingface/transformers`.

<Tip>

    O 🤗 Transformers usará as variáveis de ambiente do shell `PYTORCH_TRANSFORMERS_CACHE` ou `PYTORCH_PRETRAINED_BERT_CACHE`
    se estiver vindo de uma versão anterior da biblioteca que tenha configurado essas variáveis de ambiente, a menos que
    você especifique a variável de ambiente do shell `TRANSFORMERS_CACHE`.

</Tip>


## Modo Offline

O 🤗 Transformers também pode ser executado num ambiente de firewall ou fora da rede (offline) usando arquivos locais.
Para tal, configure a variável de ambiente de modo que `TRANSFORMERS_OFFLINE=1`.

<Tip>

Você pode adicionar o [🤗 Datasets](https://huggingface.co/docs/datasets/) ao pipeline de treinamento offline declarando
    a variável de ambiente `HF_DATASETS_OFFLINE=1`.

</Tip>

Segue um exemplo de execução do programa numa rede padrão com firewall para instâncias externas, usando o seguinte comando:

```bash
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

Execute esse mesmo programa numa instância offline com o seguinte comando:

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

O script agora deve ser executado sem travar ou expirar, pois procurará apenas por arquivos locais.

### Obtendo modelos e tokenizers para uso offline

Outra opção para usar o 🤗 Transformers offline é baixar os arquivos antes e depois apontar para o caminho local onde estão localizados. Existem três maneiras de fazer isso:

* Baixe um arquivo por meio da interface de usuário do [Model Hub](https://huggingface.co/models) clicando no ícone ↓.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)


* Use o pipeline do [`PreTrainedModel.from_pretrained`] e [`PreTrainedModel.save_pretrained`]:
    1. Baixa os arquivos previamente com [`PreTrainedModel.from_pretrained`]:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```


    2. Salve os arquivos em um diretório específico com [`PreTrainedModel.save_pretrained`]:

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. Quando estiver offline, acesse os arquivos com [`PreTrainedModel.from_pretrained`] do diretório especificado:

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* Baixando arquivos programaticamente com a biblioteca [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub):

    1. Instale a biblioteca [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub) em seu ambiente virtual:

    ```bash
    python -m pip install huggingface_hub
    ```

    2. Utiliza a função [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) para baixar um arquivo para um caminho específico. Por exemplo, o comando a seguir baixará o arquivo `config.json` para o modelo [T0](https://huggingface.co/bigscience/T0_3B) no caminho desejado:

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

Depois que o arquivo for baixado e armazenado no cachê local, especifique seu caminho local para carregá-lo e usá-lo:

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

Para obter mais detalhes sobre como baixar arquivos armazenados no Hub, consulte a seção [How to download files from the Hub](https://huggingface.co/docs/hub/how-to-downstream).

</Tip>

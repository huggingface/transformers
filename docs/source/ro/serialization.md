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

# Exportarea pentru producție

Exportă modelele Transformers în diferite formate pentru rulări și dispozitive optimizate. Dă deploy aceluiași model la furnizori de cloud sau rulează-l pe dispozitive mobile și edge. Nu trebuie să rescrii modelul de la zero pentru fiecare mediu de deployment. Dă deploy liber în orice ecosistem de inferență.

## ExecuTorch

[ExecuTorch](https://pytorch.org/executorch/stable/index.html) rulează modele PyTorch pe dispozitive mobile și edge. Exportă un model într-un graf de operatori standardizați, compilează graful într-un program ExecuTorch și îl execută pe dispozitivul țintă. Runtime-ul este ușor și calculează planul de execuție în avans.

Instalează [Optimum ExecuTorch](https://huggingface.co/docs/optimum-executorch/en/index) din codul sursă.

```bash
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install '.[dev]'
```

Exportă un model Transformers la ExecuTorch cu unealta CLI.

```bash
optimum-cli export executorch \
    --model "Qwen/Qwen3-8B" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="hf_smollm2"
```

Rulează următoarea comandă pentru a vizualiza toate opțiunile de export.

```bash
optimum-cli export executorch --help
```

## ONNX

[ONNX](http://onnx.ai) este un limbaj comun pentru descrierea modelelor din diferite framework-uri. Reprezintă modelele ca un graf de operatori standardizați cu tipuri, forme și metadate bine definite. Modelele sunt serializate în fișiere protobuf compacte pe care le poți deploya în runtimes și motoare optimizate.

[Optimum ONNX](https://huggingface.co/docs/optimum-onnx/index) exportă modele la ONNX cu obiecte de configurație. Suportă multe [arhitecturi](https://huggingface.co/docs/optimum-onnx/onnx/overview) și este ușor de extins. Exportă modele prin unealta CLI sau programatic.

Instalează [Optimum ONNX](https://huggingface.co/docs/optimum-onnx/index).

```bash
uv pip install optimum-onnx
```

### optimum-cli

Specifică un model de exportat și directorul de output cu argumentul `--model`.

```bash
optimum-cli export onnx --model Qwen/Qwen3-8B Qwen/Qwen3-8b-onnx/
```

Rulează următoarea comandă pentru a vizualiza toate argumentele disponibile sau consultă ghidul [Export a model to ONNX with optimum.exporters.onnx](https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model) pentru mai multe detalii.

```bash
optimum-cli export onnx --help
```

Pentru a exporta un model local, salvează fișierele de weights și tokenizer în același director. Pasează calea directorului argumentului `--model` și folosește argumentul `--task` pentru a specifica [task-ul](https://huggingface.co/docs/optimum/exporters/task_manager#transformers). Dacă nu furnizezi `--task`, sistemul îl inferează automat din model sau folosește o arhitectură fără un head specific task-ului.

```bash
optimum-cli export onnx --model path/to/local/model --task text-generation Qwen/Qwen3-8b-onnx/
```

Deployează modelul cu orice [runtime](https://onnx.ai/supported-tools.html#deployModel) care suportă ONNX, inclusiv ONNX Runtime.

```py
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b-onnx")
model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen3-8b-onnx")
inputs = tokenizer("Plants generate energy through a process known as ", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs))
```

### optimum.onnxruntime

Exportă modelele Transformers programatic cu Optimum ONNX. Instanțiază un [`~optimum.onnxruntime.ORTModel`] cu un model și setează `export=True`. Salvează modelul ONNX cu [`~optimum.onnxruntime.ORTModel.save_pretrained`].

```py
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

ort_model = ORTModelForCausalLM.from_pretrained("Qwen/Qwen3-8b", export=True)
tokenizer = AutoTokenizer.from_pretrained("onnx/")

ort_model.save_pretrained("onnx/")
tokenizer.save_pretrained("onnx/")
```

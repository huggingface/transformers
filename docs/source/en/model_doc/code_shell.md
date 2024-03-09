<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/6489a27bd0b2fd1f3297e5ca/3LQsqRzluBhBN2DipN6Ox.png" width="400"/>
<p>


<p align="center">
  🤗 <a href="https://huggingface.co/WisdomShell" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/WisdomShell" target="_blank">ModelScope</a> • ⭕️ <a href="https://www.wisemodel.cn/models/WisdomShell/CodeShell-7B" target="_blank">WiseModel</a> • 🌐 <a href="http://se.pku.edu.cn/kcl/" target="_blank">PKU-KCL</a> 
</p>

<div align="center">

[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/WisdomShell/codeshell/blob/main/License.pdf)
<h4 align="center">
    <p><a href="https://github.com/WisdomShell/codeshell/blob/main/README.md"><b>Chinese</b></a>|<a href="https://github.com/WisdomShell/codeshell/blob/main/README_EN.md">English</a></p>
</h4>
</div>

## Introduction

CodeShell is a code large language model (LLM) developed jointly by the [Knowledge Computing Lab at Peking University](http://se.pku.edu.cn/kcl/) and the AI team of Sichuan Tianfu Bank. CodeShell has 7 billion parameters, was trained on 500 billion tokens, and has a context window length of 8192. On authoritative code evaluation benchmarks (HumanEval and MBPP), CodeShell achieves the best performance for models of its scale. At the same time, we offer deployment solutions and IDE plugins that complement CodeShell. Please refer to the [CodeShell](https://github.com/WisdomShell/codeshell) repository for details.

The open-source models are as follows:

- <a href="https://huggingface.co/WisdomShell/CodeShell" target="_blank"><b>CodeShell Base</b></a>: The foundational model of CodeShell with strong coding capabilities.
- <a href="https://huggingface.co/WisdomShell/CodeShell-Chat" target="_blank"><b>CodeShell Chat</b></a>: A dialogue model of CodeShell that excels in code Q&A, code completion, and other downstream tasks.
- <a href="https://huggingface.co/WisdomShell/CodeShell-Chat-int4" target="_blank"><b>CodeShell Chat 4bit</b></a>: A 4bit quantized version of the CodeShell dialogue model. While preserving model performance, it consumes less memory and operates faster.
- <a href="https://github.com/WisdomShell/llama_cpp_for_codeshell" target="_blank"><b>CodeShell CPP</b></a>: A C++ version of the CodeShell dialogue model. It allows developers to use it on personal computers without GPUs. Note that the CPP version also supports quantization, allowing users to run CodeShell on PCs with a minimum of 8GB RAM.

## Main Characteristics of CodeShell

- **Powerful Performance**: CodeShell achieves optimal performance in 7B code base models on HumanEval and MBPP.
- **Complete Ecosystem**: In addition to the code model, IDE plugins for open-source (VS Code and JetBrains) are provided, forming a complete open-source technology stack.
- **Lightweight Deployment**: Supports local C++ deployment, providing a lightweight and fast local software development assistant solution.
- **Comprehensive Evaluation**: A multi-task evaluation system that supports a complete project context and covers common software development activities such as code generation, code defect detection and repair, and test case generation will be open-sourced soon.
- **Efficient Training**: Based on an efficient data governance system, CodeShell achieved excellent performance after training only 500 billion tokens from a complete cold start.

## Performance

We selected the two most popular code evaluation datasets (HumanEval and MBPP) to evaluate the model. Compared with the two most advanced 7B code models, CodeLlama and Starcoder, Codeshell achieved optimal results. The specific evaluation results are as follows.

|   Task   |  CodeShell-7b | CodeLlama-7b | Starcoder-7b |
| ------- | --------- | --------- | --------- |
| humaneval	 | **34.32** | 29.44 | 27.80 |
| mbpp		 | **38.65** | 37.60 | 34.16 |
| multiple-js	 | **33.17** | 31.30 | 27.02 |
| multiple-java	 | **30.43** | 29.24 | 24.30 |
| multiple-cpp	 | **28.21** | 27.33 | 23.04 |
| multiple-swift | 24.30 | **25.32** | 15.70 |
| multiple-php	 | **30.87** | 25.96 | 22.11 |
| multiple-d	 | 8.85 | **11.60** | 8.08 |
| multiple-jl	 | 22.08 | **25.28** | 22.96 |
| multiple-lua	 | 22.39 | **30.50** | 22.92 |
| multiple-r	 | **20.52** | 18.57 | 14.29 |
| multiple-rkt	 | **17.20** | 12.55 | 10.43 |
| multiple-rs	 | 24.55 | **25.90** | 22.82 |

## Requirements

- python 3.8 and above
- pytorch 2.0 and above are recommended
- transformers 4.32 and above
- CUDA 11.8 and above are recommended (for GPU users, flash-attention users, etc.)

## Quickstart

The CodeShell series models have been uploaded to <a href="https://huggingface.co/WisdomShell/CodeShell" target="_blank">Hugging Face</a>. Developers can quickly call CodeShell and CodeShell-Chat through Transformers.

Before starting, make sure you have set up the environment correctly, installed the necessary packages, and meet the environmental requirements from the previous section. The necessary dependencies can be installed quickly using the following code:

pip install -r requirements.txt


Next, you can use CodeShell through Transformers.

### Code Generation

Developers can use CodeShell to quickly generate code, accelerating development efficiency.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("WisdomShell/CodeShell-7B")
model = AutoModelForCausalLM.from_pretrained("WisdomShell/CodeShell-7B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
inputs = tokenizer('def merge_sort():', return_tensors='pt').cuda()
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

### Fill in the Middle
CodeShell supports the Fill-in-the-Middle mode to better assist the software development process.

```python
input_text = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
inputs = tokenizer(input_text, return_tensors='pt').cuda()
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

### Code Q&A
CodeShell has also open-sourced the CodeShell-7B-Chat code assistant model. Developers can interact with the model using the following code.

```python
model = AutoModelForCausalLM.from_pretrained('WisdomShell/CodeShell-7B-Chat', trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained('WisdomShell/CodeShell-7B-Chat')

history = []
query = 'Who are you?'
response = model.chat(query, history, tokenizer)
print(response)
history.append((query, response))

query = 'Write an HTTP server in Python'
response = model.chat(query, history, tokenizer)
print(response)
history.append((query, response))
```

Developers can also interact with CodeShell-7B-Chat through VS Code and JetBrains plugins. For details, please refer to the VSCode plugin repository and IntelliJ plugin repository.

### Model Quantization
CodeShell supports 4 bit/8 bit quantization. After 4-bit quantization, the memory footprint is approximately 6GB, allowing users to use CodeShell on GPUs with smaller memory.

```python
model = AutoModelForCausalLM.from_pretrained('WisdomShell/CodeShell-7B-Chat-int4', trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained('WisdomShell/CodeShell-7B-Chat-int4')
```

### CodeShell in c/c++
As most personal computers lack a GPU, CodeShell offers C/C++ inference support. Developers can compile based on the local environment. See CodeShell C/C++ local version. After compilation, the Web API service can be started with the following command.


## Demo
We offer demos in four forms: Web-UI, command line, API, and IDE.

### Web UI
Developers can start the Web service using the following command. After the service starts, it can be accessed at https://127.0.0.1:8000.

```
python demos/web_demo.py
```

### CLI Demo

We also offer a command-line interactive demo version. Developers can run it using the following command.

```
python cli_demo.py
```

### API

CodeShell also offers a deployment method based on the OpenAI API.

```
python openai_api.py
```

Then you can interact with CodeShell via HTTP requests:

```
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "CodeShell-7B-Chat",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ]
  }'
```

### IDE

Finally, CodeShell offers an online IDE. Developers can use the IDE for code completion, code Q&A, and other operations. IDE plugins are also released, and developers can install and use them locally. For plugin-related issues, please discuss in the VS Code plugin repository.

## Model Details

Code Shell uses GPT-2 as its basic architecture and employs technologies like Grouped-Query Attention and RoPE relative position encoding.

### Hyper-parameter

| Hyper-parameter | Value  |
|---|---|
| n_layer | 42 |
| n_embd | 4096 |
| n_inner | 16384 |
| n_head | 32 |
| num_query_groups | 8 |
| seq-length | 8192 |
| vocab_size | 70144 |


### Data

CodeShell was trained based on its own scraped Github data, the open-source Stack and StarCoder datasets from Big Code, as well as a small amount of high-quality Chinese and English data. On top of the original dataset, CodeShell used Minihash for data deduplication, KenLM, and a high-quality data selection model for data filtering and selection, resulting in a high-quality pre-training dataset.

### Tokenizer

CodeShell optimized the Starcoder vocabulary by removing infrequently used words and adding some Chinese vocabulary, significantly improving the Chinese compression rate, laying the groundwork for the training of the Chat version.


| Tokenizer | Size | Chinese  | English | Code | Total|
|---|---|---|---|---|---|
| Starcoder | 49152 | 1.22 | 3.47 | 3.30 | 2.66 |
| CodeShell | 70144 | 1.50 | 3.47 | 3.30 | 2.95 |


## License
The community's use of the CodeShell model must adhere to the ["CodeShell Model License Agreement" ](https://github.com/WisdomShell/codeshell/blob/main/License.pdf) and the [ Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). CodeShell is permitted for commercial use. However, if you plan to use the CodeShell model or its derivative products for commercial purposes, you must confirm that the entity meets the following conditions:

- The daily average active user count (DAU) of the affiliated party's service or product cannot exceed 1 million.
- The affiliated party must not be a software service provider or cloud service provider.
- There is no possibility for the affiliated party to re-license the granted commercial license to another third party without proper authorization.

Under the aforementioned conditions, you need to submit the application materials required by the "CodeShell Model License Agreement" by sending an email to codeshell.opensource@gmail.com. After approval, you will be granted a global, non-exclusive, non-transferable, non-sublicensable commercial copyright license.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WisdomShell/codeshell&type=Date)](https://star-history.com/#WisdomShell/codeshell&Date)


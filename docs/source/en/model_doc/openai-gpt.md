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

# OpenAI GPT

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[GPT (Generative Pre-trained Transformer)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) focuses on effectively learning text representations and transferring them to tasks. This model trains the Transformer decoder to predict the next word, and then fine-tuned on labeled data.

GPT can generate high-quality text, making it well-suited for a variety of natural language understanding tasks such as textual entailment, question answering, semantic similarity, and document classification.

In comparison to previous discriminative models, GPT achieves significant gains by utilizing a generative pretraining approach on a large, diverse corpus of unlabeled text, followed by fine-tuning for specific tasks. It achieves state-of-the-art results in a variety of benchmarks.

Key Improvements and Features:
Scalability: The model can scale efficiently with larger training datasets, improving performance on a wide range of NLP tasks.

Generative Pre-training: GPT leverages unsupervised pretraining on large text corpora, making it task-agnostic and highly adaptable.

Fine-Tuning Flexibility: After pretraining, GPT can be fine-tuned for specific tasks with minimal changes to the model architecture.


Official checkpoints can be found on the [Hugging Face Model Hub](https://huggingface.co/models).


## Usage tips

- Input Padding: Since this model uses absolute position embeddings, it is recommended to pad the inputs on the right side rather than the left. This ensures that the positional encodings align correctly with the token positions.

- Generative Pre-Training: The model was trained using a causal language modeling (CLM) objective, meaning it's excellent at predicting the next token in a sequence. This allows the model to generate coherent text, making it useful for tasks like text generation.

- Fine-Tuning: Fine-tuning is commonly done for specific downstream tasks like text classification, question answering, or summarization. It’s typically done by further training the model on task-specific datasets after the pre-training phase.

- Tokenization: If you wish to reproduce the original tokenization process from the model’s paper, you will need to install `ftfy` and `SpaCy`. This step ensures compatibility with the tokenization method used during the model’s original training.

  To install the necessary packages:

  ```bash
  pip install spacy ftfy==4.4.3
  python -m spacy download en
  ```

  If you skip this step, the tokenizer will default to using the BERT-style `BasicTokenizer` followed by Byte-Pair Encoding (BPE), which is sufficient for most tasks.

### Model Hyperparameters Table  

| Hyperparameter | Description | Default Value |  
|--------------|-------------|---------------|  
| **temperature** | Controls randomness; lower values make output more deterministic. | `1.0` |  
| **top_k** | Limits sampling to the top K most likely next words. | `50` |  
| **top_p** | Nucleus sampling; keeps the top cumulative probability mass of `p`. | `0.9` |  
| **max_length** | Maximum number of tokens the model generates. | `256` |  


## Resources

Here is a list of official Hugging Face and community (indicated by 🌎) resources to help you get started with OpenAI GPT. If you're interested in submitting a resource to be included here, feel free to open a Pull Request. We'll review it, especially if the resource demonstrates something new and doesn't duplicate existing resources.

### Text Classification
- [Blog: Outperforming OpenAI GPT-3 with SetFit for Text Classification](https://www.philschmid.de/getting-started-setfit)
- See also: [Text Classification Task Guide](../tasks/sequence_classification)

### Text Generation
- [Blog: Fine-Tune a Non-English GPT-2 Model with Hugging Face](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface)
- [Blog: How to Generate Text: Using Different Decoding Methods for Language Generation with Transformers](https://huggingface.co/blog/how-to-generate) (GPT-2)
- [Blog: Training CodeParrot 🦜 from Scratch](https://huggingface.co/blog/codeparrot) (Large GPT-2 model)
- [Blog: Faster Text Generation with TensorFlow and XLA](https://huggingface.co/blog/tf-xla-generate) (GPT-2)
- [Blog: How to Train a Language Model with Megatron-LM](https://huggingface.co/blog/megatron-training) (GPT-2)
- [Notebook: Fine-Tune GPT2 to Generate Lyrics in the Style of Your Favorite Artist](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb) 🌎
- [Notebook: Fine-Tune GPT2 to Generate Tweets in the Style of Your Favorite Twitter User](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb) 🌎
- [Hugging Face Course: Causal Language Modeling](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch)
- [Example Script: Causal Language Modeling with `OpenAIGPTLMHeadModel`](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)
- [Example Script: Text Generation with `OpenAIGPTLMHeadModel`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py)
- [Example Notebook: Language Modeling with `OpenAIGPTLMHeadModel`](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
- [TensorFlow Example Script: Causal Language Modeling with `TFOpenAIGPTLMHeadModel`](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy)
- [TensorFlow Example Notebook: Language Modeling with `TFOpenAIGPTLMHeadModel`](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)
- See also: [Causal Language Modeling Task Guide](../tasks/language_modeling)

### Token Classification
- [Course Material: Byte-Pair Encoding Tokenization](https://huggingface.co/course/en/chapter6/5)

## Model Usage

To use OpenAI GPT models via Hugging Face, follow these steps to get started with model inference, fine-tuning, and deployment. Hugging Face offers a simple interface for interacting with these models, whether you're using them for text generation, classification, or other NLP tasks.

### Load the Model

To begin using an OpenAI GPT model, load it directly using the Hugging Face `transformers` library. Here's how you can do it:

```python
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

# Load pre-trained model and tokenizer
model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
```

This loads the pre-trained OpenAI GPT model and tokenizer. The tokenizer is responsible for encoding text into token IDs, which are then fed into the model.

### Generate Text

Once the model and tokenizer are loaded, you can use them to generate text. Here’s an example of text generation with the GPT model:

```python
# Encode input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

This code will take an initial prompt ("Once upon a time") and generate text with a maximum length of 50 tokens.

### Fine-Tuning the Model

If you want to fine-tune the GPT model for your specific task, such as text generation for a specific style or topic, you can follow these steps:

1. **Prepare your dataset:** You need a text dataset that reflects the domain or style you want to fine-tune for.
2. **Fine-tune using `Trainer`:** Hugging Face provides the `Trainer` class for fine-tuning models efficiently.

```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to adopt during training
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size
    num_train_epochs=3,              # number of training epochs
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Provide your training dataset here
    eval_dataset=eval_dataset     # Provide your evaluation dataset here
)

# Train the model
trainer.train()
```

This will fine-tune the model on your dataset.

## Model Deployment

After fine-tuning, deploying your model is crucial to integrating it into your applications. You can deploy OpenAI GPT models using the Hugging Face `Transformers` library or through Hugging Face's Inference API.

### Deploy Using Hugging Face Inference API

The easiest way to deploy your fine-tuned GPT model is through Hugging Face's Inference API. It provides a scalable solution without requiring you to manage infrastructure. Here's how to deploy your model:

1. Push Your Model to Hugging Face Hub:

    First, ensure that your fine-tuned model is pushed to the Hugging Face Hub. You can do this by logging in to the Hugging Face website and pushing your model via the `transformers-cli` or Python API.

    ```bash
    transformers-cli login
    transformers-cli upload ./path_to_model --model_name=my_gpt_model
    ```

    After uploading, your model will be hosted on the Hugging Face Hub, and you can use the Inference API.

2. Use the Inference API:

    Once your model is on the Hugging Face Hub, you can call it using the Inference API like this:

    ```python
    from transformers import pipeline

    # Load the model from the Hugging Face Hub
    generator = pipeline('text-generation', model="username/my_gpt_model")

    # Use the model to generate text
    output = generator("Once upon a time", max_length=50)
    print(output)
    ```

    The model can now be used directly for inference without the need for managing infrastructure.

### Deploy Locally

If you prefer to deploy your model locally, you can do so using the Hugging Face `Transformers` library. This is useful for small-scale or internal applications.

Here’s how to load and deploy the model locally:

```python
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

# Load the fine-tuned model
model = OpenAIGPTLMHeadModel.from_pretrained("path_to_your_local_model")
tokenizer = OpenAIGPTTokenizer.from_pretrained("path_to_your_local_model")

# Generate text locally
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

This method ensures that the model is deployed on your machine, which is ideal if you require low-latency processing or wish to manage the model yourself.

### Deploy Using Cloud Solutions

You can also deploy the model using a cloud-based solution such as AWS, Google Cloud, or Azure. Here's how you can do this:

1. AWS SageMaker Deployment:

    AWS provides a fully managed service, **SageMaker**, for deploying machine learning models. You can upload your fine-tuned GPT model and use SageMaker to deploy and serve it with minimal overhead.

    Refer to the [AWS SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-deploy-model.html) for detailed instructions on deploying your GPT model.

2. Google Cloud AI Platform:

    Google Cloud’s AI Platform allows you to deploy models directly to their infrastructure. You can use this option if you are already using Google Cloud services.

    Refer to the [Google Cloud AI Platform documentation](https://cloud.google.com/ai-platform/docs) for detailed steps on deploying models.

3. Azure Machine Learning:

    Azure also provides a comprehensive solution for deploying models in the cloud using **Azure Machine Learning**. This is another option to consider for scalable deployment.

    Refer to the [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/) for deployment details.

## Advanced Usage

### Memory Efficient Techniques

When deploying large models like GPT, memory management becomes a crucial factor. Here are some advanced tips to help you optimize performance:

1. Model Parallelism: Split the model across multiple devices to reduce memory usage. This allows you to handle larger models and input sequences efficiently.
   
2. Mixed Precision Training: You can leverage **mixed precision** training to reduce memory usage and increase training throughput by using lower-precision arithmetic during training.

3. Gradient Checkpointing: If you need to train a larger model that does not fit into memory, you can use **gradient checkpointing** to save memory by only storing necessary activations.


### 📌 Notes
#### Model Limitations
- OpenAI's model may exhibit **biases** in its responses due to training on internet-scale data.
- Token limitations apply: **GPT models have a max token limit**, which can impact long conversations or large input texts.
- Performance trade-offs exist between **response coherence and generation diversity** when tuning parameters like `temperature` and `top_p`.

####  Optimal Hyperparameter Recommendations
| **Hyperparameter** | **Recommended Value** | **Use Case** |
|-------------------|----------------------|--------------|
| `temperature`    | 0.7 (default), 0.2 (for factual), 1.2 (for creativity) | Controls randomness |
| `top_k`         | 50 (default), 5 (for strict responses) | Limits vocab sampling |
| `top_p`         | 0.9 (default), 0.8 (for balanced), 0.95 (for diverse) | Nucleus sampling |
| `max_length`    | 512–1024 tokens | Adjust for long/short responses |
| `repetition_penalty` | 1.2 (default) | Reduces repetitive output |

For **fine-tuning**, use a **learning rate of 5e-5** and **batch size of 8–16**, depending on GPU memory availability.



## OpenAIGPTConfig

[[autodoc]] OpenAIGPTConfig

## OpenAIGPTTokenizer

[[autodoc]] OpenAIGPTTokenizer
    - save_vocabulary

## OpenAIGPTTokenizerFast

[[autodoc]] OpenAIGPTTokenizerFast

## OpenAI specific outputs

[[autodoc]] models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput

[[autodoc]] models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput

<frameworkcontent>
<pt>

## OpenAIGPTModel

[[autodoc]] OpenAIGPTModel
    - forward

## OpenAIGPTLMHeadModel

[[autodoc]] OpenAIGPTLMHeadModel
    - forward

## OpenAIGPTDoubleHeadsModel

[[autodoc]] OpenAIGPTDoubleHeadsModel
    - forward

## OpenAIGPTForSequenceClassification

[[autodoc]] OpenAIGPTForSequenceClassification
    - forward

</pt>
<tf>

## TFOpenAIGPTModel

[[autodoc]] TFOpenAIGPTModel
    - call

## TFOpenAIGPTLMHeadModel

[[autodoc]] TFOpenAIGPTLMHeadModel
    - call

## TFOpenAIGPTDoubleHeadsModel

[[autodoc]] TFOpenAIGPTDoubleHeadsModel
    - call

## TFOpenAIGPTForSequenceClassification

[[autodoc]] TFOpenAIGPTForSequenceClassification
    - call

</tf>
</frameworkcontent>

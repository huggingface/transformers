# Multimodal Large Language Models (LLMs) Overview

This document explores multimodal LLMs, which extend the capabilities of standard LLMs by integrating multiple types of inputs, such as text, images, and audio. Multimodal LLMs enable advanced applications in vision, audio, and language processing and offer unique setup and configuration options for each modality. Multimodal LLMs extend large language models are designed to understand and generate responses based on a variety of inputs, making them highly versatile for complex tasks that require interpreting visual and auditory context in addition to text.

They usually combine multiple sub-models to tackle tasks that involve various input modalities. These models are powerful tools for applications like image captioning, text-to-speech, and visual question answering, as they allow for flexible and scalable setups. This page covers what multimodal LLMs are, common types, how to set them up, and tips for effective configuration.


## 2. Types of Multimodal LLMs

Multimodal LLMs come in several forms, each designed to handle specific combinations of input data, such as text, images, and audio. These models are tailored to applications that require a blend of these data types, and their configurations can vary based on the nature of the input they process.

### Vision-Language LLMs

Vision-Language models can interpret both visual and textual information, making them ideal for tasks where understanding images in the context of descriptive text is essential. These models typically have a visual encoder (e.g., SigLIP, CLIP) paired with a text decoder (e.g., Llama, Qwen), where the visual encoder transforms images into tokens that are fed into the text processing layer.

- **Primary Use Cases**: Image captioning, visual question answering (VQA), and multimodal summarization.
- **Common Architectures**: Vision transformer combined with transformer-based language models.
- **Example Models**: [LLaVA](https://huggingface.co/docs/transformers/en/model_doc/llava), [PaliGemma](https://huggingface.co/docs/transformers/en/model_doc/paligemma).

TODO: Link to image-text-to-text guide can be here as further info

### Audio-Language LLMs

True Audio-Language LLMs are capable of directly processing spoken language inputs and generating conversational responses. Unlike simple speech-to-text models, Moshi integrates audio comprehension within a large language model, allowing it to respond contextually and dynamically to spoken inputs.

- **Primary Use Cases**: Conversational AI that responds to voice prompts in real time, voice-activated assistants, and applications requiring context-aware dialogue from audio inputs.
- **Common Architectures**: These models use a hybrid structure with a speech encoder and a generative LLM.
- **Example Model**: [Moshi](https://huggingface.co/docs/transformers/en/model_doc/moshi) and [Qwen2Audio](https://huggingface.co/docs/transformers/en/model_doc/qwen2_audio)



### Multiple Modality Models

These models integrate multiple input types, such as text, images, and audio, providing a unified framework for applications that require simultaneous interpretation of different data formats. Multiple Modality Models can leverage specialized encoders for each input type, with an adapter layer or shared cross-attention mechanism that aligns the representations across modalities.

- **Primary Use Cases**: Complex multimodal chatbots, immersive AR/VR environments, and holistic content generation that uses text, images, and audio.
- **Common Architectures**: Separate modality-specific encoders connected to a shared decoder with cross-modal attention layers.
- **Example Models**: 🤗 Transformers doesn't have a multiple modality LLM yet. Feel free to submit a PR if you have any good model in mind


## 3. Setting Up Multimodal LLMs

### Attention and Cross-Attention Mechanisms

Multimodal LLMs usually consist of modality specific encoder model and a separate language model. Sometimes one might want to use different configuration parameters to load each of the sub-models. For example, one can load the vision backbone in full precision for high-quality image feature extraction while setting the language backbone in half precision (`fp16`) to conserve memory and computational resources. This setup allows for flexible performance and memory trade-offs:

In the same way one might also want to set different attention implementations for each sub-model when loading. With 🤗 Transformers it can be achieved by passing a dictionary of  `attn_implementation` and `torch_dtype`. The dictionary keys should be identical to the keys in the model's configuration for each sub-model, and each model will then dispatch with its own `dtype` and `attn_implementation`. See below code snippet for an example usage.

```python
from transformers import LlavaForConditionalGeneration

vision_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    attn_implementation={"text_config": "flash_attention_2", "vision_config": "sdpa"},
    torch_dtype={"text_config": "float16", "vision_config": "float32"},
)
```


### Managing Input Length for Visual Inputs

Visual inputs, like images, are concatenated to the text ipnuts in some model architectures thus forming a long sequence of input embeddings.
To account for the place where each image should be concatenated, we use special `image` tokens which can be accessible via `processor.tokenizer.image_token`. When an input contains an image, it is usually embedded to be of around ~500 patches each image depending on the ViT backbone used and input image resolutions. Therefore, the `processor` expands input text by replicating an `image` placeholder token as many times as there will be image patches after embedding. That means you have to take into account how many vision inputs you are passing and make sure the input text is not truncated, otherwise it will cause index errors when tryong to merge image patches with text embeddings. 


### Chat Template Customization

Multimodal LLMs often require structured prompts to distinguish between different input types. Chat templates can help format inputs so the model knows when to expect image, text, or audio data. Multimodal models' chat template works in a similar way as LLMs with the only difference that you need to pass input images/videos as well along with the text. Therefore each "content" has to be a list containing either a text or an image/video content.

Here's an example of preparing input for using `LLaVA` model:

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id)  # You may want to use bfloat16 and/or move to GPU here
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
          {"type": "image", "image": "http://images.cocodataset.org/val2017/000000039769.jpg"},
          {"type": "text", "text": "What are these?"},
        ],
    },
]

processed_chat = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
print(processor.batch_decode(processed_chat["input_ids"][:, :30]))
```
This will yield a string in the input format that LLaVA expects with a bunch of `<image>` tokens at the end.
The `<image>`tokens are there as a placeholder and each one will be replaced by image embeddings when running the model
forward call. And the `processed_chat` can be further passed into `model.generate()` to generate text.
```text
'<|im_start|>system 
You are a friendly chatbot who always responds in the style of a pirate<|im_end|><|im_start|>user <image><image><image><image><image><image><image><image>'
```

Same way for audio model, one can pass input audio files directly into the chat template and get an already formatted and tokenized input text along with the processed audio features. 


```python
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Generate the caption in English:"},
            {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"},
        ]
    },
]
inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
print(processor.batch_decode(processed_chat["input_ids"]))
```



### Multimodal Tokenization

One might also need to set model-specific special tokens when the tokenizer is used as part of a larger multimodal model. Multimodal tokenizers with any extra special tokens is what we can use in such cases. It means that the tokenizer can hold any arbitrary tokens in its `special tokens` and thus one can have easier access to those tokens by simply getting tokenizer's attribute. For example, if the tokenizer is loaded from a vision-language model like LLaVA, you will
need access to `tokenizer.image_token_id` to obtain the special image token used as a placeholder. 

To enable extra special tokens for any type of tokenizer, you have to add the following lines and save the tokenizer. Extra special tokens do not
have to be modality related and can ne anything that the model often needs access to. In the below code, tokenizer at `output_dir` will have direct access
to three more special tokens.  

```python
vision_tokenizer = AutoTokenizer.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    extra_special_tokens={"image_token": "<image>", "boi_token": "<image_start>", "eoi_token": "<image_end>"}
)
print(vision_tokenizer.image_token, vision_tokenizer.image_token_id)
("<image>", 32000)
```


## 4. Best Practices

### Some tips for optimizing multimodal LLMs:

Memory Management: Set appropriate max lengths for each modality to prevent overloading.
Tokenization Strategy: Use specialized multimodal tokenizers to handle complex input formats.
Fine-Tuning Approaches: Train on each modality separately first, then combine for end-to-end training.

## 5. Examples and Code Snippets

### Vision-Language Model Example

```python
from transformers import VisionEncoderDecoderModel, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("google/vit-gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Process image and text
image = ...  # Preprocess the image input
text_input = tokenizer("Describe the image.", return_tensors="pt")
output = model(image, text_input)

print("Generated caption:", tokenizer.decode(output[0], skip_special_tokens=True))
```


#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# docstyle-ignore
RUN_PROMPT_TEMPLATE = """I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignment. You can print intermediate results if it makes sense to do so.

Tools:
<<all_tools>>


Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.

Answer:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
print(f"The answer is {answer}")
```

Task: "Identify the oldest person in the `document` and create an image showcasing the result."

I will use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator(answer)
```

Task: "Generate an image using the text given in the variable `caption`."

I will use the following tool: `image_generator` to generate an image.

Answer:
```py
image = image_generator(prompt=caption)
```

Task: "Summarize the text given in the variable `text` and read it out loud."

I will use the following tools: `summarizer` to create a summary of the input text, then `text_reader` to read it out loud.

Answer:
```py
summarized_text = summarizer(text)
print(f"Summary: {summarized_text}")
audio_summary = text_reader(summarized_text)
```

Task: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."

I will use the following tools: `text_qa` to create the answer, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = text_qa(text=text, question=question)
print(f"The answer is {answer}.")
image = image_generator(answer)
```

Task: "Caption the following `image`."

I will use the following tool: `image_captioner` to generate a caption for the image.

Answer:
```py
caption = image_captioner(image)
```

Task: "<<prompt>>"

I will use the following"""


# docstyle-ignore
CHAT_PROMPT_TEMPLATE = """Below are a series of dialogues between various people and an AI assistant specialized in coding. The AI assistant tries to be helpful, polite, honest, and humble-but-knowledgeable.

The job of the AI assistant is to come up with a series of simple commands in Python that will perform the task the human wants to perform.
To help with that, the AI assistant has access to a set of tools. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
The AI assistant should first explain the tools it will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignment. The AI assistant can print intermediate results if it makes sense to do so.

Tools:
<<all_tools>>

=====

Human: Answer the question in the variable `question` about the image stored in the variable `image`.

Assistant: I will use the tool `image_qa` to answer the question on the input image.

```py
answer = image_qa(text=question, image=image)
print(f"The answer is {answer}")
```

Human: I tried this code, it worked but didn't give me a good result. The question is in French

Assistant: In this case, the question needs to be translated first. I will use the tool `translator` to do this.

```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(text=translated_question, image=image)
print(f"The answer is {answer}")
```

=====

Human: Identify the oldest person in the `document`.

Assistant: I will use the tool `document_qa` to find the oldest person in the document.

```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
```

Human: Can you generate an image with the result?

Assistant: I will use the tool `image_generator` to do that.

```py
image = image_generator(answer)
```

=====

Human: Summarize the text given in the variable `text` and read it out loud.

Assistant: I will use the tool `summarizer` to create a summary of the input text, then the tool `text_reader` to read it out loud.

```py
summarized_text = summarizer(text)
print(f"Summary: {summarized_text}")
audio_summary = text_reader(text=summary)
```

Human: I got the following error: "The variable `summary` is not defined."

Assistant: My bad! Let's try this code instead.

```py
summarized_text = summarizer(text)
print(f"Summary: {summarized_text}")
audio_summary = text_reader(text=summarized_text)
```

Human: It worked! Can you translate the summary in German?

Assistant: I will use the tool `translator` to translate the text in German.

```py
translated_summary = translator(summarized_text, src_lang="English", tgt_lang="German")
```

====
"""


# docstyle-ignore
CHAT_MESSAGE_PROMPT = """
Human: <<task>>

Assistant: """

# docstyle-ignore
RUN_PROMPT_TEMPLATE = """I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignement. You can print intermediate results if it makes sense to do so.
The final result should be stored in a variable named `result`. You can also print the result if it makes sense to do so.

Tools:
<<all_tools>>


Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

I will use the following tools: `translator` to translate the question in English and then `image_qa` to answer the question on the input image.

Answer:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
result = image_qa(image=image, question=translated_question)
print(f"The answer is {result}")
```

Task: "Identify the oldest person in the `table` and create an image showcasing the result as a banner."

I wil use the following tools: `table_qa` to find the oldest person in the table, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = table_qa(table=table, question="What is the oldest person?")
print(f"The answer is {answer}.")
result = image_generator("A banner showing " + answer)
```

Task: "Generate an image using the text given in the variable `caption`."

I will use the following tool: `image_generator` to generate an image.

Answer:
```py
result = image_generator(text=caption)
```

Task: "Summarize the text given in the variable `text` and read it out loud."

I will use the following tools: `summarizer` to create a summary of the input text, then `text_reader` to read it out loud.

Answer:
```py
summarized_text = summarizer(text)
print(f"Summary: {summarized text}")
result = text_reader(summarized_text)
```

Task: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."

I will use the following tools: `text_qa` to create the answer, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = text_qa(text=text, question=question)
print(f"The answer is {answer}.")
result = image_generator(answer)
```

Task: "Caption the following `image`."

I will use the following tool: `image_captioner` to generate a caption for the image.

Answer:
```py
text = image_captioner(image)
```

Task: "<<prompt>>"

I will use the following"""


# docstyle-ignore
CHAT_PROMPT_TEMPLATE = """Below are a series of dialogues between various people and an AI assistant specialized in coding. The AI tries to be helpful, polite, honest, and humble-but-knowledgeable.

The job of the AI assistant is to come up with a series of simple commands in Python that will peform the task the human wants to perform.
To help with that, the AI assistant has access to a set of tools. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
The AI assistant should first explain the tools it will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignement. The AI assitant can print intermediate results if it makes sense to do so.
The final result should be stored in a variable named `result`. The AI assistant can also print the result if it makes sense to do so.

Tools:
<<all_tools>>

=====

Human: Answer the question in the variable `question` about the image stored in the variable `image`.

Assistant: I will use the tool `image_qa` to answer the question on the input image.

```py
result = image_qa(text=question, image=image)
print(f"The answer is {result}")
```

Human: I tried this code but it worked but didn't give me a good result. The question is in French

Assistant: In this case, the question needs to be translated first. I will use the tool `translator` to do this.

```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
result = image_qa(text=translated_question, image=image)
print(f"The answer is {result}")
```

=====

Human: Identify the oldest person in the `table`.

Assistant: I will use the tool `table_qa` to find the oldest person in the table.

```py
result = table_qa(table=table, question="What is the oldest person?")
print(f"The answer is {answer}.")
```

Human: Can you generate an image with the result?

Assistant: I will use the tool `image_generator` to do that.

```py
answer = table_qa(table=table, question="What is the oldest person?")
print(f"The answer is {answer}.")
result = image_generator(prompt="A banner showing " + answer)
```

=====

Human: Summarize the text given in the variable `text` and read it out loud.

Assistant: I will use the tool `summarizer` to create a summary of the input text, then the tool `text_reader` to read it out loud.

```py
summarized_text = summarizer(text)
print(f"Summary: {summarized text}")
audio = text_reader(text=summarized_text)
```

Human: I got the following error: "`result` not found in the code."

Assistant: My bad! Let's try this code instead.

```py
summarized_text = summarizer(text)
print(f"Summary: {summarized text}")
result = text_reader(text=summarized_text)
```

Human: I worked! Can you translate the summary in German?

Assistant: I will use the tool `translator` to translate the text in German.

```py
summarized_text = summarizer(text)
print(f"Summary: {summarized text}")
audio = text_reader(text=summarized_text)
result = translator(summarized_text, src_lang="English", tgt_lang="German)
```

====
"""


# docstyle-ignore
CHAT_MESSAGE_PROMPT = """
Human: <<task>>

Assistant: """

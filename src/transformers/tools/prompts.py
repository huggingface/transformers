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
import re

from ..utils import cached_file


# docstyle-ignore
CHAT_MESSAGE_PROMPT = """
Human: <<task>>

Assistant: """


DEFAULT_PROMPTS_REPO = "huggingface-tools/default-prompts"
PROMPT_FILES = {"chat": "chat_prompt_template.txt", "run": "run_prompt_template.txt"}


def download_prompt(prompt_or_repo_id, agent_name, mode="run"):
    """
    Downloads and caches the prompt from a repo and returns it contents (if necessary)
    """
    if prompt_or_repo_id is None:
        prompt_or_repo_id = DEFAULT_PROMPTS_REPO

    # prompt is considered a repo ID when it does not contain any kind of space
    if re.search("\\s", prompt_or_repo_id) is not None:
        return prompt_or_repo_id

    prompt_file = cached_file(
        prompt_or_repo_id, PROMPT_FILES[mode], repo_type="dataset", user_agent={"agent": agent_name}
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


DEFAULT_CODE_SYSTEM_PROMPT = """I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignment. You can print intermediate results if it makes sense to do so.

Tools:
<<tool_descriptions>>


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

Be sure to provide an 'Answer:' token, else the system will be stuck in a loop.

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
Above example were using tools that might not exist directly for you. You only have acces to those Tools:
<<tool_names>>

Now Begin !
"""


DEFAULT_REACT_SYSTEM_PROMPT = """Solve the following task as best you can. You have access to the following tools:

<<tool_descriptions>>

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (name of the tool to use) and a `action_input` key (input to the tool).

The value in the "action" field should belong to this list: <<tool_names>>.

The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is an example of a valid $ACTION_JSON_BLOB:
Action:
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You will be given:

Task: the task you are given.

You should ALWAYS use the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

ALWAYS provide a 'Thought:' and an 'Action:' sequence. You MUST provide at least the 'Action:' sequence to move forward.
To provide the final answer to the task, use an action blob with "action": 'final_answer' tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
Action:
{
  "action": 'final_answer',
  "action_input": "insert your final answer here"
}

Now begin!
"""


REACT_TEXT_SYSTEM_PROMPT = """Solve the following task as best you can. You have access to the following tools:

<<tool_descriptions>>

You will be provided a task:
Task: the task you are given.

Then you should answer as follows:

Thought: you should always think about one action to take.
Action: "$TOOL_NAME"
Action input: $ACTION_INPUT
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed)

The 'Action:' part must only use a SINGLE tool name at a time. $TOOL_NAME must have quotes.

The 'Action input:' should be the input for the tool you are using.
Make sure to have the $ACTION_INPUT as a dictionnary, with the right keys for the tool you are using, and do not put variable names as input if you know the right values.

You MUST ALWAYS provide a 'Thought:', an 'Action:', and an 'Action Input:' part to move forward.

To provide the final answer to the task, use 'Action: "final_answer"'. So your final output should look like this:

Action: "final_answer"
Action input: {"answer": "your final answer here"}

Now begin!
"""
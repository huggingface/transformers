import importlib.util
import os
import time

import requests

from .python_interpreter import evaluate


# Move to util when this branch is ready to merge
def is_openai_available():
    return importlib.util.find_spec("openai") is not None


if is_openai_available():
    import openai


BASE_PYTHON_TOOLS = {"print": print, "float": float, "int": int, "bool": bool, "str": str}


# docstyle-ignore
PROMPT_TEMPLATE = """I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
Each instruction in Python should be a simple assignement. You can print intermediate results if it makes sense to do so.
The final result should be stored in a variable named `result`. You can also print the result if it makes sense to do so.
You should only use the tools necessary to perform the task.

Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

Tools:
- tool_0: This is a tool that translates text from French to English. It takes an input named `text` which should be the text in French and returns the text translated in Enlish.
- tool_1: This is a tool that generates speech from a given text in English. It takes an input named `text` which should be the text in English and returns the path to a filename containing an audio of this text read.
- tool_2: This is a tool that answers question about images. It takes an input named `text` which should be the question in English and an input `image` which should be an image, and outputs a text that is the answer to the question.

Answer:
```py
translated_question = tool_0(text=question)
print(f"The translated question is {translated_question}.")
result = tool_2(text=translated_question, image=image)
print(f"The answer is {result}")
```

Task: "Identify the oldest person in the table and create an image showcasing the result as a banner."

Tools:
- tool_1: This is a tool that generates speech from a given text in English. It takes an input named `text` which should be the text in English and returns the path to a filename containing an audio of this text read.
- tool_1: This is a tool that creates an image according to a text description. It takes an input named `text` which contains the image description and outputs an image.
- tool_2: This is a tool that reads an English text out loud. It takes an input named `text` which whould contain the text to read (in English) and returns a waveform object containing the sound.
- tool_3: This is a tool that generates a description of an image. It takes an input named `image` which should be the image to caption, and returns a text that contains the description in English.
- tool_4: This is a tool that reads a table and answers a question related to the table. It takes an input named `table` which should be the table containing the date, as well as a `question` to be asked relative to the table. It returns the answer in text.
- tool_5: This is a tool that creates an image according to a prompt, and specializes in generating text. It takes an input named `prompt` which contains the image description as well as the specified text, and outputs an image.

Answer:
```py
answer = tool_4(table=table, question="What is the oldest person?")
print(f"The answer is {answer}.")
result = tool_5(prompt="A banner showing " + answer)
```

Task: "Generate an image using the text given in the variable `caption`."

Tools:
- tool_0: This is a tool that reads an English text out loud. It takes an input named `text` which whould contain the text to read (in English) and returns a waveform object containing the sound.
- tool_1: This is a tool that generates a description of an image. It takes an input named `image` which should be the image to caption, and returns a text that contains the description in English.
- tool_2: This is a tool that creates an image according to a text description. It takes an input named `text` which contains the image description and outputs an image.

Answer:
```py
result = tool_2(text=caption)
```

Task: "Summarize the text given in the variable `text` and read it out loud."

Tools:
- tool_0: This is a tool that summarizes long texts. It takes an input named `text` which should be the text to summarize, and returns a summary of the text.
- tool_1: This is a tool that transforms an image according to a prompt. It takes two inputs: `image`, which should be the image to transform, and `prompt`, which should be the prompt to use to change it. It returns the modified image.
- tool_2: This is a tool that reads an English text out loud. It takes an input named `text` which whould contain the text to read (in English) and returns a waveform object containing the sound.
- tool_3: This is a tool that translates text from German to French. It takes an input named `text` which should be the text in German and returns a dictionary with a single key `'translated_text'` that contains the translation in French.

Answer:
```py
summarized_text = tool_0(text)
print(f"Summary: {summarized text}")
result = tool_2(text=summarized_text)
```

Task: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."

Tools:
- tool_0: This is a tool that answers questions related to a text. It takes two arguments named `text`, which is the text where to find the answer, and `question`, which is the question, and returns the answer to the question.
- tool_1: This is a tool that creates an image according to a text description. It takes an input named `text` which contains the image description and outputs an image.

Answer:
```py
answer = tool_0(text=text, question=question)
print(f"The answer is {answer}.")
result = tool_1(text=answer)
```

Task: "Caption the following `image`."

Tools:
- tool_0: This is a tool that identifies the language of the text passed as input. It takes one input named `text` and returns the two-letter label of the identified language.
- tool_1: This is a tool that reads a table and answers a question related to the table. It takes an input named `table` which should be the table containing the date, as well as a question to be asked relative to the table. It returns the answer in text.

Answer:
```py
text = tool_1(image)
```

Task: "<<prompt>>"

Tools:
<<tools>>

Answer:
"""


class Agent:
    prompt_template = PROMPT_TEMPLATE

    def format_prompt(self, task, tools):
        tool_descs = [f"- tool_{i}: {tool.description}" for i, tool in enumerate(tools)]
        prompt = self.prompt_template.replace("<<prompt>>", task)
        return prompt.replace("<<tools>>", "\n".join(tool_descs))

    def clean_code(self, code):
        code_lines = code.split("\n")
        in_block_code = "```" in code_lines[0]
        additional_explanation = []
        if in_block_code:
            code_lines = code_lines[1:]
        for idx in range(len(code_lines)):
            if in_block_code and "```" in code_lines[idx]:
                additional_explanation = code_lines[idx + 1 :]
                code_lines = code_lines[:idx]
                break

        clean_code = "\n".join(code_lines)
        explanation = "\n".join(additional_explanation).strip()
        if explanation.startswith("Task:"):
            explanation = ""

        return clean_code, explanation

    def run(self, task, tools, return_code=False, **kwargs):
        code = self.generate_code(task, tools)
        clean_code, explanation = self.clean_code(code)

        all_tools = BASE_PYTHON_TOOLS.copy()
        all_tools.update({f"tool_{idx}": tool for idx, tool in enumerate(tools)})

        print(f"==Code generated by the agent==\n{clean_code}")
        if len(explanation) > 0:
            print(f"\n\n==Additional explanation from the agent==\n{explanation}")

        if not return_code:
            print("\n\n==Result==")
            return evaluate(clean_code, all_tools, kwargs)
        else:
            return clean_code


class EndpointAgent(Agent):
    def __init__(self, url_endpoint, token):
        self.url_endpoint = url_endpoint
        # TODO: remove the Basic support later on and then also use the HF token stored by default.
        self.token = f"Basic {token}" if "Basic" not in token else token

    def generate_code(self, task, tools):
        is_batched = isinstance(task, list)

        if is_batched:
            prompts = [self.format_prompt(one_task, one_tools) for one_task, one_tools in zip(task, tools)]
        else:
            prompts = [self.format_prompt(task, tools)]

        # Can probably batch those but can't test anymore right now as the endpoint has been limited in length.
        results = [self._generate_one(prompt) for prompt in prompts]
        return results if is_batched else results[0]

    def _generate_one(self, prompt):
        headers = {"Authorization": self.token}
        inputs = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "return_full_text": False},
        }

        response = requests.post(self.url_endpoint, json=inputs, headers=headers)
        if response.status_code == 429:
            print("Getting rate-limited, waiting a tiny bit before trying again.")
            time.sleep(1)
            return self._generate_one(prompt)
        elif response.status_code != 200:
            raise ValueError(f"Error {response.status_code}: {response.json()}")

        return response.json()[0]["generated_text"]


class OpenAiAgent(Agent):
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        if not is_openai_available():
            raise ImportError("Using `OpenAIAgent` requires `openai`: `pip install openai`.")

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "You need an openai key to use `OpenAIAgent`. You can get one here: Get one here "
                "https://openai.com/api/`. If you have one, set it in your env with `os.environ['OPENAI_API_KEY'] = "
                "xxx."
            )
        else:
            openai.api_key = api_key
        self.model = model

    def generate_code(self, task, tools):
        is_batched = isinstance(task, list)

        if is_batched:
            prompts = [self.format_prompt(one_task, one_tools) for one_task, one_tools in zip(task, tools)]
        else:
            prompts = [self.format_prompt(task, tools)]

        if "gpt" in self.model:
            results = [self._chat_generate(prompt) for prompt in prompts]
        else:
            results = self._completion_generate(prompts)

        return results if is_batched else results[0]

    def _chat_generate(self, prompt):
        result = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop="Task:",
        )
        return result["choices"][0]["message"]["content"]

    def _completion_generate(self, prompts):
        result = openai.Completion.create(
            model=self.model,
            prompt=prompts,
            temperature=0,
            stop="Task:",
            max_tokens=200,
        )
        return [answer["text"] for answer in result["choices"]]

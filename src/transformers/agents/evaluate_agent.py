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
from .agents import BASE_PYTHON_TOOLS
from .python_interpreter import InterpreterError, evaluate


### Fake tools for test
def classifier(text, labels):
    return f"This is the classification of {text} along {labels}."


def translator(text, src_lang, tgt_lang):
    return f"This is the translation of {text} from {src_lang} to {tgt_lang}."


def speaker(text):
    return f"This is actually a sound reading {text}."


def transcriber(audio):
    if "sound" not in audio:
        raise ValueError(f"`audio` ({audio}) is not a sound.")
    return f"This is the transcribed text from {audio}."


def image_generator(prompt):
    return f"This is actually an image representing {prompt}."


def image_captioner(image):
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    return f"This is a description of {image}."


def image_transformer(image, prompt):
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    return f"This is a transformation of {image} according to {prompt}."


def question_answerer(text, question):
    return f"This is the answer to {question} from {text}."


def image_qa(image, question):
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    return f"This is the answer to {question} from {image}."


def text_downloader(url):
    return f"This is the content of {url}."


def summarizer(text):
    return f"This is a summary of {text}."


def video_generator(prompt, seconds=2):
    return f"A video of {prompt}"


def document_qa(image, question):
    return f"This is the answer to {question} from the document {image}."


def image_segmenter(image, prompt):
    return f"This is the mask of {prompt} in {image}"


TEST_TOOLS = {
    "text_classifier": classifier,
    "translator": translator,
    "text_reader": speaker,
    "summarizer": summarizer,
    "transcriber": transcriber,
    "image_generator": image_generator,
    "image_captioner": image_captioner,
    "image_transformer": image_transformer,
    "text_qa": question_answerer,
    "text_downloader": text_downloader,
    "image_qa": image_qa,
    "video_generator": video_generator,
    "document_qa": document_qa,
    "image_segmenter": image_segmenter,
}


class Problem:
    """
    A class regrouping all the information to solve a problem on which we will evaluate agents.

    Args:
        task (`str` ou `list[str]`):
            One or several descriptions of the task to perform. If a list, it should contain variations on the
            phrasing, but for the same task.
        inputs (`list[str]` or `dict[str, str]`):
            The inputs that will be fed to the tools. For this testing environment, only strings are accepted as
            values. Pass along a dictionary when you want to specify the values of each inputs, or just the list of
            inputs expected (the value used will be `<<input_name>>` in this case).
        answer (`str` or `list[str`]):
            The theoretical answer (or list of possible valid answers) to the problem, as code.
    """

    def __init__(self, task, inputs, answer):
        self.task = task
        self.inputs = inputs
        self.answer = answer


### The list of problems the agent will be evaluated on.
EVALUATION_TASKS = [
    Problem(
        task=[
            "Is the following `text` (in Spanish) positive or negative?",
            "Is the text in the variable `text` (in Spanish) positive or negative?",
            "Translate the following `text` from Spanish to English then tell me if its positive or negative.",
        ],
        inputs=["text"],
        answer="""text_classifier(translator(text, src_lang="Spanish", tgt_lang="English"), labels=["positive", "negative"])""",
    ),
    Problem(
        task=[
            "Tell me out loud what the `image` contains.",
            "Describe the following `image` out loud.",
            "Find what is in the picture stored in `image` then read it out loud.",
        ],
        inputs=["image"],
        answer=[
            "text_reader(image_captioner(image))",
            "text_reader(image_qa(image, question='What is in the image?'))",
        ],
    ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`. Then transform it according to the text in `prompt`.",
            "Use the following `text_input` to generate an image, then transform it by using the text in `prompt`.",
        ],
        inputs=["text_input", "prompt"],
        answer="image_transformer(image_generator(text_input), prompt)",
    ),
    Problem(
        task=[
            "Download the content of `url`, summarize it then generate an image from its content.",
            "Use a summary of the web page at `url` to generate an image.",
            "Summarize the content of the web page at `url`, and use the result to generate an image.",
        ],
        inputs=["url"],
        answer="image_generator(summarizer(text_downloader(url)))",
    ),
    Problem(
        task=[
            "Transform the following `image` using the prompt in `text`. The prompt is in Spanish.",
            "Use the text prompt in `text` (in Spanish) to transform the following `image`.",
            "Translate the `text` from Spanish to English then use it to transform the picture in `image`.",
        ],
        inputs=["text", "image"],
        answer="image_transformer(image, translator(text, src_lang='Spanish', tgt_lang='English'))",
    ),
    Problem(
        task=[
            "Download the content of `url`, summarize it then read it out loud to me.",
            "Read me a summary of the web page at `url`.",
        ],
        inputs=["url"],
        answer="text_reader(summarizer(text_downloader(url)))",
    ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`.",
        ],
        inputs=["text_input"],
        answer="image_generator(text_input)",
    ),
    Problem(
        task=[
            "Replace the beaver in the `image` by the `prompt`.",
            "Transform the `image` so that it contains the `prompt`.",
            "Use `prompt` to transform this `image`.",
        ],
        inputs=["image", "prompt"],
        answer="image_transformer(image, prompt)",
    ),
    Problem(
        task=[
            "Provide me the summary of the `text`, then read it to me before transcribing it and translating it in French.",
            "Summarize `text`, read it out loud then transcribe the audio and translate it in French.",
            "Read me a summary of the `text` out loud. Transcribe this and translate it in French.",
        ],
        inputs=["text"],
        answer="translator(transcriber(text_reader(summarizer(text))), src_lang='English', tgt_lang='French')",
    ),
    Problem(
        task=["Generate a video of the `prompt`", "Animate a `prompt`", "Make me a short video using `prompt`."],
        inputs={"prompt": "A lobster swimming"},
        answer="video_generator('A lobster swimming')",
    ),
    Problem(
        task=[
            "Download the following file `url`, summarize it in a few words and generate a video from it."
            "Fetch the file at this `url`, summarize it, and create an animation out of it."
        ],
        inputs=["url"],
        answer="video_generator(summarizer(text_downloader(url)))",
    ),
]


def get_theoretical_tools(agent_answer, theoretical_answer, code_answer):
    if not isinstance(theoretical_answer, list):
        return {name for name in TEST_TOOLS if name in code_answer}

    if isinstance(agent_answer, dict):
        for one_answer, one_code in zip(theoretical_answer, code_answer):
            if one_answer in agent_answer.values():
                return {name for name in TEST_TOOLS if name in one_code}

    for one_answer, one_code in zip(theoretical_answer, code_answer):
        if agent_answer == one_answer:
            return {name for name in TEST_TOOLS if name in one_code}

    return {name for name in TEST_TOOLS if name in code_answer[0]}


def evaluate_code(code, inputs=None, state=None, verbose=False, return_interpretor_error=False):
    tools = BASE_PYTHON_TOOLS.copy()
    for name, tool in TEST_TOOLS.items():
        if name not in code:
            continue
        tools[name] = tool

    if isinstance(inputs, dict):
        inputs = inputs.copy()
    elif inputs is not None:
        inputs = {inp: f"<<{inp}>>" for inp in inputs}

    if state is not None:
        state.update(inputs)
    else:
        state = inputs

    try:
        return evaluate(code, tools, state)
    except InterpreterError as e:
        return str(e)
    except Exception as e:
        if verbose:
            print(e)
        return None


def score_code(agent_answer, theoretical_answer, verbose: bool = False):
    if verbose:
        print(agent_answer, theoretical_answer)
    theoretical_answer = theoretical_answer if isinstance(theoretical_answer, list) else [theoretical_answer]

    if agent_answer in theoretical_answer:
        if verbose:
            print("Perfect!")
        return 1
    elif isinstance(agent_answer, dict) and any(v in theoretical_answer for v in agent_answer.values()):
        if verbose:
            print("Almsot perfect, result in state!")
        return 0.75
    else:
        if verbose:
            print("Result is not the right one but code executed.")
        return 0.3


def evaluate_one_result(code, agent_answer, theoretical_answer, answer, verbose=False):
    tools_in_code = {name for name in TEST_TOOLS if f"`{name}`" in code}
    theoretical_tools = get_theoretical_tools(agent_answer, theoretical_answer, answer)
    if tools_in_code == theoretical_tools:
        tool_selection_score = 1.0
        tool_selection_errors = None
    else:
        missing_tools = len(theoretical_tools - tools_in_code)
        unexpected_tools = len(tools_in_code - theoretical_tools)
        tool_selection_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)

        tool_selection_errors = {
            "selected_tools": tools_in_code,
            "theoretical_tools": theoretical_tools,
        }

    tools_in_code = {name for name in TEST_TOOLS if name in code}
    if tools_in_code == theoretical_tools:
        tool_used_score = 1.0
        tool_used_errors = None
    else:
        missing_tools = len(theoretical_tools - tools_in_code)
        unexpected_tools = len(tools_in_code - theoretical_tools)
        tool_used_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)

        tool_used_errors = {
            "selected_tools": tools_in_code,
            "theoretical_tools": theoretical_tools,
        }

    score = score_code(agent_answer, theoretical_answer, verbose=verbose)
    if score < 1.0:
        code_errors = {
            "code_produced": code,
            "evaluation": agent_answer,
            "theoretical_answer": theoretical_answer,
        }
    else:
        code_errors = None

    return (tool_selection_score, tool_used_score, score), (tool_selection_errors, tool_used_errors, code_errors)


def evaluate_agent(agent, batch_size=8, verbose=False, return_errors=False):
    """
    Evaluates a new agent on all `EVALUATION_TASKS`.

    Example:

    ```py
    agent = NewOpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    bads = new_evaluate_agent(agent)
    for bad in bads:
        print(bad)
    ```
    """
    # Sanity check
    agent_tools = set(agent.toolbox.keys())
    if agent_tools != set(TEST_TOOLS):
        missing_tools = set(TEST_TOOLS) - agent_tools
        unexpected_tools = set(agent_tools) - TEST_TOOLS
        raise ValueError(
            f"Fix the test tools in the evaluate_agent module. Tools mising: {missing_tools}. Extra tools: {unexpected_tools}."
        )

    eval_tasks = []
    eval_idx = []
    for idx, pb in enumerate(EVALUATION_TASKS):
        if isinstance(pb.task, list):
            eval_tasks.extend(pb.task)
            eval_idx.extend([idx] * len(pb.task))
        else:
            eval_tasks.append(pb.task)
            eval_idx.append(idx)

    tool_selection_score = 0
    tool_used_score = 0
    code_score = 0

    if return_errors:
        tool_selection_errors = {}
        tool_used_errors = {}
        code_errors = {}

    for start_idx in range(0, len(eval_tasks), batch_size):
        end_idx = min(start_idx + batch_size, len(eval_tasks))
        batch_tasks = eval_tasks[start_idx:end_idx]

        results = [agent.run(task, return_generated_code=True) for task in batch_tasks]

        for idx, result in enumerate(results):
            problem = EVALUATION_TASKS[eval_idx[start_idx + idx]]
            if verbose:
                print(f"====Task {start_idx + idx}====\n{batch_tasks[idx]}\n")
            code = agent.extract_action(result, split_token="Answer:")

            # Evaluate agent answer and code answer
            agent_answer = evaluate_code(code, problem.inputs, verbose=verbose)
            if isinstance(problem.answer, list):
                theoretical_answer = [evaluate_code(answer, problem.inputs) for answer in problem.answer]
            else:
                theoretical_answer = evaluate_code(problem.answer, problem.inputs)

            scores, errors = evaluate_one_result(
                code, agent_answer, theoretical_answer, problem.answer, verbose=verbose
            )

            tool_selection_score += scores[0]
            tool_used_score += scores[1]
            code_score += scores[2]

            if return_errors:
                if errors[0] is not None:
                    tool_selection_errors[batch_tasks[idx]] = errors[0]
                if errors[1] is not None:
                    tool_used_errors[batch_tasks[idx]] = errors[1]
                if errors[2] is not None:
                    code_errors[batch_tasks[idx]] = errors[2]

    scores = {
        "tool selection score": 100 * (tool_selection_score / len(eval_tasks)),
        "tool used score": 100 * (tool_used_score / len(eval_tasks)),
        "code score": 100 * (code_score / len(eval_tasks)),
    }

    if return_errors:
        return scores, tool_selection_errors, tool_used_errors, code_errors
    else:
        return scores

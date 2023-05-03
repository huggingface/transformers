import random

from .agents import BASE_PYTHON_TOOLS, OUR_TOOLS, clean_code_for_run
from .python_interpreter import evaluate


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


def image_segmentor(image):
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    return f"This is the segmentation mask of {image}."


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


def search_engine(query):
    return f'This is result of the search of "{query}" on a search engine.'


_db = {}


def database_reader(key):
    global _db
    return f"db_read({_db[key]})"


def database_writer(key, value):
    global _db
    _db[key] = value
    return "200"


def video_generator(prompt, seconds=2):
    return f"A video of {prompt}"


ALL_TOOLS = {
    "text_classifier": classifier,
    "translator": translator,
    "text_reader": speaker,
    "summarizer": summarizer,
    "transcriber": transcriber,
    "image_generator": image_generator,
    "image_segmentor": image_segmentor,
    "image_captioner": image_captioner,
    "image_transformer": image_transformer,
    "test_qa": question_answerer,
    "text_downloader": text_downloader,
    "search_engine": search_engine,
    "database_reader": database_reader,
    "database_writer": database_writer,
    "table_qa": None,
    "image_qa": image_qa,
    "video_generator": video_generator,
}


def sample_num_tools(max_n):
    """
    Samples a number between 0 and max_n.

    0, 1 and 2 each have a 25% probability, the last 25% are split among the rest, geometrically decreasing.
    """
    rand_number = random.random()
    rand_number = 1 + rand_number * (2 ** (max_n) - 1)
    i = 0
    while 2**i < rand_number:
        i += 1
    result = max_n - i + 1
    if result == 1:
        rand_number = random.random()
        return 0 if rand_number < 0.5 else 1
    return result


class Problem:
    """
    A class regrouping all the information to solve a problem on which we will evaluate agents.

    Args:
        task (`str` ou `list[str]`):
            One or several descriptions of the task to perform. If a list is passed, at each random trial generated for
            this problem, one of the element of the list is selected (so the list should contain variations on the
            phrasing, but for the same task).
        minimum_tools (`list[Tool]`):
            The list of tools necessary to solve the problem. At each random trial, some random tools will be added to
            this list.
        inputs (`dict[str, str]`):
            The inputs that will be fed to the tools. For this testing environment, only strings are accepted as
            values.
        answer:
            The theoretical answer (or list of possible answers) to the problem.
    """

    def __init__(self, task, minimum_tools, inputs, answer, excluded_tools=[]):
        self.task = task
        self.minimum_tools = minimum_tools
        self.excluded_tools = excluded_tools
        self.inputs = inputs
        self.answer = answer

    def random_trial(self, max_new_tools=4):
        """
        Generates a random variation of this problem by selecting one of the phrasings of the task and adding new tools
        randomly.
        """
        all_tools = [tool for tool in ALL_TOOLS.values() if tool not in self.excluded_tools and tool is not None]
        num_new_tools = sample_num_tools(max_new_tools)
        num_new_tools = min(num_new_tools, len(all_tools) - len(self.minimum_tools))
        new_tools = list(set(all_tools) - set(self.minimum_tools))
        random.shuffle(new_tools)
        result = self.minimum_tools.copy() + new_tools[:num_new_tools]
        random.shuffle(result)

        if isinstance(self.task, list):
            random_idx = random.randint(0, len(self.task) - 1)
            task = self.task[random_idx]
        else:
            task = self.task

        return task, result

    def base_trial(self):
        task = self.task[0] if isinstance(self.task, list) else self.task
        return task, self.minimum_tools


### The list of problems the agent will be evaluated on.
EVALUATION_TASKS = [
    Problem(
        task=[
            "Is the following `text` (in Spanish) positive or negative?",
            "Is the text in the variable `text` (in Spanish) positive or negative?",
            "Translate the following `text` from Spanish to English then tell me if its positive or negative.",
        ],
        minimum_tools=[classifier, translator],
        inputs={"text": "C'est une review positive."},
        answer=classifier(
            translator("C'est une review positive.", src_lang="Spanish", tgt_lang="English"),
            labels=["positive", "negative"],
        ),
    ),
    Problem(
        task=[
            "Tell me out loud what the `image` contains.",
            "Describe the following `image` out loud.",
            "Determine what is in the pictured stored in `image` then read it out loud.",
        ],
        minimum_tools=[image_captioner, speaker],
        inputs=["image"],
        answer=[
            speaker(image_captioner("<<image>>")),
            speaker(image_qa("<<image>>", question="What is in the image?")),
        ],
    ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`. Then transform it according to the text in `prompt`.",
            "Use the following `text_input` to generate an image, then transform it by using the text in `prompt`.",
        ],
        minimum_tools=[image_generator, image_transformer],
        inputs=["text_input", "prompt"],
        answer=image_transformer(image_generator("<<text_input>>"), "<<prompt>>"),
    ),
    Problem(
        task=[
            "Download the content of `url`, summarize it then generate an image from its content.",
            "Use a summary of the web page at `url` to generate an image.",
            "Summarize the content of the web page at `url`, and use the result to generate an image.",
        ],
        minimum_tools=[summarizer, text_downloader, image_generator],
        inputs=["url"],
        answer=image_generator(summarizer(text_downloader("<<url>>"))),
    ),
    Problem(
        task=[
            "Transform the following `image` using the prompt in `text`. The prompt is in Spanish.",
            "Use the text prompt in `text` (in Spanish) to transform the following `image`.",
            "Translate the `text` from Spanish to English then use it to transform the picture in `image`.",
        ],
        minimum_tools=[translator, image_transformer],
        inputs=["text", "image"],
        answer=image_transformer("<<image>>", translator("<<text>>", src_lang="Spanish", tgt_lang="English")),
    ),
    Problem(
        task=[
            "Download the content of `url`, summarize it then read it out loud to me.",
            "Read me a summary of the web page at `url`.",
        ],
        minimum_tools=[summarizer, text_downloader, speaker],
        inputs=["url"],
        answer=speaker(summarizer(text_downloader("<<url>>"))),
    ),
    # Problem(
    #    task=[
    #        "What is the price of a Redbull can in America, converted to Korean won using last week's rate?",
    #    ],
    #    minimum_tools=[search_engine, calculator],
    #    inputs=[],
    #    answer=[
    #        calculator(
    #            "*", search_engine("Redbull can price in America"), search_engine("Last week's USD to KRW rate")
    #        ),
    #    ],
    # ),
    # Problem(
    #    task=[
    #        "What is the postal code of L'Arc de Triomphe multiplied by 3?",
    #    ],
    #    minimum_tools=[search_engine, calculator],
    #    inputs=[],
    #    answer=[
    #        calculator("*", search_engine("Postal code L'Arc de Triomphe"), "3"),
    #    ],
    # ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`, and write it into the database using the original text as key. Show me the value of the written record once this is done.",
        ],
        minimum_tools=[database_writer, database_reader, image_generator],
        inputs=["text_input"],
        answer=f"db_read({image_generator('<<text_input>>')})",
    ),
    Problem(
        task=[
            "Replace the beaver in the `image` by the `prompt`.",
            "Transform the `image` so that it contains the `prompt`.",
            "Use `prompt` to transform this `image`.",
        ],
        minimum_tools=[image_transformer],
        inputs={"image": "<image object>", "prompt": "capybara"},
        answer=image_transformer("<image object>", "capybara"),
    ),
    Problem(
        task=[
            "Provide me the summary of the `text`, then read it to me before transcribing it and translating it in French.",
            "Summarize `text`, read it out loud then transcribe the audio and translate it in French.",
            "Read me a summary of the the `text` out loud. Transcribe this and translate it in French.",
        ],
        minimum_tools=[summarizer, speaker, transcriber, translator],
        inputs={"text": "I'm a text"},
        answer=translator(transcriber(speaker(summarizer("I'm a text"))), src_lang="English", tgt_lang="French"),
    ),
    Problem(
        task=["Generate a video of the `prompt`", "Animate a `prompt`", "Make me a short video using `prompt`."],
        minimum_tools=[video_generator],
        inputs={"prompt": "A lobster swimming"},
        answer=video_generator("A lobster swimming"),
    ),
    Problem(
        task=[
            "Download the following file `url`, summarize it in a few words and generate a video from it."
            "Fetch the file at this `url`, summarize it, and create an animation out of it."
        ],
        minimum_tools=[text_downloader, summarizer, video_generator],
        inputs={"url": "url"},
        answer=video_generator(summarizer(text_downloader("url"))),
    ),
]


def evaluate_code(code, inputs, verbose=False):
    tools = BASE_PYTHON_TOOLS.copy()
    for name, tool in ALL_TOOLS.items():
        if name not in code:
            continue
        tools[name] = tool

    try:
        if isinstance(inputs, dict):
            inputs = inputs.copy()
        else:
            inputs = {inp: f"<<{inp}>>" for inp in inputs}
        return evaluate(code, tools, inputs)
    except Exception as e:
        if verbose:
            print(e)
        return None


def get_score(agent_answer, theoretical_answer, verbose: bool = False):
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

        results = agent.generate_code(batch_tasks)

        for idx, result in enumerate(results):
            problem = EVALUATION_TASKS[eval_idx[start_idx + idx]]
            if verbose:
                print(f"====Task {start_idx + idx}====\n{batch_tasks[idx]}\n")
            explanation, code = clean_code_for_run(result)

            tools_in_explanation = {name for name in OUR_TOOLS if f"`{name}`" in explanation}
            minimum_tools = {name for name, tool in ALL_TOOLS.items() if tool in problem.minimum_tools}
            if tools_in_explanation == minimum_tools:
                tool_selection_score += 1.0
            else:
                missing_tools = len(minimum_tools - tools_in_explanation)
                unexpected_tools = len(tools_in_explanation - minimum_tools)
                tool_selection_score += max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)
                if return_errors:
                    tool_selection_errors[batch_tasks[idx]] = {
                        "selected_tools": tools_in_explanation,
                        "theoretical_tools": minimum_tools,
                    }

            tools_in_code = {name for name in OUR_TOOLS if name in code}
            if tools_in_code == minimum_tools:
                tool_used_score += 1.0
            else:
                missing_tools = len(minimum_tools - tools_in_code)
                unexpected_tools = len(tools_in_code - minimum_tools)
                tool_used_score += max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)
                if return_errors:
                    tool_used_errors[batch_tasks[idx]] = {
                        "selected_tools": tools_in_code,
                        "theoretical_tools": minimum_tools,
                    }

            agent_answer = evaluate_code(code, problem.inputs, verbose=verbose)
            score = get_score(agent_answer, problem.answer, verbose=verbose)
            if return_errors and score < 1.0:
                code_errors[batch_tasks[idx]] = {
                    "code_produced": code,
                    "evaluation": agent_answer,
                    "theoretical_answer": problem.answer,
                }
            code_score += score

    scores = {
        "tool selection score": 100 * (tool_selection_score / len(eval_tasks)),
        "tool used score": 100 * (tool_used_score / len(eval_tasks)),
        "code score": 100 * (code_score / len(eval_tasks)),
    }

    if return_errors:
        return scores, tool_selection_errors, tool_used_errors, code_errors
    else:
        return scores

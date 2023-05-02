import random

from .agents import BASE_PYTHON_TOOLS
from .generative_question_answering import (
    GENERATIVE_QUESTION_ANSWERING_DESCRIPTION as generative_question_answering_description,
)
from .image_captioning import IMAGE_CAPTIONING_DESCRIPTION as image_captioning_description
from .image_segmentation import IMAGE_SEGMENTATION_DESCRIPTION as image_segmentation_description
from .image_transformation import IMAGE_TRANSFORMATION_DESCRIPTION as image_transformation_description
from .python_interpreter import InterpretorError, evaluate
from .speech_to_text import SPEECH_TO_TEXT_DESCRIPTION as speech_to_text_description
from .text_classification import TEXT_CLASSIFIER_DESCRIPTION as text_classifier_description
from .text_to_image import TEXT_TO_IMAGE_DESCRIPTION as text_to_image_description
from .text_to_speech import TEXT_TO_SPEECH_DESCRIPTION as text_to_speech_description
from .translation import TRANSLATION_DESCRIPTION as translation_description



def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        return func

    return inner


### Fake tools for test
@add_description(text_classifier_description)
def classifier(text, labels):
    for label in labels:
        if label in text:
            return {"label": label, "score": 0.99}
    
    return {"label": labels[0], "score": 0.5}


@add_description(translation_description)
def translator(text, src_lang, tgt_lang):
    return f"This is the translation of {text} from {src_lang} to {tgt_lang}."


@add_description(text_to_speech_description)
def speaker(text):
    return f"This is actually a sound reading {text}."


@add_description(speech_to_text_description)
def transcriber(audio):
    if "sound" not in audio:
        raise ValueError(f"`audio` ({audio}) is not a sound.")
    return f"This is the transcribed text from {audio}."


@add_description(text_to_image_description)
def image_generator(text):
    return f"This is actually an image representing {text}."


@add_description(image_segmentation_description)
def image_segmentor(image):
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    return f"This is the segmentation mask of {image}."


@add_description(image_captioning_description)
def image_captioner(image):
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    return f"This is a description of {image}."


@add_description(image_transformation_description)
def image_transformer(image, prompt):
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    return f"This is a transformation of {image} according to {prompt}."


@add_description(generative_question_answering_description)
def question_answerer(text, question):
    return f"This is the answer to {question} from {text}."


@add_description(
    "This is a tool that downloads the context of an url and returns the text inside. It takes an input named `url`, which is the url to download, and returns the text."
)
def text_downloader(url):
    return f"This is the content of {url}."


@add_description(
    "This is a tool that summarizes texts. It takes an input named `text`, which should be the text to summarize, and returns the summary."
)
def summarizer(text):
    return f"This is a summary of {text}."


@add_description(
    "This is a tool that performs a calculation between two numbers. It takes an input named `operation`, as well as two strings `a` and `b` representing the two numbers. The `operation` can take four values: -, +, /, *. It returns the result of that operation when applied to the two numbers."
)
def calculator(operation, a, b):
    return f"This is the solution of ({a} {operation} {b})."


@add_description(
    "This is a tool that performs a search on a search engine. It takes an input `query` and returns the first result of the search. It can be used for many searches, ranging from item prices, conversion rates to monuments location, among many other searches."
)
def search_engine(query):
    return f'This is result of the search of "{query}" on a search engine.'


_db = {}


@add_description(
    "This is a tool that reads a record in a key-value database. It takes an input `key` and returns the value in the database."
)
def database_reader(key):
    global _db
    return f"db_read({_db[key]})"


@add_description(
    "This is a tool that writes a record in a key-value database. It takes an input `key` indicating the location in the database, as well as an input `value` which will populate the database. It returns the HTTP code indicating success or failure of the write operation."
)
def database_writer(key, value):
    global _db
    _db[key] = value
    return "200"

@add_description(
    "This is a tool that generates a video (or animation) according to a `prompt`. The `prompt` is a text-based definition of the video to be generated. The returned value is a video object."
)
def video_generator(prompt):
    return f"A video of {prompt}"


ALL_TOOLS = [
    classifier,
    translator,
    speaker,
    summarizer,
    transcriber,
    image_generator,
    image_segmentor,
    image_captioner,
    image_transformer,
    question_answerer,
    text_downloader,
    calculator,
    search_engine,
    database_reader,
    database_writer,
    video_generator,
]


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
        all_tools = [tool for tool in ALL_TOOLS if tool not in self.excluded_tools]
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
        ],
        minimum_tools=[classifier, translator],
        inputs={"text": "C'est une review positive."},
        answer=[
            classifier(translator("C'est une review positive.", src_lang='Spanish', tgt_lang='English'), labels=['positive', 'negative']),
            classifier(translator("C'est une review positive.", src_lang='Spanish', tgt_lang='English'), labels=['positive', 'negative'])["label"],
        ],
    ),
    Problem(
        task=[
            "Tell me out loud what the `image` contains.",
            "Describe the following `image` out loud.",
            "Determine what is in the pictured stored in `image` then read it out loud.",
        ],
        minimum_tools=[image_captioner, speaker],
        inputs=["image"],
        answer=speaker(image_captioner("<<image>>")),
    ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`. Then transform it according to the text in `prompt`.",
        ],
        minimum_tools=[image_generator, image_transformer],
        inputs=["text_input", "prompt"],
        answer=image_transformer(image_generator("<<text_input>>"), "<<prompt>>"),
    ),
    Problem(
        task=[
            "Download the content of `url`, summarize it then generate an image from its content.",
            "Use a summary of the web page at `url` as a prompt to generate an image.",
        ],
        minimum_tools=[summarizer, text_downloader, image_generator],
        inputs=["url"],
        answer=image_generator(summarizer(text_downloader("<<url>>"))),
    ),
    Problem(
        task=[
            "Transform the following `image` using the prompt in `text. The prompt is in Spanish.",
            "Use the text prompt in `text` (in Spanish) to transform the following `image`.",
        ],
        minimum_tools=[translator, image_transformer],
        inputs=["text", "image"],
        answer=image_transformer("<<image>>", translator("<<text>>", src_lang='Spanish', tgt_lang='English')),
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
    Problem(
        task=[
            "What is the price of a Redbull can in America, converted to Korean won using last week's rate?",
        ],
        minimum_tools=[search_engine, calculator],
        inputs=[],
        answer=[
            calculator(
                "*", search_engine("Redbull can price in America"), search_engine("Last week's USD to KRW rate")
            ),
        ],
    ),
    Problem(
        task=[
            "What is the postal code of L'Arc de Triomphe multiplied by 3?",
        ],
        minimum_tools=[search_engine, calculator],
        inputs=[],
        answer=[
            calculator("*", search_engine("Postal code L'Arc de Triomphe"), "3"),
        ],
    ),
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
            "Showcase the `prompt` in the `image`.",
        ],
        minimum_tools=[image_transformer],
        inputs={"image": '<image object>', 'prompt': "capybara"},
        answer=image_transformer('<image object>', 'capybara')
    ),
    Problem(
        task=[
            "Provide me the summary of the `text`, then read it to me before transcribing it and translating it in French.",
        ],
        minimum_tools=[summarizer, speaker, transcriber, translator],
        inputs={"text": "I'm a text"},
        answer=translator(transcriber(speaker(summarizer("I'm a text"))), src_lang="English", tgt_lang="French")
    ),
    Problem(
        task=[
            "Generate a video of the `prompt`",
            "Animate a `prompt`"
        ],
        minimum_tools=[video_generator],
        inputs={"prompt": "A lobster swimming"},
        answer=video_generator('A lobster swimming')
    ),
    Problem(
        task=[
            "Download the following file `url`, summarize it in a few words and generate a video from it."
            "Fetch the file at this `url`, summarize it, and create an animation out of it."
        ],
        minimum_tools=[text_downloader, summarizer, video_generator],
        inputs={"url": "url"},
        answer=video_generator(summarizer(text_downloader("url")))
    ),
]


def get_score(problem, code, tools, verbose: bool = False):
    if verbose:
        print(code + "\n")
    all_tools = BASE_PYTHON_TOOLS.copy()
    all_tools.update({f"tool_{i}": t for i, t in enumerate(tools)})
    try:
        if isinstance(problem.inputs, dict):
            inputs = problem.inputs.copy()
        else:
            inputs = {inp: f"<<{inp}>>" for inp in problem.inputs}
        agent_answer = evaluate(code, all_tools, inputs)
    except InterpretorError as e:
        # TODO see if we score errors differently.
        if verbose:
            print(e)
        return 0
    except Exception as e:
        if verbose:
            print(e)
        return 0

    if verbose:
        print(agent_answer, problem.answer)
    theoretical_answer = problem.answer if isinstance(problem.answer, list) else [problem.answer]

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


def base_evaluate_agent(agent, batch_size=8, verbose=False):
    """
    Mostly a consistency check that all problems can be solved by an agent. Will return the list

    Example:

    ```py
    agent = OpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    bads = base_evaluate_agent(agent)
    for bad in bads:
        print(bad)
    ```
    """
    bad_problems = []
    for start_idx in range(0, len(EVALUATION_TASKS), batch_size):
        end_idx = min(start_idx + batch_size, len(EVALUATION_TASKS))
        base_tasks = [pb.base_trial() for pb in EVALUATION_TASKS[start_idx:end_idx]]
        batch_tasks = [t[0] for t in base_tasks]
        batch_tools = [t[1] for t in base_tasks]

        results = agent.generate_code(batch_tasks, tools=batch_tools)

        for idx, result in enumerate(results):
            problem = EVALUATION_TASKS[start_idx + idx]
            if verbose:
                print(f"====Task {start_idx + idx}====\n{batch_tasks[idx]}\n")
            code = agent.clean_code(result)[0]
            score = get_score(problem, code, batch_tools[idx], verbose=verbose)

            if score != 1.0:
                summary = f"====Task {start_idx + idx}====\n{batch_tasks[idx]}\n\n{code}"
                bad_problems.append(summary)

    return bad_problems


def evaluate_agent(agent, total_batches=1, batch_size=8, max_new_tools=4, verbose=False):
    """
    Evaluates an agent on random variations of the problems in `EVALUATION_TASKS`. Will generate `total_batches x
    batch_size` variations for the evaluation.

    Returns a score between 0 and 100 (100 being a perfect score).

    Example:

    ```py
    agent = OpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    score = evaluate_agent(agent, total_batches=5)
    ```
    """
    score = 0
    for _ in range(total_batches):
        batch_tasks = []
        batch_tools = []
        batch_idx = []
        for _ in range(batch_size):
            random_idx = random.randint(0, len(EVALUATION_TASKS) - 1)
            task, tool = EVALUATION_TASKS[random_idx].random_trial(max_new_tools=max_new_tools)
            batch_tasks.append(task)
            batch_tools.append(tool)
            batch_idx.append(random_idx)

        results = agent.generate_code(batch_tasks, tools=batch_tools)

        for idx, result in enumerate(results):
            print(f"Running with tools: {[tool.__name__ for tool in batch_tools[idx]]}")
            problem = EVALUATION_TASKS[batch_idx[idx]]
            if verbose:
                print(f"====Task {idx}====\n{batch_tasks[idx]}\n")
            code = agent.clean_code(result)[0]
            score += get_score(problem, code, batch_tools[idx], verbose=verbose)

    return round(score * 100 / (batch_size * total_batches), 2)

import random

from .generative_question_answering import (
    GENERATIVE_QUESTION_ANSWERING_DESCRIPTION as generative_question_answering_description,
)
from .image_captioning import IMAGE_CAPTIONING_DESCRIPTION as image_captioning_description
from .image_segmentation import IMAGE_SEGMENTATION_DESCRIPTION as image_segmentation_description
from .image_transformation import IMAGE_TRANSFORMATION_DESCRIPTION as image_transformation_description
from .python_interpreter import InterpretorError, evaluate
from .speech_to_text import SPEECH_TO_TEXT_DESCRIPTION as speech_to_text_description
from .text_classification import TEXT_CLASSIFIER_DESCRIPTION
from .text_to_image import TEXT_TO_IMAGE_DESCRIPTION as text_to_image_description
from .text_to_speech import TEXT_TO_SPEECH_DESCRIPTION as text_to_speech_description
from .translation import TRANSLATION_DESCRIPTION


text_classifier_description = TEXT_CLASSIFIER_DESCRIPTION.replace("{n_labels}", "2").replace(
    "{labels}", "'positive', and 'negative'"
)
translation_description = TRANSLATION_DESCRIPTION.replace("{src_lang}", "Spanish").replace("{tgt_lang}", "English")


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
def classifier(text):
    if "positive" in text:
        return {"label": "positive", "score": 0.99}
    elif "negative" in text:
        return {"label": "negative", "score": 0.99}
    else:
        return {"label": "positive", "score": 0.5}


@add_description(translation_description)
def translator(text):
    return {"translated_text": f"This is the translation in English of {text}"}


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


ALL_TOOLS = [
    classifier,
    translator,
    speaker,
    transcriber,
    image_generator,
    image_segmentor,
    image_captioner,
    image_transformer,
    question_answerer,
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

    def __init__(self, task, minimum_tools, inputs, answer):
        self.task = task
        self.minimum_tools = minimum_tools
        self.inputs = inputs
        self.answer = answer

    def random_trial(self, max_new_tools=4):
        """
        Generates a random variation of this problem by selecting one of the phrasings of the task and adding new tools
        randomly.
        """
        num_new_tools = sample_num_tools(max_new_tools)
        num_new_tools = min(num_new_tools, len(ALL_TOOLS) - len(self.minimum_tools))
        new_tools = list(set(ALL_TOOLS) - set(self.minimum_tools))
        random.shuffle(new_tools)
        result = self.minimum_tools.copy() + new_tools[:num_new_tools]
        random.shuffle(result)

        if isinstance(self.task, list):
            random_idx = random.randint(0, len(self.task) - 1)
            task = self.task[random_idx]
        else:
            task = self.task

        return task, result


### The list of problems the agent will be evaluated on.
EVALUATION_TASKS = [
    Problem(
        task=[
            "Is the following `text` (in French) positive or negative?",
            "Is the text in the variable `text` (in French) positive or negative?",
        ],
        minimum_tools=[classifier, translator],
        inputs={"text": "Ce text est positive."},
        answer=[
            classifier(translator("Ce text est positive.")["translated_text"]),
            classifier(translator("Ce text est positive.")["translated_text"])["label"],
        ],
    ),
    Problem(
        task=[
            "Tell me out loud what the `image` contains.",
            "Describe the following `image` out loud.",
            "Determine what is in the pictured stored in `image` then read it out loud.",
        ],
        minimum_tools=[image_captioner, speaker],
        inputs={"image": "Ceci est une image."},
        answer=speaker(image_captioner("Ceci est une image.")),
    ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`. Then transform it according to the text in `prompt`.",
        ],
        minimum_tools=[image_generator, image_transformer],
        inputs={"text_input": "initial text", "prompt": "transformation prompt"},
        answer=image_transformer(image_generator("initial text"), "transformation prompt"),
    ),
]


def evaluate_agent(agent, total_batches=1, batch_size=8, max_new_tools=4, verbose=False):
    """
    Evaluates an agent on random variations of the problems in `EVALUATION_TASKS`. Will generate `total_batches x
    batch_size` variations for the evaluation.

    Returns a score between 0 and 100 (100 being a perfect score).

    Example:

    ```py
    agent = OpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    evaluate_agent(agent, total_batches=5)
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
            problem = EVALUATION_TASKS[batch_idx[idx]]
            if verbose:
                print(f"====Task {idx}====\n{batch_tasks[idx]}\n")
            code = agent.clean_code(result)[0]
            if verbose:
                print(code + "\n")
            all_tools = {"print": print}
            all_tools.update({f"tool_{i}": t for i, t in enumerate(batch_tools[idx])})
            try:
                agent_answer = evaluate(code, all_tools, problem.inputs.copy())
            except InterpretorError as e:
                # TODO see if we score errors differently.
                if verbose:
                    print(e)
                continue
            except Exception as e:
                if verbose:
                    print(e)
                continue

            if verbose:
                print(agent_answer, problem.answer)
            theoretical_answer = problem.answer if isinstance(problem.answer, list) else [problem.answer]

            if agent_answer in theoretical_answer:
                if verbose:
                    print("Perfect!")
                score += 1
            elif isinstance(agent_answer, dict) and any(v in theoretical_answer for v in agent_answer.values()):
                if verbose:
                    print("Almsot perfect, result in state!")
                score += 0.75
            else:
                if verbose:
                    print("Result is not the right one but code executed.")
                score += 0.3

    return round(score * 100 / (batch_size * total_batches), 2)

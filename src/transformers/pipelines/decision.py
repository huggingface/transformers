import functools
import inspect
from pathlib import Path
from typing import Any

from ..decision import DecisionConfig, DecisionTemplate, Question, label_token_ids
from ..utils import add_end_docstrings, is_torch_available
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    import torch


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class DecisionPipeline(Pipeline):
    """
    Decision pipeline: answers typed questions (`choice`, `score`, `noul`) about a `state` with a probability for every
    option, from one forward pass per prompt (the System One API of Jev-like models). The decision config (see
    [`DecisionConfig`]) and the decision head (`decision_head.json`) live next to the model or in the repo given as
    `decision_config`. The head scores the options from the model it is given; without a head, the options are scored
    by the next-token logits of their labels, so any chat model can answer.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> decide = pipeline("decision", model="Qwen/Qwen2.5-0.5B-Instruct")
    >>> answers = decide(
    ...     {
    ...         "state": "I was charged twice for my order and want a refund.",
    ...         "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    ...     }
    ... )
    >>> answers["refund"]["type"]
    'noul'
    >>> decide("Where is my parcel?")["refund"]["type"]  # a bare state is asked the last questions
    'noul'
    ```

    Questions can also be set up front, with `pipeline("decision", model=..., questions={...})`.

    Call arguments:
        inputs (`dict`, `str` or `list`):
            `{"state": ..., "questions": {name: {"type": ..., "instructions": ..., "criteria": ...}}}` or its JSON, with
            `type` one of `choice` (`criteria` maps options to descriptions), `score` (`criteria` lists the levels,
            lowest first) or `noul` (optional `criteria` describe `true` and `false`). Any other input is a bare
            `state`, asked the current questions: those of the last request, or the ones set up front. Requests can
            carry `assets`, as in the System One API.
        questions (`dict`, *optional*):
            Questions to ask the input, as a bare `state`. They become the current questions.
        assets (`dict`, *optional*):
            Images keyed by a placeholder: wherever the placeholder appears in a string of the state, the instructions
            or the criteria, the model sees the image. An image is a System One asset (`{"type": "image", "format":
            "png", "data_base64": ...}`), a PIL image, a path or a URL. Needs a model with a processor that handles
            images.

    Returns `dict` (or a list of `dict`): the answers keyed by question name, in the System One format.
    """

    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def __init__(
        self,
        *args,
        decision_config: DecisionConfig | str | None = None,
        questions: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        repo = decision_config if isinstance(decision_config, (str, Path)) else self.model.name_or_path
        if not isinstance(decision_config, DecisionConfig):
            try:
                decision_config = DecisionConfig.from_pretrained(repo)
            except OSError:
                decision_config = DecisionConfig()
        self.template = DecisionTemplate(self.processor or self.tokenizer, decision_config, questions)
        # A decision model (`XxxForDecision`) scores the options itself, any other model by the logits of their labels
        self.reads_marks = "marks" in inspect.signature(self.model.forward).parameters
        if decision_config.adapter:  # a LoRA in a subfolder of the repo
            self.model.load_adapter(repo, adapter_kwargs={"subfolder": decision_config.adapter})
        self.logits_processors = {
            kind: decision_config.get_logits_processor(kind) for kind in Question.questions_types
        }
        self.label_token_ids = functools.cache(functools.partial(label_token_ids, self.tokenizer))

    def _sanitize_parameters(self, questions: dict[str, Any] | None = None, assets: dict[str, Any] | None = None):
        preprocess_params = {key: value for key, value in (("questions", questions), ("assets", assets)) if value}
        return preprocess_params, {}, {}

    def preprocess(self, inputs, questions: dict[str, Any] | None = None, assets: dict[str, Any] | None = None):
        prompts = self.template(inputs if questions is None else {"state": inputs, "questions": questions}, assets)
        return {"prompts": prompts, "inputs": [prompt.model_inputs() for prompt in prompts]}

    def _forward(self, model_inputs):
        # TODO: batch / continious batching
        scores = []
        for prompt, inputs in zip(model_inputs["prompts"], model_inputs["inputs"]):
            if self.reads_marks:
                scores.append(self.model(**inputs).split(inputs["num_options"][0].tolist()))
            else:  # the logits of the labels before each mark
                marks = inputs.pop("marks")[0]
                inputs = {
                    key: value for key, value in inputs.items() if key not in ("num_options", "question_type_ids")
                }
                logits = self.model(**inputs, use_cache=False, logits_to_keep=marks - 1).logits[0]
                labels = [self.label_token_ids(tuple(question.labels)) for question in prompt.questions]
                scores.append([logits[index, label_ids] for index, label_ids in enumerate(labels)])
        return {"prompts": model_inputs["prompts"], "scores": scores}

    def postprocess(self, model_outputs):
        answers = {}
        for prompt, prompt_scores in zip(model_outputs["prompts"], model_outputs["scores"]):
            # the tokens after the last mark, where the template writes the state
            input_ids = torch.tensor(prompt.input_ids[prompt.marks[-1] :])
            for question, scores in zip(prompt.questions, prompt_scores):
                scores = self.logits_processors[question.type](input_ids, scores.float())
                answers[question.name] = question.answer(scores.softmax(-1).tolist())
        return answers
